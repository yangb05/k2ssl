import logging
from tqdm import tqdm
import numpy as np
import argparse
from kaldiio import WriteHelper, load_mat
import torch
import joblib
from wavlm_model import WavLM, WavLMConfig
import soundfile
from librosa import resample
logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavscp", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--kmeans", type=str, required=True)
    # speed perturbation
    parser.add_argument("--speed-ratio", type=float, default=1.0)
    # parallel
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    # load model
    checkpoint = torch.load(args.model)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    if not args.no_cuda:
        model = model.cuda()
    model.eval()

    # load kmeans
    kmeans_model = joblib.load(open(args.kmeans, "rb"))
    kmeans_model.verbose = False
    
    # load wavscp
    wav_uttid_paths = []
    with open(args.wavscp, "r") as reader:
        while True:
            line = reader.readline().strip()
            if line == "": break
            uttid, wav_path = line.split(" ", maxsplit=1)
            wav_uttid_paths.append([uttid, wav_path])
    logging.info("will extract feature for {} utterances".format(len(wav_uttid_paths)))

    # speed perturbation
    augment_fn = None
    if args.speed_ratio != 1.0:
        from lhotse.augmentation.torchaudio import SoxEffectTransform
        augment_fn = SoxEffectTransform(effects=[
            ["speed", args.speed_ratio],
            ["rate", 16000]
        ])

    # decoding
    wspec = "ark,scp:{}/feats.ark,{}/feats.scp".format(args.outdir, args.outdir)
    logging.info("will write pred to {}".format(wspec))
    with WriteHelper(wspec) as writer:
        for uttid, wav_path in tqdm(wav_uttid_paths):
            # prefer using soundfile read function
            if ".ark:" in wav_path:
                sample_rate, audio = load_mat(wav_path)
                if audio.dtype == np.int16:
                    logging.info("audio format is int16, norm to float32")    
                    audio = audio.astype(np.float32)
                    audio /= (1 << (16-1))
            else:
                audio, sample_rate = soundfile.read(wav_path)

            # automatic resample
            if sample_rate != 16000:
                logging.info("resample {} to 16khz".format(uttid))
                audio = resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            # assert sample_rate == 16000, "sampling rate should be 16khz"

            if audio.shape[0] / 16000 < 0.5 or audio.shape[0] / 16000 > 30:
                logging.info(f"skip {uttid}, duration: {audio.shape[0] / 16000}")
                continue
            audio = torch.from_numpy(audio).float().unsqueeze(0)

            # apply speed perturbation
            if augment_fn != None:
                num_audio_samples = audio.shape[1] 
                # for speed perturb ratio < 1, we add extra padding
                if args.speed_ratio < 1:
                    audio = torch.nn.functional.pad(audio, (0, int(num_audio_samples/args.speed_ratio-num_audio_samples)))
                # perform augment
                audio = augment_fn(audio, 16000)
                # for speed perturb ratio > 1, we drop the padding
                if args.speed_ratio > 1:
                    audio = audio[:, :int(num_audio_samples/args.speed_ratio)]

            if not args.no_cuda:
                audio = audio.cuda()
            if cfg.normalize:
                audio = torch.nn.functional.layer_norm(audio , audio.shape)
            
            with torch.no_grad():
                # rep = model.extract_features(audio)[0]
                rep, layer_results = model.extract_features(audio, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]  # [(1, l, 1024)]
                rep = layer_reps[args.layer]

            rep = rep[0].detach().cpu().numpy() # (l, 1024)

            pred = kmeans_model.predict(rep)
            pred = np.expand_dims(pred, axis=1).astype(np.float32)

            writer[uttid] = pred


if __name__ == "__main__":
    args = get_args()
    logging.info(str(args))
    if args.rank is not None and not args.no_cuda:
        num_gpus = torch.cuda.device_count()
        assert args.rank <= num_gpus, "invalid rank: {}, total number gpus: {}".format(args.rank, num_gpus)
        torch.cuda.set_device(args.rank-1)
        logging.info("total number of gpus: {}, set current to {}".format(num_gpus, args.rank))
    main(args)
