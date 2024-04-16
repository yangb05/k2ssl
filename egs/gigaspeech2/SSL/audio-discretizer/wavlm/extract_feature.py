import logging
from tqdm import tqdm
import numpy as np
import argparse
import soundfile
from kaldiio import WriteHelper, load_mat
import torch
from wavlm_model import WavLM, WavLMConfig
logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavscp", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    # parallel
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    checkpoint = torch.load(args.model)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()
    
    # load wavscp
    wav_uttid_paths = []
    with open(args.wavscp, "r") as reader:
        while True:
            line = reader.readline().strip()
            if line == "": break
            uttid, wav_path = line.split(" ", maxsplit=1)
            wav_uttid_paths.append([uttid, wav_path])
    logging.info("will extract feature for {} utterances".format(len(wav_uttid_paths)))

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

            assert sample_rate == 16000, "sampling rate should be 16khz"
            audio = torch.from_numpy(audio).float().unsqueeze(0)
            if cfg.normalize:
                audio = torch.nn.functional.layer_norm(audio , audio.shape)
            audio = audio.cuda()
            
            with torch.no_grad():
                # rep = model.extract_features(audio)[0]
                rep, layer_results = model.extract_features(audio, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]  # [(1, l, 1024)]
                rep = layer_reps[args.layer]
                
            rep = rep[0].detach().cpu().numpy() # (l, 1024)

            writer[uttid] = rep


if __name__ == "__main__":
    args = get_args()
    logging.info(str(args))
    if args.rank is not None:
        num_gpus = torch.cuda.device_count()
        assert args.rank <= num_gpus, "invalid rank: {}, total number gpus: {}".format(args.rank, num_gpus)
        torch.cuda.set_device(args.rank-1)
        logging.info("total number of gpus: {}, set current to {}".format(num_gpus, args.rank))
    main(args)