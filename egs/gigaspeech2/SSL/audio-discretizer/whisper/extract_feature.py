import argparse
import logging
import math
from pathlib import Path

import torch
from kaldiio import WriteHelper, load_mat
from tqdm import tqdm

import whisper


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav-scp",
        type=Path,
        default=Path("data/ml_superb.scp"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ml_superb_whisper_large_v3"),
        help="Dictionary path to save audio files.",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size (large, medium, small, base, tiny)",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
    )

    return parser.parse_args()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(args.model_size, device=device)

    # load wav.scp
    wav_uttid_paths = []
    with open(args.wav_scp, "r") as reader:
        while True:
            line = reader.readline().strip()
            if line == "":
                break
            uttid, wav_path = line.split(" ", maxsplit=1)
            wav_uttid_paths.append([uttid, wav_path])

    logging.info(f"About to extract feature for {len(wav_uttid_paths)} utterances")

    # decoding
    wspec = f"ark,scp:{args.output_dir}/feats.ark,{args.output_dir}/feats.scp"
    logging.info(f"About to write pred to {wspec}")
    with WriteHelper(wspec) as writer:
        for uttid, wav_path in tqdm(wav_uttid_paths):
            with torch.no_grad():
                try:
                    audio = whisper.load_audio(wav_path)
                    audio_len = audio.shape[0]
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(
                        audio, n_mels=128, device=device
                    ).unsqueeze(0)
                    feature = (
                        model.embed_audio(mel).squeeze(0).detach().cpu().numpy()
                    )  # (T, C)
                    feature_no_pad_len = math.ceil(
                        feature.shape[0] * audio_len / 16000 / 30
                    )

                    writer[uttid] = feature[:feature_no_pad_len]
                except:
                    logging.info(f"Error when processing {uttid} {wav_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    args = get_args()
    logging.info(str(args))

    if args.rank is not None:
        num_gpus = torch.cuda.device_count()
        assert (
            args.rank <= num_gpus
        ), f"invalid rank: {args.rank}, total number gpus: {args.num_gpus}"
        torch.cuda.set_device(args.rank - 1)
        logging.info(f"total number of gpus: {num_gpus}, set current to {args.rank}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
