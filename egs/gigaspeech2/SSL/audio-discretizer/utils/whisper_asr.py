import argparse
import json
import os

import numpy as np
import torch
import whisper
from kaldiio import load_mat
from tqdm import tqdm

########## object ##########
whisper_model = None


def load_model():
    global whisper_model
    whisper_model = whisper.load_model("medium")
    whisper_model.transcribe


def asr_decode(audio_file: str):
    global whisper_model
    result = whisper_model.transcribe(
        audio_file,
        language="English",
        word_timestamps=True,
        prepend_punctuations="",
        append_punctuations="'",
    )
    return result


def get_files_from_dir(wavdir: str):
    wav_files = [f for f in os.listdir(wavdir) if ".wav" in f]
    wav_uttids = [f.removesuffix(".wav") for f in wav_files]
    wav_paths = [os.path.join(wavdir, f) for f in wav_files]
    return wav_uttids, wav_paths


def get_files_from_scp(wavscp: str):
    wav_uttids = []
    wav_paths = []
    with open(wavscp, "r") as reader:
        while True:
            line = reader.readline().strip()
            if line == "":
                break
            uttid, path = line.split(" ", maxsplit=1)
            wav_uttids.append(uttid)
            wav_paths.append(path)
    return wav_uttids, wav_paths


def main(args):
    # load model
    load_model()
    # get all wavs
    if args.wavdir is not None:
        wav_uttids, wav_paths = get_files_from_dir(args.wavdir)
    else:
        wav_uttids, wav_paths = get_files_from_scp(args.wavscp)

    # with open(args.out, "w") as writer:
    for uttid, wav_path in tqdm(zip(wav_uttids, wav_paths), total=len(wav_uttids)):
        # load wav first
        sampling_freq, wav_data = load_mat(wav_path)
        wav_data = wav_data.astype(np.float32) / 32768.0
        assert (
            sampling_freq == 16000
        ), "sampling rate should be 16khz, but got: {} for: {}".format(
            sampling_freq, uttid
        )

        result = asr_decode(wav_data)
        result_str = json.dumps(result)

        print("{} {}".format(uttid, result_str))
        # writer.write("{} {}\n".format(uttid, result_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default=None)
    parser.add_argument("--wavscp", type=str, default=None)
    # parser.add_argument("--out", type=str)
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()
    # setting gpus
    if args.rank is not None:
        total_num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.rank - 1)
        print(
            "total: {} gpus; setting current to: {}".format(
                total_num_gpus, args.rank - 1
            )
        )
    # must have wavdir or wavscp
    assert (args.wavdir is not None) or (
        args.wavscp is not None
    ), "one of wavdir or wavscp must be set"
    assert (args.wavdir is not None) ^ (
        args.wavscp is not None
    ), "only provide one of wavdir or wavscp"
    main(args)
