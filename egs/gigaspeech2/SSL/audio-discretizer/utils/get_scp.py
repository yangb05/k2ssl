import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Generate wav.scp for a given dataset, using absolute paths"
)
parser.add_argument(
    "--corpus-dir", required=True, help="Directory of the audio and text files"
)

args = parser.parse_args()

output_filename = "wav.scp"

corpus_path = Path(args.corpus_dir)

wav_files = list(corpus_path.rglob("*.wav"))

with open(output_filename, "w") as output_file:
    for wav_path in wav_files:
        abs_wav_path = wav_path.resolve()
        utt_id = wav_path.stem
        output_file.write(f"{utt_id} {abs_wav_path}\n")

print(f"Created {output_filename} with {len(wav_files)} entries.")
