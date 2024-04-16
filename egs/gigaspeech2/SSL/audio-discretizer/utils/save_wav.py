import os
import sys

import soundfile as sf
from kaldiio import ReadHelper

rspec = sys.argv[1]
outdir = sys.argv[2]

# NOTE, when reading wav.ark, should pass dtype=np.int16 in soundfile read function!
with ReadHelper(rspec) as reader:
    for uttid, (sampling_rate, audio_data) in reader:
        # audio data is int16
        wav_path = os.path.join(outdir, "{}.wav".format(uttid))
        sf.write(wav_path, audio_data, sampling_rate)
