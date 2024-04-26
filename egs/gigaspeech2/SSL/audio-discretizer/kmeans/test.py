from lhotse import CutSet
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def test(cut):
    fbank = cut.load_features()
    kmeans = cut.custom['kmeans'].split()
    return len(fbank), len(kmeans)


def run(cut_file):
    cutset = CutSet.from_file(cut_file)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for len_f, len_k in tqdm(ex.map(test, cutset), desc='testing'):
            assert len_f == len_k


if __name__ == '__main__':
    run('/data_a100/userhome/yangb/data/kms/librispeech_cuts_train-clean-360.jsonl.gz')