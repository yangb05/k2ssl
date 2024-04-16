import fileinput
import json


def get_word_align(json_data):
    ret_info = []
    segments = json_data["segments"]
    for seg in segments:
        start, end = seg["start"], seg["end"]
        for info in seg["words"]:
            word, word_start, word_end = info["word"], info["start"], info["end"]
            word = word.strip()
            if word_start >= end:
                break
            ret_info.append([word, word_start, word_end])

    ret_info_str = " ".join(["|".join(map(str, info)) for info in ret_info])
    return ret_info_str


def main():
    for line in fileinput.input():
        uttid, data_str = line.strip().split(" ", maxsplit=1)
        data = json.loads(data_str)
        out = get_word_align(data)
        print("{} {}".format(uttid, out))


if __name__ == "__main__":
    main()
