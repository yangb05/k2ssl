import fileinput
import json

for line in fileinput.input():
    uttid, data_str = line.strip().split(" ", maxsplit=1)
    data = json.loads(data_str)
    text = ""
    for seg in data["segments"]:
        end_time = seg["end"]
        for w in seg["words"]:
            if w["start"] >= end_time:
                break
            text += w["word"]
    text = text.strip()
    print("{} {}".format(uttid, text))
