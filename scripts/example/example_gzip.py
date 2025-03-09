import json
import gzip

# load json data
with open("scripts/example/example_episode.json", "r") as f:
    json_data = json.load(f)

# write json data to a gzip file
with gzip.open("scripts/example/example_episode.json.gz", "wt") as f:
    json.dump(json_data, f)