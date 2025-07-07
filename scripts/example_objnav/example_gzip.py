import json
import gzip

scene_name = "1LXtFkjw3qL"

# load json data
with open(f"scripts/example_objnav/{scene_name}.json", "r") as f:
    goal_data = json.load(f)
with open("scripts/example_objnav/example_episode.json", "r") as f:
    episode_data = json.load(f)
with open("scripts/example_objnav/category.json", "r") as f:
    category_data = json.load(f)

# write json data to a gzip file
with gzip.open("scripts/example_objnav/example_episode.json.gz", "wt") as f:
    json.dump({
        **goal_data,
        **episode_data,
        **category_data
    }, f)