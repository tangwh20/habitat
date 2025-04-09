import gzip
import json

data_path = "/home/tangwenhao/Workspace/habitat/data/datasets/vln/mp3d/rxr/v1"
split = "train"
role = "follower" # "guide"

with gzip.open(f"{data_path}/{split}/{split}_{role}.json.gz", "rt") as f:
    data = json.load(f)

with gzip.open(f"{data_path}/{split}/{split}_{role}_gt.json.gz", "rt") as f:
    gt = json.load(f)


breakpoint()
print(data.keys())
print(gt.keys())

"""
data['episodes'][0]
{
    'episode_id': '1', 
    'trajectory_id': '0', 
    'scene_id': 'mp3d/SN83YJsR3w2/SN83YJsR3w2.glb', 
    'info': {'role': 'guide'}, 
    'instruction': {
        'instruction_id': '0', 
        'instruction_text': "You will start by standing in front of a glass door and on your right is a doorway. Turn around and you will see a doorway to the washroom. Walk towards the doorway and inside the washroom. Once you're there, stand in between the sink and the bathtub and once you're there, you're done.", 
        'language': 'en-US', 
        'annotator_id': '0', 
        'edit_distance': 0.07692307692307693, 
        'timed_instruction': [{'end_time': 1.5, 'word': 'You', 'start_time': 1.0}, {'end_time': 1.6, 'word': 'will', 'start_time': 1.5}, {'start_time': 1.6, 'end_time': 2.2, 'word': 'start'}, {'start_time': 2.2, 'word': 'by', 'end_time': 2.5}, {'word': 'standing', 'start_time': 2.5, 'end_time': 3.3}, {'start_time': 3.3, 'end_time': 3.6, 'word': 'in'}, {'word': 'front', 'start_time': 3.6, 'end_time': 3.8}, {'word': 'of', 'start_time': 3.8, 'end_time': 4.4}, {'word': 'a', 'start_time': 4.4, 'end_time': 5.5}, {'end_time': 6.0, 'word': 'glass', 'start_time': 5.5}, {'start_time': 6.0, 'word': 'door', 'end_time': 6.2}, {'word': 'and', 'end_time': 7.7, 'start_time': 6.2}, {'word': 'on', 'start_time': 7.7, 'end_time': 8.3}, {'start_time': 8.3, 'end_time': 8.5, 'word': 'your'}, {'start_time': 8.5, 'word': 'right', 'end_time': 8.6}, {'word': 'is', 'end_time': 9.1, 'start_time': 8.6}, {'word': 'a', 'start_time': 9.1, 'end_time': 9.5}, {'word': 'doorway.', 'start_time': 9.5, 'end_time': 10.3}, {'start_time': 10.3, 'end_time': 12.0, 'word': 'Turn'}, {'start_time': 12.0, 'end_time': 12.7, 'word': 'around'}, {'word': 'and', 'start_time': 12.7, 'end_time': 12.9}, {'word': 'you', 'end_time': 13.0, 'start_time': 12.9}, {'start_time': 13.0, 'word': 'will', 'end_time': 13.0}, {'word': 'see', 'end_time': 13.4, 'start_time': 13.0}, {'word': 'a', 'start_time': 13.4, 'end_time': 14.9}, {'end_time': 15.3, 'start_time': 14.9, 'word': 'doorway'}, {'start_time': 15.3, 'end_time': 15.6, 'word': 'to'}, {'word': 'the', 'start_time': 15.6, 'end_time': 15.9}, {'word': 'washroom.', 'end_time': 16.5, 'start_time': 15.9}, {'start_time': 18.2, 'word': 'Walk', 'end_time': 18.9}, {'word': 'towards', 'start_time': 20.6, 'end_time': 21.2}, {'start_time': 21.2, 'word': 'the', 'end_time': 21.4}, {'word': 'doorway', 'end_time': 22.8, 'start_time': 21.4}, {'word': 'and', 'start_time': 22.8, 'end_time': 23.3}, {'start_time': 23.3, 'end_time': 23.8, 'word': 'inside'}, {'word': 'the', 'start_time': 23.8, 'end_time': 24.0}, {'start_time': 24.0, 'word': 'washroom.', 'end_time': 24.5}, {'word': 'Once', 'end_time': 26.7, 'start_time': 26.2}, {'word': "you're", 'end_time': 26.9, 'start_time': 26.7}, {'word': 'there,', 'end_time': 27.2, 'start_time': 26.9}, {'word': 'stand', 'start_time': 27.2, 'end_time': 28.2}, {'start_time': 29.0, 'word': 'in', 'end_time': 29.2}, {'word': 'between', 'start_time': 29.2, 'end_time': 29.3}, {'start_time': 29.3, 'word': 'the', 'end_time': 30.4}, {'word': 'sink', 'end_time': 30.8, 'start_time': 30.4}, {'end_time': 31.2, 'start_time': 30.8, 'word': 'and'}, {'word': 'the', 'end_time': 31.3, 'start_time': 31.2}, {'word': 'bathtub', 'end_time': 31.7, 'start_time': 31.3}, {'end_time': 33.9, 'word': 'and', 'start_time': 33.2}, {'word': 'once', 'start_time': 33.9, 'end_time': 34.2}, {'word': "you're", 'end_time': 34.3, 'start_time': 34.2}, {'word': 'there,', 'start_time': 34.3, 'end_time': 34.5}, {'start_time': 34.5, 'end_time': 35.1, 'word': "you're"}, {'word': 'done.', 'start_time': 35.1, 'end_time': 35.3}]
    }, 
    'reference_path': [[0.25628501176834106, 3.8914501667022705, -16.086700439453125], [0.547003984451294, 3.8914501667022705, -17.49679946899414], [1.1071385145187378, 3.8914501667022705, -18.532249450683594]], 
    'start_position': [0.25628501176834106, 3.8914501667022705, -16.086700439453125], 
    'goals': [{'position': [1.1071385145187378, 3.8914501667022705, -18.532249450683594], 'radius': 3.0}], 
    'start_rotation': [-0.0, 0.967659801213298, -0.0, -0.2522588137525439]
}
"""

"""
gt['1']
{
    'locations': [[0.25628501176834106, 3.8914501667022705, -16.086700439453125], [0.2596777081489563, 3.8914501667022705, -16.33667755126953], [0.26307040452957153, 3.8914501667022705, -16.586654663085938], [0.390997052192688, 3.8914501667022705, -16.80144500732422], [0.5189236998558044, 3.8914501667022705, -17.0162353515625], [0.5223163962364197, 3.8914501667022705, -17.266212463378906], [0.6502430438995361, 3.8914501667022705, -17.481002807617188], [0.6536357402801514, 3.8914501667022705, -17.730979919433594], [0.7815623879432678, 3.8914501667022705, -17.945770263671875], [0.9094890356063843, 3.8914501667022705, -18.160560607910156]], 
    'actions': [2, 2, 2, 2, 2, 1, 1, 3, 1, 1, 2, 1, 3, 1, 2, 1, 3, 1, 1, 0], 
    'forward_steps': 9
}
"""