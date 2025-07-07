import json
import gzip

scene_name = "1LXtFkjw3qL"
path = f"/home/tangwenhao/Workspace/habitat/data/datasets/objectnav/mp3d/v1/train/content/{scene_name}.json.gz"
output_path = f"/home/tangwenhao/Workspace/habitat/scripts/example_objnav/{scene_name}.json"
category_output_path = "/home/tangwenhao/Workspace/habitat/scripts/example_objnav/category.json"

with gzip.open(path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({"goals_by_category": data["goals_by_category"]}, f, indent=4)

with open(category_output_path, 'w', encoding='utf-8') as f:
    json.dump({
        "category_to_task_category_id": data["category_to_task_category_id"],
        "category_to_mp3d_category_id": data["category_to_mp3d_category_id"]
    }, f, indent=4)

breakpoint()
print(data.keys())

"""
Data Structure:

objectnav/mp3d/v1/train/content/1LXtFkjw3qL.json.gz

{
    "goals_by_category": {
        (xxx.glb_target)
        "1LXtFkjw3qL.glb_tv_monitor":[
            (list of all goals in this room)
            GOAL 1,
            GOAL 2,
            ...
        ]
        ...
    }

    "episodes": [
        (list of all episodes)
        EPISODE 1,
        EPISODE 2,
        ...
    ]

    (All scene configs have the same)
    "category_to_task_category_id":
        {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14, 'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18, 'seating': 19, 'clothes': 20}
    "category_to_mp3d_category_id":
        {'chair': 3, 'table': 5, 'picture': 6, 'cabinet': 7, 'cushion': 8, 'sofa': 10, 'bed': 11, 'chest_of_drawers': 13, 'plant': 14, 'sink': 15, 'toilet': 18, 'stool': 19, 'towel': 20, 'tv_monitor': 22, 'shower': 23, 'bathtub': 25, 'counter': 26, 'fireplace': 27, 'gym_equipment': 33, 'seating': 34, 'clothes': 38}

}

GOAL Structure:
{
    'position': [-3.04617, 4.44737, 5.29812], 
    'radius': None, 
    'object_id': 6, 
    'object_name': '2_0_6', 
    'object_category': 'tv_monitor', 
    'room_id': None, 
    'room_name': None, 
    'view_points': [
        {'agent_state': {'position': [-3.03801, 3.60679, 5.61729], 'rotation': [0.0, 0.01279, 0.0, 0.99992]}, 'iou': 2.10818},
        {'agent_state': {'position': [-2.94801, 3.5806, 5.61729], 'rotation': [0.0, 0.14863, 0.0, 0.98889]}, 'iou': 2.0306},
        ...
    ],
}

EPISODE Structure:
{
    'episode_id': '0', 
    'scene_id': 'mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb', 
    'start_position': [-5.73303, 0.08441, 6.67048], 
    'start_rotation': [0, 0.92732, 0, -0.37428], 
    'info': {
        'geodesic_distance': 24.10707, 
        'euclidean_distance': 4.77366, 
        'closest_goal_object_id': 6, 
        'navigation_bounds': [[-7.62605, -3.11559, -4.36096], [4.15602, 10.20153, 17.4473]], 
        'best_viewpoint_position': [-2.51318, 3.48441, 5.88465]}, 
        'goals': [], 
        'start_room': None, 
        'shortest_paths': [[3, 3, 3, 1, 3, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, None]], 
    'object_category': 'tv_monitor'
}



"""