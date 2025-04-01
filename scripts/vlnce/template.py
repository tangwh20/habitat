INTRO_REFERENCE = (
    "You are an expert in providing natural language instructions to guide a robot in an indoor environment. \n"
    "Now your robot is given a very long instruction to follow. You need to break down the instruction into "
    "smaller subtasks, and provide natural language instructions for a certain sub-trajetory. \n"
    # "Specifically, you will be given the overall instruction, the starting and ending view images, and "
    # "a list of locations representing the movement of this sub-trajectory. "
    # "Each location corresponds to a point in [x, y] coordinate system within the current view. \n"
    # "The x-axis represents the vertical direction: positive values indicate distance in meters ahead, "
    # "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    # "positive values indicate distance to the left, and negative values indicate distance to the right.\n"
    # "IMPORTANT: Do not mix up left and right! Remember positive y values are to the left, and negative y values are to the right.\n"
    "Specifically, you will be given the overall instruction and a list of images representing the view of " 
    "the robot at each reference step. \n"
    "You need to analyze the list of images and provide the most precise instruction for each sub-trajectory" \
    "between two consecutive reference steps. \n" 
    "IMPORTANT: Each sub-trajectory is represented by a starting and ending view image, " \
    "ensure to provide the most precise instruction for the sub-trajectory based on the starting and ending view images.\n"
)

INTRO_FULL = (
    "You are an expert in providing natural language instructions to guide a robot in an indoor environment. \n"
    "Now your robot is given a very long instruction to follow. You need to break down the instruction into "
    "smaller subtasks, and provide natural language instructions for a certain sub-trajetory. \n"
    "Specifically, you will be given the overall instruction and a list of images representing the view of "
    "the robot at each step. \n"
    "You need to provide the subtask instructions based on the overall instruction, and provide the crucial "
    "frames for each subtask, that is, the starting frame and ending frame for each subtask. \n"
)

RESPONSE_TEMPALTE_REASON_BY_STEPS = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess your current view. Are you in an open space, or at an intersection with branching pathways? Is there any obstacle close by?"
    "2) Identify the trajectory pattern. Based on x and y values, determine if the trajectory is moving forward or backward, or turning left or right."
    "3) Check for clear pathways or targets. Is the trajectory consistently moving towards a direction? Or is it following the shape of some pathway or moving around any obstacle?"
    '4) Based on your observations, select the most precise instruction for the given trajectory.}",\n'
    '    "instruction": "{the most precise instruction}",\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)

RESPONSE_TEMPALTE_REFERENCE = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess the overall instruction, and try to split it into smaller subtasks."
    "2) Based on views of the reference steps, identify which part of the instruction each sub-trajectory corresponds to."
    # "3) Analyze the list of locations. Are they moving in a straight line, turning, or following a specific path?"
    # '4) Based on your observations, select the most precise instruction for the given sub-trajectory.}",\n'
    '3) Based on your observations, select one most precise instruction for each sub-trajectory.}",\n'
    '    "instruction": ["{the most precise instruction for sub-trajectory 1}", '
    '"{the most precise instruction for sub-trajectory 2}",...],\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers. \n"
)

RESPONSE_TEMPALTE_FULL = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess the overall instruction, and try to split it into smaller subtasks."
    "2) Based on views of the whole trajectory, identify which part of the trajectory each subtask corresponds to."
    '3) Based on your observations, provide the starting and ending frame for each subtask.}",\n'
    '    "instruction": ["{the most precise instruction for subtask 1}", '
    '"{the most precise instruction for subtask 2}",...],\n'
    '    "frames": [[{the starting frame for subtask 1}, {the ending frame for subtask 1}], '
    '[{the starting frame for subtask 2}, {the ending frame for subtask 2}],...],\n'
    "}\n"
    "Remember item 'frames' is a list of lists, each containing two integers representing the starting and ending frame for each subtask. \n"
    "Remember to count the frames from 1, not 0. \n"
    "Remember the ending frame must be smaller than the total number of frames. \n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers. \n"
)

FORMAT_ACTION = [
    "move forward",
    "move towards {describe the movement goal}",
    "move across {describe open space}",
    "turn {left, right} at {describe the intersection}",
    "move along {describle the pathway}",
    "turn backwards",
    "skirt {left, right} around {describle the obstacle object}",
    "go through {space in between {object_1} and {object_2}, an open door, entrance to hallway}",
]