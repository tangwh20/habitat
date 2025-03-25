INTRO = (
    "You are an expert in providing natural language instructions to guide a robot in an indoor environment. \n"
    "Now your robot is given a very long instruction to follow. You need to break down the instruction into "
    "smaller subtasks, and provide natural language instructions for a certain sub-trajetory. \n"
    "Specifically, you will be given the overall instruction, the starting and ending view images, and "
    "a list of locations representing the movement of this sub-trajectory. "
    "Each location corresponds to a point in [x, y] coordinate system within the current view. \n"
    "The x-axis represents the vertical direction: positive values indicate distance in meters ahead, "
    "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    "positive values indicate distance to the left, and negative values indicate distance to the right.\n"
    "IMPORTANT: Do not mix up left and right! Remember positive y values are to the left, and negative y values are to the right.\n"
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

RESPONSE_TEMPALTE = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess the overall instruction, and try to split it into smaller subtasks."
    "2) Based on the starting and ending views, identify which part of the instruction this sub-trajectory corresponds to."
    "3) Analyze the list of locations. Are they moving in a straight line, turning, or following a specific path?"
    '4) Based on your observations, select the most precise instruction for the given sub-trajectory.}",\n'
    '    "instruction": "{the most precise instruction}",\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
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