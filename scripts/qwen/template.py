INTRO_8_OBS_AND_ACTION_STRING = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. \n"
    # "You will be presented with a trajectory of 8 steps that represent the movement you want the robot "
    # "to follow, consisting of 8 image observations and a list of locations corresponding to points in "
    # "an [x, y] coordinate system within the current view."
    # "The x axis represents the vertical direction: positive values indicate distance in meters ahead, "
    # "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    # "positive values indicate distance to the right, and negative values indicate distance to the left.\n"
    # "You will be presented with a sequence of 8 actions that represent the movement you want the robot to follow. "
    # "Each action is an integer from 0 to 3, where 0 represents stopping, 1 represents moving forward, "
    # "2 represents turning left, and 3 represents turning right. \n"
    "You will be presented with 8 images that represent the robot's view at each step. "
    "Each image is a snapshot of the robot's front view, showing the layout of the indoor environment. \n"
    "Along with the images, you will be provided with a list of actions that represent the movement the robot "
    "actually took. Each action is an integer from 0 to 3, where 0 represents stopping, 1 represents moving forward for one step, "
    "2 represents turning left for a small degree, and 3 represents turning right for a small degree. \n"
    "Your task is to provide a brief natural language instruction to summarize the whole sequence "
    "of actions based on the given images. \n"
)


GENERATION_GUIDE = (
    "IMPORTANT:\n"
    "1.	Be succinct and answer questions directly with one of the given options.\n"
    "2.	Be precise when referring to the movement goal. Make sure the reference is easily identifiable and "
    "clearly indicates the extending direction of the last few steps. For example, if the movement is extending "
    "toward the far left, it likely isn't heading toward something directly in front. Avoid referring to large "
    "objects or open spaces that span a wide area, as they can can obscure the direction. Use specific, "
    "functional, or descriptive terms for objects, and steer clear of vague phrases like 'the goal' or 'your "
    "destination.' Do not mention or create objects that aren't visible in the current view. \n"
    "3.	Be precise in your word choices. Phrases like 'move along are typically used with pathways or barriers, "
    "rarely with open spaces or areas. Similarly, phrases like 'skirt around' are usually associated with "
    "obstacles, not open spaces. Additionally, taking a turn should only be described at intersections or "
    "corners, not in straight hallways.\n"
    "4.	Do not confuse doors, hallways and glass walls. A hallway is an open passage that doesn't have any "
    "barriers to open or close and extends further into another area, with walls as tall as the surrounding "
    "structure and no top frame. A door, in contrast, has a frame with a top that is usually lower than the "
    "ceiling height. You can interact with doors by opening or closing them. Glass walls, however, are "
    "transparent barriers that may look like doors or open spaces but cannot be used to move through. "
    "They serve as dividers or enclosures, without providing an entrance or exit.\n"
    # "5.	Do not mix up left and right! In mose cases, y value of the last few steps determines the general"
    # "direction of movement. Large negative y values indicate you are moving to the left, while large positive "
    # "y values indicate a rightward movement. If the y values show little change compared to the x values, "
    # "especially in a hallway where forward is the only valid option, it generally means you are moving "
    # "straight forward.\n"
    # "6.	Pay close attention to trajectories with sharply decreasing x values, as these typically indicate "
    # "backward movements. In cases where x values change much less than y values (whether increasing or "
    # "decreasing), it is more likely a left or right turn rather than a backward movement.\n"
    # "7.	When giving instructions to move through an open door or a narrow space between two objects, "
    # "ensure that the trajectory consistently leads toward and through the door or space. Otherwise, it may "
    # "simply indicate movement toward, but not through, the door or space. \n"
    # "4.	Distinguish between turning and moving toward a direction. Turning involves a noticeable shift in "
    # "orientation within a few steps, typically causing a much larger change in y values compared to x values. "
    # "Turning may begin with forward steps followed by turning steps. Turning usually occurs near intersections, "
    # "corners, or when facing walls. Turning can also involve choosing a path, so if you're at an intersection "
    # "and the movement clearly follows one of the paths, it's likely a turn. In a sharp turn, the movement is "
    # "completed in 2-3 steps, and in a vertical turn, the x values remain around 0. In contrast, moving toward "
    # "a direction indicates maintaining a consistent orientation toward an object, with a steady ratio between "
    # "x and y values.\n"
)


RESPONSE_TEMPALTE_REASON_BY_STEPS = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess your current view. Are you in an open space, or at an intersection with branching pathways? Is there any obstacle close by?"
    "2) Identify the trajectory pattern. Based on given observations, determine if the trajectory is moving forward or backward, or turning left or right."
    "3) Check for clear pathways or targets. Is the trajectory consistently moving towards a direction? Or is it following the shape of some pathway or moving around any obstacle?"
    '4) Based on your observations, select the most precise instruction for the given trajectory.}",\n'
    '    "instruction": "{the most precise instruction}",\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)


FORMAT_ACTION = [
    "move forward",
    "move towards {describe the movement goal}",
    "move across {describe open space}",
    "turn {left, right} at {describe the intersection}",
    "turn {left, right} and move forward",
    "turn {left, right} and move towards {describe the movement goal}",
    "turn {left, right} and move across {describe open space}",
    "move along {describle the pathway}",
    "turn backwards",
    "skirt {left, right} around {describle the obstacle object}",
    "go through {space in between {object_1} and {object_2}, an open door, entrance to hallway}",
]