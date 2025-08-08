INTRO_INSTRUCTION = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "You will be given an image showing the current view, a list of 8 locations and 8 yaw angles "
    "representing the movement you want the robot to execute. \n"
    "Each location corresponds to a point in [x, y] coordinate system within the current view, and each yaw "
    "angle is in radians, indicating the robot's orientation at that point. \n"
    "The x-axis represents the vertical direction: positive values indicate distance in meters ahead, "
    "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    "positive values indicate distance to the left, and negative values indicate distance to the right.\n"
    "The yaw angle indicates the robot's orientation, where 0 radians means facing forward, "
    "positive values indicate a counterclockwise rotation, and negative values indicate a clockwise rotation.\n"
    "You need to provide two types of instructions: One containing the robot's target object or goal, "
    "and the other containing no object or goal. \n"
)

REMINDER_INSTRUCTION = (
    "IMPORTANT:\n"
    "1.	Be succinct and answer questions directly with one of the given options.\n"
    "2.	Be precise when referring to the movement goal. Make sure the reference is easily identifiable and "
    "clearly indicates the extending direction of the last few steps. For example, if the movement is extending "
    "toward the far left, it likely isn’t heading toward something directly in front. Avoid referring to large "
    "objects or open spaces that span a wide area, as they can can obscure the direction. Use specific, "
    "functional, or descriptive terms for objects, and steer clear of vague phrases like ‘the goal’ or ‘your "
    "destination.’ Do not mention or create objects that aren’t visible in the current view. \n"
    "3.	Be precise in your word choices. Phrases like 'move along are typically used with pathways or barriers, "
    "rarely with open spaces or areas. Similarly, phrases like 'skirt around' are usually associated with "
    "obstacles, not open spaces. Additionally, taking a turn should only be described at intersections or "
    "corners, not in straight hallways.\n"
    "4.	Do not confuse doors, hallways and glass walls. A hallway is an open passage that doesn’t have any "
    "barriers to open or close and extends further into another area, with walls as tall as the surrounding "
    "structure and no top frame. A door, in contrast, has a frame with a top that is usually lower than the "
    "ceiling height. You can interact with doors by opening or closing them. Glass walls, however, are "
    "transparent barriers that may look like doors or open spaces but cannot be used to move through. "
    "They serve as dividers or enclosures, without providing an entrance or exit.\n"
    "5.	Do not mix up left and right! In mose cases, y value of the last few steps determines the general"
    "direction of movement. Large negative y values indicate you are moving to the right, while large positive "
    "y values indicate a leftward movement. If the y values show little change compared to the x values, "
    "especially in a hallway where forward is the only valid option, it generally means you are moving "
    "straight forward.\n"
    "6.	Pay close attention to trajectories with sharply decreasing x values, as these typically indicate "
    "backward movements. In cases where x values change much less than y values (whether increasing or "
    "decreasing), it is more likely a left or right turn rather than a backward movement.\n"
)

RESPONSE_TEMPALTE_INSTRUCTION = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step:'
    "1) Assess your current view. Are you in an open space, or at an intersection with branching pathways? Is there any obstacle close by?"
    "2) Identify the trajectory pattern. Based on x and y values, determine if the trajectory is moving forward or backward, or turning left or right."
    "3) Check for clear pathways or targets. Is the trajectory consistently moving towards a direction? Or is it following the shape of some pathway or moving around any obstacle?"
    "4) Based on your observations, select the most precise instruction with target object for the given trajectory."
    '5) Based on your observations, select the most precise instruction without target object for the given trajectory.}",\n'
    '    "instruction_with_object": "{the most precise instruction with target object}",\n'
    '    "instruction_without_object": "{the most precise instruction without target object}"\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)


# ===== Format Instruction Templates =====
FORMAT_INSTRUCTION_NO_OBJECT = [
    "move forward",
    "turn {left, right}", 
    "move along", 
    "turn backwards",
    "skirt {left, right}",
    "stop",
]

FORMAT_INSTRUCTION_OBJECT = [
    "move forward towards {describe the movement goal}",
    "turn {left, right} at {describe the intersection}",
    "move along {describle the pathway}",
    "turn backwards at {describe the intersection}",
    "skirt {left, right} around {describle the obstacle object}",
    "go through {space in between {object_1} and {object_2}, an open door, entrance to hallway}",
    "stop at {describe the object}",
]


# ===== Summary =====
TEMPLATE_INSTRUCTION = (
    INTRO_INSTRUCTION
    + REMINDER_INSTRUCTION
    + RESPONSE_TEMPALTE_INSTRUCTION
)

EXAMPLE_INSTRUCTION = "Examples with target object: \n" + "\n".join(
    [f"{i}. {instruction}" for i, instruction in enumerate(FORMAT_INSTRUCTION_OBJECT)]
) + "\n\n" + "Examples without target object: \n" + "\n".join(
    [f"{i}. {instruction}" for i, instruction in enumerate(FORMAT_INSTRUCTION_NO_OBJECT)]
)