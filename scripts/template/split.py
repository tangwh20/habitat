INTRO_SPLIT = (
    "You are an expert AI specializing in stateful image sequence analysis and interpretation. "
    "Your primary task is to conduct a detailed step-by-step analysis of a photo sequence. "
    "You must provide two key pieces of information for every step:\n"
    "1. **History Summary**: A concise, observational summary of the robot's journey before this step. "
    "This summary must focus exclusively on the **places the robot has passed and the significant objects it has seen**, based on visual evidence from all *preceding* frames.\n"
    "2. **Current Instruction**: The specific instruction text that is being executed at this step.\n"
    "You will receive the following inputs:\n"
    "1. **Image Sequence**: A sequence of N frames, indexed from 0 to N-1.\n"
    "2. **Instructions**: An ordered list of text commands describing the robot's actions.\n"
)

REMINDER_SPLIT = (
    "IMPORTANT: \n"
    "1. **History Summary Generation**:\n"
    "   * For each frame `i`, the `history_summary` must be a purely descriptive summary of the visual information from frames `0` to `i-1`.\n"
    "   * **Focus exclusively on**: The path taken, locations visited, and significant objects or features observed "
    "   (e.g., 'Started in a kitchen with marble countertops, moved past a stainless steel refrigerator, and entered a living room with a brown sofa.').\n"
    "   * The summary must include visual information for **ALL preceding frames** (0 to i-1)."
    "   * The summary must not include information from the current frame `i` or any future frames.\n"
    "   * **Edge Case**: For frame 0, the history must be a simple statement indicating there is no prior history (e.g., \"This is the starting frame, so there is no history yet.\").\n"
    "2. **Identify Instruction Segments**: First, internally determine the `start_frame` and `end_frame` for each instruction to correctly identify the current instruction for any given frame.\n"
    "   * The `start_frame` is the very first frame where an action becomes visually apparent.\n"
    "   * The `end_frame` is the last frame where the action is still relevant.\n"
    "3. **Output List Length**: The final output list in your response must contain exactly **N** objects, one for each frame.\n"
)

RESPONSE_TEMPLATE_SPLIT = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step-by-step: '
    'First, I will process the sequence frame by frame from 0 to N-1. '
    'For each frame \'i\', I will analyze **ALL preceding frames** (0 to i-1) to build a purely observational summary of the path and significant objects seen. '
    'Then, I will analyze the intent of each instruction, and scan the image sequence for key visual cues corresponding to these instructions, such as movement, rotation, or object interaction. '
    'I will pinpoint the exact start frame where each action visually begins, and the exact end frame where each action visually ends. '
    'For each frame, I will determine the current instruction based on the identified start and end frames of the actions.}",\n'
    '    "segments": [[0, 1], [2, 4], ..., [N-3, N-1]],\n'
    '    "frame_analysis": [\n'
    '        {\n'
    '            "history_summary": "This is the starting frame, so there is no history yet.",\n'
    '            "current_instruction": "Move forward towards the door"\n'
    '        },\n'
    '        {\n'
    '            "history_summary": "I started in a brightly lit office with a blue chair and a wooden desk.",\n'
    '            "current_instruction": "Move forward towards the door"\n'
    '        },\n'
    '        {\n'
    '            "history_summary": "I have moved across the office, passing the blue chair and the wooden desk. '
    'It is now positioned in front of a white door with a silver handle.",\n'
    '            "current_instruction": "Open the door"\n'
    '        }\n'
    '    ]\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)


TEMPLATE_SPLIT = (
    INTRO_SPLIT
    + REMINDER_SPLIT
    + RESPONSE_TEMPLATE_SPLIT
)