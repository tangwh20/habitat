INTRO_SPLIT = (
    "You are an AI expert in image sequence segmentation. \n"
    "Now you are given a sequence of N consecutive photos taken by a moving robot along with a set of ordered instructions. \n"
    "Your task is to divide the photo sequence into segments, where each segment corresponds to one specific instruction. \n"
    "For each segment, identify the starting frame number and the ending frame number of the corresponding instruction. \n"
    "The first frame of the photo sequence is numbered 0, and the last frame is numbered N-1, where N is the total number of frames. \n"
)

REMINDER_SPLIT = (
    "IMPORTANT: \n"
    "1) Carefully analyze each instruction and the corresponding visual cues in the photo sequence. \n"
    "2) Ensure that the starting frame of each segment corresponds to the first frame where the action described in the instruction becomes visually apparent. \n"
    "3) The ending frame of each segment is not necessarily the starting frame of the next instruction, but it should be the last frame where the current instruction is still relevant. \n"
    "4) The starting frame of the first instruction is always 0, and the ending frame of the last instruction is N-1. \n"
)

RESPONSE_TEMPLATE_SPLIT = (
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{Think step by step: '
    '1) Analyze Instructions: Read and understand each instruction’s intent (e.g., "turn left," "move forward"). '
    '2) Observe Visual Cues: Scan the photo sequence for key changes (e.g., robot movement, object interaction, direction shifts). '
    '3) Match Instructions to Frames: Align each instruction’s expected action with the first frame where the action becomes visually apparent. '
    '4) Validate Starting Frames: Ensure the previous instruction ends before the next instruction begins. '
    '5) Output: Based on the analysis, provide a list of starting frame numbers for each instruction segment.}",\n'
    '    "start_step": [{the start step of the first instruction}, {the start step of the second instruction}, ...],\n'
    "}\n"
    "Remember to count the images from 0, not 1.\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)


TEMPLATE_SPLIT = (
    INTRO_SPLIT
    + REMINDER_SPLIT
    + RESPONSE_TEMPLATE_SPLIT
)