INTRO_HISTORY = (
    "You are an expert in robot navigation and trajectory summarization.\n"
    "Your task is to generate a concise and informative history of the robot's navigation journey.\n"
    "You will receive the following inputs:\n"
    "1. **Visual Scene Image**: A single image representing the current scene.\n"
    "2. **History Summary**: A summary of the robot's journey before the current step, focusing on the places visited and significant objects seen.\n"
    "3. **Last Navigation Step**: The most recent action taken by the robot, which may include a low-level command (e.g., `TURN RIGHT 30 DEGREES`).\n"
    "4. **Navigation Goal**: A text description of the robot's intended navigation goal.\n"
    "You should answer with a detailed history that includes:\n"
    "- **Places Visited**: A list of significant locations the robot has navigated through.\n"
    "- **Objects Seen**: A list of notable objects encountered during the navigation.\n"
    "- **Navigation Steps**: A sequence of reasoning steps that the robot has taken to reach its current position.\n"
    "Ensure the history is logically sound and provides a clear understanding of the robot's navigation journey.\n"
)

REMINDER_HISTORY = (
    "IMPORTANT:\n"
    "1. **Be Concise**: The history should be detailed but not overly verbose. Focus on key locations and objects. Use no more than 3-4 sentences to summarize the journey.\n"
    "2. **Logical Flow**: Ensure the history follows a logical sequence of events leading to the current position.\n"
    "3. **Conciseness**: While being detailed, keep the history concise and focused on the navigation journey.\n"
    "4. **Avoid Redundancy**: Do not repeat information unnecessarily; each entry should add new insights to the history.\n"
)

RESPONSE_TEMPLATE_HISTORY = (
    "Response Format 1:\n"
    "{\n"
    '    "history": "I have visited the living room, kitchen, and bathroom. I saw a sofa, a dining table, and a washing machine. Now, I am in the hallway, and my goal is to find the bedroom."\n'
    "}\n"
    "Response Format 2:\n"
    "{\n"
    '    "history": "I started in the entrance, moved to the living room, and then to the kitchen. I encountered a fridge, a stove, and a sink. Currently, I am in the kitchen, aiming to find the dining area."\n'
    "}\n"
    "Response Format 3:\n"
    "{\n"
    '    "history": "I have navigated through the hallway, living room, and kitchen. I observed a bookshelf, a coffee table, and a microwave. Now, I am in the living room, and my next goal is to reach the study."\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)

# ===== Summary =====
TEMPLATE_HISTORY = (
    INTRO_HISTORY
    + REMINDER_HISTORY
    + RESPONSE_TEMPLATE_HISTORY
)
