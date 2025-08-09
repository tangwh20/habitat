from .split import TEMPLATE_SPLIT
from .history import TEMPLATE_HISTORY
from .instruction import TEMPLATE_INSTRUCTION, EXAMPLE_INSTRUCTION
from .reasoning import TEMPLATE_REASONING
from .reflection import TEMPLATE_REFLECTION

TEMPLATES = {
    "split": TEMPLATE_SPLIT,
    "history": TEMPLATE_HISTORY,
    "instruction": TEMPLATE_INSTRUCTION,
    "reasoning": TEMPLATE_REASONING,
    "reflection": TEMPLATE_REFLECTION,
}

EXAMPLES = {
    "instruction": EXAMPLE_INSTRUCTION,
}

ACTION_MAP = ["STOP", "MOVE FORWARD 0.25 METERS", "TURN LEFT 30 DEGREES", "TURN RIGHT 30 DEGREES"]