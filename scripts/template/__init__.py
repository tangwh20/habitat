from .split import TEMPLATE_SPLIT
from .instruction import TEMPLATE_INSTRUCTION, EXAMPLE_INSTRUCTION
from .reasoning import TEMPLATE_REASONING, EXAMPLE_REASONING
from .reflection import TEMPLATE_REFLECTION

TEMPLATES = {
    "split": TEMPLATE_SPLIT,
    "instruction": TEMPLATE_INSTRUCTION,
    "reasoning": TEMPLATE_REASONING,
    "reflection": TEMPLATE_REFLECTION,
}

EXAMPLES = {
    "instruction": EXAMPLE_INSTRUCTION,
    "reasoning": EXAMPLE_REASONING,
}