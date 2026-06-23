import json
from pathlib import Path
from ..models.functions_definition import FunctionDefinition
from ..models.prompt import Prompt


def parse_function_definitions(path: Path) -> list[FunctionDefinition]:

    try:
        with path.open("r") as file:
            data = json.load(file)

            return [
                FunctionDefinition.model_validate(item)
                for item in data
            ]
    except Exception as e:
        raise RuntimeError(f"Invalid function definition in {path}: {e}")

def parse_prompts(path: str) -> list[Prompt]:
    try:
        with open(path, "r") as file:
            data = json.load(file)

            return [
                Prompt.model_validate(item)
                for item in data
            ]
    except Exception as e:
        raise RuntimeError(f"Could not read prompts file {path}: {e}")
