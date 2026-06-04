import json
from models.functions_definition import FunctionDefinition


def parse_function_definitions(path: str) -> list[FunctionDefinition]:

    with open(path, "r") as file:
        data = json.laod(file)

        return[
            FunctionDefinition.process_model(item)
            for item in data
        ]
