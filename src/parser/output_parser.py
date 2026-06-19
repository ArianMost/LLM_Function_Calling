import json
from typing import Any
from models.function_call import FunctionCall
from models.functions_definition import FunctionDefinition, ParameterType

def parse_output(
        prompt: str,
        function_definition: FunctionDefinition,
        decoded: str
) -> FunctionCall:
    decoded_dict = json.loads(decoded)

    params: dict[str, Any] = {}
    for name, value in decoded_dict["parameters"].items():
        param_type = function_definition.parameters[name].type
        if param_type == ParameterType.NUMBER:
            params[name] = float(value)
        elif param_type == ParameterType.BOOLEAN:
            params[name] = bool(value)
        else:
            params[name] = str(value)

    return FunctionCall(prompt=prompt, name=decoded_dict["name"], parameters=params)
