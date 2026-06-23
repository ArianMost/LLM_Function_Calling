from enum import Enum
from pydantic import BaseModel


class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"

class FunctionParams(BaseModel):
    type: str


class FunctionReturns(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParams]
    returns: FunctionReturns
