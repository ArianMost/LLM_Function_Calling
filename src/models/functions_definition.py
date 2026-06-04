from pydantic import BaseModel


class FunctionParams(BaseModel):
    type: str


class FunctionReturns(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParams]
    return_values: FunctionReturns
