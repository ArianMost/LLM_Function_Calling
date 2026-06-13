from enum import Enum, auto

class State(Enum):
    START = auto()
    OPEN_ROOT_OBJECT = auto()
    NAME_KEY = auto()
    NAME_COLON = auto()
    FUNCTION_NAME = auto()
    COMMA_AFTER_NAME = auto()
    PARAMETERS_KEY = auto()
    PARAMETERS_COLON = auto()
    OPEN_PARAMETERS_OBJECT = auto()
    PARAMETER_KEY = auto()
    PARAMETER_COLON = auto()
    PARAMETER_VALUE = auto()
    PARAMETER_COMMA = auto()
    CLOSE_PARAMETERS_OBJECT = auto()
    CLOSE_ROOT_OBJECT = auto()
    DONE = auto()
