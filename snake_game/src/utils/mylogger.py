import enum

class LOG_LEVEL(enum.Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2

MY_LOG_LEVEL = LOG_LEVEL.INFO

def log(loc, msg, log_level = LOG_LEVEL.DEBUG):
    if log_level.value > MY_LOG_LEVEL.value: return
    print(f"[{loc}]\t\t{msg}")
