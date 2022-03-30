
import time
from datetime import datetime


def log(s):
    now = datetime.now()
    print("{}: {}".format(now, s))
