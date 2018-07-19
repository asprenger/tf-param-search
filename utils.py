import time
import random

def ts_rand():
    ts = current_time_ms()
    random_num = random.randint(1e6, 1e7-1)
    return '%d_%d' % (ts, random_num)

def current_time_ms():
    return int(time.time()*1000.0)
