
from typing import Optional, Callable, Dict, List
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile


class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)



def max_fill(grid, capacity):
    n = len(grid)
    m = len(grid[0])
    res = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                res += capacity - 1
                for k in range(capacity):
                    if grid[i+k][j] == 0:
                        res += 1
    return res

def get_result():
    test_result = []

    try:
        with time_limit(0.1):
            test_result.append(max_fill([[0,0,1,0], [0,1,0,0], [1,1,1,1]],1))
    except Exception as e:
        test_result.append(None)

    try:
        with time_limit(0.1):
            test_result.append(max_fill([[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]],2))
    except Exception as e:
        test_result.append(None)

    try:
        with time_limit(0.1):
            test_result.append(max_fill([[0,0,0], [0,0,0]],5))
    except Exception as e:
        test_result.append(None)

    return test_result

global final_test_result
final_test_result = get_result()
def check():
    pass_result = []

    try:
        with time_limit(0.1):
            assert max_fill([[0,0,1,0], [0,1,0,0], [1,1,1,1]],1) == 6                    
            pass_result.append(True)
    except Exception as e:
        pass_result.append(False)

    try:
        with time_limit(0.1):
            assert max_fill([[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]],2) == 5                    
            pass_result.append(True)
    except Exception as e:
        pass_result.append(False)

    try:
        with time_limit(0.1):
            assert max_fill([[0,0,0], [0,0,0]],5) == 0                    
            pass_result.append(True)
    except Exception as e:
        pass_result.append(False)

    return pass_result

global final_result
final_result = check()

if __name__=="__main__":
    print(final_result,final_test_result)