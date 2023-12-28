
from typing import Optional, Callable, Dict, List, Tuple
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

def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each
    other and return them in order (smaller number, larger number).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
    (2.0, 2.0)
    """
    pairwise_diff = (num1 - num2 for num1, num2 in zip(numbers, numbers[1:]))
    min_diff = min(pairwise_diff)
    closest_pairs = [(num, min_diff) for num in numbers]
    print(closest_pairs)
    return tuple(closest_pairs)

def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
    print(len(set(s0)))
    print(len(set(s1)))
    print(len(set(s0) & set(s1)))
    # print(len(set(s0) & set(s1)) == len(s1))
    return len(set(s0) & set(s1)) == len(s1)

def even_odd_count(num):
    """Given an integer. return a tuple that has the number of even and odd digits respectively.

     Example:
        even_odd_count(-12) ==> (1, 1)
        even_odd_count(123) ==> (1, 2)
    """
    if num < 0:
        num = abs(num)
    even_count, odd_count = 0, 0
    while num > 0:
        if num % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
        num = num // 10
    return even_count, odd_count

if __name__=="__main__":
    # print(final_result,final_test_result)
    # assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True
    # assert same_chars('abcd', 'dddddddabc') == True
    # assert same_chars('dddddddabc', 'abcd') == True
    # assert same_chars('eabcd', 'dddddddabc') == False
    # assert same_chars('abcd', 'dddddddabcf') == False
    # assert same_chars('eabcdzzzz', 'dddzzzzzzzddddabc') == False
    # assert same_chars('aabb', 'aaccc') == False
    assert even_odd_count(7) == (0, 1)
    assert even_odd_count(-78) == (1, 1)
    assert even_odd_count(3452) == (2, 2)
    assert even_odd_count(346211) == (3, 3)
    assert even_odd_count(-345821) == (3, 3)
    assert even_odd_count(-2) == (1, 0)
    assert even_odd_count(-45347) == (2, 3)
    assert even_odd_count(0) == (1, 0)