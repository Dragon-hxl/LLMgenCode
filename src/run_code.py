def prime_fib(n: int):
    import math

    def is_prime(p):
        if p < 2:
            return False
        for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
            if p % k == 0:
                return False
        return True
    f = [0, 1]
    while True:
        f.append(f[-1] + f[-2])
        if is_prime(f[-1]):
            n -= 1
        if n == 0:
            return f[-1]

def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> make_a_pile(3)
    [3, 5, 7]
    """
    return [n + 2*i for i in range(n)]

def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, return a list.
    Each element i of the output should be "the number of odd elements in the
    string i of the input." where all the i's should be replaced by the number
    of odd digits in the i'th string of the input.

    >>> odd_count(['1234567'])
    ["the number of odd elements 4n the str4ng 4 of the 4nput."]
    >>> odd_count(['3',"11111111"])
    ["the number of odd elements 1n the str1ng 1 of the 1nput.",
     "the number of odd elements 8n the str8ng 8 of the 8nput."]
    """
    res = []
    for arr in lst:
        n = sum(int(d)%2==1 for d in arr)
        res.append("the number of odd elements " + str(n) + "n the str"+ str(n) +"ng "+ str(n) +" of the "+ str(n) +"nput.")
    return res

def decimal_to_binary(decimal):
    """You will be given a number in decimal form and your task is to convert it to
    binary format. The function should return a string, with each character representing a binary
    number. Each character in the string will be '0' or '1'.

    There will be an extra couple of characters 'db' at the beginning and at the end of the string.
    The extra characters are there to help with the format.

    Examples:
    decimal_to_binary(15)   # returns "db1111db"
    decimal_to_binary(32)   # returns "db100000db"
    """
    return "db" + bin(decimal)[2:] + "db"

def match_parens(lst):
    '''
    You are given a list of two strings, both strings consist of open
    parentheses '(' or close parentheses ')' only.
    Your job is to check if it is possible to concatenate the two strings in
    some order, that the resulting string will be good.
    A string S is considered to be good if and only if all parentheses in S
    are balanced. For example: the string '(())()' is good, while the string
    '())' is not.
    Return 'Yes' if there's a way to make a good string, and return 'No' otherwise.

    Examples:
    match_parens(['()(', ')']) == 'Yes'
    match_parens([')', ')']) == 'No'
    '''
    def check(s):
        val = 0
        for i in s:
            if i == '(':
                val = val + 1
            else:
                val = val - 1
            if val < 0:
                return False
        return True if val == 0 else False

    S1 = lst[0] + lst[1]
    S2 = lst[1] + lst[0]
    return 'Yes' if check(S1) or check(S2) else 'No'

def intersection(interval1, interval2):
    """You are given two intervals,
    where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
    The given intervals are closed which means that the interval (start, end)
    includes both start and end.
    For each given interval, it is assumed that its start is less or equal its end.
    Your task is to determine whether the length of intersection of these two 
    intervals is a prime number.
    Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
    which its length is 1, which not a prime number.
    If the length of the intersection is a prime number, return "YES",
    otherwise, return "NO".
    If the two intervals don't intersect, return "NO".


    [input/output] samples:
    intersection((1, 2), (2, 3)) ==> "NO"
    intersection((-1, 1), (0, 4)) ==> "NO"
    intersection((-3, -1), (-5, 5)) ==> "YES"
    """
    def is_prime(num):
        if num == 1 or num == 0:
            return False
        if num == 2:
            return True
        for i in range(2, num):
            if num%i == 0:
                return False
        return True

    l = max(interval1[0], interval2[0])
    r = min(interval1[1], interval2[1])
    length = r - l
    if length > 0 and is_prime(length):
        return "YES"
    return "NO"

def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
    Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) =  tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    For example:
    tri(2) = 1 + (2 / 2) = 2
    tri(4) = 3
    tri(3) = tri(2) + tri(1) + tri(4)
           = 2 + 3 + 3 = 8 
    You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    Examples:
    tri(3) = [1, 3, 2, 8]
    """
    if n == 0:
        return [1]
    my_tri = [1, 3]
    for i in range(2, n + 1):
        if i % 2 == 0:
            my_tri.append(i / 2 + 1)
        else:
            my_tri.append(my_tri[i - 1] + my_tri[i - 2] + (i + 3) / 2)
    return my_tri

def add(lst):
    """Given a non-empty list of integers lst. add the even elements that are at odd indices..


    Examples:
        add([4, 2, 6, 7]) ==> 2 
    """
    return sum([lst[i] for i in range(1, len(lst), 2) if lst[i]%2 == 0])


def encode_cyclic(s: str):
    """
    returns encoded string by cycling groups of three characters.
    """
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # cycle elements in each group. Unless group has fewer elements than 3.
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


def decode_cyclic(s: str):
    """
    takes as input string encoded with encode_cyclic function. Returns decoded string.
    """
    return encode_cyclic(encode_cyclic(s))

def do_algebra(operator, operand):
    """
    Given two lists operator, and operand. The first list has basic algebra operations, and 
    the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.

    The basic algebra operations:
    Addition ( + ) 
    Subtraction ( - ) 
    Multiplication ( * ) 
    Floor division ( // ) 
    Exponentiation ( ** ) 

    Example:
    operator['+', '*', '-']
    array = [2, 3, 4, 5]
    result = 2 + 3 * 4 - 5
    => result = 9

    Note:
        The length of operator list is equal to the length of operand list minus one.
        Operand is a list of of non-negative integers.
        Operator list has at least one operator, and operand list has at least two operands.

    """
    result = operand[0]
    for i in range(len(operand)-1):
        result = eval(operator[i]+' '+str(result)+' '+str(operand[i+1]))
    return result

def check(candidate):

    # Check some simple cases
    assert candidate(['**', '*', '+'], [2, 3, 4, 5]) == 37
    assert candidate(['+', '*', '-'], [2, 3, 4, 5]) == 9
    assert candidate(['//', '*'], [7, 3, 4]) == 8, "This prints if this assert fails 1 (good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert True, "This prints if this assert fails 2 (also good for debugging!)"

# def encode(message):
#     """
#     Write a function that takes a message, and encodes in such a 
#     way that it swaps case of all letters, replaces all vowels in 
#     the message with the letter that appears 2 places ahead of that 
#     vowel in the english alphabet. 
#     Assume only letters. 
    
#     Examples:
#     >>> encode('test')
#     'TGST'
#     >>> encode('This is a message')
#     'tHKS KS C MGSSCGG'
#     """
#     def swap_case(s):
#         return s.upper() if s.islower() else s.lower() if s.isupper() else s
#     def replace_vowels(s):
#         return ''.join([swap_case(c) if c.isalpha() else next(alphabet) for c in s])
#     def get_next_vowel(letter):
#         i = alphabet.index(letter) + 1
#         while i < len(alphabet) and alphabet[i] == letter:
#             i += 1
#         return chr(ord('a') + i)
#     message = swap_case(message)
#     return ''.join([''.join(reversed(word)) if word.isalpha() else get_next_vowel(word[0]) for word in re.findall(r'\w+', message)])

# def by_length(arr):
#     sorted_arr = sorted(arr, key=lambda x: x >= 1 and x <= 9)
#     return ''.join([' '.join(sorted_arr[:i+1]), ' ' + 'One' + ' ' + ' '.join(sorted_arr[i+1:]) for i in range(len(sorted_arr))])

def main():
    from typing import List
    def below_zero(operations: List[int]) -> bool:
        for i in range(len(operations)):
            if operations[i] < 0:
                return True
        return False
    assert below_zero([]) == False
    assert below_zero([1, 2, -3, 1, 2, -3]) == False
    assert below_zero([1, 2, -4, 5, 6]) == True
    assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert below_zero([1, -2, 2, -2, 5, -5, 4, -4]) == True

def string_sequence(n: int) -> str:

    if n == 0:
        return '0'
    else:
        return ' '.join([str(i) for i in range(n)])
def smallest_change(arr):
    if len(arr) == 1:
        return 0
    else:
        min_change = float('inf')
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                change = arr[i] ^ arr[j]
                new_arr = arr[:i] + [change] + arr[j+1:]
                if len(new_arr) == len(arr):
                    min_change = min(min_change, smallest_change(new_arr))
        return min_change

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0

from typing import List, Any
def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
    return [x for x in values if isinstance(x, int)]

def mbpp_get_chosen_idx(tid:int):
    mbpp_chosen_idx2 = [240, 93, 372, 296, 155, 102, 454, 370, 209, 387, 366, 388, 135, 272, 125, 325, 416, 376, 255, 181, 212, 269, 497, 315, 111, 158, 278, 360, 169, 265, 38, 374, 396, 443, 105, 352, 385, 477, 239, 363, 425, 446, 334, 75, 486, 108, 444, 210, 29, 394, 178, 321, 213, 238, 63, 371, 380, 71, 390, 167, 199, 471, 176, 406, 494, 166, 218, 479, 162, 290, 109, 208, 117, 104, 20, 383, 115, 441, 9, 132, 258, 163, 395, 291, 411, 361, 215, 314, 57, 438, 457, 310, 399, 118, 120, 237, 187, 69, 103, 188, 252, 304, 448, 72, 134, 198, 319, 172, 171, 362, 364, 458, 86, 350, 356, 67, 410, 465, 297, 351, 33, 50, 88, 2, 77, 224, 472, 405, 179, 427, 41, 100, 145, 122, 355, 236, 308, 417, 246, 268, 223, 339, 432, 435, 36, 154, 354, 142, 402, 289, 338, 128, 478, 51, 253, 475, 368, 450, 90, 263, 114, 418, 480, 23, 496, 473, 193, 324, 37, 60, 492, 28, 470, 64, 107, 412, 44, 419, 377, 462, 249, 298, 84, 82, 323, 326, 53, 398, 287, 309, 15, 312, 55, 286, 92, 409, 161, 0, 62, 143]
    
    task_idx= tid- 11
    
    idx = mbpp_chosen_idx2.index(task_idx)
    
    # print(idx)
    return idx
    


if __name__=="__main__":
    # print( smallest_change([1,2,3,5,4,7,9,6]) == 4)
    # print( smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1)
    # print( smallest_change([1, 2, 3, 2, 1]) == 0)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) == 3)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 6)
    # print( smallest_change([3, 4, 5, 6, 7]) == 2)
    # print( smallest_change([1, 2, 3, 4, 5]) == 1)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 4)
    # print( smallest_change([3, 4, 5, 6, 7]) == 2)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 4)
    # print( smallest_change([1, 2, 5, 6, 7, 8, 4, 3]) == 4)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 5)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8]) == 4)
    # print( smallest_change([1, 2, 3, 4, 5]) == 1)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 7)
    # print( smallest_change([1, 2, 3, 4, 5]) == 2)
    # print( smallest_change([1, 2, 3, 4, 5]) == 1)
    # print( smallest_change([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 3)
    # assert 1+1==3, "Test case 1 failed"
    # assert truncate_number(2.0) == 2.0,"Test case 2 failed"
    assert filter_integers(["apple", "banana", "cherry", 5, 6.0]) == [5]
    assert filter_integers(["hello", "world", 123, 4.5, "bye"]) == [123]
    assert filter_integers(range(1, 6)) == list(range(1, 6))
    assert filter_integers([1, "a", 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert filter_integers(["a", "b", 1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    def check():
        pass_result = []
        try:
            assert filter_integers(["apple", "banana", "cherry", 5, 6.0]) == [5]                   
            pass_result.append(True)
        except Exception as e:
            pass_result.append(False)
        return pass_result
    print(check())
    lack = [0, 1, 2, 4, 6, 7, 12, 15, 23, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 67, 72, 74, 94, 104, 105, 107, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 162]
    ignore = [0, 2, 4, 7, 12, 15, 23, 28, 29, 30, 31, 34, 35, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 124, 162]#, 72, 6, 39, 105,115
    lack_2 = [0, 2, 4, 7, 12, 15, 23, 28, 29, 30, 31, 34, 35, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 94, 105, 115, 124, 129, 162]
    lack_3 = list(set(lack_2) - set(ignore))
    lack_3 = sorted(lack_3)
    passed = [28, 39, 44, 48, 50, 49, 49, 49, 49, 9]
    pass1 = [0, 8, 24, 34, 39, 42, 43, 43, 43, 43, 43]
    pass1  = sorted([x+27 for x in pass1])
    passed = sorted([x+27 for x in passed])
    print(lack_3)
    print(passed)
    print(len(ignore))
    print(pass1)
    
    mbpp_chosen_idx2 = [240, 93, 372, 296, 155, 102, 454, 370, 209, 387, 366, 388, 135, 272, 125, 325, 416, 376, 255, 181, 212, 269, 497, 315, 111, 158, 278, 360, 169, 265, 38, 374, 396, 443, 105, 352, 385, 477, 239, 363, 425, 446, 334, 75, 486, 108, 444, 210, 29, 394, 178, 321, 213, 238, 63, 371, 380, 71, 390, 167, 199, 471, 176, 406, 494, 166, 218, 479, 162, 290, 109, 208, 117, 104, 20, 383, 115, 441, 9, 132, 258, 163, 395, 291, 411, 361, 215, 314, 57, 438, 457, 310, 399, 118, 120, 237, 187, 69, 103, 188, 252, 304, 448, 72, 134, 198, 319, 172, 171, 362, 364, 458, 86, 350, 356, 67, 410, 465, 297, 351, 33, 50, 88, 2, 77, 224, 472, 405, 179, 427, 41, 100, 145, 122, 355, 236, 308, 417, 246, 268, 223, 339, 432, 435, 36, 154, 354, 142, 402, 289, 338, 128, 478, 51, 253, 475, 368, 450, 90, 263, 114, 418, 480, 23, 496, 473, 193, 324, 37, 60, 492, 28, 470, 64, 107, 412, 44, 419, 377, 462, 249, 298, 84, 82, 323, 326, 53, 398, 287, 309, 15, 312, 55, 286, 92, 409, 161, 0, 62, 143]
    
    mbpp_chosen_idx1 = [240, 93, 372, 296, 155, 102, 454, 370, 209, 387, 366, 388, 135, 272, 125, 325, 416, 376, 255, 181, 212, 269, 497, 315, 111, 158, 278, 360, 169, 265, 38, 374, 396, 443, 105, 352, 385, 477, 239, 363, 425, 446, 334, 75, 486, 108, 444, 210, 29, 394, 178, 321, 213, 238, 63, 371, 380, 71, 390, 167, 199, 471, 176, 406, 494, 166, 218, 479, 162, 290, 109, 208, 117, 104, 20, 383, 115, 441, 9, 132, 258, 163, 395, 291, 411, 361, 215, 314, 57, 438, 457, 310, 399, 118, 120, 237, 187, 69, 103, 188, 252, 304, 448, 72, 134, 198, 319, 172, 171, 362, 364, 458, 86, 350, 356, 67, 410, 465, 297, 351, 33, 50, 88, 2, 77, 224, 472, 405, 179, 427, 41, 100, 145, 122, 355, 236, 308, 417, 246, 268, 223, 339, 432, 435, 36, 154, 354, 142, 402, 289, 338, 128, 478, 51, 253, 475, 368, 450, 90, 263, 114, 418, 480, 23, 496, 473, 193, 324, 37, 60, 492, 28, 470, 64, 107, 412, 44, 419, 377, 462, 249, 298, 84, 82, 323, 326, 53, 398, 287, 309, 15, 312, 55, 286, 92, 409, 161, 0, 62, 143]
    
    mbpp_chosen_idx3 = [0, 2, 9, 15, 20, 23, 28, 29, 33, 36, 37, 38, 41, 44, 50, 51, 53, 55, 57, 60, 62, 63, 64, 67, 69, 71, 72, 75, 77, 82, 84, 86, 88, 90, 92, 93, 100, 102, 103, 104, 105, 107, 108, 109, 111, 114, 115, 117, 118, 120, 122, 125, 128, 132, 134, 135, 142, 143, 145, 154, 155, 158, 161, 162, 163, 166, 167, 169, 171, 172, 176, 178, 179, 181, 187, 188, 193, 198, 199, 208, 209, 210, 212, 213, 215, 218, 223, 224, 236, 237, 238, 239, 240, 246, 249, 252, 253, 255, 258, 263, 265, 268, 269, 272, 278, 286, 287, 289, 290, 291, 296, 297, 298, 304, 308, 309, 310, 312, 314, 315, 319, 321, 323, 324, 325, 326, 334, 338, 339, 350, 351, 352, 354, 355, 356, 360, 361, 362, 363, 364, 366, 368, 370, 371, 372, 374, 376, 377, 380, 383, 385, 387, 388, 390, 394, 395, 396, 398, 399, 402, 405, 406, 409, 410, 411, 412, 416, 417, 418, 419, 425, 427, 432, 435, 438, 441, 443, 444, 446, 448, 450, 454, 457, 458, 462, 465, 470, 471, 472, 473, 475, 477, 478, 479, 480, 486, 492, 494, 496, 497]
    print(set(mbpp_chosen_idx1) - set(mbpp_chosen_idx2))
    print(set(mbpp_chosen_idx3) - set(mbpp_chosen_idx1))
    
    print(mbpp_chosen_idx2[0],mbpp_chosen_idx2[50],mbpp_chosen_idx2[100],mbpp_chosen_idx2[150])
    
    mbpp_lack_task = [26,154]
    mbpp_lack_task_idx = [mbpp_get_chosen_idx(i) for i in mbpp_lack_task]
    print(sorted(mbpp_lack_task_idx))
    