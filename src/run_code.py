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


if __name__=="__main__":
    print(tri(2))
    print(tri(10))
    print(tri(4))
    print(tri(11))
    print(tri(12))

    print(intersection((-4, 2), (-2, 6)))
    print(intersection((5, 7), (4, 11)))
    print(intersection((1, 12), (2, 7)))
    print(intersection((-11, 12), (-3, 3)))
    print(intersection((3, 8), (4, 10)))
    print("="*30)
    print(add([4, 5, 6]))
    print(add([4, 3, 6666, 6, 2, 122]))
    print(add([4, 6, 7, 2, 122]))
    print(add([11, 12, 13, 14, 15, 16]))
    print(add([7, 7, 7, 8, 8]))

    print(encode_cyclic("1234"))
    print(decode_cyclic("2314"))

    # check(do_algebra)
    # main()
    from typing import List
    # def mean_absolute_deviation(numbers: List[float]) -> float:
    #     from typing import List
    #     import numpy as np
    #     from statistics import mean
    #     sum_of_squares = sum([(x - np.mean(numbers))**2 for x in numbers])
    #     return np.sqrt(sum_of_squares)

    # code_ans_0 = mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    # assert mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0
    # def factorize(n: int) -> List[int]:
    #     factors = []
    #     while n > 1:
    #         for i in range(1, n):
    #             if n % i == 0:
    #                 print(n,i)
    #                 factors.append(i)
    #                 n =int( n / i)
    #                 break
    #         else:
    #             factors.append(n)
    #     return factors

    # code_ans_0 = factorize(8)
    # assert factorize(8) == [2, 2, 2]
    # assert factorize(25) == [5, 5]
    # assert factorize(70) == [2, 5, 7]
    solution = "init"
    cir = 0
    while cir<10:
        complete = solution
        sol = solution+f" {cir}"
        solution = sol
        cir += 1
    print(complete)
    print(solution)
    print(sol)
    
    print(string_sequence(5))