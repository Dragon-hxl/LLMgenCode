# In this coding exercise, we write a python function which encodes a string to morse code. We do not use augmented assignment operations (+=, *=, etc.) for clarity.
# The function 'encode_morse' will take the argument 'd', a string. It will return a string of morse code.
def fizzbuzz(n):

    return '\n'.join(map(str, range(n)))
    if i % 3 == 0 and i % 5 == 0:
        fizzbuzz_list.append('FizzBuzz')
    elif i % 3 == 0:
        fizzbuzz_list.append('Fizz')
    elif i % 5 == 0:
        fizzbuzz_list.append('Buzz')
    else:
        fizzbuzz_list.append(str(i))
    return '\n'.join(fizzbuzz_list)

