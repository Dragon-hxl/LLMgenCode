
def average_Odd(n) :
    count = 0
    sum = 0
    for i in range(1,n+1):
        if i % 2 != 0:
            count += 1
            sum += i
    return sum/count

def check(average_Odd):
    assert average_Odd(9) == 5
    assert average_Odd(5) == 3
    assert average_Odd(11) == 6

check(average_Odd)