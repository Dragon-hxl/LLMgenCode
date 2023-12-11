import json
from collections import defaultdict,Counter

def data_analysis(data:list):
    data = sorted(data)
    n = len(data)
    max_num = max(data)
    min_num = min(data)
    if n%2:
        mid = data[int(n/2)]
    else:
        mid = (data[int(n/2)-1]+data[int(n/2)])
    mean = 1.0*sum(data)/n
    print(f"Data with length {n}, max_num:{max_num}, min_num:{min_num}, mid_num:{mid}, mean:{mean}")
    return

def Counter_with_base(data,base=10):
    uniform_data = [int(x/base) for x in data]
    return Counter(uniform_data)

    