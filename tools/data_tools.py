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
        mid = (data[int(n/2)-1]+data[int(n/2)])/2
    mean = 1.0*sum(data)/n
    print(f"Data with length {n}, max_num:{max_num}, min_num:{min_num}, mid_num:{mid}, mean:{mean}, diff:{max_num-min_num}")
    # list_neighbor_diff(data)
    return

def Counter_with_base(data,base=10):
    uniform_data = [int(x/base) for x in data]
    print(f"Counter_with_base {base} : {Counter(uniform_data)}")
    return Counter(uniform_data)

def list_neighbor_diff(data:list):
    n = len(data)
    if n <= 1:
        return [0]
    diff_list = []
    for i in range(1,n):
        diff = data[i] - data[i-1]
        if diff < 0:
            diff = -diff
        diff_list.append(diff)
    print(f"list_neighbor_diff : {diff_list}")
    return diff_list
    