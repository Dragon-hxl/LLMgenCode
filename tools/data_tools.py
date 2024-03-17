import json
from collections import defaultdict,Counter
def data_analysis(data:list):
    """
    return and print the min,max,mid,mean,max-min of data
    data为列表，表中元素为整数或浮点数。函数计算并打印列表元素的最大最小值，中位数，均值，以及最大最小值之差
    """
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
    return max_num,min_num,mid,mean,max_num-min_num

def Counter_with_base(data,base=10):
    """
    data中的元素先除以base取整，然后统计其分布
    for x in data, x = x/base and then Counter(data)
    """
    uniform_data = [int(x/base) for x in data]
    print(f"Counter_with_base {base} : {Counter(uniform_data)}")
    return Counter(uniform_data)

def list_neighbor_diff(data:list):
    """
    获取data相邻元素的差值形成列表返回
    for i in range(1,len(data)) return a list of data[i] - data[i-1]
    """
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
    


    
# 本模块函数说明
# data_analysis(data:list)  data为列表，表中元素为整数或浮点数。函数计算并打印列表元素的最大最小值，中位数，均值，以及最大最小值之差
# Counter_with_base(data,base=10)  data中的元素先除以base取整，然后统计其分布
# list_neighbor_diff(data:list)  获取data相邻元素的差值形成列表返回