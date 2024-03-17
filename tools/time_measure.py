import json
from data_tools import data_analysis
import numpy as np

def load_time_record(res_file):
    time_record = {}
    with open(res_file, 'r') as f:
        res = f.readlines()
        for line in res:
            data = json.loads(line)
            tid = data["task_id"]
            time_record[tid] = data["time_record"]
    return time_record

def time_compare(time_record1, time_record2):
    for tid in time_record1.keys():
        if tid not in time_record2.keys():
            continue
        print("=====================================")
        print(f"tid:{tid}")
        t1 = time_record1[tid]
        t2 = time_record2[tid]
        n = min(len(t1), len(t2))
        inference_time1 = 0
        inference_time2 = 0
        print(f"len t1:{len(t1)}, len t2:{len(t2)}, n:{n}")
        for i in range(n):
            cir = int(t1[i]["cir"])
            print(f"cir{cir}")
            if  cir == 0 or cir==1:
                continue
            inference_time1 += t1[i]["model_inference_time_pure"]/(t1[i]["gened_nodes_num"]/8)
            inference_time2 += t2[i]["model_inference_time_pure"]/(t2[i]["gened_nodes_num"]/8)
        if inference_time2 == 0:
            print("inference_time2 is 0, skip")
            continue
        inference_time1 = inference_time1 / (n-1)
        inference_time2 = inference_time2 / (n-1)
        speed_up = (inference_time2 - inference_time1) / inference_time2 * 100
        print(f"inference_time1:{inference_time1}, inference_time2:{inference_time2}, speed_up:{speed_up}%")
    return

def load_speed_percent(res_file):
    speed_up_list = []
    with open(res_file, 'r') as f:
        res = f.readlines()
        for line in res:
            data = json.loads(line)
            speed_up = data["average_speed_up"]
            if not np.isnan(speed_up):
                speed_up_list.append(speed_up)
    data_analysis(speed_up_list)
    return speed_up


if __name__=="__main__":
    # res_file1 = "../try/fastdebug_final.jsonl"
    # res_file2 = "../try/fastdebug_base.jsonl"
    # fast_time_record = load_time_record(res_file1)
    # base_time_record = load_time_record(res_file2)
    # time_compare(fast_time_record, base_time_record)
    load_speed_percent("../try/fastdebug.jsonl")
    
    