
import json
from evaluate import load_results
from time_measure import load_time_record,time_analysis
from tools.data_tools import data_analysis
from collections import defaultdict
# from tools.computation import computation_expr1,computation_expr2,computation_expr3,load_length2

def computation_expr1(n,b=1,d=4096,V=32000):
    one_block = 24*b*n*d*d + 4*b*n*n*d
    total = 32*one_block + 2*b*n*d*V
    return total

def computation_expr2(n,m,b=1,d=4096,V=32000):
    one_block = 24*b*d*d*(n-1) + 2*b*d*(n*n+m*m-n-m)
    total = 32*one_block + 2*b*d*V*(n-1)
    return total

def load_length_record(res_file):
    len_record = {}
    with open(res_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            len_record[tid] = data["fix_record"]
            
    return len_record
def fix_percent_analysis(len_record):
    fix_percent = []
    fix_percent_sq = []
    for tid,fix_record in len_record.items():
        for record in fix_record:
            len_r = record["len_record"] #(input_length,fix_input_len,fix_percent,output_length,i)
            for r in len_r:
                input_length = r[0]
                fix_input_length = r[1]
                fix_percent_sq.append((fix_input_length*(fix_input_length - 1.0))/(input_length*(input_length - 1.0)))
                fix_percent.append(fix_input_length/input_length)
    print("fix_percent")
    data_analysis(fix_percent)
    print("fix_percent_square")
    data_analysis(fix_percent_sq)
    return fix_percent

def computation_analysis(len_record):
    computation_time = []
    for tid,fix_record in len_record.items():
        for record in fix_record:
            len_r = record["len_record"] #(input_length,fix_input_len,fix_percent,output_length,i)
            for r in len_r:
                input_length = r[0]
                fix_input_length = r[1]
                output_length = r[3]
                fix_com = computation_expr1(fix_input_length)
                total_com = computation_expr2(output_length,input_length)
                computation_time.append(fix_com/total_com)
    print("computation_percent")
    data_analysis(computation_time)
    return computation_time

def show_distribute_percent(data):
    f0_t10 = [d for d in data if d<=10 and d>=0]
    f10_t20 = [d for d in data if d<=20 and d>10]
    f20_t30 = [d for d in data if d<=30 and d>20]
    f30_t40 = [d for d in data if d<=40 and d>30]
    f40_t50 = [d for d in data if d<=50 and d>40]
    print(f"0-10 : {len(f0_t10)/len(data)}")
    print(f"10-20 : {len(f10_t20)/len(data)}")
    print(f"20-30 : {len(f20_t30)/len(data)}")
    print(f"30-40 : {len(f30_t40)/len(data)}")
    print(f"40-50 : {len(f40_t50)/len(data)}")
    
    m10_0 = [d for d in data if d<0 and d>=-10]
    m20_m10 = [d for d in data if d<-10 and d>=-20]
    m30_m20 = [d for d in data if d<-20 and d>=-30]
    m40_m30 = [d for d in data if d<-30 and d>=-40]
    m50_m40 = [d for d in data if d<-40 and d>=-50]
    print(f"-10-0 : {len(m10_0)/len(data)}")
    print(f"-20-10 : {len(m20_m10)/len(data)}")
    print(f"-30-20 : {len(m30_m20)/len(data)}")
    print(f"-40-30 : {len(m40_m30)/len(data)}")
    print(f"-50-40 : {len(m50_m40)/len(data)}")
    
    return
    

def load_speedup_percent(res_file):
    speedup_percent = []
    task_speedup = defaultdict(list)
    with open(res_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            speedup = data["speed_up"]
            speedup_percent += speedup
            task_speedup[tid] += speedup
    speedup_percent = [s for s in speedup_percent if s<50 and s>=-50]
    print("speedup_percent")
    data_analysis(speedup_percent)
    print("speedup_percent distribution")
    show_distribute_percent(speedup_percent)
    average_task_speedup = []
    for tid,speedup in task_speedup.items():
        if len(speedup)>0:
            average_task_speedup.append(sum(speedup)/len(speedup))
    print("average_task_speedup")
    data_analysis(average_task_speedup)
    return speedup_percent

if __name__=="__main__":
    #fix percent analysis
    # res_file = "/home/S/hexiaolong/codex/self-debug/res/"+"mtpbTS_SBSP10_codellama7bpy.jsonl"#"humanevalTS_SBSP10_7b16k.jsonl"
    # len_record = load_length_record(res_file=res_file)
    # fix_percent_analysis(len_record)
    # computation_analysis(len_record)
    #speed up percent analysis
    log_files = [
        "fastdebug_7b16k_prefill.jsonl",
        "fastdebug_7b16k_mbpp_prefill.jsonl",
        "fastdebug_7b16k_mtpb_prefill.jsonl",
        "fastdebug_7b16k_bigbench_prefill.jsonl",
        
        
        "fastdebug_mtpb_codellama7bpy.jsonl",
        "fastdebug_bigbench_codellama7bpy.jsonl",
        "fastdebug_cola7bpy_bigbench_prefill.jsonl",
        "fastdebug_cola7bpy_humaneval_prefill.jsonl",
        "fastdebug_cola7bpy_mbpp_prefill.jsonl",
    ]
    res_file = "/home/S/hexiaolong/codex/self-debug/res/"+log_files[1]#"humanevalTS_SBSP10_7b16k.jsonl"
    speedup_percent = load_speedup_percent(res_file=res_file)
            