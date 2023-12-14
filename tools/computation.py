import json
from collections import defaultdict
import matplotlib.pyplot as plt
from pylab import mpl

def computation_expr1(n,b=1,d=4096,V=32000):
    one_block = 24*b*n*d*d + 4*b*n*n*d
    total = 32*one_block + 2*b*n*d*V
    return total

def computation_expr2(n,m,b=1,d=4096,V=32000):
    one_block = 24*b*d*d*(n-1) + 2*b*d*(n*n+m*m-n-m)
    total = 32*one_block + 2*b*d*V*(n-1)
    return total

def load_length(resfile):
    task_length_record = {}
    task_computation = {}
    with open(resfile,"r") as f:
        for line in f.readlines():
            computation = []
            length_record = []
            data = json.loads(line)
            tid = data["task_id"]
            completions = data["completion"]
            input_tokens_len = data["input_tokens_len"]
            length_record.append([input_tokens_len])
            step_one_computation = computation_expr1(input_tokens_len)
            computation.append({"cir":0,"com":step_one_computation})
            fix_record = data["fix_record"]
            for record in fix_record:
                cir = record["cir"]
                fix_percents = record["fix_percents"]
                length_record.append(fix_percents)
                total_input_com = 0
                total_fix_com = 0
                numbers = 0
                for r in fix_percents:
                    input_length = r[0]
                    fix_length = r[1]
                    input_com = computation_expr1(input_length)
                    fix_com = computation_expr1(fix_length)
                    total_input_com += input_com
                    total_fix_com += fix_com
                    numbers += 1
                computation.append({"cir":cir,"com":(total_input_com,total_fix_com,numbers)})
            task_length_record[tid] = length_record
            task_computation[tid] = computation
            print(f"task {tid} record {len(length_record)} cirs's length")
            print(f"task {tid} record {len(computation)} cirs's computation")
    return task_computation

def load_length2(resfile):
    task_length_record = {}
    task_computation = {}
    with open(resfile,"r") as f:
        for line in f.readlines():
            computation = []
            length_record = []
            data = json.loads(line)
            tid = data["task_id"]
            completions = data["completion"]
            input_tokens_len = data["step_one_tokens_len"]
            length_record.append([input_tokens_len])
            step_one_computation = computation_expr2(input_tokens_len[1],input_tokens_len[0])
            print(f"step1:{step_one_computation}")
            computation.append({"cir":0,"com":step_one_computation})
            fix_record = data["fix_record"]
            for record in fix_record:
                cir = record["cir"]
                fix_percents = record["fix_percents"]
                length_record.append(fix_percents)
                total_input_com = 0
                total_fix_com = 0
                numbers = 0
                for r in fix_percents:
                    input_length = r[0]
                    fix_length = r[1]
                    output_length = r[3]
                    input_com = computation_expr2(output_length,input_length)
                    fix_com = computation_expr1(fix_length)
                    total_input_com += input_com
                    total_fix_com += fix_com
                    numbers += 1
                computation.append({"cir":cir,"com":(total_input_com,total_fix_com,numbers)})
            task_length_record[tid] = length_record
            task_computation[tid] = computation
            print(f"task {tid} record {len(length_record)} cirs's length")
            print(f"task {tid} record {len(computation)} cirs's computation")
    return task_computation

def compute_percents(task_computation):
    bt = 0
    ct = 0
    ave_a = 0
    ave_b = []
    ave_c = []
    percents = [0]
    step2_percents = [0]
    tids  = list(task_computation.keys())
    n = len(tids)
    a = 0
    for tid in tids:
        a +=task_computation[tid][0]["com"]
    ave_a = (1.0*a)/n
    for i in range(1,11):
        total_b = 0
        total_c = 0
        m = 0
        for tid in tids:
            if i < len(task_computation[tid]):
                total_b += task_computation[tid][i]["com"][0]
                total_c += task_computation[tid][i]["com"][1]
                m += 1
        print(f"for cir {i} compute {m} task")
        bt += total_b
        ct += total_c
        b = (1.0*total_b)/m
        c = (1.0*total_c)/m
        ave_b.append(b)
        ave_c.append(c)
        percent = (c/(ave_a+b))*100
        percents.append(percent)
        step2_percent = (ave_a/(ave_a+b))*100
        step2_percents.append(step2_percent)
    print(percents)
    print(step2_percents)
    print((ct/(a+bt))*100)
    return percents
    
def draw_plots_percent(data,color=None,image_path=""):
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = [round(p,4) for p in v]
        ys.append(y)
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("Cirs",fontsize='large')
    plt.ylabel("percent:%",fontsize='large')
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    
    plots = []
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker='o',color="black",linewidth=2)
        plots.append(p)
        for xz,yz in zip(x,y):
            plt.text(xz,yz+0.5,yz,fontsize='large')
        # plt.text(10+0.1,y[10],labels[i],fontsize="x-large",color="black")
    plt.legend(handles=plots,labels=labels,loc="upper left")#填best自动找到最好的位置
    plt.xticks(range(14),[str(i) for i in range(14)])
    fig.savefig(image_path)
    return

fix_com_percents = {
    1:[0, 62.162943546352345, 65.36767087157672, 64.01600952505848, 63.26178853591353, 62.803607362082616, 62.6334021117471, 62.27913714979484, 61.83789417394393, 61.64282775152817, 61.15799337009814],
}
other = {
    1:[0, 94.57844393557944, 99.20674091173184, 99.43065235012286, 99.43833215092668, 99.44513201275672, 99.45420065650703, 99.46151845761389, 99.47129909318733, 99.47296686981197, 99.47711210115497],
    2:[ 5.421556064420557, 0.7932590882681655, 0.5693476498771456, 0.5616678490733055, 0.554867987243292, 0.5457993434929751, 0.5384815423861051, 0.5287009068126751, 0.5270331301880408, 0.5228878988450234],
}


if __name__=="__main__":
    # file = "../res/UTfeedback_multiCODETfilter3_7b16k_pT.jsonl"
    # task_computation = load_length(file)
    # compute_percents(task_computation)
    # file = "../res/UTfeedback_multiCODETfilter4_7b16k_pT.jsonl"
    # task_computation = load_length2(file)
    # percents = compute_percents(task_computation)
    # draw_plots_percent({"fix_computation_percent":percents},image_path = "../image/fix_computation_percent.jpg")
    draw_plots_percent({"first_step_computation_percent":other[2]},image_path = "../image/first_step_computation_percent.jpg")