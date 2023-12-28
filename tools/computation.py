import json
from collections import defaultdict
import matplotlib.pyplot as plt
from pylab import mpl
from data_tools import data_analysis,list_neighbor_diff,Counter_with_base

def computation_expr1(n,b=1,d=4096,V=32000):
    one_block = 24*b*n*d*d + 4*b*n*n*d
    total = 32*one_block + 2*b*n*d*V
    return total

def computation_expr2(n,m,b=1,d=4096,V=32000):
    one_block = 24*b*d*d*(n-1) + 2*b*d*(n*n+m*m-n-m)
    total = 32*one_block + 2*b*d*V*(n-1)
    return total


def computation_expr3(ave_m2,ave_m,Q,b=1,d=4096,V=32000):
    res = 32*24*b*d*d*(2*ave_m-Q-1)+2*b*d*V*(2*ave_m-Q-1)
    res = res +32*2*b*d*(5*ave_m2-(4*Q-7)*ave_m+Q*Q)
    return res

def sita(Q,m1,m11,b=1,d=4096,V=32000):
    r = m11*m11-m1*m1-m11+m1-2*Q*m11+2*Q*m1+Q*Q
    res = 32*24*b*d*d*(m11-m1)+2*b*d*V*(m11-m1) + 32*2*b*d*r
    return res

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
    task_computation = {}
    with open(resfile,"r") as f:
        for line in f.readlines():
            list_m = []
            list_m2 = []
            list_n = []
            m1 = 0
            m11 = 0
            fix_len = 0
            length_record = []
            data = json.loads(line)
            tid = data["task_id"]
            print(f"----------------{tid}----------------")
            completions = data["completion"]
            input_tokens_len = data["step_one_tokens_len"]
            length_record.append([input_tokens_len])
            step_one_computation = computation_expr2(input_tokens_len[1],input_tokens_len[0])
            print(f"step_one_computation:{step_one_computation}")
            fix_record = data["fix_record"]
            for record in fix_record:
                cir = record["cir"]
                fix_percents = record["fix_percents"]
                length_record.append(fix_percents)
                numbers = 0
                input_len_record = []
                input2_len_record = []
                output_len_record = []
                total_fix_len = 0
                for r in fix_percents:
                    input_length = r[0]
                    fix_length = r[1]
                    output_length = r[3]
                    total_fix_len += fix_length
                    input_len_record.append(input_length)
                    output_len_record.append(output_length)
                    input2_len_record.append(input_length*input_length)
                    numbers += 1
                ave_in = sum(input_len_record)/numbers
                ave_in2 = sum(input2_len_record)/numbers
                ave_out = sum(output_len_record)/numbers
                fix_len = total_fix_len/numbers
                list_m.append(ave_in)
                list_m2.append(ave_in2)
                list_n.append(ave_out)
                if cir==1:
                    m1 = ave_in
            print(f"m:{len(list_m)},n:{len(list_n)}")
            len_m = len(list_m)
            if len_m == 0:
                continue
            list_Q = []
            for i in range(len_m-1):
                list_Q.append(list_m[i]+list_m[i+1]-list_n[i])
            if len(list_Q)==0:
                Q = 2*list_m[0] - list_n[0]
            else:
                Q = sum(list_Q)/len(list_Q)
            m11 = list_n[-1] + Q - list_m[-1]
            ave_m = sum(list_m)/len(list_m)
            list_m_m = [m*m for m in list_m]
            ave_m2 = sum(list_m_m)/len(list_m_m)
            max_percent = computation_expr1(fix_len)/computation_expr3(ave_m2,ave_m,Q)
            task_computation[tid] = {"step_one_computation":step_one_computation,"ave_m":ave_m,"ave_m2":ave_m2,"Q":Q,"sita":sita(Q,m1,m11),"fix_len":fix_len,"max_percent":max_percent}
            print(task_computation[tid])
    return task_computation

def load_length3(resfile):
    task_input_length = {}
    task_output_length = {}
    total_diff = []
    with open(resfile,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            completions = data["completion"]
            fix_record = data["fix_record"]
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            cir_in_len = []
            for record in fix_record:
                cir = record["cir"]
                fix_percents = record["fix_percents"]
                numbers = 0
                input_length_record = []
                for r in fix_percents:
                    input_length = r[0]
                    fix_length = r[1]
                    output_length = r[3]
                    input_length_record.append(input_length)
                    numbers += 1
                print(f"task {tid} cir {cir} : input length")
                data_analysis(input_length_record)
                ave_len = sum(input_length_record)/len(input_length_record)
                cir_in_len.append(ave_len)
            total_diff.append(list_neighbor_diff(cir_in_len))
    print(total_diff)
    total_diff = sum(total_diff,[])
    Counter_with_base(total_diff)
    data_analysis(total_diff)
    return 


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


def draw_plots_percent2(data,color=None,image_path=""):
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    for d in data:
        labels.append(d["label"])
        x = range(len(d["value"]))
        xs.append(x)
        y = [round(p,4) for p in d["value"]]
        ys.append(y)
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("Cirs",fontsize='large')
    plt.ylabel("percent:%",fontsize='large')
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    
    plots = []
    for i in range(len(data)):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker='o',color=data[i]["color"],linewidth=2)
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
    file = "../res/UTfeedback_CODETPass_7b16k_pT.jsonl"
    task_computation = load_length2(file)
    task_max = {}
    for tid,com in task_computation.items():#{"step_one_computation":step_one_computation,"ave_m":ave_m,"ave_m2":ave_m2,"Q":Q,"sita":sita(Q,m1,m11),"fix_len":fix_len,"max_percent":max_percent}
        task_max[tid] = com["max_percent"]
    print(task_max)
    task_max_sorted = sorted(task_max.items(),key=lambda x:x[1])
    min_task_max_id = task_max_sorted[0][0]
    max_task_max_id = task_max_sorted[len(task_max_sorted)-1][0]
    print(f"min_task_max_id:{min_task_max_id}max_task_max_id:{max_task_max_id}")
    max_y = []
    min_y = []
    for i in range(10):
        j = i+1
        com1 = task_computation[min_task_max_id]
        com2 = task_computation[max_task_max_id]
        y1 = computation_expr1(com1["fix_len"])/(com1["step_one_computation"]/j+computation_expr3(com1["ave_m2"],com1["ave_m"],com1["Q"])+com1["sita"]/j)
        y2= computation_expr1(com2["fix_len"])/(com2["step_one_computation"]/j+computation_expr3(com2["ave_m2"],com2["ave_m"],com2["Q"])+com2["sita"]/j)
        max_y.append(y2)
        min_y.append(y1)
    ave_y = []
    for i in range(10):
        j = i+1
        total = 0
        numbers = 0
        for k in task_computation.keys():
            com1 = task_computation[k]
            y = computation_expr1(com1["fix_len"])/(com1["step_one_computation"]/j+computation_expr3(com1["ave_m2"],com1["ave_m"],com1["Q"])+com1["sita"]/j)
            total += y
            numbers += 1
        ave_y.append(total/numbers)
    data = []
    data.append({"label":"max_percent","value":max_y,"color":"green"})
    data.append({"label":"min_percent","value":min_y,"color":"red"})
    data.append({"label":"average_percent","value":ave_y,"color":"blue"})
    draw_plots_percent2(data=data,image_path="../image/fix_computation_percent_last2.jpg")
    # task_computation = load_length(file)
    # compute_percents(task_computation)
    # file = "../res/UTfeedback_multiCODETfilter4_7b16k_pT.jsonl"
    # task_computation = load_length2(file)
    # percents = compute_percents(task_computation)
    # draw_plots_percent({"fix_computation_percent":percents},image_path = "../image/fix_computation_percent.jpg")
    # draw_plots_percent({"first_step_computation_percent":other[2]},image_path = "../image/first_step_computation_percent.jpg")