import json
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

pass_record = {
    "total": range(164),
    "UT_7b16k_t1": ['0', '2', '4', '7', '12', '13', '14', '15', '22', '23', '28', '29', '30', '31', '34', '35', '40', '42', '43', '44', '45', '47', '48', '51', '52', '53', '55', '58', '60', '101', '124', '139', '143', '152', '162'],
    "UT_7b16k_trueTest_t1": ['0', '7', '8', '11', '12', '13', '15', '18', '22', '23', '27', '28', '29', '30', '31', '34', '35', '40', '42', '43', '45', '48', '51', '53', '55', '58', '60', '67', '77', '82', '87', '96', '98', '101', '124', '143', '157', '162'],
    "UT_7b16k_codeTTest_t1": ['0', '4', '7', '8', '12', '13', '15', '16', '22', '23', '27', '28', '29', '30', '31', '34', '35', '40', '42', '43', '44', '45', '48', '51', '52', '53', '55', '58', '60', '101', '105', '124', '152', '162'],
    "UT_cola34bpy_t1": ['0', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '33', '34', '35', '36', '37', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '77', '79', '80', '85', '86', '87', '89', '90', '91', '92', '94', '95', '96', '97', '98', '101', '104', '105', '106', '107', '111', '113', '116', '117', '121', '122', '123', '124', '133', '136', '142', '143', '146', '147', '151', '152', '153', '155', '156', '158', '160', '162'],
    "UT_cola34bpy_t1_trueTest": ['0', '1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '27', '28', '29', '30', '31', '33', '34', '35', '37', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '51', '52', '53', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '71', '72', '73', '74', '76', '78', '79', '80', '85', '86', '87', '89', '90', '91', '92', '96', '97', '101', '104', '105', '106', '107', '111', '113', '115', '116', '117', '121', '122', '124', '127', '128', '133', '136', '142', '143', '144', '146', '147', '148', '149', '151', '152', '153', '155', '156', '157', '159', '161', '162'],
    "UT_SBP10_7b16k_pT": [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 65, 67, 72, 79, 80, 84, 85, 86, 87, 89, 95, 96, 101, 105, 120, 121, 124, 133, 136, 143, 152, 158, 159, 162],
    "UT_SBS10_7b16k_pT": [0, 2, 4, 7, 8, 12, 13, 14, 15, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 67, 72, 79, 82, 87, 88, 96, 98, 101, 116, 121, 124, 133, 134, 136, 143, 152, 162],
    "UT_SBS10_7b16k_pT": [0, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 42, 43, 45, 47, 48, 51, 52, 53, 54, 55, 58, 60, 67, 79, 82, 84, 87, 89, 98, 101, 121, 124, 127, 143, 152, 157, 159, 162],
    "UT_SBP10_7b16k_tT": [0, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 63, 66, 67, 68, 69, 72, 76, 77, 78, 79, 80, 82, 84, 85, 86, 87, 89, 90, 91, 92, 95, 96, 97, 98, 101, 105, 108, 109, 110, 111, 112, 116, 120, 121, 122, 124, 127, 133, 135, 136, 139, 142, 143, 150, 152, 157, 159, 162],
    "UT_SBSP10_7b16k_pT":[0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 58, 60, 61, 62, 66, 67, 72, 78, 79, 82, 84, 85, 86, 87, 89, 94, 95, 96, 97, 101, 112, 115, 117, 120, 121, 122, 124, 127, 132, 133, 134, 137, 143, 152, 159, 162],
    "UT_SBSP10_7b16k_tT":[0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 66, 67, 72, 78, 79, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 97, 98, 99, 100, 101, 105, 108, 109, 110, 112, 116, 120, 121, 122, 124, 127, 128, 133, 135, 136, 138, 139, 142, 143, 149, 150, 152, 153, 155, 157, 159, 162]
}

cir_record = {
    "UT_SBP10_7b16k_pT":[27, 39, 53, 58, 65, 66, 69, 69, 69, 69, 69],
    "UT_SBP10_7b16k_tT":[27, 27, 41, 63, 73, 77, 83, 86, 91, 91, 93],
    "UT_SBS10_7b16k_pT":[27, 38, 44, 46, 47, 47, 47, 50, 49, 50, 51],
    "UT_SBS10_7b16k_tT":[27, 41, 41, 43, 43, 46, 46, 48, 48, 51, 53],
    "UT_SBSP10_7b16k_pT":[27, 36, 51, 61, 63, 66, 70, 71, 71, 72, 74],
    "UT_SBSP10_7b16k_pT_pass@1":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
    "UT_SBSPCODET_7b16k_pT":[27, 40, 51, 57, 64, 65, 66, 69, 69, 70, 70],
    "UT_SBSPCODET_7b16k_pT_pass@1":[27, 39, 50, 55, 63, 64, 65, 68, 68, 69, 69],
    "UT_SBSPCODET2_7b16k_pT":[27, 40, 53, 55, 60, 61, 61, 61, 61, 61, 62],
    "UT_SBSPCODET2_7b16k_pT_pass@1":[27, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36],
    "UT_SBSP10_7b16k_tT":[27, 40, 58, 69, 78, 81, 85, 89, 92, 94, 96],
    "UT_SBSP10_7b16k_tT_pass@1":[27, 40, 55, 65, 75, 78, 82, 86, 89, 90, 92],
    "UT_7b16k_t1_pT": [27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
    "UT_7b16k_t1_tT": [27, 28, 32, 33, 35, 35, 35, 36, 36,38, 38],
    "UT_7b16k_t1_cT": [27, 27, 26, 27, 27, 29, 33, 32, 34, 34, 34],
}

color_map = {
    "UT_SBP10_7b16k_pT":"red",
    "UT_SBP10_7b16k_tT":"darkred",
    "UT_SBS10_7b16k_pT":"gold",
    "UT_SBS10_7b16k_tT":"darkorange",
    "UT_SBSP10_7b16k_pT":"grey",
    "UT_SBSP10_7b16k_pT_pass@1":"black",
    "UT_SBSP10_7b16k_tT":"green",
    "UT_SBSP10_7b16k_tT_pass@1":"darkgreen",
    "UT_SBSPCODET_7b16k_pT":"red",
    "UT_SBSPCODET_7b16k_pT_pass@1":"darkred",
    "UT_SBSPCODET2_7b16k_pT":"chocolate",
    "UT_SBSPCODET2_7b16k_pT_pass@1":"sienna",
    "UT_7b16k_t1_pT": "c",
    "UT_7b16k_t1_tT": "darkblue",
    "UT_7b16k_t1_cT": "green",
}


def displaySpecifiedCode(solution_file,task_list):
    with open(solution_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            task_id_int = int(task_id.split("/")[1])
            if task_id_int not in task_list:
                continue
            else:
                for cir,solutions in data["completion"]:
                    print(f"-----------------task {task_id} cir {cir}-----------------------")
                    for solution,passed in solutions:
                        print(f"{solution}\npassed: {passed}")
                    print("-----------------------------------------------------------------")
def draw_plots(data,image_path):
    # data format: {"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(9,6),dpi=400)
    plt.xlabel("Cirs")
    plt.ylabel("number of task")
    plt.title(image_path.split(".")[0])
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)
        for xz,yz in zip(x,y):
            plt.text(xz,yz,yz)
    plt.legend(labels,loc="upper left")
    fig.savefig(image_path)
    return

def draw_plots_percent(data,color,image_path):
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("Cirs",fontsize='large')
    plt.ylabel("Pass percent:%",fontsize='large')
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    
    plots = []
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker='o',color=color[labels[i]],linewidth=2)
        plots.append(p)
        for xz,yz in zip(x,y):
            plt.text(xz,yz+0.5,yz,fontsize='large')
        plt.text(10+0.1,y[10],labels[i],fontsize="x-large",color=color[labels[i]])
    plt.legend(handles=plots,labels=labels,loc="upper left")#填best自动找到最好的位置
    plt.xticks(range(14),[str(i) for i in range(14)])
    fig.savefig(image_path)
    return


def showDifferent(label1,label2):
    print(f'\nshow difference between {label1} and {label2}')
    v1 = pass_record[label1]
    v2 = pass_record[label2]
    if type(v1[0]) is str:
        v1 = [int(x) for x in v1]
    if type(v2[0]) is str:
        v2 = [int(x) for x in v2]
    v1 = set(v1)
    v2 = set(v2)
    print(f"label1 - label2 = {len(v1 - v2)}\n{sorted(v1 - v2)}")
    print(f"label2 - label1 = {len(v2 - v1)}\n{sorted(v2 - v1)}")
    print(f"label1 + label2 = {len(v1 | v2)}\n{sorted(v1 | v2)}")
    total = set(pass_record["total"])
    print(f"total - label1 - label2 = {len(total - (v1 | v2))}\n{sorted(total - (v1 | v2))}")
    return
    


if __name__=="__main__":
    # showDifferent("UT_SBSP10_7b16k_tT","UT_SBSP10_7b16k_pT")
    # showDifferent("total","UT_cola34bpy_t1_trueTest")
    total_pass = set()
    for id,pass_list in pass_record.items():
        if id == "total" or ("7b16k" not in id) or ("SBSP" not in id):
            continue
        for t in pass_list:
            if type(t)==str:
                t = int(t)
            total_pass.add(t)
    total = set(pass_record["total"])
    print(f"total pass num is {len(total_pass)} never pass num is {len(total-total_pass)}:\n{sorted(total-total_pass)}")
    data = {}
    for label,value in cir_record.items():
        if "SBSP" not in label:
            continue
        data[label] = [round((1.0*x)/164*100,1) for x in value]
    draw_plots_percent(data, color_map, "../image/UTfeedback_7b16k_CODET.jpg")
    