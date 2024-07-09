import json
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt
from pylab import mpl
from draw_tools import draw_plots_mean_std
import numpy as np

from result_store import *
 
# # 设置中文显示字体
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# # 设置正常显示符号
# mpl.rcParams["axes.unicode_minus"] = False
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cir_record = {
    "UT_SBSP10_7b16k_pT_pass@1":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
    "UT_SBSP10_7b16k_tT_pass@1":[27, 40, 55, 65, 75, 78, 82, 86, 89, 90, 92],
    
    "humanevalTS_SBSP1_7b16k_pT": [27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
    "humanevalNTS_SBSP10_7b16k_pT":[27, 35, 50, 56, 61, 64, 67, 68, 68, 68, 68],
    "humanevalTS_SBSP10_7b16k_pT":[27, 36, 51, 60, 63, 66, 69, 70, 72, 72, 72],
    "humanevalTFTS_SBSP10_7b16k_pT":[27, 40, 57, 63, 65, 69, 71, 72, 74, 75, 75],
    
    "mbppTS_SBSP1_7b16k":[38, 40, 41, 43, 44, 46, 48, 48, 48, 50, 51],
    "mbppNTS_SBSP10_7b16k_pass@1":[38, 45, 47, 49, 51, 52, 52, 53, 54, 54, 54],
    "mbppTS_SBSP10_7b16k_pass@1":[38, 48, 51, 55, 58, 58, 58, 58, 58, 58, 58],
    "mbppTFTS_SBSP10_7b16k_pass@1":[38, 50, 54, 59, 59, 59, 60, 60, 60, 60, 60],
    
    "1_mtpbTFTS_SBSP10_7b16k_pass@1":[6, 11, 15, 17, 18, 19, 19, 19, 19, 19, 19],
    "2_mtpbTS_SBSP10_7b16k_pass@1":[6, 11, 14, 16, 17, 17, 17, 17, 17, 17, 17],
    "3_mtpbNTS_SBSP10_7b16k_pass@1":[6, 10, 11, 13, 16, 16, 16, 16, 16, 16, 16],
    "4_mtpbTS_SBSP1_7b16k":[6, 8, 9, 11, 12, 14, 14, 14, 14, 14, 14],
    
    "bigbenchTS_SBSP1_7b16k":[7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "bigbenchNTS_SBSP10_7b16k_pass@1":[7, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11],
    "bigbenchTS_SBSP10_7b16k_pass@1":[7, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11],
    "bigbenchTFTS_SBSP10_7b16k_pass@1":[7, 7, 8, 11, 12, 12, 12, 12, 12, 12, 12],
    
    "humanevalTS_SBSP1_codellama7bpy_pass@1":[63, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67],
    "humanevalNTS_SBSP10_codellama7bpy_pass@1":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],
    "humanevalTS_SBSP10_codellama7bpy_pass@1":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],
    "humanevalTFTS_SBSP10_codellama7bpy_pass@1":[63, 81, 100, 105, 109, 111, 111, 112, 112, 113, 115],
    
    "humanevalTS_SBSP1_codellama34bpy_pass@1":[85, 93, 95, 95, 95, 95, 95, 95, 95, 95, 95],
    "humanevalNTS_SBSP10_codellama34bpy_pass@1":[85, 106, 113, 115, 115, 116, 116, 116, 116, 116, 116],
    "humanevalTS_SBSP10_codellama34bpy_pass@1": [85, 106, 113, 115, 115, 116, 116, 116, 116, 116, 116],
    "humanevalTFTS_SBSP10_codellama34bpy_pass@1":[85, 107, 116, 117, 119, 120, 120, 120, 120, 121, 121],
    
}

color_map = {
    "humanevalTS_SBSP1_7b16k_pT_pass@1": "blue",
    "humanevalNTS_SBSP10_7b16k_pT_pass@1":"grey",
    "humanevalTS_SBSP10_7b16k_pT_pass@1":"orange",
    "humanevalTFTS_SBSP10_7b16k_pT_pass@1":"green",
    
    "mbppTS_SBSP1_7b16k":"blue",
    "mbppNTS_SBSP10_7b16k_pass@1":"grey",
    "mbppTS_SBSP10_7b16k_pass@1":"orange",
    "mbppTFTS_SBSP10_7b16k_pass@1":"green",
    
    
    "4_mtpbTS_SBSP1_7b16k":"blue",
    "3_mtpbNTS_SBSP10_7b16k_pass@1":"grey",
    "2_mtpbTS_SBSP10_7b16k_pass@1":"orange",
    "1_mtpbTFTS_SBSP10_7b16k_pass@1":"green",
    
    
    "bigbenchTS_SBSP1_7b16k":"blue",
    "bigbenchNTS_SBSP10_7b16k_pass@1":"grey",
    "bigbenchTFTS_SBSP10_7b16k_pass@1":"green",
    "bigbenchTS_SBSP10_7b16k_pass@1":"orange",
}

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

markers = [
    'o', 'v', '^','s', 'p', '*', '1', '2' , 'h', 'H', '+', 'x', 'D', 'd', '_'
]

def draw_plots_percent(data,color,image_path,legends=[],title=""):
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    if legends:
        changed_legends = False
    else:
        changed_legends = True
    for k,v in data.items():
        print(f"{k}:{v}")
        labels.append(k)
        if changed_legends:
            if "TS_SBSP1_" in k:
                legends.append("自反馈")
            elif "NTS" in k:
                legends.append("非树搜索自反馈")
            elif "TFTS" in k:
                legends.append("基于树搜索和测试用例筛选的自反馈")
            elif "TS_SBSP10" in k:
                legends.append("树搜索自反馈")
            elif "pT" in k:
                legends.append("使用问题描述中的测试用例")
            elif "tT" in k:
                legends.append("使用用来验证的测试用例")
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("迭代轮次",fontsize=34)
    plt.ylabel("代码生成正确率(pass@1):%",fontsize=34)
    # title = image_path.split("/")[-1].split(".")[0]
    # plt.title(title,fontsize=28)
    
    plots = []
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker=markers[i],markersize=12,color=color[labels[i]],linewidth=2)
        plots.append(p)
        for xz,yz in zip(x,y):
            plt.text(xz-0.2,yz+0.5,yz,fontsize=24,color=color[labels[i]])
        # plt.text(10+0.1,y[10],labels[i],fontsize="xx-large",color=color[labels[i]])
    plt.legend(handles=plots,labels=legends,loc="best",fontsize=32,frameon=False)#填best自动找到最好的位置
    plt.xticks(range(12),[str(i) for i in range(12)],fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True,linestyle="--",alpha=0.5)
    fig.savefig(image_path)
    return

def draw_bars(data,image_path):
    xs = []
    ys = []
    for k,v in data.items():
        xs.append(k)
        ys.append(v)
    fig = plt.figure(figsize=(5,3),dpi=400)
    plt.xlabel("fix percent:%")
    plt.ylabel("numbers")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    plt.bar(x=xs,height=ys,color="blue")
    for xt,yt in zip(xs,ys):
        plt.text(xt,yt,yt,va="bottom",ha="center")
    fig.savefig(image_path)
    return

if __name__=="__main__":
    data = {}
    true_testcase_pack = {
        "labels":["UT_SBSP10_7b16k_tT_pass@1","UT_SBSP10_7b16k_pT_pass@1",],
        "image_path":"../image/true_testcase.png",
        "num_task":164,
        "legends":["使用用来验证的测试用例","使用问题描述中的测试用例"],
    }
    humaneval_7b16k_pack = {
        "labels":["humanevalTFTS_SBSP10_7b16k_pT_pass@1","humanevalTS_SBSP10_7b16k_p_pass@1T","humanevalNTS_SBSP10_7b16k_pT_pass@1","humanevalTS_SBSP1_7b16k_pT_pass@1"],
        "image_path":"../image/humaneval_7b16k_pass@1.svg",
        "num_task":164,
        "legends":["基于树搜索和测试用例筛选的自反馈","树搜索自反馈","非树搜索自反馈","自反馈"],
    }
    mtpb_7b16k_pack = {
        "labels":["1_mtpbTFTS_SBSP10_7b16k_pass@1","2_mtpbTS_SBSP10_7b16k_pass@1","3_mtpbNTS_SBSP10_7b16k_pass@1","4_mtpbTS_SBSP1_7b16k",],
        "image_path":"../image/mtpb_7b16k_pass@1.svg",
        "num_task":115,
        "legends":["基于树搜索和测试用例筛选的自反馈","树搜索自反馈","非树搜索自反馈","自反馈"],
    }
    bigbench_7b16k_pack = {
        "labels":["bigbenchTFTS_SBSP10_7b16k_pass@1","bigbenchTS_SBSP10_7b16k_pass@1","bigbenchNTS_SBSP10_7b16k_pass@1","bigbenchTS_SBSP1_7b16k",],
        "image_path":"../image/bigbench_7b16k_pass@1.svg",
        "num_task":32,
        "legends":["基于树搜索和测试用例筛选的自反馈","树搜索自反馈","非树搜索自反馈","自反馈"],
    }
    mbpp_7b16k_pack = {
        "labels":["mbppTFTS_SBSP10_7b16k_pass@1","mbppTS_SBSP10_7b16k_pass@1","mbppNTS_SBSP10_7b16k_pass@1","mbppTS_SBSP1_7b16k",],
        "image_path":"../image/mbpp_7b16k_pass@1.svg",
        "num_task":200,
        "legends":["基于树搜索和测试用例筛选的自反馈","树搜索自反馈","非树搜索自反馈","自反馈"],
    }
    
    pack = mbpp_7b16k_pack
    show_label = pack["labels"]
    num_task = pack["num_task"]
    image_path = pack["image_path"]
    legends = pack["legends"]
    for label in show_label:
        if label in cir_record.keys():
            value = cir_record[label]
            data[label] = [round((1.0*x)/num_task*100,1) for x in value]
    ys = []
    ys = np.array(ys)
    draw_plots_percent(data,color_map=[],image_path=image_path,legends=[],title="Vicuna-7b-16k在BIG-bench上的pass@1正确率")
    