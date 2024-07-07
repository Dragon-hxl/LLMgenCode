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
from show_debug_process import draw_plots_percent


cir_record = {
    "humanevalNTS_SBSP10_7b16k_pT":[27, 35, 50, 56, 61, 64, 67, 68, 68, 68, 68],
    "humanevalTS_SBSP1_7b16k_pT": [27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
    "humanevalTS_SBSP10_7b16k_pT":[27, 36, 51, 60, 63, 66, 69, 70, 72, 72, 72],
    "humanevalTFTS_SBSP10_7b16k_pT":[27, 40, 57, 63, 65, 69, 71, 72, 74, 75, 75],
}

color_map = {
    "humanevalNTS_SBSP10_7b16k_pT":"grey",
    "humanevalTS_SBSP1_7b16k_pT": "blue",
    "humanevalTS_SBSP10_7b16k_pT":"orange",
    "humanevalTFTS_SBSP10_7b16k_pT":"green",
    
}

if __name__=="__main__":
    data = {}
    show_label = [
        "humanevalTFTS_SBSP10_7b16k_pT",
        "humanevalTS_SBSP10_7b16k_pT",
        "humanevalNTS_SBSP10_7b16k_pT",
        "humanevalTS_SBSP1_7b16k_pT",
        ]
    num_task = 164
    # for label,value in cir_record.items():
    #     if label in show_label:
    #         data[label] = [round((1.0*x)/num_task*100,1) for x in value]
    for label in show_label:
        if label in cir_record.keys():
            value = cir_record[label]
            data[label] = [round((1.0*x)/num_task*100,1) for x in value]
    # draw_plots_mean_std(data, data2,color_map, "../image/CODETv3.jpg")
    draw_plots_percent(data,color_map,"../image/humaneval_TFTS_7b16k.svg")