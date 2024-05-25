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

pass_task_record = {
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
    "UT_SBSP10_7b16k_tT":[0, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 66, 67, 72, 78, 79, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 97, 98, 99, 100, 101, 105, 108, 109, 110, 112, 116, 120, 121, 122, 124, 127, 128, 133, 135, 136, 138, 139, 142, 143, 149, 150, 152, 153, 155, 157, 159, 162],
    "UT_moretests_7b16k":[0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 40, 41, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 66, 67, 72, 73, 78, 79, 80, 82, 84, 85, 86, 88, 95, 96, 98, 101, 105, 108, 111, 112, 117, 121, 122, 124, 126, 135, 136, 137, 140, 143, 150, 152, 159, 162],
    "treesearch_SBSP10_7b16k_pT":[0, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 38, 40, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 59, 60, 62, 66, 68, 72, 73, 78, 79, 84, 85, 86, 87, 96, 101, 111, 112, 114, 117, 120, 121, 122, 124, 133, 136, 143, 152, 154, 159, 162],
    "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT":[0, 2, 4, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 67, 68, 72, 78, 79, 80, 84, 85, 86, 87, 96, 98, 100, 101, 105, 111, 112, 117, 120, 121, 122, 123, 124, 133, 134, 137, 139, 143, 152, 156, 159, 161, 162],
    "UTfeedback_CODETv3_t8_7b16k_pT":[0, 2, 4, 7, 8, 11, 12, 13, 14, 15, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 58, 60, 61, 66, 67, 72, 74, 77, 78, 79, 80, 84, 85, 86, 87, 89, 95, 96, 98, 101, 105, 109, 111, 112, 117, 121, 122, 124, 127, 133, 134, 136, 137, 143, 151, 152, 159, 162],
}

cir_record = {
    "UT_SBP10_7b16k_pT":[27, 39, 53, 58, 65, 66, 69, 69, 69, 69, 69],
    "UT_SBP10_7b16k_tT":[27, 27, 41, 63, 73, 77, 83, 86, 91, 91, 93],
    "UT_SBS10_7b16k_pT":[27, 38, 44, 46, 47, 47, 47, 50, 49, 50, 51],
    "UT_SBS10_7b16k_tT":[27, 41, 41, 43, 43, 46, 46, 48, 48, 51, 53],
    "UT_SBSP10_7b16k_pT":[27, 36, 51, 61, 63, 66, 70, 71, 71, 72, 74],
    "UT_SBSP10_7b16k_pT_pass@1":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
    "UT_SBSPCODET1_7b16k_pT":[27, 40, 51, 57, 64, 65, 66, 69, 69, 70, 70],
    "UT_SBSPCODET1_7b16k_pT_pass@1":[27, 39, 50, 55, 63, 64, 65, 68, 68, 69, 69],
    "UT_SBSPCODET2_7b16k_pT":[27, 40, 53, 55, 60, 61, 61, 61, 61, 61, 62],
    "UT_SBSPCODET2_7b16k_pT_pass@1":[27, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36],
    "UT_SBSPCODET4_7b16k_pT":[27, 35, 50, 56, 58, 58, 59, 59, 61, 62, 62],
    "UT_SBSPCODET4_7b16k_pT_pass@1":[27, 34, 48, 55, 57, 57, 58, 58, 60, 61, 61],
    "UT_SBSPCODET5_7b16k_pT":[27, 35, 49, 57, 60, 62, 64, 65, 65, 66, 66],
    "UT_SBSPCODET5_7b16k_pT_pass@1":[27, 35, 49, 57, 60, 62, 64, 65, 65, 66, 66], # 71
    "UTfeedback_multiCODETfilter3_7b16k_pT_pass@10":[27,37,51,61,65,65,66,66,67,67,68],
    "UTfeedback_multiCODETfilter3_7b16k_pT_pass@1":[27,38,51,62,65,65,66,66,67,68,68],
    "UTfeedback_moretests_7b16k_pT_@10":[27,38,55,63,70,73,74,78,80,81,82],
    "UTfeedback_moretests_7b16k_pT_@1":[27,38,53,60,66,70,71,75,77,79,79],
    "UT_SBSP10_7b16k_tT":[27, 40, 58, 69, 78, 81, 85, 89, 92, 94, 96],
    "UT_SBSP10_7b16k_tT_pass@1":[27, 40, 55, 65, 75, 78, 82, 86, 89, 90, 92],
    "UT_7b16k_t1_pT": [27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
    "UT_7b16k_t1_tT": [27, 28, 32, 33, 35, 35, 35, 36, 36,38, 38],
    "UT_7b16k_t1_cT": [27, 27, 26, 27, 27, 29, 33, 32, 34, 34, 34],
    
    "CODETRatefirst_alltestcase_pT_pass@10":[27,36,47,50,51,52,53,53,53,55,55],#94,126
    "CODETRatefirst_alltestcase_pT_pass@1":[27,35,41,41,41,42,43,43,43,43,44],#94,126
    "CODETPointfirst_rm_pT_pass@10":[27,43,49,58,61,61,62,63,64,64,64],#94,126,148
    "CODETPointfirst_rm_pT_pass@1":[27,38,44,53,56,56,57,58,59,59,59],#94,126,148
    # "CODETPointfirst_rm_tT_pass@10":[27,43,49,58,61,61,62,63,64,64,64],
    "UTfeedback_CODETRate_alltestcasesrm_7b16k_pT_pass@10":[27,36,48,54,54,56,57,58,61,62,64],#78,94,113,129,126
    "UTfeedback_CODETRate_alltestcasesrm_7b16k_pT_pass@1": [27,35,45,50,50,52,53,54,57,58,59],
    "UT_SBSP10_7b16k_pT":[27, 36, 51, 61, 63, 66, 70, 71, 71, 72, 74],
    
    "UT_SBSP10_7b16k_pT_pass@1":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
    "UTfeedback_CODETRate_correctrm_7b16k_pT_pass@10":[27,40,53,58,65,67,68,68,70,73,73],# 67,94,126,148,156
    "UTfeedback_CODETRate_correctrm_7b16k_pT_pass@1":[27,38,49,54,60,61,63,63,63,65,66],
    "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT_pass@10":[27,38,52,58,60,64,64,67,69,69,71],# 78,94,95,105,115,126,148,129
    "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT_pass@1":[27,37,49,54,55,61,61,64,65,65,67],
    "UTfeedback_CODETRate_correctrm_mix05_7b16k_pT_pass@10":[27,36,47,57,62,66,67,67,67,68,68],
    
    "UTfeedback_checkrate_7b16k_pT_pass@10":[27,37,54,60,64,66,66,67,70,75,75],# 38,118,126 ,129,148,156(没有对过),94   ,105,115,117(对过)
    "UTfeedback_CODETPoint_alltestcaserm_7b16k_tT_pass@10":[27,38,51,62,67,70,72,73,73,74,77],# 17,19,20,36,94,115,129
    "UTfeedback_CODETPoint_alltestcaserm_7b16k_tT_pass@1":[27,32,45,55,60,63,65,67,67,68,70],
    # - 68,69,70,110,114,116,126,129,/// + 115 ///[110,114,115,116,126,128,129,138,139,140,141,163]


    "UTfeedback_SBSP_7b16k_halftT":[27, 38, 56, 66, 75, 80, 82, 83, 84, 87, 89],
    "UTfeedback_SBSP_7b16k_halftT2":[27, 36, 55, 67, 73, 75, 76, 77, 79, 81, 82],#115,129,130,148
    "UTfeedback_SBSP_7b16k_halftT3_s1000":[27, 40, 56, 66, 72, 76, 79, 80, 82, 82, 82],#129,130
    "UTfeedback_SBSP_7b16k_halftT4": [27, 43, 53, 64, 71, 74, 77, 78, 79, 82, 82],# s20000
    "UTfeedback_SBSP_7b16k_halftT5_s10":[27, 36, 54, 64, 68, 73, 73, 75, 76, 79, 79],
    "UTfeedback_SBSP_7b16k_halftT6s256":[27, 37, 53, 62, 66, 70, 70, 74, 75, 76, 77],
    "UTfeedback_PassRate_mix05_10_7b16k_pT":[27, 37, 60, 66, 69, 73, 76, 76, 79, 81, 81], # 49.096%
    "UTfeedback_PassRate_mix09_10_7b16k_pT":[27, 36, 52, 61, 70, 74, 76, 78, 81, 81, 84],# 33,77,105,107,115,116,120,129,153,163 ##87.997%
    "UTfeedback_PassRate_mix092_10_7b16k_pT":[27, 41, 54, 66, 73, 79, 78, 78, 79, 81, 83],
    "UTfeedback_PassRate_mix10_10_7b16k_pT":[27,34,53,59,62,62,63,64,66,66,66],# [26, 37, 52, 56, 59, 61, 63, 63, 63, 63, 63]
    "UTfeedback_PassRate_mix10_10_7b16k_pT2":[27, 38, 49, 55, 57, 62, 62, 63, 64, 66, 66],
    
    "humanevalTS_SBSP_codellama7bpy_pT_pass@1":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],
    "humanevalTS_SBSP_codellama7bpy_pT_pass@1":[63, 81, 105, 111, 112, 113, 113, 113, 113, 113, 114],
    "humanevalTFTS_SBSP_codellama7bpy_pT_pass@1":[63, 81, 98, 101, 104, 106, 106, 107, 107, 108, 108],
    "humanevalTFTS_SBSP_codellama7bpy_pT":[63, 83, 103, 108, 111, 113, 113, 114, 114, 115, 115],
    
    "UTfeedback_CODETv3_7b16k_pT":[27, 39, 46, 53, 54, 55, 57, 58, 61, 62, 62],
    "UTfeedback_CODETv3_t2_7b16k_pT":[27, 35, 47, 51, 54, 57, 59, 60, 62, 62, 62],
    "UTfeedback_CODETv3_t3_7b16k_pT":[27, 39, 48, 52, 54, 57, 57, 59, 61, 62, 62],
    
    "UTfeedback_CODETv3_t7_7b16k_pT_pass@10":[27, 36, 55, 66, 71, 75, 76, 76, 76, 76, 77],
    "UTfeedback_CODETv3_t7_7b16k_pT_pass@1":[27, 35, 51, 61, 66, 69, 70, 70, 70, 70, 70],
    "UTfeedback_CODETv3_t8_7b16k_pT_pass@10":[27, 40, 57, 63, 65, 69, 71, 72, 74, 75, 77],
    "UTfeedback_CODETv3_t8_7b16k_pT_pass@1":[27, 35, 53, 59, 60, 64, 66, 67, 70, 71, 73],
    "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@10":[27, 37, 47, 59, 65, 70, 71, 73, 74, 75, 76],
    "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@1":[27, 37, 47, 58, 63, 67, 68, 69, 70, 71, 72],
    
    "treesearch_SBSP10_7b16k_pT@10":[27,35,50,58,61,64,67,70,70,70,70],
    "treesearch_SBSP10_7b16k_pT@1":[27,35,49,57,60,63,66,70,70,70,70],
    
    "mbppTS_SBSP1_7b16k":[38, 40, 41, 43, 44, 46, 48, 48, 48, 50, 51],
    "mbppNTS_SBSP10_7b16k_pass@1":[38, 45, 47, 49, 51, 52, 52, 53, 54, 54, 54],#[44, 48, 51, 53, 55, 56, 56, 56, 57, 57, 57]
    "mbppNTS_SBSP10_7b16k_pass@10":[44, 60, 65, 67, 69, 69, 69, 69, 70, 70, 71],
    "mbppTFTS_SBSP10_7b16k_pass@1":[38, 50, 54, 59, 59, 59, 60, 60, 60, 60, 60],
    "mbppTFTS_SBSP10_7b16k_pass@10":[38, 57, 62, 70, 71, 72, 72, 72, 72, 72, 72],
    "mbppTS_SBSP10_7b16k_pass@1":[38, 48, 51, 55, 58, 58, 58, 58, 58, 58, 58],#[37, 48, 51, 55, 58, 58, 58, 58, 58, 58, 58],
    "mbppTS_SBSP10_7b16k_pass@10":[38, 57, 65, 69, 70, 70, 70, 70, 70, 70, 70],#[37, 57, 65, 69, 70, 70, 70, 70, 70, 70, 70]
    
    "1_mtpbTFTS_SBSP10_7b16k_pass@1":[6, 11, 15, 17, 18, 19, 19, 19, 19, 19, 19],#[4, 9, 13, 15, 18, 19, 19, 19, 19, 19, 19],
    "1_mtpbTFTS_SBSP10_7b16k_pass@10":[4, 16, 25, 33, 33, 33, 33, 33, 33, 33, 33],
    "2_mtpbTS_SBSP10_7b16k_pass@1":[6, 11, 14, 16, 17, 17, 17, 17, 17, 17, 17],
    "2_mtpbTS_SBSP10_7b16k_pass@10":[6, 15, 22, 23, 26, 26, 27, 27, 28, 28, 28],
    "3_mtpbNTS_SBSP10_7b16k_pass@1":[6, 10, 11, 13, 16, 16, 16, 16, 16, 16, 16],#[6, 11, 15, 17, 17, 17, 17, 17, 17, 17, 17]
    "3_mtpbNTS_SBSP10_7b16k_pass@10":[6, 15, 24, 26, 28, 28, 28, 28, 28, 28, 28],
    "4_mtpbTS_SBSP1_7b16k":[6, 8, 9, 11, 12, 14, 14, 14, 14, 14, 14],
    
    
    "bigbenchTS_SBSP1_7b16k":[7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "bigbenchNTS_SBSP10_7b16k_pass@1":[7, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11],#[8, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13]#[9, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    "bigbenchNTS_SBSP10_7b16k_pass@10":[9, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13],
    "bigbenchTFTS_SBSP10_7b16k_pass@1":[7, 7, 8, 11, 12, 12, 12, 12, 12, 12, 12],
    "bigbenchTFTS_SBSP10_7b16k_pass@10":[7, 10, 11, 13, 13, 13, 13, 13, 13, 13, 13],
    "bigbenchTS_SBSP10_7b16k_pass@1":[7, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11],#[8, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13]
    "bigbenchTS_SBSP10_7b16k_pass@10":[8, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14],
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
    "UT_SBSPCODET1_7b16k_pT":"red",
    "UT_SBSPCODET1_7b16k_pT_pass@1":"darkred",
    "UT_SBSPCODET2_7b16k_pT":"chocolate",
    "UT_SBSPCODET2_7b16k_pT_pass@1":"sienna",
    "UT_SBSPCODET4_7b16k_pT":"chocolate",
    "UT_SBSPCODET4_7b16k_pT_pass@1":"sienna",
    "UTfeedback_multiCODETfilter3_7b16k_pT_pass@10":"green",
    "UTfeedback_multiCODETfilter3_7b16k_pT_pass@1":"c",
    "UTfeedback_moretests_7b16k_pT_@10":"green",
    "UTfeedback_moretests_7b16k_pT_@1":"c",
    "CODETRatefirst_alltestcase_pT_pass@10":"chocolate",
    "CODETRatefirst_alltestcase_pT_pass@1":"sienna",
    "UTfeedback_CODETRate_alltestcasesrm_7b16k_pT_pass@10":"greenyellow",
    "UTfeedback_CODETRate_alltestcasesrm_7b16k_pT_pass@1":"c",
    "UTfeedback_CODETPoint_alltestcaserm_7b16k_tT_pass@10":"gold",
    "UTfeedback_CODETPoint_alltestcaserm_7b16k_tT_pass@1":"darkorange",
    "UT_7b16k_t1_pT": "c",
    "UT_7b16k_t1_tT": "darkblue",
    "UT_7b16k_t1_cT": "green",
    
    "CODETPointfirst_rm_pT_pass@10":"royalblue",
    "CODETPointfirst_rm_pT_pass@1":"blue",
    "UTfeedback_CODETRate_correctrm_7b16k_pT_pass@10":"red",
    "UTfeedback_CODETRate_correctrm_7b16k_pT_pass@1":"darkred",
    "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT_pass@10":"chocolate",
    "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT_pass@1":"sienna",
    "UTfeedback_CODETRate_correctrm_mix05_7b16k_pT_pass@10":"darkblue",
    
    "UTfeedback_checkrate_7b16k_pT_pass@10":"orchid",
    "UTfeedback_PassRate_correctrm10_7b16k_pT": "teal",
    "UTfeedback_PassRate_mix05_10_7b16k_pT":"red",
    "UTfeedback_PassRate_mix09_10_7b16k_pT":"teal",
    "UTfeedback_PassRate_mix10_10_7b16k_pT":"darkorange",
    "UTfeedback_PassRate_mix10_10_7b16k_pT2":"gold",
    
    "UTfeedback_SBSP_7b16k_halftT": "olive",
    "UTfeedback_SBSP_7b16k_halftT2": "c",#115,129,130,148
    "UTfeedback_SBSP_7b16k_halftT3_s1000":"red",
    "UTfeedback_SBSP_7b16k_halftT4": "black",# s20000
    "UTfeedback_SBSP_7b16k_halftT5_s10":"blue",
    "UTfeedback_SBSP_7b16k_halftT6s256":"pink",
    "halftT":"black",
    
    "UTfeedback_CODETv3_t2_7b16k_pT":"olive",
    "UTfeedback_CODETv3_t3_7b16k_pT":"c",
    "UTfeedback_CODETv3_t7_7b16k_pT_pass@10":"red",
    "UTfeedback_CODETv3_t7_7b16k_pT_pass@1":"darkred",
    "UTfeedback_CODETv3_t8_7b16k_pT_pass@10":"orange",
    "UTfeedback_CODETv3_t8_7b16k_pT_pass@1":"darkorange",
    "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@10":"blue",
    "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@1":"darkblue",
    "treesearch_SBSP10_7b16k_pT@10":"grey",
    "treesearch_SBSP10_7b16k_pT@1":"black",
    
    "mbppTS_SBSP1_7b16k":"blue",
    "mbppNTS_SBSP10_7b16k_pass@1":"grey",
    "mbppNTS_SBSP10_7b16k_pass@10":"dark",
    "mbppTS_SBSP10_7b16k_pass@1":"orange",
    "mbppTS_SBSP10_7b16k_pass@10":"darkorange",
    "mbppTFTS_SBSP10_7b16k_pass@1":"green",
    "mbppTFTS_SBSP10_7b16k_pass@10":"darkgreen",
    
    
    "4_mtpbTS_SBSP1_7b16k":"blue",
    "3_mtpbNTS_SBSP10_7b16k_pass@1":"grey",
    "3_mtpbNTS_SBSP10_7b16k_pass@10":"dark",
    "1_mtpbTFTS_SBSP10_7b16k_pass@1":"green",
    "1_mtpbTFTS_SBSP10_7b16k_pass@10":"darkgreen",
    "2_mtpbTS_SBSP10_7b16k_pass@1":"orange",
    "2_mtpbTS_SBSP10_7b16k_pass@10":"darkorange",
    
    "bigbenchTS_SBSP1_7b16k":"blue",
    "bigbenchNTS_SBSP10_7b16k_pass@1":"grey",
    "bigbenchNTS_SBSP10_7b16k_pass@10":"dark",
    "bigbenchTFTS_SBSP10_7b16k_pass@1":"green",
    "bigbenchTFTS_SBSP10_7b16k_pass@10":"darkgreen",
    "bigbenchTS_SBSP10_7b16k_pass@1":"orange",
    "bigbenchTS_SBSP10_7b16k_pass@10":"darkorange",
    
}

fix_com_percents = {
    1:[0, 62.162943546352345, 65.36767087157672, 64.01600952505848, 63.26178853591353, 62.803607362082616, 62.6334021117471, 62.27913714979484, 61.83789417394393, 61.64282775152817, 61.15799337009814],
}
other = {
    1:[0, 94.57844393557944, 99.20674091173184, 99.43065235012286, 99.43833215092668, 99.44513201275672, 99.45420065650703, 99.46151845761389, 99.47129909318733, 99.47296686981197, 99.47711210115497],
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

markers = [
    'o', 'v', '^','s', 'p', '*', '1', '2' , 'h', 'H', '+', 'x', 'D', 'd', '_'
]
# colors = [
#     "blue","grey","orange","green"
# ]
#'.', 
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
                legends.append("传统自反馈")
            elif "NTS" in k:
                legends.append("非树搜索自反馈")
            elif "TFTS" in k:
                legends.append("使用测试用例筛选的树搜索自反馈")
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

def showDifferent(label1,label2):
    print(f'\nshow difference between {label1} and {label2}')
    v1 = pass_task_record[label1]
    v2 = pass_task_record[label2]
    if type(v1[0]) is str:
        v1 = [int(x) for x in v1]
    if type(v2[0]) is str:
        v2 = [int(x) for x in v2]
    v1 = set(v1)
    v2 = set(v2)
    print(f"label1 - label2 = {len(v1 - v2)}\n{sorted(v1 - v2)}")
    print(f"label2 - label1 = {len(v2 - v1)}\n{sorted(v2 - v1)}")
    print(f"label1 + label2 = {len(v1 | v2)}\n{sorted(v1 | v2)}")
    total = set(pass_task_record["total"])
    print(f"total - label1 - label2 = {len(total - (v1 | v2))}\n{sorted(total - (v1 | v2))}")
    return
    


if __name__=="__main__":
    # showDifferent("UT_SBSP10_7b16k_tT","UT_SBSP10_7b16k_pT")
    # showDifferent("total","UT_moretests_7b16k")
    
    # total_pass = set()
    # for id,pass_list in pass_task_record.items():
    #     if id == "total" or ("7b16k" not in id):
    #         continue
    #     for t in pass_list:
    #         if type(t)==str:
    #             t = int(t)
    #         total_pass.add(t)
    # total = set(pass_task_record["total"])
    # print(f"total pass num is {len(total_pass)} never pass num is {len(total-total_pass)}:\n{sorted(total-total_pass)}")
    data = {}
    true_testcase_pack = {
        "labels":["UT_SBSP10_7b16k_tT_pass@1","UT_SBSP10_7b16k_pT_pass@1",],
        "image_path":"../image/true_testcase.png",
        "num_task":164,
        "legends":["使用用来验证的测试用例","使用问题描述中的测试用例"],
    }
    humaneval_7b16k_pack = {
        "labels":"",
        "num_task":164,
    }
    mtpb_7b16k_pack = {
        "labels":["1_mtpbTFTS_SBSP10_7b16k_pass@1","2_mtpbTS_SBSP10_7b16k_pass@1","3_mtpbNTS_SBSP10_7b16k_pass@1","4_mtpbTS_SBSP1_7b16k",],
        "image_path":"../image/mtpb_7b16k_pass@1.svg",
        "num_task":115,
        "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","传统自反馈"],
    }
    bigbench_7b16k_pack = {
        "labels":["bigbenchTFTS_SBSP10_7b16k_pass@1","bigbenchTS_SBSP10_7b16k_pass@1","bigbenchNTS_SBSP10_7b16k_pass@1","bigbenchTS_SBSP1_7b16k",],
        "image_path":"../image/bigbench_7b16k_pass@1.svg",
        "num_task":32,
        "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","传统自反馈"],
    }
    mbpp_7b16k_pack = {
        "labels":["mbppTFTS_SBSP10_7b16k_pass@1","mbppTS_SBSP10_7b16k_pass@1","mbppNTS_SBSP10_7b16k_pass@1","mbppTS_SBSP1_7b16k",],
        "image_path":"../image/mbpp_7b16k_pass@1.svg",
        "num_task":200,
        "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","传统自反馈"],
    }
    show_label = [
        # "UT_SBSP10_7b16k_tT",
        # "UT_SBSP10_7b16k_pT",
        # "UT_SBSP10_7b16k_tT_pass@1",
        # "UT_SBSP10_7b16k_pT_pass@1",
        # "UTfeedback_CODETRate_alltestcasesrm_7b16k_pT_pass@10",
        # "UTfeedback_CODETRate_correctrm_7b16k_pT_pass@10",
        # "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT_pass@10",
        # "UTfeedback_CODETRate_correctrm_mix05_7b16k_pT_pass@10",
        # "UTfeedback_checkrate_7b16k_pT_pass@10",
        # "UTfeedback_PassRate_correctrm10_7b16k_pT",
        # "UTfeedback_PassRate_mix05_10_7b16k_pT",
        # "UTfeedback_PassRate_mix05_10_7b16k_pT",
        # "UTfeedback_PassRate_mix09_10_7b16k_pT",
        # "UTfeedback_PassRate_mix10_10_7b16k_pT",
        # "UTfeedback_PassRate_mix10_10_7b16k_pT2",
        # "UTfeedback_SBSP_7b16k_halftT",
        # "UTfeedback_SBSP_7b16k_halftT2",#115,129,130,148
        # "UTfeedback_SBSP_7b16k_halftT3_s1000",
        # "UTfeedback_SBSP_7b16k_halftT4",# s20000
        # "UTfeedback_SBSP_7b16k_halftT5_s10",
        # "UTfeedback_SBSP_7b16k_halftT6s256",
        # "UTfeedback_CODETv3_t2_7b16k_pT",
        # "UTfeedback_CODETv3_t3_7b16k_pT",
        # "UTfeedback_PassRate_mix05_10_7b16k_pT",
        # "UT_7b16k_t1_pT",
        # "UTfeedback_CODETv3_t7_7b16k_pT_pass@10",
        # "UTfeedback_CODETv3_t7_7b16k_pT_pass@10",
        # "UTfeedback_CODETv3_t8_7b16k_pT_pass@10",
        # "UTfeedback_CODETv3_t8_7b16k_pT_pass@10",
        # "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@10",
        # "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT@10",
        # "mbppTS_SBSP1_7b16k",
        # "mbppNTS_SBSP10_7b16k_pass@1",
        # "mbppNTS_SBSP10_7b16k_pass@10",
        # "mbppTFTS_SBSP10_7b16k_pass@1",
        # "mbppTFTS_SBSP10_7b16k_pass@10",
        # "mbppTS_SBSP10_7b16k_pass@1",
        # "mbppTS_SBSP10_7b16k_pass@10",
        
        # "mtpbTFTS_SBSP1_7b16k",
        # "3_mtpbNTS_SBSP10_7b16k_pass@1",
        # # "3_mtpbNTS_SBSP10_7b16k_pass@10",
        # "1_mtpbTFTS_SBSP10_7b16k_pass@1",
        # # "1_mtpbTFTS_SBSP10_7b16k_pass@10",
        # "2_mtpbTS_SBSP10_7b16k_pass@1",
        # # "2_mtpbTS_SBSP10_7b16k_pass@10",
        
        # "bigbenchTS_SBSP1_7b16k",
        # "bigbenchNTS_SBSP10_7b16k_pass@1",
        # # "bigbenchNTS_SBSP10_7b16k_pass@10",
        # "bigbenchTFTS_SBSP10_7b16k_pass@1",
        # # "bigbenchTFTS_SBSP10_7b16k_pass@10",
        # "bigbenchTS_SBSP10_7b16k_pass@1",
        # # "bigbenchTS_SBSP10_7b16k_pass@10",
        ]
    show_label2 = [
        # "UTfeedback_SBSP_7b16k_halftT",
        # "UTfeedback_SBSP_7b16k_halftT2",#115,129,130,148
        # "UTfeedback_SBSP_7b16k_halftT3_s1000",
        # "UTfeedback_SBSP_7b16k_halftT4",# s20000
        # "UTfeedback_SBSP_7b16k_halftT5_s10",
        # "UTfeedback_SBSP_7b16k_halftT6s256",
    ]
    pack = mbpp_7b16k_pack
    show_label = pack["labels"]
    num_task = pack["num_task"]
    image_path = pack["image_path"]
    legends = pack["legends"]
    # for label,value in cir_record.items():
    #     if label in show_label:
    #         data[label] = [round((1.0*x)/num_task*100,1) for x in value]
    for label in show_label:
        if label in cir_record.keys():
            value = cir_record[label]
            data[label] = [round((1.0*x)/num_task*100,1) for x in value]
    ys = []
    for label in show_label2:
        d = cir_record[label]
        ys.append([round((1.0*x)/num_task*100,1) for x in d])
    ys = np.array(ys)
    mean_ys = np.mean(ys,axis=0)
    mean_ys = np.around(mean_ys,decimals=1)
    std_ys = np.std(ys,axis=0)
    print(f"mean_ys:{mean_ys}, std_ys:{std_ys}")
    data2 = {"x":range(11),"mean":mean_ys,"std":std_ys,"label":"halftT"}
    # draw_plots_mean_std(data, data2,color_map, "../image/CODETv3.jpg")
    draw_plots_percent(data,color_map,image_path=image_path,legends=[],title="Vicuna-7b-16k在BIG-bench上的pass@1正确率")
    # fix_lengths = {9: 2740, 10: 930, 8: 897, 11: 457, 12: 176, 13: 86}
    # fix_percent = {6: 2094, 5: 1650, 7: 1169, 4: 225, 8: 141, 3: 7}
    
    # draw_bars(fix_lengths,"../image/fix_lengths.jpg")
    # draw_bars(fix_percent,"../image/fix_percents.jpg")
    
    # ms_time = {3: 232, 4: 213, 2: 52, 5: 24, 1: 19, 6: 2}
    # draw_plots_percent("fix computation percent")
    dv4 = [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 58, 60, 63, 82, 84, 86, 87, 89, 96, 97, 101, 111, 117, 120, 121, 122, 124, 133, 135, 136, 139, 143, 146, 152, 159, 162]
    pmix05 = [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 25, 27, 28, 29, 30, 31, 34, 35, 40, 42, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 60, 61, 66, 67, 68, 72, 73, 74, 78, 79, 84, 85, 86, 87, 92, 93, 95, 96, 97, 98, 101, 109, 111, 112, 120, 121, 122, 124, 127, 133, 134, 135, 136, 137, 139, 142, 143, 151, 152, 154, 157, 162]
    v3t6 = [0, 2, 4, 7, 8, 12, 13, 14, 15, 23, 24, 25, 28, 29, 30, 31, 34, 35, 37, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 66, 67, 86, 87, 89, 96, 98, 101, 117, 121, 122, 133, 137, 139, 143, 152, 153]
    v3t6_lack = ['HumanEval/1', 'HumanEval/3', 'HumanEval/5', 'HumanEval/6', 'HumanEval/10', 'HumanEval/11', 'HumanEval/17', 'HumanEval/20', 'HumanEval/21', 'HumanEval/22', 'HumanEval/26', 'HumanEval/27', 'HumanEval/32', 'HumanEval/33', 'HumanEval/36', 'HumanEval/47', 'HumanEval/68', 'HumanEval/69', 'HumanEval/70', 'HumanEval/71', 'HumanEval/72', 'HumanEval/73', 'HumanEval/74', 'HumanEval/75', 'HumanEval/76', 'HumanEval/77', 'HumanEval/78', 'HumanEval/79', 'HumanEval/80', 'HumanEval/81', 'HumanEval/82', 'HumanEval/83', 'HumanEval/84', 'HumanEval/85', 'HumanEval/88', 'HumanEval/93', 'HumanEval/94', 'HumanEval/95', 'HumanEval/103', 'HumanEval/104', 'HumanEval/105', 'HumanEval/106', 'HumanEval/107', 'HumanEval/108', 'HumanEval/109', 'HumanEval/110', 'HumanEval/111', 'HumanEval/112', 'HumanEval/113', 'HumanEval/114', 'HumanEval/115', 'HumanEval/123', 'HumanEval/124', 'HumanEval/125', 'HumanEval/126', 'HumanEval/127', 'HumanEval/128', 'HumanEval/129', 'HumanEval/130', 'HumanEval/131', 'HumanEval/132', 'HumanEval/134', 'HumanEval/156', 'HumanEval/157', 'HumanEval/158', 'HumanEval/159', 'HumanEval/160', 'HumanEval/161', 'HumanEval/162', 'HumanEval/163']
    v3t6_lack = [int(x.split("/")[1]) for x in v3t6_lack]
    baseline = [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 58, 60, 61, 62, 66, 67, 72, 78, 79, 82, 84, 85, 86, 87, 89, 94, 95, 96, 97, 101, 112, 115, 117, 120, 121, 122, 124, 127, 132, 133, 134, 137, 143, 152, 159, 162]
    pmix10 = [0, 2, 4, 7, 8, 11, 12, 13, 14, 15, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 40, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 78, 79, 82, 84, 86, 87, 90, 96, 98, 101, 105, 112, 120, 121, 124, 126, 135, 143, 152, 158, 162]
    # dif = set(pmix05) - set(dv4)
    # dif = sorted(list(dif)) #[16, 18, 54, 61, 66, 67, 68, 72, 73, 74, 78, 79, 85, 92, 93, 95, 98, 109, 127, 134, 137, 142, 151, 154, 157]
    # print(f"dif:{dif}")
    # dif2 = set(dv4) - set(pmix05)# [24, 41, 46, 63, 82, 89, 117, 146, 159]
    # dif2 = sorted(list(dif2))
    # print(f"dif2:{dif2}")
    future_t = []
    for t in v3t6_lack:
        if t in pmix05:
            future_t.append(t)
    print(f"future_t:{future_t}, len:{len(future_t)}")
    print(f"v3t6_lack:{v3t6_lack}, len:{len(v3t6_lack)}")
    
    print("---------------------")
    print(sorted(set(pmix05) - set(pmix10)))
    
    p0 = set(pass_task_record['UT_SBSP10_7b16k_pT'])
    p1 = set(pass_task_record['UTfeedback_CODETv3_t8_7b16k_pT'] )
    p2 = set(pass_task_record['UTfeedback_CODETv3_sortby_solution_num_7b16k_pT'])
    print(f"p1 - p0 : {sorted(p1 - p0)}")
    print(f"p2 - p0 : {sorted(p2 - p0)}")
    print(f"p2 - p1 : {sorted(p0 - p1)}")
    
    
    mbpp_chosen_idx1 = [240, 93, 372, 296, 155, 102, 454, 370, 209, 387, 366, 388, 135, 272, 125, 325, 416, 376, 255, 181, 212, 269, 497, 315, 111, 158, 278, 360, 169, 265, 38, 374, 396, 443, 105, 352, 385, 477, 239, 363, 425, 446, 334, 75, 486, 108, 444, 210, 29, 394, 178, 321, 213, 238, 63, 371, 380, 71, 390, 167, 199, 471, 176, 406, 494, 166, 218, 479, 162, 290, 109, 208, 117, 104, 20, 383, 115, 441, 9, 132, 258, 163, 395, 291, 411, 361, 215, 314, 57, 438, 457, 310, 399, 118, 120, 237, 187, 69, 103, 188, 252, 304, 448, 72, 134, 198, 319, 172, 171, 362, 364, 458, 86, 350, 356, 67, 410, 465, 297, 351, 33, 50, 88, 2, 77, 224, 472, 405, 179, 427, 41, 100, 145, 122, 355, 236, 308, 417, 246, 268, 223, 339, 432, 435, 36, 154, 354, 142, 402, 289, 338, 128, 478, 51, 253, 475, 368, 450, 90, 263, 114, 418, 480, 23, 496, 473, 193, 324, 37, 60, 492, 28, 470, 64, 107, 412, 44, 419, 377, 462, 249, 298, 84, 82, 323, 326, 53, 398, 287, 309, 15, 312, 55, 286, 92, 409, 161, 0, 62, 143]