import json
from data_tools import Counter_with_base,data_analysis
from draw_tools import draw_hist,draw_scatters,draw_bars,draw_corr_hist
from collections import defaultdict,Counter
import pandas as pd
import numpy as np
from scipy import stats
def load_pass_testcase(res_file:str):
    pass_testcase = {}
    with open(res_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid= data["task_id"]
            completion = data["completion"]
            CODET_pass_testcase = []
            for cir,v in completion.items():
                pass_testcase_cir = v[0]["CODET_pass_testcase"]
                CODET_pass_testcase.append({"cir":cir,"pass_testcase":pass_testcase_cir})
            pass_testcase[tid] = CODET_pass_testcase
            print(tid)
    return pass_testcase

def load_chosen_testcase(res_file:str):
    chosen_testcase = {}
    with open(res_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid= data["task_id"]
            completion = data["completion"]
            chosen_testcase_ids = []
            for cir,v in completion.items():
                chosen_testcase_id = v[0]["chosen_testcase_id"]
                chosen_testcase_ids.append({"cir":cir,"chosen_testcase_id":chosen_testcase_id})
            chosen_testcase[tid] = chosen_testcase_ids
            print(tid)
    return chosen_testcase

def pass_testcase_analysis(pass_testcase):
    group_num_count = []
    group_and_percent = []
    for tid,CODET_pass_testcase in pass_testcase.items():
        print(f"Analysis for task {tid}")
        for t in CODET_pass_testcase:
            cir = t["cir"]
            pass_testcase_cir = t["pass_testcase"]
            n = len(pass_testcase_cir)
            group_num_count.append(n)
            print(f"    there are {n} CODET group in cir {cir}")
            for i,p in enumerate(pass_testcase_cir):
                p1 = set(p)
                if i < n-1 and i==0:
                    p2 = set(pass_testcase_cir[i+1])
                    p3 = p1 & p2
                    k,j,l = len(p1),len(p2),len(p3)
                    print(f"        {i}: {k} ,{i+1}: {j} ,{i}&{i+1}: {l}.")
                    if i == 0:
                        if l==0:
                            group_and_percent.append(0.001)
                        else:
                            group_and_percent.append(round(l/min(k,j),2))
                    break
    
    zero_percent = len([x for x in group_and_percent if x-0.001 < 0.00000000001])
    one_percent = len([x for x in group_and_percent if 1.0-x < 0.00000000001])
    print(f"group_and_percent has {len(group_and_percent)} nums, {zero_percent} are 0.0, {one_percent} are 1.0:\n {group_and_percent}")
    Counter_with_base(group_num_count,10)
    draw_hist(group_num_count,image_path="group_num_count_more.png")
    draw_hist(group_and_percent,image_path="group_and_percent_more.png")
    data_analysis(group_and_percent)
    data_analysis(group_num_count)
    return
def pass_testcase_analysis_range3(pass_testcase):
    group_num_count = []
    group_and_percent = []
    for tid,CODET_pass_testcase in pass_testcase.items():
        print(f"Analysis for task {tid}")
        for t in CODET_pass_testcase:
            cir = t["cir"]
            pass_testcase_cir = t["pass_testcase"]
            n = len(pass_testcase_cir)
            group_num_count.append(n)
            print(f"    there are {n} CODET group in cir {cir}")
            for i,p in enumerate(pass_testcase_cir):
                p1 = set(p)
                if i < n-2:
                    p2 = set(pass_testcase_cir[i+1])
                    p3 = set(pass_testcase_cir[i+2])
                    p4 = p1 & p2
                    h,k,j,l = len(p1),len(p2),len(p3),len(p4)
                    print(f"        {i}: {h} ,{i+1}: {k} ,{i+2}: {j} ,{i}&{i+1}&{i+2}: {l}.")
                    if i == 0:
                        if l==0:
                            group_and_percent.append(0.001)
                        else:
                            group_and_percent.append(round(l/min(h,j),2))
    
    zero_percent = len([x for x in group_and_percent if x-0.001 < 0.00000000001])
    one_percent = len([x for x in group_and_percent if 1.0-x < 0.00000000001])
    print(f"group_and_percent has {len(group_and_percent)} nums, {zero_percent} are 0.0, {one_percent} are 1.0:\n {group_and_percent}")
    group_num_count = [x for x in group_num_count if x <=10]
    Counter_with_base(group_num_count,1)
    draw_hist(group_num_count,image_path="group_num_count_range3&1.png")
    draw_hist(group_and_percent,image_path="group_and_percent_range3&1.png")
    data_analysis(group_and_percent)
    data_analysis(group_num_count)
    return

def get_correctp(ts,cts,return_set=False):
    n = len(ts)
    m = 0
    correct_ts = []
    for t in ts:
        if t in cts:
            m += 1
            correct_ts.append(t)
    # print(f"n:{n}  m:{m}")
    if n==0:
        correct_rate = 0.0
    else:
        correct_rate = (1.0*m)/n
    if return_set:
        return correct_rate,correct_ts
    else:
        return correct_rate

def pass_testcase_analysis_idx1(pass_testcase,idx_data):
    #分析排名（根据排序将测试用例集合记s1,s2,s3......）,s1,s1&s2,s1&s2&s3的正确率情况
    
    group_num_count = []
    s1_p = []
    s1_n = []
    s12_p = []
    s12_n = []
    s123_p = []
    s123_n = []
    valid_data = 0
    for tid,CODET_pass_testcase in pass_testcase.items():
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Analysis for task {tid}")
        # s1_p = []
        # s12_p = []
        # s123_p = []
        cts = idx_data[tid]["correct_idx"]
        for t in CODET_pass_testcase:
            cir = t["cir"]
            pass_testcase_cir = t["pass_testcase"]
            n = len(pass_testcase_cir)
            group_num_count.append(n)
            print(f"    there are {n} CODET group in cir {cir}")
            for i,p in enumerate(pass_testcase_cir):
                p1 = set(p)
                if i < n-2 and i==0:
                    valid_data += 1
                    p2 = set(pass_testcase_cir[i+1])
                    p3 = set(pass_testcase_cir[i+2])
                    p4 = p1 & p2
                    p5 = p1 & p2 & p3
                    j,k,l = len(p1),len(p4),len(p5)
                    if l!=0:
                        s1_n.append(j)
                        s12_n.append(k)
                        s123_n.append(l)
                        p1_p = get_correctp(p1,cts)
                        p4_p = get_correctp(p4,cts)
                        p5_p = get_correctp(p5,cts)
                        # if i!=0 and j!=0 and k!=0:
                        s1_p.append(p1_p)
                        s12_p.append(p4_p)
                        s123_p.append(p5_p)
    if s1_p!=[]:
        print("### s1_p ###")
        data_analysis(s1_p)
        print(f"zero num {len([x for x in s1_n if x==0])}")
    if s12_p!=[]:
        print("### s12_p ###")
        data_analysis(s12_p)
        print(f"zero num {len([x for x in s12_n if x==0])}")
    if s123_p!=[]:
        print("### s123_p ###")
        data_analysis(s123_p)
        print(f"zero num {len([x for x in s123_n if x==0])}")
    return

def pass_testcase_analysis_idx2(pass_testcase,idx_data):
    s1_cp = defaultdict(list)
    cts_cp = defaultdict(list)
    total_cts_op_list = defaultdict(list)
    cts_gp_list = defaultdict(list)
    for tid,pts in pass_testcase.items():
        print("-----------------------------------------")
        cts = idx_data[tid]["correct_idx"]
        all_ts = set(idx_data[tid]["correct_idx"] + idx_data[tid]["wrong_idx"])
        for pt in pts:
            cir = pt["cir"]
            print(f"+++++++cir:{cir}++++++++")
            pass_test_cir = pt["pass_testcase"]
            n = len(pass_test_cir)
            for i,g in enumerate(pass_test_cir):
                # if i > 10:
                #     break
                if i == 0:
                    s1 = set(g)
                    ds1 = all_ts - s1
                    s1_p = get_correctp(s1,cts)
                    cts_s1p = get_correctp(cts,s1)
                    ds1_p = get_correctp(ds1,cts)
                    cts_op = get_correctp(cts,ds1)
                    s1_cp[cir].append(s1_p)
                    cts_cp[cir].append(cts_s1p)
                    total_cts_op_list[cir].append(cts_op)
                    print(f"s1_p: {s1_p}\tds1_p: {ds1_p}\tcts_s1p: {cts_s1p}\tcts_op: {cts_op}\ts1 len:{len(s1)}\tds1 len:{len(ds1)}\tcts len:{len(cts)}\tall ts:{len(all_ts)}")
                elif i==1:
                    cts_gp = get_correctp(cts,g)
                else:
                    g = set(g)
                    gcp = get_correctp(g,cts)
                    
                    print(f"gcp: {gcp}")
            # total_cts_op_list[cir].append(total_cts_op_list)
    s1_cp_average = [sum(v)/len(v) for cir,v in s1_cp.items()]
    cts_cp_ave = [sum(v)/len(v) for cir,v in cts_cp.items()]
    total_cts_op_list_ave = [sum(v)/len(v) for cir,v in total_cts_op_list.items()]
    print(s1_cp_average)
    data_analysis(s1_cp_average)
    print(cts_cp_ave)
    data_analysis(cts_cp_ave)
    print(total_cts_op_list_ave)
    data_analysis(total_cts_op_list_ave)                
    return           
def pass_testcase_analysis_idx3(pass_testcase,idx_data):
    # num_to_ts = defaultdict(list)
    # cnum_to_ts = defaultdict(list)
    data_per_cir = {}
    for d in range(11):
        data_per_cir[d] = {"num":{},"cnum":{}}
    print(data_per_cir)
    # cts_cp = defaultdict(list)
    # total_cts_op_list = defaultdict(list)
    invalid_num =0 
    for tid,pts in pass_testcase.items():
        print("-----------------------------------------")
        cts = idx_data[tid]["correct_idx"]
        # all_ts = set(idx_data[tid]["correct_idx"] + idx_data[tid]["wrong_idx"])
        if len(pts) == 1:
            invalid_num+=1
            continue
        for pt in pts:
            cir = int(pt["cir"])
            ts_num = defaultdict(int)
            gnum_ts = defaultdict(set)
            gnum_cts = defaultdict(set)
            num_to_ts = defaultdict(list)
            cnum_to_ts = defaultdict(list)
            print(f"+++++++cir:{cir}++++++++")
            pass_test_cir = pt["pass_testcase"]
            n = len(pass_test_cir)
            print(f"{n} CODET group")
            for i,g in enumerate(pass_test_cir):
                g = set(g)
                for gt in g:
                    ts_num[gt] += 1
            ts_num = dict(sorted(ts_num.items(),key=lambda x:x[1],reverse=True))
            for gt,gt_num in ts_num.items():
                if gt in cts:
                    gnum_cts[gt_num].add(gt)
                    cnum_to_ts[gt_num].append(gt)
                gnum_ts[gt_num].add(gt)
                num_to_ts[gt_num].append(gt)
            sort_gnum_ts = sorted(gnum_ts.items(),key=lambda x:x[0],reverse=True)
            for gnum,tlist in sort_gnum_ts:
                cn = len(gnum_cts[gnum])
                cp = len(gnum_cts[gnum])/len(tlist)
                print(f"{len(tlist)} testcases are in {gnum} groups. {cn} of them are correct, the correct percent is {cp}.")
            # total_cts_op_list[cir].append(total_cts_op_list)
            for gt_num in num_to_ts.keys():
                if gt_num in data_per_cir[cir]["num"].keys():
                    data_per_cir[cir]["num"][gt_num] += num_to_ts[gt_num]
                    data_per_cir[cir]["cnum"][gt_num] += cnum_to_ts[gt_num]
                else:
                    data_per_cir[cir]["num"][gt_num] = num_to_ts[gt_num]
                    data_per_cir[cir]["cnum"][gt_num] = cnum_to_ts[gt_num]
    # data_per_cir[cir]
    num_to_num = {}
    num_to_cnum = {}
    for cir in data_per_cir.keys():
        if cir==0:
            print(data_per_cir[cir])
        num_to_ts = data_per_cir[cir]["num"]
        cnum_to_ts = data_per_cir[cir]["cnum"]
        num_to_cp = {}
        
        for gt_num in num_to_ts.keys():
            if num_to_ts[gt_num]==[]:
                continue
            else:
                cp = len(cnum_to_ts[gt_num])/len(num_to_ts[gt_num])
                num_to_cp[gt_num] = cp
                num_to_num[gt_num] = len(set(num_to_ts[gt_num]))
                num_to_cnum[gt_num] = len(set(cnum_to_ts[gt_num]))
                if gt_num>200:
                    print(len(set(num_to_ts[gt_num])),len(set(cnum_to_ts[gt_num])))
            
        image_path = f"../image/test_num_to_cp/num_to_cp_cir{cir}.png"
        draw_bars(data=num_to_cp,image_path=image_path)
    draw_bars(data=num_to_num,image_path="../image/test_num_to_cp/num_to_num.png")
    draw_bars(data=num_to_cnum,image_path="../image/test_num_to_cp/num_to_cnum.png")
    # s1_cp_average = [sum(v)/len(v) for cir,v in s1_cp.items()]
    # cts_cp_ave = [sum(v)/len(v) for cir,v in cts_cp.items()]
    # total_cts_op_list_ave = [sum(v)/len(v) for cir,v in total_cts_op_list.items()]
    # print(s1_cp_average)
    # print(cts_cp_ave)
    # print(total_cts_op_list_ave)
    print("invalid num:",invalid_num)
    return

def pass_testcase_analysis_idx4(pass_testcase,idx_data):
    num_to_ts = defaultdict(list)
    cnum_to_ts = defaultdict(list)
    data_per_cir = {}
    num_cp_corr = []
    num_cp_p = []
    num_cp_scorr = []
    num_cp_sp = []
    for d in range(11):
        data_per_cir[d] = {"num":{},"cnum":{}}
    invalid_task = []
    for tid,pts in pass_testcase.items():
        print(f"-------------------{tid}----------------------")
        cts = idx_data[tid]["correct_idx"]
        all_ts = set(idx_data[tid]["correct_idx"] + idx_data[tid]["wrong_idx"])
        # if tid=="HumanEval/76":
        #     invalid_task.append(tid)
        #     continue
        for pt in pts:
            cir = int(pt["cir"])
            # if cir == 0:
            #     continue
            ts_num = {}
            gnum_ts = defaultdict(set)
            gnum_cts = defaultdict(set)
            gnum_cp = defaultdict(int)
            for t in all_ts:
                ts_num[t] = 0
            print(f"+++++++cir:{cir}++++++++")
            pass_test_cir = pt["pass_testcase"]
            n = len(pass_test_cir)
            print(f"{n} CODET group")
            for i,g in enumerate(pass_test_cir):
                g = set(g)
                for gt in g:
                    ts_num[gt] += 1
            ts_num = dict(sorted(ts_num.items(),key=lambda x:x[1],reverse=True))
            for gt,gt_num in ts_num.items():
                if gt in cts:
                    gnum_cts[gt_num].add(gt)
                    cnum_to_ts[gt_num].append(gt)
                gnum_ts[gt_num].add(gt)
                num_to_ts[gt_num].append(gt)
            sort_gnum_ts = sorted(gnum_ts.items(),key=lambda x:x[0],reverse=True)
            for gnum,tlist in sort_gnum_ts:
                # if gnum>150:
                #     print(f"In task {tid},cir {cir},with gnum{gnum},{gnum_cts[gnum]}:{tlist}")
                cn = len(gnum_cts[gnum])
                cp = len(gnum_cts[gnum])/len(tlist)
                print(f"{len(tlist)} testcases are in {gnum} groups. {cn} of them are correct, the correct percent is {cp}.")
                gnum_cp[gnum] = cp
            gnum_list = list(gnum_cp.keys())
            if len(gnum_list) <= 2:
                continue
            cp_list = [gnum_cp[x] for x in gnum_list]
            c,p,sc,sp = dataCorrelation2(gnum_list,cp_list)
            if not np.isnan(c):
                num_cp_corr.append(c)
                num_cp_p.append(p)
            if not np.isnan(sc):
                num_cp_scorr.append(sc)
                num_cp_sp.append(sp)
            # total_cts_op_list[cir].append(total_cts_op_list)
    print("num cp corr")
    data_analysis(num_cp_corr)
    Counter_with_base(num_cp_corr,0.1)
    draw_corr_hist(num_cp_corr,image_path="../image/test_num_to_cp/num_cp_corr3.png")
    print("num cp p")
    data_analysis(num_cp_p)
    Counter_with_base(num_cp_p,0.1)
    draw_corr_hist(num_cp_p,image_path="../image/test_num_to_cp/num_cp_p3.png")
    
    print("num cp scorr")
    data_analysis(num_cp_scorr)
    Counter_with_base(num_cp_scorr,0.1)
    draw_corr_hist(num_cp_scorr,image_path="../image/test_num_to_cp/num_cp_scorr3.png")
    print("num cp sp")
    data_analysis(num_cp_sp)
    Counter_with_base(num_cp_sp,0.1)
    draw_corr_hist(num_cp_sp,image_path="../image/test_num_to_cp/num_cp_sp3.png")
    
    num_to_cp = {}
    for gt_num in num_to_ts.keys():
        if num_to_ts[gt_num]==[]:
            continue
        else:
            cp = len(cnum_to_ts[gt_num])/len(num_to_ts[gt_num])
            num_to_cp[gt_num] = cp
            # if gt_num >= 180:
            #     print(f"gt_num : {gt_num} with {num_to_ts[gt_num]} : {cnum_to_ts[gt_num]}")
    image_path = f"../image/test_num_to_cp/num_to_cp_sc3_without_t76.png"
    draw_scatters(data=num_to_cp,image_path=image_path)
    print(f"invalid num:{len(invalid_task)}\n{invalid_task}")
    num = list(num_to_cp.keys())
    cp = [num_to_cp[x] for x in num]
    dataCorrelation2(num,cp,verbose=True)
    return
    
def pass_testcase_analysis_idx5(pass_testcase,idx_data):
    num_to_ts = defaultdict(list)
    cnum_to_ts = defaultdict(list)
    num_to_ts_76 = defaultdict(list)
    cnum_to_ts_76 = defaultdict(list)
    data_76 = []
    data_per_cir = {}
    for d in range(11):
        data_per_cir[d] = {"num":{},"cnum":{}}
    invalid_task = []
    for tid,pts in pass_testcase.items():
        print(f"-------------------{tid}----------------------")
        cts = idx_data[tid]["correct_idx"]
        all_ts = set(idx_data[tid]["correct_idx"] + idx_data[tid]["wrong_idx"])
        for pt in pts:
            cir = int(pt["cir"])
            if cir == 0:
                continue
            ts_num = {}
            gnum_ts = defaultdict(set)
            gnum_cts = defaultdict(set)
            gnum_cp = defaultdict(int)
            for t in all_ts:
                ts_num[t] = 0
            print(f"+++++++cir:{cir}++++++++")
            pass_test_cir = pt["pass_testcase"]
            n = len(pass_test_cir)
            print(f"{n} CODET group")
            for i,g in enumerate(pass_test_cir):
                g = set(g)
                for gt in g:
                    ts_num[gt] += 1
            ts_num = dict(sorted(ts_num.items(),key=lambda x:x[1],reverse=True))
            for gt,gt_num in ts_num.items():
                if gt in cts:
                    gnum_cts[gt_num].add(gt)
                    cnum_to_ts[gt_num].append(gt)
                gnum_ts[gt_num].add(gt)
                if tid == "HumanEval/76":
                    data_76.append(gt_num)
                num_to_ts[gt_num].append(gt)
            sort_gnum_ts = sorted(gnum_ts.items(),key=lambda x:x[0],reverse=True)
            for gnum,tlist in sort_gnum_ts:
                # if gnum>150:
                #     print(f"In task {tid},cir {cir},with gnum{gnum},{gnum_cts[gnum]}:{tlist}")
                cn = len(gnum_cts[gnum])
                cp = len(gnum_cts[gnum])/len(tlist)
                print(f"{len(tlist)} testcases are in {gnum} groups. {cn} of them are correct, the correct percent is {cp}.")
                gnum_cp[gnum] = cp
            # total_cts_op_list[cir].append(total_cts_op_list)
    
    num_to_cp = {}
    for gt_num in num_to_ts.keys():
        if num_to_ts[gt_num]==[]:
            continue
        else:
            cp = len(cnum_to_ts[gt_num])/len(num_to_ts[gt_num])
            num_to_cp[gt_num] = cp
            # if gt_num >= 180:
            #     print(f"gt_num : {gt_num} with {num_to_ts[gt_num]} : {cnum_to_ts[gt_num]}")
    image_path = f"../image/test_num_to_cp/num_to_cp_sc2_without_t76.png"
    draw_scatters(data=num_to_cp,image_path=image_path)
    print(f"invalid num:{len(invalid_task)}\n{invalid_task}")
    num = list(num_to_cp.keys())
    cp = [num_to_cp[x] for x in num]
    dataCorrelation2(num,cp,verbose=True)
    return

def chosen_testcase_analysis(pass_testcase,idx_data):
    num_to_ts = defaultdict(list)
    cnum_to_ts = defaultdict(list)
    cp_list = []
    cir_cp_list = defaultdict(list)
    valid_num = 0
    data_per_cir = {}
    for d in range(11):
        data_per_cir[d] = {"num":{},"cnum":{}}
    invalid_num = 0
    for tid,pts in pass_testcase.items():
        print(f"-------------------{tid}----------------------")
        cts = idx_data[tid]["correct_idx"]
        all_ts = set(idx_data[tid]["correct_idx"] + idx_data[tid]["wrong_idx"])
        for pt in pts:
            cir = int(pt["cir"])
            print(f"+++++++cir:{cir}++++++++")
            chosen_testcase_id = pt["chosen_testcase_id"]
            n = len(chosen_testcase_id)
            if n == 0:
                invalid_num+=1
                continue
            valid_num += 1
            cnum = 0
            for i,g in enumerate(chosen_testcase_id):
                if g in cts:
                    cnum+= 1
            print(f"chosen {n} testcase, {cnum} of them are correct.")
            cp = cnum/n
            cp_list.append(cp)
            cir_cp_list[cir].append(cp)
    for cir in cir_cp_list.keys():
        print(f"cir {cir} has {len(cir_cp_list[cir])} data.")
        data_analysis(cir_cp_list[cir])
    data_analysis(cp_list)
    draw_hist(cp_list,"../image/chosen_testcase_cp.png")
    return

def dataCorrelation(x:list,y:list):
    print(stats.spearmanr(x,y))
    print(stats.pearsonr(x,y))
    x,y = pd.Series(x),pd.Series(y)
    data = pd.DataFrame({"num":x,"cp":y})
    corr = data.corr()
    spear_corr = data.corr(method="spearman")
    print(len(data["num"]),len(data["cp"]))
    print(f"corr:\n{corr}")
    print(f"spearman corr:\n{spear_corr}")
    print(f"corr value : {corr['num']['cp']}")
    return corr['num']['cp'],spear_corr['num']['cp']

def dataCorrelation2(x:list,y:list,verbose=False):
    scorr,sp = stats.spearmanr(x,y)
    corr,p = stats.pearsonr(x,y)
    if verbose:
        print(f"pearsonr:{corr} with p-value {p}\nspearmanr:{scorr} with p-value {sp}")
    return corr,p,scorr,sp



def main():
    testcase_idx_file  = "../try/gen_test_t0.8_topp0.95_sample100_max300_idx.jsonl"#"../try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5_idx.jsonl"#     gen_test_t0.8_topp0.95_sample100_max300_idx.jsonl
    idx_data = {}
    with open(testcase_idx_file,"r") as df:
        # idx_data = json.loads(df.readline())
        for line in df.readlines():
            data = json.loads(line)
            for tid,value in data.items():
                correct_idx = value["correct_idx"]
                wrong_idx = value["wrong_idx"]
                idx_data[tid] = {"correct_idx":correct_idx,"wrong_idx":wrong_idx}
            # tid = data["task_id"]
            # correct_idx = data["correct_idx"]
            # wrong_idx = data["wrong_idx"]
            # idx_data[tid] = {"correct_idx":correct_idx,"wrong_idx":wrong_idx}
    
    res_file = "../res/UTfeedback_CODETPointtry_7b16k_pT.jsonl"
    pass_testcase = load_pass_testcase(res_file=res_file)
    # pass_testcase_analysis(pass_testcase=pass_testcase)
    pass_testcase_analysis_idx4(pass_testcase=pass_testcase,idx_data=idx_data)
    # res_file = "../res/UTfeedback_CODETv3_t8_7b16k_pT.jsonl"
    # chosen_testcase = load_chosen_testcase(res_file=res_file)
    # chosen_testcase_analysis(pass_testcase=chosen_testcase,idx_data=idx_data)
    
    
    
if __name__=="__main__":
    main()