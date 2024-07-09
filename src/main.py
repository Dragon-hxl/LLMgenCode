import time
import random
import hydra
from omegaconf import DictConfig, OmegaConf

import faulthandler
faulthandler.enable(all_threads=True)

# dataset class
from dataset import taskBase,read_problems

# import self-debug strategy
from tree_search import *
from testcase_filter_first import *
from base_generate import *
from not_tree_search import *
from fastdebug_test import *
from tree_search_cached import *
from tree_search2 import *
from testcase_filter_cached import *
from code_expl_gen import *
from mcts_search import run_mcts_search
from testcase_generate import *

humaneval_7bpy_base = [0, 2, 3, 4, 5, 7, 9, 12, 13, 15, 16, 17, 21, 22, 23, 24, 27, 28, 29, 30, 31, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 53, 55, 56, 58, 60, 61, 63, 66, 71, 72, 74, 77, 79, 80, 86, 87, 104, 107, 112, 113, 116, 121, 122, 124, 136, 142, 147, 152, 153, 156, 159, 162]
mbpp_7bpy_base = [11, 34, 49, 52, 55, 61, 62, 66, 68, 71, 80, 82, 88, 93, 95, 113, 115, 116, 125, 126, 128, 131, 133, 153, 166, 172, 173, 174, 187, 192, 199, 204, 210, 221, 223, 224, 226, 234, 248, 249, 250, 257, 263, 269, 283, 301, 302, 309, 319, 330, 332, 334, 335, 336, 345, 361, 362, 365, 366, 371, 373, 377, 379, 381, 388, 394, 399, 401, 409, 422, 427, 428, 452, 455, 459, 465, 476, 481, 484, 489, 491, 505, 507]
@hydra.main(version_base=None, config_path="../configs/", config_name="UTfeedback_config.yaml")
def main(cfg: DictConfig):
    #读取配置,获取参数
    print(OmegaConf.to_yaml(cfg))
    output_file = cfg.output
    model_path = cfg.model_path
    debug_temp = cfg.multi.debug.temperature
    debug_maxLen = cfg.multi.debug.max_new_tokens
    debug_top_p = cfg.multi.debug.top_p
    debug_do_sample = cfg.multi.debug.do_sample
    # debug_num_return_sequences = cfg.multi.debug.num_return_sequences
    sample_num = cfg.sample_num
    filter_num = cfg.filter_num
    feedback_type = cfg.feedback_type
    Strategy = cfg.Strategy
    dataset_type = cfg.dataset
    first_num = cfg.first_num
    
    # load task
    dataset = []
    print(f"load dataset:{dataset_type}")
    if dataset_type == "humaneval":
        print("load dataset : humaneval")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/data/humaneval.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            #important
            if num_id < 139 or num_id > 164:
                continue
            #important
            dataset.append(problem)
    elif dataset_type == "mbpp":
        print("load dataset : mbpp")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/MBPP/mbpp_humaneval_format_11_510.jsonl")
        for tid,problem in problems.items():
            dataset.append(problem)
        n = len(dataset)
        random.seed(2024)
        chosen_idx = random.sample(range(n), 200)
        # chosen_idx = sorted(chosen_idx)
        print("mbpp chosen idx:",chosen_idx)
        with open("mbpp_chosen_idx.txt","w") as f:
            f.write(json.dumps(chosen_idx))
        #important
        chosen_idx = chosen_idx[0:199]
        #important
        dataset = [dataset[i] for i in chosen_idx]
        
    elif dataset_type == "mtpb":
        print("load dataset : mtpb")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/benchmarks/mtpb_humaneval_format.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            #important
            if num_id < 58 or num_id > 116 or num_id==20 or num_id==47:
                continue
            #important
            dataset.append(problem)
    elif dataset_type == "bigbench":
        print("load dataset : bigbench")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/benchmarks/bigbench_humaneval_format.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            #important
            if num_id < 0 or num_id > 32:  
                continue
            #important
            dataset.append(problem)
    print(f"load {len(dataset)} problems")
    
    time_start = time.time()
    cir_times = 10
    if feedback_type=="expl" or dataset_type!="humaneval":
        cir_times = 5
    if Strategy == "TS":
        run_tree_search(dataset,model_path, output_file,
                        sample_num=sample_num, filter_num=filter_num,feedback_type=feedback_type,
                        first_num = first_num,
                        cir_times=cir_times ,verbose=True)
    elif Strategy == "TS2":
        run_tree_search2(dataset,model_path, output_file,
                        sample_num=sample_num, filter_num=filter_num,feedback_type=feedback_type,
                        first_num = first_num,
                        cir_times=cir_times ,verbose=True)
    elif Strategy == "NTS":
        run_not_tree_search(dataset,model_path, output_file, 
                            sample_num=sample_num, filter_num=filter_num,feedback_type=feedback_type,
                            first_num = first_num,
                            cir_times=cir_times ,verbose=True)
    elif Strategy == "TFTS":
        run_testcase_filter(dataset,model_path, output_file,
                            sample_num=sample_num, filter_num=filter_num,feedback_type=feedback_type,
                            first_num = first_num,
                            cir_times=cir_times ,verbose=True)
    elif Strategy == "BASE":
        run_base_generate(dataset,model_path, output_file ,verbose=True)
    elif Strategy == "debug":
        run_fastdebug_test(dataset,model_path, output_file, sample_num=sample_num,cir_times=10 ,verbose=True)
    elif Strategy == "TSC":
        run_tree_search_cached(dataset,model_path, output_file, sample_num=sample_num, filter_num=filter_num,cir_times=10 ,verbose=True)
    elif Strategy == "TFTC":
        run_testcase_filter_cached(dataset,model_path, output_file, sample_num=sample_num, filter_num=filter_num,cir_times=10 ,verbose=True)
    elif Strategy == "test":
        gen_code_expl(dataset,model_path, output_file, sample_num=sample_num, filter_num=filter_num,cir_times=10 ,verbose=True)
    elif Strategy == "mcts":
        run_mcts_search(dataset,model_path, output_file, sample_num=sample_num, filter_num=filter_num,cir_times=10 ,verbose=True)
    elif Strategy == "TG":
        testcase_generate(dataset,model_path, output_file, sample_num=sample_num, filter_num=filter_num,cir_times=10 ,verbose=True)
    else:
        print("Strategy not found")
        return 1
    time_end = time.time()
    time_cost = (time_end - time_start) / 60
    print('time cost:', time_cost, 'min')
    return 0

if __name__ == "__main__":
    main()