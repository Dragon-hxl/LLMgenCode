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
    Strategy = cfg.Strategy
    dataset_type = cfg.dataset
    
    # load task
    dataset = []
    print(f"load dataset:{dataset_type}")
    if dataset_type == "humaneval":
        print("load dataset : humaneval")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/data/humaneval.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            if num_id < 132 or num_id > 164:
                continue
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
        chosen_idx = chosen_idx[129:150]
        dataset = [dataset[i] for i in chosen_idx]
    elif dataset_type == "mtpb":
        print("load dataset : mtpb")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/benchmarks/mtpb_humaneval_format.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            if num_id < 0 or num_id > 116 or num_id==20 or num_id==47:
                continue
            dataset.append(problem)
    elif dataset_type == "bigbench":
        print("load dataset : bigbench")
        problems = read_problems("/home/S/hexiaolong/codex/self-debug/benchmarks/bigbench_humaneval_format.jsonl")
        for tid,problem in problems.items():
            num_id = int(tid.split("/")[1])
            if num_id < 20 or num_id > 32:
                continue
            dataset.append(problem)
    print(f"load {len(dataset)} problems")
    
    time_start = time.time()
    if Strategy == "TS":
        run_tree_search(dataset,model_path, output_file, sample_num=sample_num, cir_times=10 ,verbose=True)
    elif Strategy == "TFTS":
        run_testcase_filter(dataset,model_path, output_file, sample_num=sample_num, cir_times=10 ,verbose=True)
    elif Strategy == "BASE":
        run_base_generate(dataset,model_path, output_file ,verbose=True)
    elif Strategy == "NTS":
        run_not_tree_search(dataset,model_path, output_file, sample_num=sample_num, cir_times=10 ,verbose=True)
    else:
        print("Strategy not found")
        return 1
    time_end = time.time()
    time_cost = (time_end - time_start) / 60
    print('time cost:', time_cost, 'min')
    return 0

if __name__ == "__main__":
    main()