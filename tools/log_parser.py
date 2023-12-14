import json
import re
from collections import defaultdict
def extract_chosen_nodes_idx(log_file):
    cir = 0
    chosen_nodes_idx_dict = defaultdict([])
    chosen_nodes_parent_idx_dict = defaultdict([])
    with open(log_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            if "get solution for task" in line:
                tid = line[len("get solution for task : "):].strip()
            elif "total unique nodes:" in line:
                cir_match = re.search(r", cir:(.*?), total nodes",line)
                cir = int(cir_match.group(1))
            elif "chosen nodes idx is " in line:
                s = line[len("chosen nodes idx is "):].strip()
                chosen_nodes_idx = json.loads(s)
                chosen_nodes_idx_dict[tid].append({"cir":cir,"idx":chosen_nodes_idx})
            elif "chosen nodes's parent's idx is " in line:
                s = line[len("chosen nodes's parent's idx is "):].strip()
                chosen_nodes_parent_idx = json.loads(s)
                chosen_nodes_parent_idx_dict[tid].append({"cir":cir,"idx":chosen_nodes_parent_idx})
    return chosen_nodes_idx_dict,chosen_nodes_parent_idx_dict

def analysis_nodes_idx(chosen_nodes_idx_dict,chosen_nodes_parent_idx_dict):
    
                
                