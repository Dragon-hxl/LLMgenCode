
humaneval_7b16k_pack = {
    "pass@1":{
        "humanevalTS_SBSP1_7b16k_pT.jsonl":[27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
        "humanevalNTS_SBSP10_7b16k_pT.jsonl":[27, 35, 50, 56, 61, 64, 67, 68, 68, 68, 68],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "humanevalTS_SBSP10_7b16k_pT.jsonl":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
        "humanevalTFTS_SBSP10_7b16k_pT.jsonl":[27, 40, 57, 63, 65, 69, 71, 72, 74, 75, 77],
    },
    "pass@10":{
        "humanevalTS_SBSP1_7b16k_pT.jsonl":[27, 28, 31, 31, 32, 32, 32, 32, 33, 34, 35],
        "humanevalNTS_SBSP10_7b16k_pT.jsonl":[27, 35, 50, 56, 61, 64, 67, 68, 68, 68, 68],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "humanevalTS_SBSP10_7b16k_pT.jsonl":[27, 36, 51, 60, 63, 66, 69, 71, 71, 72, 73],
        "humanevalTFTS_SBSP10_7b16k_pT.jsonl":[27, 40, 57, 63, 65, 69, 71, 72, 74, 75, 77],
    },
    
    "image_path":"../image/humaneval_7b16k_pass@1.png",
    "num_task":164,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}


humaneval_cola7bpy_pack = {
    "pass@1":{
        "humanevalTS_SBSP1_codellama7bpy_pT.jsonl":[63, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67],
        "humanevalNTS_SBSP10_codellama7bpy_pT.jsonl":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "humanevalTS_SBSP10_codellama7bpy_pT.jsonl":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],
        "humanevalTFTS_SBSP10_codellama7bpy_pT.jsonl":[63, 81, 98, 101, 104, 106, 106, 107, 107, 108, 108],
    },
    
    "image_path":"../image/humaneval_cola7bpy_pass@1.png",
    "num_task":164,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}

mbpp_cola7bpy_pack  = {
    "pass@1":{
        "mbppTS_SBSP1_codellama7bpy_pT.jsonl":[41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
        "mbppNTS_SBSP10_codellama7bpy_pT.jsonl":[83, 86, 90, 90, 91, 92, 92, 92, 92, 92, 92],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "mbppTS_SBSP10_codellama7bpy_pT.jsonl":[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94],
        "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl":[83, 92, 95, 95, 96, 96, 97, 97, 97, 97, 97],
    },
    "pass@10":{
        "mbppNTS_SBSP10_codellama7bpy_pT.jsonl":[83, 103, 108, 108, 108, 108, 109, 109, 109, 109, 109],#[156, 367]
        "mbppTS_SBSP10_codellama7bpy_pT.jsonl":[83, 103, 108, 108, 108, 108, 109, 109, 109, 109, 109],#[120, 367]
        "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl":[79, 104, 111, 111, 112, 112, 113, 113, 113, 113, 113],
    },
    
    
    "image_path":"../image/mbpp_cola7bpy_pass@1.png",
    "num_task":200,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}

mtpb_cola7bpy_pack  = {
    "pass@1":{
        "mtpbTS_SBSP1_codellama7bpy_pT.jsonl":[27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
        "mtpbNTS_SBSP10_codellama7bpy_pT.jsonl":[27, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "mtpbTS_SBSP10_codellama7bpy_pT.jsonl":[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94],
        "mtpbTFTS_SBSP10_codellama7bpy_pT.jsonl":[83, 92, 95, 95, 96, 96, 97, 97, 97, 97, 97],
    },
    "image_path":"../image/mtpb_cola7bpy_pass@1.png",
    "num_task":115,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}

bigbench_cola7bpy_pack  = {
    "pass_task":{
        "bigbenchNTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "bigbenchTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],
        "bigbenchTFTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],
    },
    
    "pass@1":{
        "bigbenchTS_SBSP1_codellama7bpy_pT.jsonl":[14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        "bigbenchNTS_SBSP10_codellama7bpy_pT.jsonl":[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "bigbenchTS_SBSP10_codellama7bpy_pT.jsonl":[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17],
        "bigbenchTFTS_SBSP10_codellama7bpy_pT.jsonl":[14, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
    },
    
    "pass@10":{
        "bigbenchNTS_SBSP10_codellama7bpy_pT.jsonl":[14, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18],
        "bigbenchTS_SBSP10_codellama7bpy_pT.jsonl":[14, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18],
        "bigbenchTFTS_SBSP10_codellama7bpy_pT.jsonl":[14, 17, 19, 19, 19, 20, 20, 20, 20, 20, 20],
    },
    
    "image_path":"../image/bigbench_cola7bpy_pass@1.png",
    "num_task":32,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}