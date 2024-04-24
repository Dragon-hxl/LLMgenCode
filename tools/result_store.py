

humaneval_result = [
    "humanevalTS_SBSP_codellama7bpy_pT_pass@1":[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113],
    "humanevalTS_SBSP_codellama7bpy_pT_pass@10":[63, 81, 105, 111, 112, 113, 113, 113, 113, 113, 114],
    "humanevalTFTS_SBSP_codellama7bpy_pT_pass@1":[63, 82, 104, 110, 111, 112, 112, 113, 113, 113, 114],
    "humanevalTFTS_SBSP_codellama7bpy_pT_pass@10":[63, 83, 105, 111, 112, 113, 113, 114, 114, 114, 115],
]

mbpp_cola7bpy_pack  = {
    "data":{
        "mbppNTS_SBSP10_codellama7bpy_pT.jsonl":[83, 86, 90, 90, 91, 92, 92, 92, 92, 92, 92],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "mbppTS_SBSP10_codellama7bpy_pT.jsonl":[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94],
        "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl":[83, 92, 95, 95, 96, 96, 97, 97, 97, 97, 97],
    },
    
    "image_path":"../image/mbpp_cola7bpy_pass@1.png",
    "num_task":200,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}

bigbench_cola7bpy_pack  = {
    "pass_task":{
        "bigbenchNTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "bigbenchTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],
        "bigbenchTFTS_SBSP10_codellama7bpy_pT.jsonl":[1, 2, 3, 4, 5, 6, 8, 11, 14, 15, 16, 17, 19, 20, 21, 22, 27],
    },
    
    "data":{
        "bigbenchNTS_SBSP10_codellama7bpy_pT.jsonl":[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17],#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
        "bigbenchTS_SBSP10_codellama7bpy_pT.jsonl":[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17],
        "bigbenchTFTS_SBSP10_codellama7bpy_pT.jsonl":[14, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
    },
    
    "image_path":"../image/bigbench_cola7bpy_pass@1.png",
    "num_task":32,
    "legends":["使用测试用例筛选的树搜索自反馈","树搜索自反馈","非树搜索自反馈","原始自反馈"],
}