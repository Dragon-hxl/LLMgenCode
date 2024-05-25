

res_root = "/home/S/hexiaolong/codex/self-debug/res/"

data_files = {
        "humaneval":"/home/S/hexiaolong/codex/self-debug/data/humaneval.jsonl",
        "mbpp":"/home/S/hexiaolong/codex/self-debug/MBPP/mbpp_humaneval_format_11_510.jsonl",
        "mbpt":"/home/S/hexiaolong/codex/self-debug/benchmarks/mtpb_humaneval_format.jsonl",
        "bigbench":"/home/S/hexiaolong/codex/self-debug/benchmarks/bigbench_humaneval_format.jsonl",
}



res_7b16k = [
        "humanevalTSC_SBSP_7b16k.jsonl",
        "humanevalTS_SBSP10_7b16k_0.jsonl",
        "humanevalTS_SBSP1_7b16k.jsonl",
        "humanevalNTS_SBSP10_7b16k.jsonl",
        "humanevalTS_SBSP10_7b16k.jsonl",
        "humanevalTFTS_SBSP10_7b16k.jsonl",
        
        "mbppTS_SBSP1_7b16k.jsonl",
        "mbppNTS_SBSP10_7b16k.jsonl",
        "mbppTS_SBSP10_7b16k.jsonl",
        "mbppTFTS_SBSP10_7b16k.jsonl",
        
        "mtpbTS_SBSP1_7b16k.jsonl",
        "mtpbNTS_SBSP10_7b16k.jsonl",
        "mtpbTS_SBSP10_7b16k.jsonl",
        "mtpbTFTS_SBSP10_7b16k.jsonl",
        
        "bigbenchTS_SBSP1_7b16k.jsonl",
        "bigbenchNTS_SBSP10_7b16k2.jsonl",
        "bigbenchTS_SBSP10_7b16k2.jsonl",
        "bigbenchTFTS_SBSP10_7b16k.jsonl",
    ]

res_cola7bpy = [
    "humanevalTS_SBSP1_codellama7bpy.jsonl",
    "humanevalNTS_SBSP10_codellama7bpy_pT.jsonl",
    "humanevalTS_SBSP10_codellama7bpy_pT.jsonl",
    "humanevalTFTS_SBSP10_codellama7bpy_pT.jsonl",
    
    "mbppTS_SBSP1_codellama7bpy.jsonl",
    "mbppNTS_SBSP10_codellama7bpy_pT.jsonl",
    "mbppTS_SBSP10_codellama7bpy_pT.jsonl",
    "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl",
    
    "mtpbTS_SBSP1_codellama7bpy.jsonl",
    "mtpbNTS_SBSP10_codellama7bpy.jsonl",
    "mtpbTS_SBSP10_codellama7bpy.jsonl",
    "mtpbTFTS_SBSP10_codellama7bpy.jsonl",
    
    "bigbenchTS_SBSP1_codellama7bpy.jsonl",
    "bigbenchNTS_SBSP10_codellama7bpy.jsonl",
    "bigbenchTS_SBSP10_codellama7bpy.jsonl",
    "bigbenchTFTS_SBSP10_codellama7bpy.jsonl",
]


res_cola34bpy = [
    "humanevalNTS_SBSP10_codellama34bpy_pT.jsonl",
    "humanevalTS_SBSP10_codellama34bpy_pT.jsonl",
    "humanevalTFTS_SBSP10_codellama34bpy_pT.jsonl",
    
    "mbppNTS_SBSP10_codellama7bpy_pT.jsonl",
    "mbppTS_SBSP10_codellama7bpy_pT.jsonl",
    "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl",
 ]

res_llama7b = [
    "humanevalNTS_SBSP10_llama7b.jsonl",
    "humanevalTS_SBSP10_llama7b.jsonl",
    "humanevalTFTS_SBSP10_llama7b.jsonl",
]

tmp = [
    "humanevalTSA_SBSP10_codellama7bpy_0.jsonl",
    "humanevalTSA2_SBSP10_codellama7bpy.jsonl",
    
    "humanevalTFTSA_SBSP10_codellama7bpy_0.jsonl",
    "humanevalTFTSA2_SBSP10_codellama7bpy_0.jsonl",
    
    
    "humanevalNTSA_SBSP1_codellama7bpy_0.jsonl",#[63, 80, 89, 92, 92, 93, 93, 93, 93, 93, 93]
    "humanevalTSA_SBSP1_codellama7bpy.jsonl",#[63, 80, 89, 92, 92, 93, 93, 93, 93, 93, 93]
    "humanevalTSA2_SBSP1_codellama7bpy_0.jsonl",#[0, 17, 26, 30, 30, 31, 31, 31, 31, 31, 31]
    
    "humanevalTSA_SBSP2_codellama7bpy_0.jsonl",
    "humanevalTSA_SBSP3_codellama7bpy_0.jsonl",#[0, 17, 27, 28, 31, 31, 31, 31, 31, 31, 31]
    "humanevalTSA_SBSP5_codellama7bpy_0.jsonl",#[0, 17, 29, 31, 32, 32, 32, 32, 32, 32, 32]
    "humanevalTSA_F1_S15_codellama7bpy_0.jsonl",#[0, 26, 34, 34, 34, 36, 36, 36, 37, 37, 37]
    "humanevalTSA_F1_S20_codellama7bpy_0.jsonl",#[0, 27, 33, 35, 35, 36, 36, 36, 36, 36, 36]
    "humanevalTSA_F1_S25_codellama7bpy_0.jsonl",#[0, 28, 33, 34, 34, 35, 36, 36, 36, 37, 37]
    "humanevalTSA_F1_S30_codellama7bpy_0.jsonl",#[0, 32, 40, 41, 42, 43, 43, 43, 43, 43, 43]
    
    "humanevalNTSA_F5_S2_codellama7bpy_0.jsonl",
    "humanevalNTS_simple_F1_S10_codellama7bpy.jsonl",#[0, 22, 25, 29, 29, 29, 30, 30, 30, 30, 30]
    "humanevalTS_simple_F1_S10_codellama7bpy.jsonl",#[0, 22, 26, 27, 27, 27, 27, 27, 27, 27, 27]
    "humanevalTS_simple_F5_S2_codellama7bpy.jsonl",#[0, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    "humanevalTFTSA_F2_S5_codellama7bpy_0.jsonl",
    "humanevalNTS_simple_F1_S1_codellama7bpy.jsonl",#[0, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7]
]