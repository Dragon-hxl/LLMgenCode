

res_root = "/home/S/hexiaolong/codex/self-debug/res/"

data_files = {
        "humaneval":"/home/S/hexiaolong/codex/self-debug/data/humaneval.jsonl",
        "mbpp":"/home/S/hexiaolong/codex/self-debug/MBPP/mbpp_humaneval_format_11_510.jsonl",
        "mbpt":"/home/S/hexiaolong/codex/self-debug/benchmarks/mtpb_humaneval_format.jsonl",
        "bigbench":"/home/S/hexiaolong/codex/self-debug/benchmarks/bigbench_humaneval_format.jsonl",
}



res_7b16k = [
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
    "humanevalTS_SBSP1_codellama7bpy.jsonl",#[63, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67]
    "humanevalNTS_SBSP10_codellama7bpy_pT.jsonl",#[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113]
    "humanevalTS_SBSP10_codellama7bpy_pT.jsonl",#[63, 80, 104, 110, 111, 112, 112, 112, 112, 112, 113]
    "humanevalTFTS_SBSP10_codellama7bpy_pT.jsonl",#[63, 81, 98, 101, 104, 106, 106, 107, 107, 108, 108]
    
    "mbppTS_SBSP1_codellama7bpy.jsonl",
    "mbppNTS_SBSP10_codellama7bpy_pT.jsonl",#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
    "mbppTS_SBSP10_codellama7bpy_pT.jsonl",#[83, 88, 93, 93, 93, 94, 94, 94, 94, 94, 94]
    "mbppTFTS_SBSP10_codellama7bpy_pT.jsonl",#[79, 92, 95, 95, 96, 96, 97, 97, 97, 97, 97]
    
    "mtpbTS_SBSP1_codellama7bpy.jsonl",#[27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28]
    "mtpbNTS_SBSP10_codellama7bpy.jsonl",#[26, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    
    "mtpbTS_SBSP10_codellama7bpy.jsonl",#[26, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    #TAG [20, 99, 114, 115]
    
    "mtpbTFTS_SBSP10_codellama7bpy.jsonl",#[14, 29, 31, 33, 34, 34, 34, 34, 34, 34, 34]
    
    "bigbenchTS_SBSP1_codellama7bpy.jsonl",#[14, 14,14, 14, 14, 14, 14, 14, 14, 14, 14]
    "bigbenchNTS_SBSP10_codellama7bpy.jsonl",#[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17]
    "bigbenchTS_SBSP10_codellama7bpy.jsonl",#[14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17]
    "bigbenchTFTS_SBSP10_codellama7bpy.jsonl",#[11, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
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
    "humanevalTSA_SBSP10_codellama7bpy_0.jsonl",#0
    "humanevalTSA2_SBSP10_codellama7bpy.jsonl",#1
    "humanevalTFTSA_SBSP10_codellama7bpy_0.jsonl",
    "humanevalTFTSA2_SBSP10_codellama7bpy_0.jsonl",
    "humanevalNTSA_SBSP1_codellama7bpy_0.jsonl",#[63, 80, 89, 92, 92, 93, 93, 93, 93, 93, 93]
    "humanevalTSA_SBSP1_codellama7bpy.jsonl",#[63, 80, 89, 92, 92, 93, 93, 93, 93, 93, 93]
    "humanevalTSA2_SBSP1_codellama7bpy_0.jsonl",#[0, 17, 26, 30, 30, 31, 31, 31, 31, 31, 31]
    "humanevalTSA_SBSP2_codellama7bpy_0.jsonl",#7
    "humanevalTSA_SBSP3_codellama7bpy_0.jsonl",#[0, 17, 27, 28, 31, 31, 31, 31, 31, 31, 31]
    "humanevalTSA_SBSP5_codellama7bpy_0.jsonl",#[0, 17, 29, 31, 32, 32, 32, 32, 32, 32, 32]
    #TAG [6, 8, 14, 18, 19, 25, 26, 37, 40, 50, 51, 54, 59, 68, 82, 84, 85, 90, 95, 96, 97, 100, 101, 105, 106, 109, 114, 117, 135, 143, 146, 158]
    
    #开始新标准
    #TAG UT反馈
    "humanevalTSA_F1_S15_codellama7bpy_0.jsonl",#[0, 26, 34, 34, 34, 36, 36, 36, 37, 37, 37]
    "humanevalTSA_F1_S20_codellama7bpy_0.jsonl",#[0, 27, 33, 35, 35, 36, 36, 36, 36, 36, 36]
    "humanevalTSA_F1_S25_codellama7bpy_0.jsonl",#[0, 28, 33, 34, 34, 35, 36, 36, 36, 37, 37]
    "humanevalTSA_F1_S30_codellama7bpy_0.jsonl",#[0, 32, 40, 41, 42, 43, 43, 43, 43, 43, 43]
    
    "humanevalNTSA_F5_S2_codellama7bpy_0.jsonl",#14
    #TAG [8, 14, 18, 26, 40, 50, 51, 59, 82, 95, 96, 106, 109, 133, 138, 143]
    "humanevalTFTSA_F2_S5_codellama7bpy_0.jsonl",#[0, 9, 21, 23, 23, 23, 25, 25, 25, 25, 25]
    "humanevalTFTSA_F5_S2_codellama7bpy_0.jsonl",#[0, 6, 8, 12, 12, 12, 12, 12, 12, 12, 12]
    "humanevalTFTS_F1_S10_codellama7bpy.jsonl",#[0, 16, 21, 21, 21, 26, 28, 29, 31, 31, 31]
    "humanevalNTSA_F5_S2_codellama7bpy_0.jsonl",#22
    "humanevalTS_UT_F1_S10_codellama7bpy_origin.jsonl",#[0, 17, 26, 29, 29, 30, 30, 30, 30, 30, 30]
    "humanevalTFTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 9, 21, 23, 23, 27, 29, 29, 29, 29, 29]
    "humanevalTFTS_UT_F1_S10_codellama7bpy_origin.jsonl",#[0, 18, 23, 23, 24, 29, 31, 32, 33, 33, 33]
    "humanevalNTS_UT_F1_S1_codellama7bpy.jsonl",
    "humanevalTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 12, 25, 28, 32, 32, 33, 34, 34, 34, 34]
    
    
    #HH simple反馈
    "humanevalNTS_simple_F1_S10_codellama7bpy.jsonl",#[0, 22, 25, 29, 29, 29, 30, 30, 30, 30, 30]
    "humanevalTS_simple_F1_S10_codellama7bpy.jsonl",#[0, 22, 26, 27, 27, 27, 27, 27, 27, 27, 27]
    "humanevalTS_simple_F5_S2_codellama7bpy.jsonl",#[0, 17, 24, 26, 27, 28, 28, 28, 28, 28, 28]
    "humanevalTS_simple_F2_S5_codellama7bpy2.jsonl",#[0, 17, 24, 26, 27, 28, 28, 28, 28, 28, 28]
    
    "humanevalNTS_simple_F1_S1_codellama7bpy.jsonl",#[0, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    "humanevalNTS_simple_F1_S1_codellama7bpy2.jsonl",#[0, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    "humanevalNTS_simple_F5_S2_codellama7bpy.jsonl",#[0, 9, 16, 16, 17, 17, 17, 17, 17, 17, 17]
    "humanevalNTS_simple_F1_S10_codellama7bpy3.jsonl",#[0, 22, 26, 27, 27, 28, 28, 29, 29, 29, 30]
    "humanevalNTS_simple_F5_S2_codellama7bpy2.jsonl",
    #[0, 9, 16, 16, 17, 17, 17, 18, 19, 19, 19]
    #[8, 14, 18, 19, 25, 37, 50, 51, 67, 82, 96, 97, 102, 105, 106, 123, 133, 138, 143]
    "humanevalNTS_simple_F2_S5_codellama7bpy.jsonl",
    "humanevalNTS_simple_F1_S10_codellama7bpy3.jsonl",
    
    "humanevalTFTS_simple_F1_S10_codellama7bpy_origin.jsonl",#[0, 19, 24, 25, 25, 25, 26, 27, 29, 30, 30]
    
    #HH expl反馈
    "humanevalNTS_expl_F1_S1_codellama7bpy2_51.jsonl",
    
    #TAG 测试位置
    
    "humanevalTS_UT_F2_S5_codellama7bpy_NPT_new.jsonl",#[33, 38, 40, 40, 40, 41, 41, 41, 41, 41, 41]
    "humanevalTFTS_UT_F1_S10_codellama7bpy_new_G20.jsonl",#[42, 43, 43, 43, 43, 44, 46, 46, 46, 46, 46]
    "humanevalTFTS_UT_F5_S2_codellama7bpy_new0.jsonl",#[40, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48]
    
    "humanevalTFTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 9, 21, 23, 23, 27, 29, 29, 29, 29, 29]
    "humanevalTS_UT_F1_S10_codellama7bpy_new.jsonl",#[38, 42, 45, 46, 46, 46, 46, 46, 46, 46, 46]
    "humanevalTFTS_UT_F1_S10_codellama7bpy_new.jsonl",#[38, 41, 43, 46, 46, 46, 46, 46, 46, 46, 46]
    "humanevalNTS_UT_F1_S10_codellama7bpy_new.jsonl",#[38, 42, 45, 46, 46, 46, 46, 46, 46, 46, 46]
    "humanevalNTS_UT_F1_S10_codellama7bpy_new_sortlen.jsonl",#[38, 40, 42, 43, 43, 43, 44, 44, 44, 44, 44]
    "humanevalTS_UT_F1_S10_codellama7bpy_new_sortlen.jsonl",#[38, 40, 42, 43, 43, 43, 43, 43, 43, 43, 43]
    "humanevalNTS_UT_F5_S2_codellama7bpy.jsonl",
    "humanevalNTS_UT_F2_S5_codellama7bpy.jsonl",
    "humanevalTFTS_UT_F2_S5_codellama7bpy.jsonl",
    "humanevalNTS_UT_F5_S2_codellama7bpy.jsonl",
    "humanevalTFTS_UT_F5_S2_codellama7bpy_NPT_new.jsonl",#[37, 44, 45, 46, 47, 48, 48, 48, 48, 48, 48]
    
    "humanevalTFTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 9, 21, 23, 23, 27, 29, 29, 29, 29, 29]
    
    "humanevalTFTS_UT_F2_S5_codellama7bpy_NPT_new.jsonl",#[30, 33, 34, 35, 35, 36, 37, 37, 37, 37, 37]+7
    #lack :[83, 123, 146, 148, 149, 150, 151, 154, 155, 157, 158, 160, 161, 163]
    #[1, 6, 8, 14, 18, 19, 25, 37, 38, 40, 41, 50, 51, 52, 54, 57, 59, 62, 67, 68, 69, 73, 78, 82, 84, 85, 88, 90, 92, 94, 96, 97, 101, 106, 117, 133, 143]
    
    
    "humanevalTS_UT_F5_S2_codellama7bpy_NPT_new.jsonl",#[31, 35, 37, 37, 37, 37, 37, 37, 37, 37, 37] lack[163]
    "humanevalTS_UT_F2_S5_codellama7bpy_NPT_new.jsonl",
    #[8, 10, 11, 14, 18, 19, 25, 40, 41, 50, 51, 54, 57, 59, 62, 70, 73, 78, 82, 83, 88, 90, 92, 94, 96, 97, 101, 105, 106, 114, 120, 123, 140, 143, 146, 151, 154, 155, 157, 158, 160]
    "humanevalTFTS_UT_F5_S2_codellama7bpy_NPT_new.jsonl",
    "humanevalTS_simple_F5_S2_codellama7bpy_new.jsonl",
    "humanevalNTS_simple_F5_S2_codellama7bpy_new.jsonl",
    # "humanevalTS2_simple_F5_S2_codellama7bpy_new.jsonl",
    
    # "humanevalTFTS_simple_F5_S2_codellama7bpy_new.jsonl",
    # "humanevalNTS_simple_F5_S2_codellama7bpy_new.jsonl",
    "humanevalNTS_UT_F5_S2_codellama7bpy.jsonl",#[0, 17, 28, 30, 31, 31, 31, 31, 31, 31, 31]
    # "humanevalNTS_UT_F2_S5_codellama7bpy.jsonl",
    "humanevalTS_UT_F5_S2_codellama7bpy_origin.jsonl",
    "humanevalNTS_simple_F5_S2_codellama7bpy_new.jsonl",
    "humanevalTS_UT_F5_S2_codellama7bpy_new.jsonl",#[39, 48, 49, 49, 49, 50, 50, 50, 50, 50, 50]
    "humanevalTFTS_UT_F5_S2_codellama7bpy_new0.jsonl",
    "humanevalNTS_UT_F5_S2_codellama7bpy_new.jsonl",#[39, 48, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    "humanevalTS_simple_F5_S2_codellama7bpy_new_050.jsonl",#[34, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38]
    
    "humanevalTFTS_UT_F5_S2_codellama7bpy_new_0.jsonl",
    
    "humanevalNTS_UT_F1_S6_codellama7bpy.jsonl",#[30, 36, 40, 42, 42, 42, 42, 42, 42, 42, 42]
    "humanevalTS_UT_F1_S6_codellama7bpy.jsonl",#[30, 36, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    "humanevalNTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 12, 25, 28, 32, 32, 32, 33, 33, 33, 33]
    "humanevalTS_UT_F2_S5_codellama7bpy.jsonl",#[0, 12, 25, 28, 32, 32, 33, 34, 34, 34, 34]
    "humanevalNTS_simple_I10_F1_S10_codellama7bpy_new.jsonl",#[38, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]
    "humanevalNTS_simple_F5_S2_codellama7bpy_new.jsonl",#[38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39]
    "humanevalTSA_SBSP5_codellama7bpy_0.jsonl",#[0, 17, 29, 31, 32, 32, 32, 32, 32, 32, 32]
    "humanevalNTS_UT_F5_S2_codellama7bpy.jsonl",
    "humanevalNTS_expl_I10_F1_S10_codellama7bpy_new.jsonl",#[39, 42, 44, 44, 44, 44, 44, 44, 44, 44, 44]
    "humanevalNTS_UT_I10_F1_S10_codellama7bpy_new_0.jsonl",#[38, 43, 47, 48, 48, 48, 48, 48, 48, 48, 48]
    "humanevalTFTS_I6_UT_F1_S6_codellama7bpy.jsonl",
    "humanevalNTS_UT_F1_S6_codellama7bpy.jsonl",
    
    
    "humanevalNTS_UT_I5_F1_S5_codellama7bpy.jsonl",#[26, 30, 34, 34, 34, 34, 34, 34, 34, 34, 34]
    "humanevalTFTS_UT_I5_F1_S5_codellama7bpy.jsonl",#[25, 31, 31, 31, 31, 33, 34, 35, 35, 35, 35]
    "humanevalTS_UT_I5_F1_S5_codellama7bpy.jsonl",#[26, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35]
    "humanevalTS_UT_I1_F10_S10_vicuna7b16k.jsonl",#[27, 36, 50, 52, 56, 57, 58, 58, 58, 58, 58]
    "humanevalTFTS_UT_I1_F10_S10_vicuna7b16k.jsonl",
    "humanevalTS_UT_I1_F10_S10_vicuna7b16k_seed1024.jsonl",
    "humanevalTS_UT_I1_F10_S10_vicuna7b16k_noseed.jsonl",
    "humanevalTS_UT_I1_F10_S10_vicuna7b16k_long.jsonl",
    "UTfeedback_CODETv3_7b16k_pT.jsonl",
    "UTfeedback_multiSBSP10_7b16k_pT.jsonl",
    "UTfeedback_CODETv3_t8_7b16k_pT.jsonl",
    "humanevalTS_UT_I1_F10_S10_vicuna7b16k_long1024.jsonl",
    
    "humanevalTS_SBSP1_codellama34bpy.jsonl",
    "humanevalNTS_SBSP10_codellama34bpy_pT.jsonl",
    "humanevalTS_SBSP10_codellama34bpy_pT.jsonl",
    "humanevalTFTS_SBSP10_codellama34bpy_pT.jsonl",
    
    
    
]
sim_res = [
    "humanevalNTS_simple_F5_S2_codellama7bpy.jsonl",
]