



python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-7bpy +output=../res/humanevalTS_SBSP10_codellama7bpy_pT_0.jsonl +sample_num=10 +Strategy=TS +dataset=humaneval

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-7bpy +output=../res/mbppTS_SBSP10_codellama7bpy_pT_0.jsonl +sample_num=10 +Strategy=TS +dataset=mbpp

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-7bpy +output=../res/bigbenchTS_SBSP10_codellama7bpy_0.jsonl +sample_num=10 +Strategy=TS +dataset=bigbench > ../log/bigbenchTS_SBSP10_codellama7bpy_0.out 2>&1

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-7bpy +output=../res/bigbenchNTS_SBSP10_codellama7bpy_0.jsonl +sample_num=10 +Strategy=NTS +dataset=bigbench > ../log/bigbenchNTS_SBSP10_codellama7bpy_0.out 2>&1

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-34bpy +output=../res/humanevalTS_SBSP10_codellama34bpy_pT_0.jsonl +sample_num=10 +Strategy=TS +dataset=humaneval

python3.9 main.py +model_path=/lustre/S/hexiaolong/llama7b +output=../res/humanevalTS_SBSP10_llama7b_pT_0.jsonl +sample_num=10 +Strategy=TS +dataset=humaneval

python3.9 main.py +model_path=/lustre/S/hexiaolong/llama7b +output=../res/humanevalTFTS_SBSP10_llama7b_pT_0.jsonl +sample_num=10 +Strategy=TFTS +dataset=humaneval > ../log/humanevalTFTS_SBSP10_llama7b_pT_0.out 2>&1

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-34bpy +output=../res/mbppTS_SBSP10_codellama34bpy_pT_0.jsonl +sample_num=10 +Strategy=TS +dataset=mbpp > ../log/mbppTS_SBSP10_codellama34bpy_pT_0.out 2>&1

python3.9 main.py +model_path=/lustre/S/hexiaolong/codellama-7bpy +output=../res/fastdebug_test.jsonl +sample_num=10 +Strategy=debug +dataset=humaneval > ../log/fastdebug_test.out 2>&1

python3.9 main.py +model_path=/lustre/S/hexiaolong/vicuna-7b-16k +output=../res/fastdebug_test_7b16k.jsonl +sample_num=10 +Strategy=debug +dataset=humaneval > ../log/fastdebug_test_7b16k.out 2>&1