#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J mix10_153              # The job name
#SBATCH -o ../log/humaneval_UTfeedback_PassRate_mix10_10_7b16k_pT_153.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ../log/humaneval_UTfeedback_PassRate_mix10_10_7b16k_pT_153.out        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw                    # Submit to 'nv-gpu' Partitiion
#SBATCH -t 0-30:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:4                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-normal             # Request QOS Type

####SBATCH --ntasks-per-node=48

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
#SBATCH --cpus-per-task=64            # Request K cores
####SBATCH -n 50
####SBATCH --mem-per-cpu=100000
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
#SBATCH --nodelist=r8a100-c01          # Request a specific list of hosts 
#SBATCH --constraint="80G" # Request GPU Type: Volta(V100 or V100S) or RTX8000
###

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cmake/3.21.7

##- language
## module load python3/3.8.16

##- CUDA
module load cuda-cudnn/11.7-8.5.0

##- virtualenv

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Use GPU ${CUDA_VISIBLE_DEVICES}"                              # which gpus
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM
nvidia-smi

#- Job step
cd /home/S/hexiaolong/codex/self-debug/src
python3.9 job-record.py -jobID $SLURM_JOBID -comment "use gened testcase for feedback with 100 percent correctness.prompt testcase for passrate." -output_file "../res/UTfeedback_PassRate_mix10_10_7b16k_pT_153.jsonl"
python3.9 humaneval_UTfeedback_multiCodeT9.py +model_path=/lustre/S/hexiaolong/vicuna-7b-16k/ +output=../res/UTfeedback_PassRate_mix10_10_7b16k_pT_153.jsonl
#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
