#!/bin/bash

#- Job parameters
# (TODO)
# Please modify job name

#SBATCH -J base              # The job name
#SBATCH -o humaneval_base_7bv0.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e humaneval_base_7bv0.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu                    # Submit to 'nv-gpu' Partitiion
#SBATCH -t 0-12:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:2                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-short             # Request QOS Type

####SBATCH --ntasks-per-node=48

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
#SBATCH --cpus-per-task=16            # Request K cores
####SBATCH -n 50
####SBATCH --mem-per-cpu=100000
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
####SBATCH --nodelist=gpu-v20          # Request a specific list of hosts 
#SBATCH --constraint="24G" # Request GPU Type: Volta(V100 or V100S) or RTX8000
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
module load python3/3.8.16

##- CUDA
module load cuda-cudnn/11.6-8.4.1

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


#- Job step
cd ~/codex/human-eval
./run_selfdebug.sh $1 $2
#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
