#!/bin/bash
#SBATCH --job-name=ukrlm_mlm
#SBATCH --account=plgexaile-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH -C memfs
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source modules.sh
source scratch/masters/masters-venv/bin/activate

cd masters/ukr-lm

python3 main-ml.py \
    accelerator=gpu \
    datamodule.batch_size=64 \
    huggingface_cache_dir=/net/people/plgrid/plggoader/scratch/masters/huggingface_cache/ \
    datasets.cc100.streaming=true \
    task.strategy=ddp \
    datasets.cc100.num_shards=1 \
    datamodule.num_workers=1 \
    devices=-1
