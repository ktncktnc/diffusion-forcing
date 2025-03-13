#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=dc_rnn_dmlab
#SBATCH --output=./logs/dc_rnn_dmlab.out
#BATCH --error=./logs/error/dc_rnn_dmlab.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
python -m main \
    +name=rnn_dc_minecraft \
    algorithm=rnn_dc_video \
    experiment=exp_video \
    dataset=video_minecraft dataset.context_length=8 dataset.frame_skip=1 dataset.n_frames=72 \
    experiment.training.batch_size=2 algorithm.original_algo.frame_stack=8 algorithm.original_algo.model.network_size=64 algorithm.original_algo.z_shape=[32,128,128] \
    "algorithm.original_algo.ckpt_path=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/2025-03-10/08-13-30/checkpoints/epoch\=0-step\=200000.ckpt"