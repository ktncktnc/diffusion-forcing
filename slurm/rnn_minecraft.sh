#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=rnn_minecraft
#SBATCH --output=./logs/rnn_minecraft.out
#BATCH --error=./logs/error/rnn_minecraft.err
#SBATCH --time=72:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/

python -m main +name=rnn_minecraft_video algorithm=rnn_video experiment=exp_video dataset=video_minecraft experiment.training.batch_size=16 algorithm.frame_stack=8 dataset.context_length=8 dataset.frame_skip=1 dataset.n_frames=72 algorithm.model.network_size=64 algorithm.z_shape=[32,128,128]