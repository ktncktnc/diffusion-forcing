#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=rnn_dmlab
#SBATCH --output=./logs/rnn_dmlab.out
#BATCH --error=./logs/error/rnn_dmlab.err
#SBATCH --time=72:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
# python -m main +name=rnn_dmlab_video algorithm=rnn_video experiment=exp_video dataset=video_dmlab algorithm.model.num_gru_layers=0 dataset.context_length=4 dataset.n_frames=36 dataset.frame_skip=1

python -m main +name=rnn_dmlab_video algorithm=rnn_video experiment=exp_video dataset=video_error_dmlab 