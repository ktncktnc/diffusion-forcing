#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=dc_rnn_minecraft-target_data-startat0-pred_x0
#SBATCH --output=./logs/dc_rnn_minecraft/dc_rnn_minecraft-target_data-startat0-pred_x0-%j.out
#SBATCH --error=./logs/dc_rnn_minecraft/dc_rnn_minecraft-target_data-startat0-pred_x0-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
python -m main \
    +name=rnn_dc_minecraft algorithm=rnn_dc_video experiment=exp_video \
    dataset=video_minecraft dataset.context_length=4 dataset.frame_skip=1 dataset.n_frames=16 \
    algorithm.correction_size=8 algorithm.target_type=data algorithm.random_start=start algorithm.diffusion.objective=pred_x0 \
    algorithm.original_algo.frame_stack=1 algorithm.original_algo.model.network_size=48 algorithm.original_algo.z_shape=[2,128,128] \
    "algorithm.original_algo.ckpt_path=/vast/s224075134/temporal_diffusion/diffusion-forcing/checkpoints/video/minecraft-fs1/rnn/epoch\=0-step\=230000.ckpt" \
    experiment.training.batch_size=8 experiement.training.lr=3e-4