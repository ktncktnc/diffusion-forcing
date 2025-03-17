#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=dc_rnn_dmlab_startat0_pred_x0
#SBATCH --output=./logs/dc_rnn_dmlab/dc_rnn_dmlab_startat0_pred_x0-%j.out
#SBATCH --error=./logs/dc_rnn_dmlab/dc_rnn_dmlab_startat0_pred_x0-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
# python -m main +name=rnn_dc_dmlab algorithm=rnn_dc_video experiment=exp_video dataset=video_dmlab algorithm.original_algo.model.num_gru_layers=0 dataset.context_length=4 dataset.n_frames=36 dataset.frame_skip=1 "algorithm.original_algo.ckpt_path=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/2025-03-10/07-25-58/checkpoints/epoch\=0-step\=200000.ckpt"

# python -m main +name=rnn_dc_dmlab algorithm=rnn_dc_video experiment=exp_video \
#             dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
#             algorithm.frame_stack=1 \
#             algorithm.original_algo.model.num_gru_layers=0 algorithm.original_algo.z_shape=[4,64,64] algorithm.original_algo.model.network_size=48 algorithm.original_algo.frame_stack=1 \
#             "algorithm.original_algo.ckpt_path=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-14/04-50-30/checkpoints/epoch\=0-step\=150000.ckpt" 

python -m main +name=no_random_start algorithm=rnn_dc_video experiment=exp_video \
        dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
        algorithm.diffusion.objective=pred_x0 algorithm.random_start=start \
        algorithm.original_algo.frame_stack=1 algorithm.original_algo.model.num_gru_layers=0 algorithm.original_algo.z_shape=[4,64,64] algorithm.original_algo.model.network_size=48 \
        "algorithm.original_algo.ckpt_path=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-14/04-50-30/checkpoints/epoch\=0-step\=150000.ckpt" \
        experiment.training.batch_size=64