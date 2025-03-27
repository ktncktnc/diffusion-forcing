#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=batch-short
#SBATCH --job-name=dc_rnn_dmlab-target_data-startat0-pred_x0-linear_increasing-causal-anneal_gt
#SBATCH --output=./logs/dc_rnn_dmlab/dc_rnn_dmlab-target_data-startat0-pred_x0-linear_increasing-causal-anneal_gt-%j.out
#SBATCH --error=./logs/dc_rnn_dmlab/dc_rnn_dmlab-target_data-startat0-pred_x0-linear_increasing-causal-anneal_gt-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

python -m main +name=dc_rnn_dmlab-target_data-startat0-pred_x0-linear_increasing-causal-anneal_gt algorithm=rnn_dc_video experiment=exp_video \
        dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
        algorithm.target_type=data algorithm.random_start=start algorithm.diffusion.objective=pred_x0 algorithm.finetune_org_model=False \
        algorithm.diffusion.noise_level_sampling=linear_increasing \
        algorithm.original_algo.frame_stack=1 algorithm.original_algo.model.num_gru_layers=1 algorithm.original_algo.z_shape=[4,64,64] algorithm.original_algo.model.network_size=24 \
        "algorithm.original_algo.ckpt_path=/vast/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-22/21-30-08/checkpoints/epoch\=0-step\=170000.ckpt" \
        experiment.training.batch_size=24 experiment.training.lr=1e-4