#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=rnn_dmlab
#SBATCH --output=./logs/rnn_dmlab.out
#BATCH --error=./logs/error/rnn_dmlab.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/

# Main script
python -m main +name=rnn_dmlab_video \
        algorithm=rnn_video experiment=exp_video dataset=video_dmlab algorithm.model.num_gru_layers=0 algorithm.frame_stack=1 \
        algorithm.weight_decay=1e-4 algorithm.model.network_size=32 algorithm.z_shape=[2,128,128]\
        dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
        experiment.training.max_steps=1304297 experiment.training.lr=2e-4 experiment.training.batch_size=8

# Resume script, train until done
# python -m main +name=rnn_dmlab_video algorithm=rnn_video experiment=exp_video dataset=video_dmlab algorithm.model.num_gru_layers=0 dataset.context_length=4 dataset.n_frames=36 dataset.frame_skip=1 resume=yne5na9o experiment.training.max_steps=1304297 wandb.mode=disabled