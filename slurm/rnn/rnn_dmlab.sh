#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=rnn_dmlab-gru1layer
#SBATCH --output=./logs/rnn_dmlab/rnn_dmlab-gru1layer-%j.out
#SBATCH --error=./logs/rnn_dmlab/rnn_dmlab-gru1layer-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Main script
python -m main +name=rnn_dmlab_video-gru1layer \
        algorithm=rnn_video experiment=exp_video dataset=video_dmlab algorithm.model.num_gru_layers=1 algorithm.frame_stack=1 \
        algorithm.weight_decay=1e-4 algorithm.model.network_size=24 algorithm.z_shape=[4,64,64] \
        dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
        experiment.training.lr=1e-4 experiment.training.batch_size=16

# Resume script, train until done
# python -m main +name=rnn_dmlab_video algorithm=rnn_video experiment=exp_video dataset=video_dmlab algorithm.model.num_gru_layers=0 dataset.context_length=4 dataset.n_frames=36 dataset.frame_skip=1 resume=yne5na9o experiment.training.max_steps=1304297 wandb.mode=disabled