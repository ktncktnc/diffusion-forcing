#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=teacher_forcing_dmlab-pred_x0
#SBATCH --output=./logs/teacher_forcing_dmlab/teacher_forcing_dmlab-pred_x0-%j.out
#SBATCH --error=./logs/teacher_forcing_dmlab/teacher_forcing_dmlab-pred_x0-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

python -m main +name=df_rnn_dmlab-pred_v experiment=exp_video algorithm=rnn_df_video dataset=video_dmlab experiment.tasks=[validation] \
       dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
       algorithm.diffusion.num_gru_layers=0 algorithm.diffusion.objective=pred_v algorithm.is_teacher_forcing=True \
       "load=/vast/s224075134/temporal_diffusion/diffusion-forcing/checkpoints/video/dmlab-fs1/df-rnn/epoch\=0-step\=170000.ckpt"