#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=batch-short
#SBATCH --job-name=teacher_forcing_dmlab-small-pred_v
#SBATCH --output=./logs/teacher_forcing_dmlab/teacher_forcing_dmlab-pred_v-%j.out
#SBATCH --error=./logs/teacher_forcing_dmlab/teacher_forcing_dmlab-pred_v-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

python -m main +name=teacher_forcing_dmlab-pred_v algorithm=rnn_df_video dataset=video_dmlab experiment=exp_video \
       experiment.training.lr=1e-4 experiment.training.batch_size=32 \
       dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 \
       algorithm.diffusion.objective=pred_v algorithm.diffusion.num_gru_layers=0 algorithm.diffusion.objective=pred_v algorithm.gt_cond_prob=0.0
