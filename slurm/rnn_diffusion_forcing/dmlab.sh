#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=batch-short
#SBATCH --job-name=df_rnn_dmlab-pred_x0-gru1layer
#SBATCH --output=./logs/df_rnn_dmlab/df_rnn_dmlab-pred_x0-gru1layer-%j.out
#SBATCH --error=./logs/df_rnn_dmlab/df_rnn_dmlab-pred_x0-gru1layer-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
python -m main +name=df_rnn_dmlab-pred_x0-gru1layer algorithm=rnn_df_video experiment=exp_video dataset=video_dmlab algorithm.diffusion.num_gru_layers=1 dataset.context_length=4