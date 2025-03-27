#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=df_rnn_dmlab
#SBATCH --output=./logs/eval_df_rnn_dmlab/df_rnn_dmlab-chunk1-%j.out
#SBATCH --error=./logs/eval_df_rnn_dmlab/df_rnn_dmlab-chunk1-%j.err
#SBATCH --time=120:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
python  -m main +name=sample-rnn_df_dmlab-chunk1 algorithm=rnn_df_video experiment=exp_video dataset=video_dmlab experiment.tasks=[validation] \
        algorithm.diffusion.num_gru_layers=0 algorithm.chunk_size=1 \
        dataset.context_length=0 dataset.validation_multiplier=1 \
        experiment.validation.batch_size=2 experiment.validation.limit_batch=1 \
        algorithm.evaluation.frame_wise=False \
        "load=/vast/s224075134/temporal_diffusion/diffusion-forcing/checkpoints/video/dmlab-fs1/df-rnn/epoch\=0-step\=170000.ckpt"