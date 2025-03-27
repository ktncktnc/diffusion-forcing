#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=df_dmlab
#SBATCH --output=./logs/df_dmlab.out
#SBATCH --error=./logs/error/df_dmlab.err
#SBATCH --time=72:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
# python -m main +name=dmlab_video algorithm=df_video experiment=exp_video dataset=video_dmlab algorithm.diffusion.num_gru_layers=0
python -m main +name=dmlab_video algorithm=df_video dataset=video_dmlab algorithm.weight_decay=1e-3 algorithm.diffusion.architecture.network_size=48 algorithm.diffusion.architecture.attn_dim_head=32 algorithm.diffusion.architecture.attn_resolutions=[8,16,32,64] algorithm.diffusion.beta_schedule=cosine
