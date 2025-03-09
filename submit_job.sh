#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=batch-short
#SBATCH --job-name=minecraft
#SBATCH --output=./logs/load_minecraft.out
#SBATCH --time=72:00:00

# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# Navigate to directory and run script
cd /scratch/s224075134/temporal_diffusion/diffusion-forcing/
python -m main +name=minecraft_video algorithm=df_video experiment=exp_video dataset=video_minecraft experiment.training.batch_size=16 algorithm.frame_stack=8 algorithm.diffusion.network_size=64 algorithm.diffusion.beta_schedule=sigmoid algorithm.diffusion.cum_snr_decay=0.96 algorithm.z_shape=[32,128,128]