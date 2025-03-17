#!/bin/bash
# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df

# python -m main +name=sample_dmlab_pretrained algorithm=rnn_video \
#         algorithm.model.num_gru_layers=0 algorithm.frame_stack=1 \
#         dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 dataset.validation_multiplier=20 \
#         experiment.tasks=[validation] \
#         "load=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-14/04-50-30/checkpoints/epoch\=0-step\=250000.ckpt" 

# python -m main +name=sample_dmlab_rnn_dc algorithm=rnn_dc_video experiment=exp_video \
#         dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 dataset.validation_multiplier=20 \
#         algorithm.original_algo.frame_stack=1 algorithm.random_start=false \
#         algorithm.original_algo.model.num_gru_layers=0 algorithm.original_algo.z_shape=[4,64,64] algorithm.original_algo.model.network_size=48 \
#         "algorithm.original_algo.ckpt_path=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-14/04-50-30/checkpoints/epoch\=0-step\=150000.ckpt" \
#         experiment.tasks=[validation] \
#         "load=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_dc_video/2025-03-16/18-15-02/checkpoints/epoch\=0-step\=20000.ckpt" 

python -m main +name=rnn_minecraft_video algorithm=rnn_video experiment=exp_video experiment=exp_video \
        algorithm.frame_stack=1 algorithm.model.network_size=48 algorithm.z_shape=[2,128,128] \
        dataset=video_minecraft dataset.n_frames=16 dataset.context_length=4 dataset.frame_skip=1 dataset.validation_multiplier=20 \
        experiment.tasks=[validation] experiment.training.batch_size=8 \
        "load=/weka/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_minecraft-fs1/rnn_video/2025-03-14/08-11-00/checkpoints/epoch\=0-step\=230000.ckpt"