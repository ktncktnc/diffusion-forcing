#!/bin/bash
# Initialize conda first
eval "$(conda shell.bash hook)"
# Alternative method: source $(conda info --base)/etc/profile.d/conda.sh

# Now activate your environment
conda activate df


python -m main +name=sample_dmlab_rnn_dc algorithm=rnn_dc_video experiment=exp_video experiment.tasks=[validation] \
        dataset=video_dmlab dataset.context_length=4 dataset.n_frames=24 dataset.frame_skip=1 dataset.validation_multiplier=5 \
        algorithm.random_start=start algorithm.diffusion.objective=pred_v algorithm.target_type=data algorithm.diffusion.noise_level_sampling=linear_increasing algorithm.correction_size=8 \
        algorithm.original_algo.frame_stack=1 algorithm.original_algo.model.num_gru_layers=1 algorithm.original_algo.z_shape=[4,64,64] algorithm.original_algo.model.network_size=24 \
        experiment.validation.batch_size=16 experiment.validation.limit_batch=1 \
        "algorithm.original_algo.ckpt_path=/vast/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_video/2025-03-22/21-30-08/checkpoints/epoch\=0-step\=90000.ckpt" \
        "load=/vast/s224075134/temporal_diffusion/diffusion-forcing/outputs/video_dmlab-fs1/rnn_dc_video/2025-03-24/23-04-37/checkpoints/last.ckpt"