#!/bin/bash
echo "Experiment 1: VAE model"
CUDA_VISIBLE_DEVICES=5 python -u gen_model_impl.py -e 20 -l "VAELoss" -m "VAE" -x 5 -T 0.08 -v 0.08 > vae_train.log 2>&1 &
vae_pid=$!

echo "Experiment 2: GAN model"      # try full epoch 50
CUDA_VISIBLE_DEVICES=6 python -u gen_model_impl.py -e 20 -l "BCELoss" -m "GAN" -x 5 -T 0.08 -v 0.08 > gan_train.log 2>&1 &
gan_pid=$!

echo "Experiment 3: Diffusion model"
CUDA_VISIBLE_DEVICES=7 python -u gen_model_impl.py -e 20 -l "MSELoss" -m "DIFFUSION" -x 5 -T 0.08 -v 0.08 > diffusion_train.log 2>&1 &
diff_pid=$!

echo "Training jobs started:"
echo "VAE PID: $vae_pid"
echo "GAN PID: $gan_pid"
echo "Diffusion PID: $diff_pid"

wait $vae_pid
echo "VAE training finished."

wait $gan_pid
echo "GAN training finished."

wait $diff_pid
echo "Diffusion training finished."

echo "All experiments completed!"

echo "Starting inference for VAE"
python -u gen_model_inference.py -m "VAE" -p "vae_final_model_decoder.onnx" -t "vae"

echo "Starting inference for GAN"
python -u gen_model_inference.py -m "GAN" -p "gan_final_model.onnx" -t "gan"

echo "Starting inference for Diffusion"
python -u gen_model_inference.py -m "DIFFUSION" -p "diffusion_final_model.onnx" -t "diffusion"

echo "Inference completed!"