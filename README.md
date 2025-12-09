RAVE_Data link: <a>https://drive.google.com/drive/folders/1SSCSCa5x5D-RTbMxqPnORDXHkt2yNpUU?usp=sharing</a>

To train and run the VAE model: 

python vae_spec.py --data_dir "Normalized Breaks\Normalized_Breaks\spectrograms"--epochs 60 --batch_size 32 --lr 2e-4 --beta 0.1 --kl_warmup_epochs 5 --z_dim 128 --n_mels 128 --target_frames 172 --specaug_time_mask 0 --specaug_freq_mask 0 --num_workers 2 --save_dir "checkpoints" --recon l1

To train and run the VQ-VAE model: 

python vqvae_spec.py --data_dir "Normalized Breaks\Normalized_Breaks\spectrograms" --epochs 60 --batch_size 32 --lr 2e-4 --z_dim 128 --n_mels 128 --target_frames 172 --specaug_time_mask 1 --specaug_freq_mask 1 --num_workers 2 --save_dir "checkpoints_vq" --recon l1 --num_codes 512 --vq_beta 0.25

The spectrograms folder contains .npy files of the drum breaks and are already preprocessed. 

After running the model, convert the .npy files to .wav files using audio_reconstruction.py. Change the path of the .wav folder output to your desired folder in lines 62 and 65.

To generate images of the spectrograms, run wav_to_spec.py. Change the paths accordingly, similar to the audio reconstruction script, this time in lines 46 and 50
