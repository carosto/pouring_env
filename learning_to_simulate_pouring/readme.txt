python train.py     --data_path=datasets/Pouring_sdf_newTest,datasets/Pouring_sdf_MartiniBottle_2701_lessPt1     --model_path=models/multi_data_test --batch_size=20

python train.py --data_path=/shared_data/Pouring_sdf_fullPose_1002 --model_path=models/sdf_fullpose_lessPt_2412 --output_path=output/test --mode=eval_rollout

det experiment create test_run.yaml .  --project_id=129



#mpc

python mpc.py --model_path='/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl' \
            --out_path='/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/output/'  \
            --data_path='/shared_data/Pouring_mpc_1D_1902/'  \
            --out_name='baseline4'
#for 42

# Switch to the desired virtual environment
eval "$(conda shell.bash hook)"
conda activate myenv

python mpc.py --model_path='/server/models/model_checkpoint_globalstep_430000.pkl' \
            --out_path='/server/models/'  \
            --data_path='/server/Pouring_mesh_mpc/'  \
            --out_name='baseline'


# rl
python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --buffer_size 10000 --total_timesteps 100000 --learning_starts 15000 --capture_video
python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --capture_video
python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --total_timesteps 100000 --learning_starts 15000 --capture_video

python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --capture_video --noise_clip 0.05 --exploration_noise 0.05 --policy_noise 0.1
python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --total_timesteps 100000 --capture_video

python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --buffer_size 50000 --total_timesteps 500000 --capture_video
python td3_continuous_action_jax.py --env-id PouringEnv-v0 --seed 42 --save-model --buffer_size 50000 --total_timesteps 500000 --capture_video --noise_clip 0.05 --exploration_noise 0.05 --policy_noise 0.1