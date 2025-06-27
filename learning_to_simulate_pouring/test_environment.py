import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import gymnasium as gym
import learning_to_simulate_pouring.register_env  # this triggers the registration
import jax
import numpy as np
import jax.numpy as jnp

import imageio.v3 as iio


gnn_model_path = '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl'
data_path = '/shared_data/Pouring_mpc_1D_1902/'

out_path = f'/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/output/rl/test6'

"""gt_dict = np.load(f"/home/carola/learning_to_simulate_pouring/output/mpc/baseline/final_output.npz")
actions = gt_dict['params']"""

os.makedirs(out_path, exist_ok=True)

# Initialize the pouring environment
env_kwargs = {
    "gnn_model_path": gnn_model_path,
    "data_path": data_path,
    "render_mode": "rgb_array"
}
env = gym.make("PouringEnv-v0", **env_kwargs)

# Reset the environment to start
state = env.reset(seed=42)
print("Initial state:", state)

# Perform a few steps in the environment
saved_rewards = []
saved_states = []
frames = []
step = 0
while True:
    # Example action (replace with a valid action for your environment)
    """
    # fill level = 0.6
    if step <= 100:
        action = jnp.array([-0.01]) #env.action_space.sample(rng_key)
    elif step > 100 and step <= 200:
        action = jnp.array([0.01])
    elif step > 240 and step <= 340:
        action = jnp.array([0.01])
    elif step > 340:
        action = jnp.array([-0.01])
    else:
        action = jnp.array([0])"""
    
    # fill level = 0.23
    if step <= 100:
        action = jnp.array([-0.01]) #env.action_space.sample(rng_key)
    elif step > 100 and step <= 300:
        action = jnp.array([0.01])
    elif step > 300 and step <= 400:
        action = jnp.array([-0.01])
    else:
        action = jnp.array([0])

    """# fill level = 0.23
    if step <= 96:
        action = jnp.array([-0.01]) #env.action_space.sample(rng_key)
    elif step > 96 and step <= 288:
        action = jnp.array([0.01])
    elif step > 288 and step <= 378:
        action = jnp.array([-0.01])
    else:
        action = jnp.array([0])"""

    next_state, reward, terminated, truncated , info = env.step(action=action)
    print(f"Step {step + 1}:", " Action:", action, " Done:", terminated or truncated, " Reward:", reward)
    # Save the data for visualization
    """visualization_data = info['visualization_data']
    filename_save = f'{out_path}/final_output.npz'
    np.savez(filename_save, **visualization_data)"""

    saved_rewards.append(reward)
    filename_save = f'{out_path}/saved_rewards.npz'
    np.savez(filename_save, saved_rewards)

    """saved_states.append(next_state[-1])
    filename_save = f'{out_path}/saved_states.npz'
    np.savez(filename_save, saved_states)"""

    """saved_states.append(info['particle_positions'])
    filename_save = f'{out_path}/saved_particles_final_state.npz'
    np.savez(filename_save, saved_states[-1])
"""

    frame = env.render()
    frames.append(frame)


    state = next_state
    
    if terminated or truncated:
        print("Episode finished.")
        break
    step += 1
print('Final state: ', next_state)
print('Final fill level: ', info['current_fill_level'])
env.close()
iio.imwrite(uri=f"{out_path}/episode.mp4", image=frames, fps=env.metadata["render_fps"])

"""
close to ideal movement: (-> test2)
if step <= 100:
    action = jnp.array([-0.01]) #env.action_space.sample(rng_key)
elif step > 100 and step <= 200:
    action = jnp.array([0.01])
elif step > 240 and step <= 340:
    action = jnp.array([0.01])
elif step > 340:
    action = jnp.array([-0.01])
else:
    action = jnp.array([0])
    """