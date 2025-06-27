
import jax.numpy as jnp
import jax
import pouring_env
import json, os, functools, collections, pickle, h5py, trimesh
import learned_simulator, model_utils
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import jax.profiler

import determined as det
from determined import core
import time
import argparse

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# initialise_tracking()

parser = argparse.ArgumentParser()
parser.add_argument('--out_path')
args = parser.parse_args()

Stats = collections.namedtuple('Stats', ['mean', 'std'])
def get_stats(metadata, acc_noise_std, vel_noise_std):
    cast = lambda v: jnp.array(v, dtype=jnp.float32)
    acceleration_stats = Stats(cast(metadata['acc_mean']), _combine_std(cast(metadata['acc_std']), acc_noise_std))
    velocity_stats = Stats(cast(metadata['vel_mean']), _combine_std(cast(metadata['vel_std']), vel_noise_std))

    normalization_stats = {'acceleration': acceleration_stats,
                            'velocity': velocity_stats}

    context_stats = Stats(
            cast(metadata['context_mean']), cast(metadata['context_std']))
    normalization_stats['context'] = context_stats

    return normalization_stats

def _combine_std(std_x, std_y):
  return jnp.sqrt(std_x**2 + std_y**2)


def load_model(path):
    with open(path, 'rb') as f:
        numpy_params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params) 

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
  

def get_particles_in_cup(particles, bb_cup):
    #particles are in frame of ref of cup. simple test to check if particles in the boundingbox of cup
    x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
    x_in_range = (particles[:, 0] >= x_min) & (particles[:, 0] <= x_max)
    y_in_range = (particles[:, 1] >= y_min) & (particles[:, 1] <= y_max)
    z_in_range = (particles[:, 2] >= z_min) & (particles[:, 2] <= z_max)

    inside_cup_mask = x_in_range & y_in_range & z_in_range
    return inside_cup_mask


def get_particles_on_floor(particles, z_floor):
    #check the z-coordinate for spilled particles
    delta = 0.01
    distance_to_floor = jnp.abs(particles[:, 2] - z_floor)
    spilled_particles_mask = (distance_to_floor <= delta)
    return spilled_particles_mask

# @jax.checkpoint
def cost_pouring(particles, action):
    particles_count = particles.shape[0]

    particles_cup_mask = get_particles_in_cup(particles, bb_cup)
    particles_spilled_mask = get_particles_on_floor(particles, z_floor=0.1)
    #inside jug.. flowing...
    other_particles_mask = ~jnp.logical_or(particles_cup_mask, particles_spilled_mask)

    count_particles_cup = jnp.sum(particles_cup_mask)/particles_count
    count_particles_spill = jnp.sum(particles_spilled_mask)/particles_count
    count_particles_other = jnp.sum(other_particles_mask)/particles_count

    cost = -count_particles_cup**2 + count_particles_spill**2 + count_particles_other**2 
    # cost = -count_particles_cup**2 + count_particles_spill**2 + count_particles_other**2 + 0.1*jnp.mean(action**2)
    return cost


@jax.jit
def opt_step(params, opt_state):
  grads, aux = jax.grad(run, has_aux=True)(params)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, aux

@jax.jit
def run(control_params):
    step_context = pouring_env.create_step_contexts_acc(jug_start_pose, control_params)

    # state_cur = state_full_traj[step]

    final_graph, out_trajectory_dict = pouring_env.rollout(input_graph, step_context, model, network_params)

    pred_positions = out_trajectory_dict['pred_pos']
    pred_positions_liq = pred_positions[-1, :liq_particles_count]

    loss = cost_pouring(pred_positions_liq, control_params)

    # auxiliaries to keep track of for plotting
    aux = {
        'design': control_params,
        'loss': loss,
        'input_positions': out_trajectory_dict['input_positions'],
        'step_context': step_context,
        'traj': pred_positions,
    }
    return loss, aux

def find_latest_params(model_dir):
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.startswith('output_step_') and f.endswith('.npz')]
    if not model_files:
        return None
    
    def get_step_from_fname(_file):
        return int(_file.split('_')[2])

    def get_iter_from_fname(_file):
        return int(_file.split('_')[-1].split(".")[0])


    steps = [get_step_from_fname(f) for f in model_files]
    latest_step = max(steps)
    model_files_step = [f for f in os.listdir(model_dir) if f.startswith(f'output_step_{latest_step}') and f.endswith('.npz')]
    iters = [get_iter_from_fname(f) for f in model_files_step]
    latest_iter = max(iters)

    latest_model = f'output_step_{latest_step}_iter_{latest_iter}.npz'
    
    return os.path.join(model_dir, latest_model), latest_step



bb_cup_dict = {
    'Martini': [-0.59958, -0.59958,  0.7, 0.59958,  0.59958,  1.78 ]
}

bb_cup = bb_cup_dict['Martini']
liq_particles_count = 13547

data_path = '/server/Pouring_mesh_mpc/'
model_path = '/server/models/model_checkpoint_globalstep_430000.pkl'
out_path = f'/server/models/{args.out_path}/'
noise_std = 6.7e-4
batch_size = 1

if not os.path.exists(out_path):
    os.makedirs(out_path)


#traj params
jug_position = np.array([0,-1.5,2.5])
jug_init_pose = np.array([0,-1.5,2.5, 0, 0, 0]) #situated above and side of the cup, such that rotation along x fills it
dt = 1/60
max_rot = 120



roll_steps = 200 
optim_iter_steps = 10
# define optimizer as adam with learning rate
LEARNING_RATE = 0.05
optimizer = optax.adam(learning_rate=LEARNING_RATE)
rng_key = jax.random.PRNGKey(42)

if __name__ == '__main__':
    metadata = _read_metadata(data_path)
    connectivity_radius = metadata["default_connectivity_radius"]
    boundaries = metadata["bounds"]
    max_n_node_per_graph = int(metadata["max_n_node"]) #  can be read from position as well.. ignore for now
    max_edges =int( metadata["max_n_edge"])
    # read the fixed obstacle edges of the meshes(already offsetted) by liq postion & obj pos
    fixed_obstacle_edges = np.load(f'{data_path}/fixed_obstacle_edges_offseted.npy') 

    normalization_stats = get_stats(metadata, acc_noise_std=noise_std, vel_noise_std=noise_std)

    graph_network_kwargs = dict(
        include_sent_messages_in_node_update=False,
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        use_layer_norm=True)
    model_kwargs = {'graph_network_kwargs': graph_network_kwargs}
    flatten_kwargs = {'apply_normalization': True}
    
    flatten_fn = functools.partial(model_utils.flatten_features, **flatten_kwargs)
    haiku_model = functools.partial(learned_simulator.LearnedSimulator, connectivity_radius=connectivity_radius, 
                                    boundaries=boundaries, 
                                    max_edges=int(max_edges*batch_size), 
                                    normalization_stats=normalization_stats, 
                                    flatten_features_fn=flatten_fn, **model_kwargs)

    model = hk.transform(lambda x: haiku_model()(x))
    network_params = load_model(model_path)

    initial_graph, particle_types = pouring_env.build_initial_graph(data_path, max_n_node_per_graph, max_edges, fixed_obstacle_edges)

    #check if params already exists from previous run..
    output = find_latest_params(out_path)
    if output is not None:
        latest_params_path, start_step = output
        print(f'Loading latest params: {latest_params_path}')
        data_dict = np.load(latest_params_path)
        acc_params = jnp.array(data_dict['params'])
        state_full_traj = jnp.array(data_dict['positions'])

        assert len(acc_params) == roll_steps
    else:
        print('init with random params')
        acc_params = jax.random.normal(rng_key,shape=(roll_steps,)) * 0.1 - 0.04
        start_step = 0
        state_full_traj = jnp.zeros((roll_steps, max_n_node_per_graph, 6, 3))
        state_full_traj = state_full_traj.at[0].set(initial_graph.nodes["position"][:-1])


    step_context_full_traj = pouring_env.create_step_contexts_acc(jug_init_pose, acc_params)



    pred_full_traj = jnp.zeros((roll_steps, max_n_node_per_graph, 3))
    
    opt_traj = []
    loss_verbose = 100.
    for step in range(start_step, roll_steps):

        print("Step: %d / %d" % (step, roll_steps))

        state_cur = state_full_traj[step]
        acc_curr_params = acc_params[step:]
        jug_start_pose = step_context_full_traj[step, :6]
        input_graph = pouring_env.build_graph(state_cur, particle_types, step_context_full_traj[step], max_n_node_per_graph, max_edges, fixed_obstacle_edges)

        opt_state = optimizer.init(acc_curr_params)

        for i in range(optim_iter_steps):
            t = time.time()
            acc_curr_params, opt_state, aux = opt_step(acc_curr_params, opt_state)

            # if i % 5 == 0:
            loss_step = aux['loss']
            if loss_step < loss_verbose:
               loss_verbose = loss_step
               save_data = {'params':np.array(acc_params), 'positions':np.array(state_full_traj)}
               filename_save = f'{out_path}/output_step_{step}_iter_{i}.npz'
               np.savez(filename_save, **save_data)

            print("  Iter %d / %d: Loss %.6f, time %.3f" % (i, optim_iter_steps, loss_step, time.time() - t))

        acc_params =  acc_params.at[step:].set(acc_curr_params)
        state_full_traj =  state_full_traj.at[step:].set(aux['input_positions'][:, :-1])
        step_context_full_traj =  step_context_full_traj.at[step:].set(aux['step_context'])
        pred_full_traj =  pred_full_traj.at[step:].set(aux['traj'][:, :-1])