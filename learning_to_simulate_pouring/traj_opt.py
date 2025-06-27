
import jax.numpy as jnp
import jax
import pouring_env
import json, os, functools, collections, pickle, h5py
from . import learned_simulator, model_utils
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import jax.profiler
import time
import argparse
import open3d as o3d
from datetime import datetime
from jax.scipy.optimize import minimize
from tensorboardX import SummaryWriter
from splinex import BSpline as Bspline_jax
from  jax.nn import sigmoid

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
# jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_disable_jit", True)

d_t = datetime.now()
date_str = d_t.strftime("%Y%m%d")
time_str = d_t.strftime("%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--out_path')
parser.add_argument('--out_name', default=f'run_{date_str}_{time_str}')
parser.add_argument('--data_path')
parser.add_argument('--model_path')
args = parser.parse_args()


def load_model(path):
    with open(path, 'rb') as f:
        numpy_params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params) 

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
  
def read_mesh_vertices(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    scale_factor = 10 # isaac is 10x reality
    return jnp.array(vertices * scale_factor)

def chamfer_loss(predicted: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Chamfer loss between two point clouds.

    Args:
        predicted (jnp.ndarray): Predicted point cloud of shape (N, 3).
        goal (jnp.ndarray): Goal point cloud of shape (M, 3).

    Returns:
        jnp.ndarray: Scalar Chamfer loss.
    """
    # Compute pairwise distances between predicted and goal points
    dists = jnp.linalg.norm(predicted[:, None, :] - goal[None, :, :], axis=-1)  # Shape: (N, M)

    # Compute min distances from predicted to goal (forward Chamfer)
    min_pred_to_goal = jnp.min(dists, axis=1)  # Shape: (N,)
    
    # Compute min distances from goal to predicted (backward Chamfer)
    min_goal_to_pred = jnp.min(dists, axis=0)  # Shape: (M,)

    # Chamfer loss is the mean of both directional distances
    loss = jnp.mean(min_pred_to_goal) + jnp.mean(min_goal_to_pred)
    
    return loss

def debug_grad_print(grads):
    # Compute and print gradient norm
    grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
    jax.debug.print("Gradient norm: {}", grad_norm)
    # Print the structure of grads (shapes of individual elements in the PyTree)
    def print_grad_shape(grad):
        jax.debug.print("Grad shape: {}", grad.shape)
        return grad  # Return unmodified
    # Apply the printing function to all elements in the PyTree
    grads = jax.tree_util.tree_map(print_grad_shape, grads)
    # Print the entire grads PyTree (note: may be verbose)
    jax.debug.print("Gradients PyTree: {}", grads)


def get_particles_in_cup_smooth(particles, bb_cup, alpha = 7.):
    #particles are in frame of ref of cup. simple test to check if particles in the boundingbox of cup
    x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
    px, py, pz = particles[:, 0], particles[:, 1], particles[:, 2]
    x_in_range = sigmoid(alpha*(px - x_min)) - sigmoid(alpha*(px - x_max))
    y_in_range = sigmoid(alpha*(py - y_min)) - sigmoid(alpha*(py - y_max))
    z_in_range = sigmoid(alpha*(pz - z_min)) - sigmoid(alpha*(pz - z_max))

    comb = jnp.mean(x_in_range * y_in_range * z_in_range) + 1e-6
    return comb

    # eps = 1e-6  # small epsilon to avoid log(0)
    # log_x = jnp.log(x_in_range + eps)
    # log_y = jnp.log(y_in_range + eps)
    # log_z = jnp.log(z_in_range + eps)

    # log_sum = (log_x + log_y + log_z) / 3.0  # divide by 3 to normalize
    # combined = jnp.exp(log_sum)

    # return jnp.mean(combined)




def get_particles_on_floor_smooth(particles, z_floor, alpha=7.):
    return jnp.mean(sigmoid(alpha*( z_floor - particles[:, 2]))) + 1e-6


def get_particles_in_cup_smooth_relu(particles, bb_cup):
    #particles are in frame of ref of cup. simple test to check if particles in the boundingbox of cup
    x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
    px, py, pz = particles[:, 0], particles[:, 1], particles[:, 2]

    act_func = jax.nn.tanh
    x_in_range = act_func((px - x_min)) - act_func((px - x_max))
    y_in_range = act_func((py - y_min)) - act_func((py - y_max))
    z_in_range = act_func((pz - z_min)) - act_func((pz - z_max))
    
    eps = 1e-6  # small epsilon to avoid log(0)
    log_x = jnp.log(x_in_range + eps)
    log_y = jnp.log(y_in_range + eps)
    log_z = jnp.log(z_in_range + eps)

    log_sum = (log_x + log_y + log_z) / 3.0  # divide by 3 to normalize
    combined = jnp.exp(log_sum)

    return jnp.mean(combined)

def get_particles_on_floor_smooth_relu(particles, z_floor):
    return jnp.mean(jax.nn.relu(( z_floor - particles[:, 2]))) + 1e-6



# TRY TO MAKE COST CONTINUOUS... SDF FROM CENTER OF CUP??? THINKKK! COST RIGHT NOW IS VERY VERY NON LINEAR
goal_state_in_cup = jnp.array(np.load('particles_cup_full.npy'))
def cost_pouring(particles, action, jug_pose):

    cost_cup = pt_cup_wgt*get_particles_in_cup_smooth(particles, bb_cup, alpha=alpha)
    cost_other = pt_spill_wgt*get_particles_on_floor_smooth(particles, z_floor=0.5, alpha=alpha)

    # cost_cup = pt_cup_wgt*get_particles_in_cup_smooth_relu(particles, bb_cup)
    # cost_other = pt_spill_wgt*get_particles_on_floor_smooth_relu(particles, z_floor=0.5)


    # cost = jax.lax.stop_gradient(cost_cup + cost_other) + action_cost*jnp.mean(action**2) + jug_resting_wgt*jnp.mean(jug_pose[3]**2)
    cost = cost_cup + cost_other + action_cost*jnp.mean(action**2) + jug_resting_wgt*jnp.mean(jug_pose[3]**2)
    # cost = cost_other + action_cost*jnp.mean(action**2) + jug_resting_wgt*jnp.mean(jug_pose[3]**2)

    # chamfer_l = chamfer_loss_(particles, goal_state_in_cup)
    # cost = chamfer_l

    return cost

def clip_grads(grads):
    # First handle the extreme values with an initial clip
    # Use a very large initial range to preserve some ratio between values
    pre_clipped = jnp.clip(grads, -1e15, 1e15)
    
    # Now normalize the gradients to control their scale
    grad_norm = jnp.linalg.norm(pre_clipped) + 1e-8  # avoid division by zero
    normalized = pre_clipped / grad_norm
    
    # Apply final clip to a reasonable range
    max_grad = 1.0
    final_clipped = jnp.clip(normalized, -max_grad, max_grad)
    
    # Apply time-dependent scaling (earlier control points need more scaling)
    # Use log space since your gradients vary by many orders of magnitude
    time_scale = jnp.logspace(-4, 0, len(grads))  # [0.0001, ..., 1.0]
    
    return final_clipped * time_scale


@jax.jit
def opt_step(params, opt_state):
    grads, aux = jax.grad(run, has_aux=True)(params)

    jax.debug.print("\nRaw gradient norm: {}", jnp.linalg.norm(grads))
    jax.debug.print("Raw gradients: {}", grads)

    grads = clip_grads(grads)
    # grads = jnp.clip(grads, -1, 1)

    # Debug prints after clipping
    jax.debug.print("Clipped gradient norm: {}", jnp.linalg.norm(grads))
    jax.debug.print("Clipped gradients: {}\n", grads)



    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, aux

@jax.jit
def run(control_points):
    control_params = get_spline_traj(control_points)
# def run(control_params):
#     def clamp_control(u):
#         return 1.*jax.nn.tanh(u)
#     control_params = clamp_control(control_params)

    jug_pose_traj = pouring_env.create_jug_control_acc(jug_start_pose, control_params)
    final_graph, out_trajectory_dict = pouring_env.rollout(input_graph, jug_pose_traj, model, network_params, jug_vertices, config_dict)
    #last node is padded.. shape is (timesteps, #liq+1, hist, 3) --> (#liq, 3) at last step
    pred_positions_liq = out_trajectory_dict['liq_position'][-1, :-1, -1] 
    jug_final_pose = out_trajectory_dict['mesh_pose'][-1, 0, -1] 

    loss = cost_pouring(pred_positions_liq, control_params, jug_final_pose)
    # auxiliaries to keep track of for plotting
    aux = {
        'design': control_params,
        'loss': loss,
        'liq_position': out_trajectory_dict['liq_position'],
        'mesh_position': out_trajectory_dict['mesh_position'],
        'mesh_pose': out_trajectory_dict['mesh_pose'],
    }
    return loss, aux


def get_spline_traj(control_points):
    def clamp_control(u):
        return 0.1*jax.nn.tanh(u)
    control_points_clamped = clamp_control(control_points)
    comb = jnp.column_stack((knot_locs, control_points_clamped))
    splinex_curve, _, _ = spline(comb)
    return splinex_curve[:,1]
    
    


@jax.jit
def opt_step_b(params):
    """using L-BFGS"""
    # Store the latest auxiliary data
    # latest_aux = []
    def objective_with_aux(params):
        loss, aux = run(params)
        # latest_aux.clear()
        # latest_aux.append(aux)
        return loss

    results = minimize(
        objective_with_aux,
        params,
        method='BFGS',
        options={
            'maxiter': 100, 
        }
    )
    
    # Update params
    new_params = results.x
    
    # Get the auxiliaries from the last evaluation
    loss, aux = run(new_params)
    return new_params, aux


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


INPUT_SEQUENCE_LENGTH = 6

bb_cup_dict = {
    'Martini': [-0.59958, -0.59958,  0.7, 0.59958,  0.59958,  1.78 ]
}

bb_cup = bb_cup_dict['Martini']
# liq_particles_count = 13547
liq_particles_count = 1047

data_path = args.data_path
# model_path = '/home/niteesh/Documents/source_codes/learning_to_simulate_pouring/models/model_checkpoint_globalstep_430000.pkl'
model_path =  args.model_path
out_base_path = args.out_path
out_path = f'{args.out_path}/mpc/{args.out_name}/'


metadata = _read_metadata(data_path)

jug_name = metadata['collision_mesh']['0']
jug_path = f'./ObjectFiles/{jug_name}'

noise_std = 6.7e-4
batch_size = 1
print(out_path)
if not os.path.exists(out_base_path):
    os.makedirs(out_base_path)
if not os.path.exists(f'{args.out_path}/mpc'):
    os.makedirs(f'{args.out_path}/mpc')
if not os.path.exists(out_path):
    os.makedirs(out_path)

jug_vertices = read_mesh_vertices(jug_path)
config_dict = {'jug_vertices_count':len(jug_vertices),}

#traj params
jug_position = np.array([0,-1.5,2.5])
jug_init_pose = np.array([0,-1.5,2.5, 0, 0, 0]) #situated above and side of the cup, such that rotation along x fills it
dt = 1/60
max_rot = 120



roll_steps = 350 
# roll_steps = np.load('jug_acc_hdata_mpc.npy').shape[0]
#fixed knot locations

knot_locs = jnp.array([ 0., 0.,0., 0., 100., 200., 250., 300., 350., 399., 399., 399., 399.])
# knot_locs = jnp.array([ 0., 0., 100., 200., 250., 350., 399., 399])


spline = Bspline_jax(n_in= knot_locs.shape[0], n_out=roll_steps, degree=3)


control_horizon = 2
optim_iter_steps = 1000

pt_cup_wgt = -1.
pt_spill_wgt = 0.
pt_other_wgt = 1.
action_cost = .1
jug_resting_wgt = 0
alpha = 1.
LEARNING_RATE = 1e-5
# LEARNING_RATE = 1e-3

tf_flag = True


#tensorboard
if tf_flag:
    log_dir = f"{out_path}/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

# define optimizer as adam with learning rate
optimizer = optax.adam(learning_rate=LEARNING_RATE)
# optimizer = optax.chain(
#     optax.clip_by_global_norm(1.0),  # Clip the global gradient norm to 1.0
#     optax.adam(learning_rate=LEARNING_RATE)
# )


rng_key = jax.random.PRNGKey(42)

if __name__ == '__main__':
    device = jax.devices()[0]
    print('\n\n JAX devices', device)
    assert device.device_kind != 'cpu'
    
    connectivity_radius = metadata["default_connectivity_radius"]
    max_n_liq_node_per_graph = int(metadata["max_n_liq_node"]) #  can be read from position as well.. ignore for now
    max_edges_l_per_graph =int( metadata["max_n_edge_l"])
    max_edges_m_per_graph =int( metadata["max_n_edge_m"])
    collision_mesh_dict = metadata["collision_mesh"]

    floor_pose = jnp.array(metadata['floor_pose'])[jnp.newaxis]

    graph_network_kwargs = dict(
        include_sent_messages_in_node_update=False,
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        node_types = ['v_l', 'v_m', 'v_o'],
        edge_types = ['e_l', 'e_mo', 'e_om', 'e_ol'],
        use_layer_norm=True)
    model_kwargs = {'graph_network_kwargs': graph_network_kwargs}
    flatten_kwargs = {'apply_normalization': True}
    
    flatten_fn = functools.partial(model_utils.flatten_features, **flatten_kwargs)
    haiku_model = functools.partial(learned_simulator.LearnedSimulator, connectivity_radius=connectivity_radius, 
                                    collision_mesh_dict=collision_mesh_dict, 
                                    max_nodes_edges_info=[batch_size,max_n_liq_node_per_graph, max_edges_l_per_graph, max_edges_m_per_graph], 
                                    flatten_features_fn=flatten_fn, **model_kwargs)


    model = hk.transform_with_state(lambda x: haiku_model()(x))
    network_params = load_model(model_path)['network']
    with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
        logging_file.write(f"Loading gnn model at path: {model_path}")

    print(f"Loading gnn model at path: {model_path}")
    print("PARAMS for this run:")
    print(f"\n learning_rate:{LEARNING_RATE}\n pt_cup_wgt:{pt_cup_wgt}\n pt_spill_wgt:{pt_spill_wgt}\n pt_other_wgt:{pt_other_wgt}\n action_cost:{action_cost}\n jug_resting_wgt:{jug_resting_wgt}")
    print(f"\n roll_steps:{roll_steps}\n control_horizon:{control_horizon}\n optim_iter_steps:{roll_steps}\n alpha:{alpha}")

    with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
        logging_file.write("\nPARAMS for this run:")
        logging_file.write(f"\n learning_rate:{LEARNING_RATE}\n pt_cup_wgt:{pt_cup_wgt}\n pt_spill_wgt:{pt_spill_wgt}\n pt_other_wgt:{pt_other_wgt}\n action_cost:{action_cost}\n jug_resting_wgt:{jug_resting_wgt}")
        logging_file.write(f"\n roll_steps:{roll_steps}\n control_horizon:{control_horizon}\n optim_iter_steps:{roll_steps}\n alpha:{alpha}")
    particle_types = pouring_env.get_particle_types(data_path)

    #check if params already exists from previous run..
    output = find_latest_params(out_path)
    # if output is not None:
    if False:
        latest_params_path, start_step = output
        print(f'Loading latest params: {latest_params_path}')
        data_dict = np.load(latest_params_path)
        acc_params = jnp.array(data_dict['params'])
        state_liq_pos_full_traj = jnp.array(data_dict['liq_position'])
        state_mesh_node_pos_full_traj = jnp.array(data_dict['mesh_position'])
        state_mesh_pose_full_traj = jnp.array(data_dict['mesh_pose'])
        assert len(acc_params) == roll_steps
    else:
        print('init with random params')
        initial_features = pouring_env.build_initial_features(data_path, floor_pose)
        # acc_params = jax.random.normal(rng_key,shape=(roll_steps,)) * 0.1 
        #load best params from previous run
        # rollout_dict = np.load("output/mpc/run_20250115_162748/init.npz")
        # acc_params = jnp.array(rollout_dict['params'][:roll_steps])

        # control_points_params = jnp.array([  -0.01561766,  0.00187812,  0.00646471,-0.01817663,0.02995969,  0.10898339, -0.06484577,  0. ])*10
        control_points_params = jnp.array([ -0.09118116, 0.06222875, -0.01561766,  0.00187812,  0.00646471,-0.01817663,0.02995969,  0.10898339, -0.06484577,  0. ,0.,0.,0.])*10
        # control_points_params = jax.random.normal(rng_key,shape=knot_locs.shape) * 0.1 
        

        start_step = 0
        state_liq_pos_full_traj = jnp.zeros((roll_steps, max_n_liq_node_per_graph, INPUT_SEQUENCE_LENGTH, 3))
        state_mesh_node_pos_full_traj = jnp.zeros((roll_steps, max_edges_m_per_graph, INPUT_SEQUENCE_LENGTH, 3))
        state_mesh_pose_full_traj = jnp.zeros((roll_steps, 3, INPUT_SEQUENCE_LENGTH, 6)) # 3 objects


        state_liq_pos_full_traj = state_liq_pos_full_traj.at[0].set(initial_features["liq_position"])
        state_mesh_node_pos_full_traj = state_mesh_node_pos_full_traj.at[0].set(initial_features["mesh_position"][0])
        state_mesh_pose_full_traj = state_mesh_pose_full_traj.at[0].set(initial_features["mesh_pose"][0])

    print(state_liq_pos_full_traj.shape, state_mesh_node_pos_full_traj.shape, state_mesh_pose_full_traj.shape, )

    opt_traj = []
    loss_verbose = 1e10
    for step in range(start_step, roll_steps, control_horizon):

        print("Step: %d / %d" % (step, roll_steps))
        with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
            logging_file.write("\nStep: %d / %d" % (step, roll_steps))
        state_liq_pos_cur = state_liq_pos_full_traj[step]
        state_mesh_node_pos_cur = state_mesh_node_pos_full_traj[step]
        state_mesh_pose_cur = state_mesh_pose_full_traj[step]
        # print(state_liq_pos_cur.shape, state_mesh_node_pos_cur.shape, state_mesh_pose_cur.shape, )
        input_features = {'liq_position': state_liq_pos_cur,
                        'mesh_position': state_mesh_node_pos_cur[jnp.newaxis],
                        'mesh_pose': state_mesh_pose_cur[jnp.newaxis],
                        'particle_type': particle_types,}

        # acc_curr_params = acc_params[step:]
        jug_start_pose = state_mesh_pose_cur[0, -1] # 0th is jug, -1 time position.
        print(jug_start_pose)
        input_graph = pouring_env.build_graph(input_features, max_n_liq_node_per_graph, max_edges_l_per_graph)

        # opt_state = optimizer.init(acc_curr_params)


        opt_state = optimizer.init(control_points_params)

        # acc_params =  acc_params.at[step:].set(acc_curr_params)
        # _, aux = run(acc_params)
        acc_params =  get_spline_traj(control_points_params)
        _, aux = run(control_points_params)
        state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:].set(aux['liq_position'][:, :-1])
        state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:].set(aux['mesh_position'])
        state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:].set(aux['mesh_pose'])
        #full traj with hist needs to be saved since mpc can be started from between.
        save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
                    'mesh_position':np.array(state_mesh_node_pos_full_traj),
                    'mesh_pose':np.array(state_mesh_pose_full_traj),
                    'control_points_param':np.array(control_points_params),
                    }
        filename_save = f'{out_path}/init.npz'
        np.savez(filename_save, **save_data)
        
        for i in range(optim_iter_steps):
            t = time.time()
            # acc_curr_params, opt_state, aux  = opt_step(acc_curr_params, opt_state)
            control_points_params, opt_state, aux  = opt_step(control_points_params, opt_state)
            # control_points_params,aux = opt_step_b(control_points_params)

            loss_step = aux['loss']
            if loss_step < loss_verbose:
                loss_verbose = loss_step
                # acc_params =  acc_params.at[step:].set(acc_curr_params)
                acc_params =  get_spline_traj(control_points_params)
                state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:].set(aux['liq_position'][:, :-1])
                state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:].set(aux['mesh_position'])
                state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:].set(aux['mesh_pose'])
                #full traj with hist needs to be saved since mpc can be started from between.
                save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
                            'mesh_position':np.array(state_mesh_node_pos_full_traj),
                            'mesh_pose':np.array(state_mesh_pose_full_traj),
                            'control_points_param':np.array(control_points_params),
                            }
                filename_save = f'{out_path}/output_step_{step}_iter_{i}.npz'
                np.savez(filename_save, **save_data)

            print("  Iter %d / %d: Loss %.6f, min so far %.6f, time %.3f" % (i, optim_iter_steps, loss_step, loss_verbose, time.time() - t))
            with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
                logging_file.write("\n  Iter %d / %d: Loss %.6f, min so far %.6f, time %.3f" % (i, optim_iter_steps, loss_step, loss_verbose, time.time() - t))
            if tf_flag:
                writer.add_scalar('loss', loss_step, i)

            if i%1 == 0:
                #save every 100 steps
                # acc_params =  acc_params.at[step:].set(acc_curr_params)
                acc_params =  get_spline_traj(control_points_params)
                state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:].set(aux['liq_position'][:, :-1])
                state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:].set(aux['mesh_position'])
                state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:].set(aux['mesh_pose'])
                save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
                            'mesh_position':np.array(state_mesh_node_pos_full_traj),
                            'mesh_pose':np.array(state_mesh_pose_full_traj),
                            'control_points_param':np.array(control_points_params),
                            }
                filename_save = f'{out_path}/output_step_{step}_iter_{i}.npz'
                np.savez(filename_save, **save_data)  

        jax.clear_caches()
        #to satisfy mpc
        # acc_params =  acc_params.at[step:].set(acc_curr_params)
        acc_params =  get_spline_traj(control_points_params)
        state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:].set(aux['liq_position'][:, :-1])
        state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:].set(aux['mesh_position'])
        state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:].set(aux['mesh_pose'])
        break

    save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
                            'mesh_position':np.array(state_mesh_node_pos_full_traj),
                            'mesh_pose':np.array(state_mesh_pose_full_traj),
                            'control_points_param':np.array(control_points_params),
                            }
    filename_save = f'{out_path}/final_output.npz'
    np.savez(filename_save, **save_data)

if tf_flag:
    writer.close()