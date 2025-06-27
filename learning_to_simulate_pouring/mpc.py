
import jax.numpy as jnp
import jax
import pouring_env
import json, os, functools, collections, pickle, h5py
import learned_simulator, model_utils
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
# from splinex import BSpline as Bspline_jax
from  jax.nn import sigmoid
from jax.scipy.spatial.transform import Rotation

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


def transform_to_local_coordinates(points, pose):
    """
    Transform points from world frame to local frame.
    Args:
        points: Array of shape (N,3) world frame points
        pose: Array of shape (6,) [x,y,z, roll,pitch,yaw] in degrees
    Returns:
        local_points: Array of shape (N,3) local frame points
    """
    # Extract position and angles
    position = pose[:3]
    angles = pose[3:]
    
    # Get rotation matrix using inbuilt function
    R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    
    # Construct transformation matrix
    T = jnp.zeros((4, 4))
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(position)
    T = T.at[3, 3].set(1.0)
    
    # Get inverse transform
    T_inv = jnp.linalg.inv(T)
    
    # Transform points
    points_h = jnp.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    transformed_h = (T_inv @ points_h.T).T
    
    return transformed_h[:, :3]



def get_particles_ratio_in_cup(particles, bb_cup):
    #particles are in frame of ref of cup. simple test to check if particles in the boundingbox of cup

    particles_count = particles.shape[0]
    x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
    x_in_range = (particles[:, 0] >= x_min) & (particles[:, 0] <= x_max)
    y_in_range = (particles[:, 1] >= y_min) & (particles[:, 1] <= y_max)
    z_in_range = (particles[:, 2] >= z_min) & (particles[:, 2] <= z_max)

    inside_cup_mask = x_in_range & y_in_range & z_in_range

    particles_cup_ratio = jnp.sum(inside_cup_mask)/particles_count
    return particles_cup_ratio

def cost_cup_attractor(particles, mesh_final_pos):
    return jnp.mean((particles - mesh_final_pos)**2)
    # return jnp.mean((particles - mesh_final_pos)**2) + 1.*jnp.mean((particles - cup_bb_center)**2)

def cost_cup_attractor_fullHorizon(particles, mesh_final_pos):
    squared_diff = (particles - mesh_final_pos[:, None, :]) ** 2  # Shape: (T, 1047, 3)
    # Mean over particles (axis=1) and dimensions (axis=2)
    mse_per_timestep = jnp.mean(squared_diff, axis=(1, 2))  # Shape: (T,)
    return jnp.sum(mse_per_timestep)  


def cost_pouring(particles, mesh_final_pos, action, jug_pose):
    # cost = pt_cup_wgt_set*cost_cup_attractor(particles[-1], mesh_final_pos[-1]) + action_cost*jnp.mean(action**2) + jug_resting_wgt_set*jnp.mean(jug_pose[3]**2)
    cost = pt_cup_wgt_set*cost_cup_attractor_fullHorizon(particles, mesh_final_pos) + action_cost*jnp.mean(action**2) + jug_resting_wgt_set*jnp.mean(jug_pose[3]**2)

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

    # jax.debug.print("\nRaw gradient norm: {}", jnp.linalg.norm(grads))
    # jax.debug.print("Raw gradients: {}", grads)

    # grads = clip_grads(grads)

    # # Debug prints after clipping
    # jax.debug.print("Clipped gradient norm: {}", jnp.linalg.norm(grads))
    # jax.debug.print("Clipped gradients: {}\n", grads)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, aux

@jax.jit
def run(control_params):
    def clamp_control(u):
        return 1.*jax.nn.tanh(u)
    #jax.debug.print('control params: {}', control_params)
    control_params = clamp_control(control_params)
    print('inputs control acc')
    print(jug_start_pose.shape) # (6,)
    print(control_params.shape) # (400,)
    jug_pose_traj = pouring_env.create_jug_control_acc(jug_start_pose, control_params)
    final_graph, out_trajectory_dict = pouring_env.rollout(input_graph, jug_pose_traj, model, network_params, jug_vertices, config_dict)
    #last node is padded.. shape is (timesteps, #liq+1, hist, 3) --> (#liq, 3) at last step
    pred_positions_liq = out_trajectory_dict['liq_position'][:, :-1, -1] 
    jug_final_pose = out_trajectory_dict['mesh_pose'][-1, 0, -1] 
    mesh_final_pos = out_trajectory_dict['mesh_position'][:, 315, -1, :] # 315 is the index of the jug node(elongated) closest to the cup.. cehck untitled.ipynb in legion machine

    loss = cost_pouring(pred_positions_liq, mesh_final_pos, control_params, jug_final_pose)
    # auxiliaries to keep track of for plotting
    aux = {
        'design': control_params,
        'loss': loss,
        'liq_position': out_trajectory_dict['liq_position'],
        'mesh_position': out_trajectory_dict['mesh_position'],
        'mesh_pose': out_trajectory_dict['mesh_pose'],
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


INPUT_SEQUENCE_LENGTH = 6

bb_dict = {
    'Martini': [-0.59958, -0.59958,  0.7, 0.59958,  0.59958,  1.78 ],
    # 'Elongated': [-0.44801, -0.44801, -0.81116,  0.44801,  0.44801,  0.96884],
    'Elongated': [-0.44801, -0.44801, -0.81116,  0.44801,  0.44801,  0.65], #reducing z height to account for flowing liq
}

# bb_cup = bb_dict['Martini']


bb_jug = bb_dict['Elongated']


def get_center_from_ext(_min,_max):
    return _min + (_max - _min)/2

# #center of the BB of the cup forms the attractor for pouring
# x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
# cup_bb_center = jnp.array([get_center_from_ext(x_min, x_max),get_center_from_ext(y_min, y_max), get_center_from_ext(z_min, z_max) ])[np.newaxis]



# liq_particles_count = 13547
liq_particles_count = 1047

data_path = args.data_path
# model_path = '/home/niteesh/Documents/source_codes/learning_to_simulate_pouring/models/model_checkpoint_globalstep_430000.pkl'
model_path =  args.model_path
out_base_path = args.out_path
out_path = f'{args.out_path}/mpc/{args.out_name}/'


metadata = _read_metadata(data_path)

jug_name = metadata['collision_mesh'][0][0]
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



roll_steps = 400


# control_horizon = 2
# planning_horizon = 20
# optim_iter_steps = 50
# pt_cup_wgt = 30.
# jug_resting_wgt = 10.

control_horizon = 2
planning_horizon = 10
optim_iter_steps = 50
pt_cup_wgt = 20.
jug_resting_wgt = 5.

pt_spill_wgt = 0.
action_cost = .5


LEARNING_RATE = 1e-2
pour_percentage = 0.7


tf_flag = False

pt_cup_wgt_set = pt_cup_wgt
jug_resting_wgt_set = 0.

#tensorboard
if tf_flag:
    log_dir = f"{out_path}/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

# define optimizer as adam with learning rate
optimizer = optax.adam(learning_rate=LEARNING_RATE)

rng_key = jax.random.PRNGKey(np.random.randint(0,100))

if __name__ == '__main__':
    device = jax.devices()[0]
    print('\n\n JAX devices', device)
    assert device.device_kind != 'cpu'
    
    collision_mesh_info_list = metadata["collision_mesh"] 
    mesh_pt_type_list = [ z[1] for z in collision_mesh_info_list] #mesh pt type for handling in v_o

    connectivity_radius = metadata["default_connectivity_radius"]
    max_n_liq_node_per_graph = int(metadata["max_n_liq_node"]) #  can be read from position as well.. ignore for now
    max_edges_l_per_graph =int( metadata["max_n_edge_l"])
    max_edges_m_per_graph =int( metadata["max_n_edge_m"])

    max_nodes_edges_info = [len(collision_mesh_info_list),max_n_liq_node_per_graph, max_edges_l_per_graph, max_edges_m_per_graph]

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
                                    collision_mesh_info_lists=[collision_mesh_info_list,], 
                                    max_nodes_edges_info=max_nodes_edges_info, 
                                    flatten_features_fn=flatten_fn, **model_kwargs)


    model = hk.transform_with_state(lambda x: haiku_model()(x))
    network_params = load_model(model_path)['network']
    with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
        logging_file.write(f"Loading gnn model at path: {model_path}")

    print(f"Loading gnn model at path: {model_path}")
    print("PARAMS for this run:")
    print(f"\n learning_rate:{LEARNING_RATE}\n pt_cup_wgt:{pt_cup_wgt}\n pt_spill_wgt:{pt_spill_wgt}\n action_cost:{action_cost}\n jug_resting_wgt:{jug_resting_wgt}\n pour_percentage:{pour_percentage}")
    print(f"\n roll_steps:{roll_steps}\n control_horizon:{control_horizon}\n planning_horizon:{planning_horizon} \n optim_iter_steps:{optim_iter_steps}")

    with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
        logging_file.write("\nPARAMS for this run:")
        logging_file.write(f"\n learning_rate:{LEARNING_RATE}\n pt_cup_wgt:{pt_cup_wgt}\n pt_spill_wgt:{pt_spill_wgt}\n action_cost:{action_cost}\n jug_resting_wgt:{jug_resting_wgt}\n pour_percentage:{pour_percentage}")
        logging_file.write(f"\n roll_steps:{roll_steps}\n control_horizon:{control_horizon}\n planning_horizon:{planning_horizon} \n optim_iter_steps:{optim_iter_steps}")
    particle_types = pouring_env.get_particle_types(data_path)

    #check if params already exists from previous run..
    output = find_latest_params(out_path)
    # if output is not None:
    if False:
        # latest_params_path, start_step = output
        latest_params_path = '/home/niteesh/Documents/source_codes/learning_to_simulate_pouring/output/mpc/run_20250118_051327/output_step_165_iter_0.npz'
        start_step = 165
        print(f'Loading latest params: {latest_params_path}')
        data_dict = np.load(latest_params_path)
        acc_params = jnp.array(data_dict['params'])
        state_liq_pos_full_traj = jnp.array(data_dict['liq_position'])
        state_mesh_node_pos_full_traj = jnp.array(data_dict['mesh_position'])
        state_mesh_pose_full_traj = jnp.array(data_dict['mesh_pose'])
        assert len(acc_params) == roll_steps
    else:
        print('init with random params')
        initial_features = pouring_env.build_initial_features(data_path, mesh_pt_type_list)
        acc_params = jax.random.normal(rng_key,shape=(roll_steps,)) * 0.1 
        # #load best params from previous run
        # rollout_dict = np.load("output/mpc/init.npz")
        # acc_params = jnp.array(rollout_dict['params'][:roll_steps])

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
        print('state_mesh_pose_cur shape:', state_mesh_pose_cur.shape) # state_mesh_pose_cur shape: (3, 6, 6)
        # print(state_liq_pos_cur.shape, state_mesh_node_pos_cur.shape, state_mesh_pose_cur.shape, )
        input_features = {'liq_position': state_liq_pos_cur,
                        'mesh_position': state_mesh_node_pos_cur[jnp.newaxis],
                        'mesh_pose': state_mesh_pose_cur[jnp.newaxis],
                        'particle_type': particle_types,
                        'particle_type_obj': jnp.array(mesh_pt_type_list),
                        }
        print('input features: ', {key : value.shape for key, value in input_features.items()})
        #input features:  {'liq_position': (1047, 6, 3), 'mesh_position': (1, 1271, 6, 3), 'mesh_pose': (1, 3, 6, 6), 'particle_type': (1271,), 'particle_type_obj': (3,)}
        planning_end_step = min(step+planning_horizon, roll_steps)
        acc_curr_params = acc_params[step:planning_end_step ]
        jug_start_pose = state_mesh_pose_cur[0, -1] # 0th is jug, -1 time position.
        print(jug_start_pose)
        with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
            logging_file.write(f"\n{jug_start_pose}")
        input_graph = pouring_env.build_graph(input_features, max_n_liq_node_per_graph, max_edges_l_per_graph)

        opt_state = optimizer.init(acc_curr_params)

        if step == 0:
            _, aux = run(acc_params)
            print('aux shape: ', {key : value.shape for key, value in aux.items()})
            #aux shape:  {'design': (400,), 'liq_position': (400, 1048, 6, 3), 'loss': (), 'mesh_pose': (400, 3, 6, 6), 'mesh_position': (400, 1271, 6, 3)}
            print('test: ', aux['liq_position'][:, :-1].shape) # test:  (400, 1047, 6, 3)
            state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:].set(aux['liq_position'][:, :-1])
            state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:].set(aux['mesh_position'])
            state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:].set(aux['mesh_pose'])
            #full traj with hist needs to be saved since mpc can be started from between.
            save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj[:,:,-1]), 
                        'mesh_position':np.array(state_mesh_node_pos_full_traj[:,:,-1]),
                        'mesh_pose':np.array(state_mesh_pose_full_traj[:,:,-1]),
                        }
            filename_save = f'{out_path}/init.npz'
            np.savez(filename_save, **save_data)
        

        #two stage cost.. similar to Schenk&Fox pouring cost

        # particles_in_cup_ratio = get_particles_ratio_in_cup(state_liq_pos_cur[:,-1,:], bb_cup)
        #check particles leaving jug instead of inside cup
        jug_pose = state_mesh_pose_cur[0,-1,:] #of shape (3,6,6) or (#obj, hist, 6D)..
        liq_gt_jug_frame = transform_to_local_coordinates(state_liq_pos_cur[:,-1,:], jug_pose)
        particles_in_jug_ratio = get_particles_ratio_in_cup(liq_gt_jug_frame, bb_jug)
        particles_in_cup_ratio = (1. - particles_in_jug_ratio)* .7421 # 0.7421 is the ratio of particles in cup for FULL level..



        if particles_in_cup_ratio <=pour_percentage:
            # Stage 1.. pour until x percent inside cup. TODO should be jug based. flowing liq will fall in the cup
            pt_cup_wgt_set = pt_cup_wgt
            jug_resting_wgt_set = 0.
            print(f"STAGE 1 PHASE, {particles_in_cup_ratio}")
            with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
                logging_file.write(f"\nSTAGE 1 PHASE, {particles_in_cup_ratio}")
        else:
            # Stage 2. Cost to return cup to origin.
            pt_cup_wgt_set = 0.
            jug_resting_wgt_set = jug_resting_wgt   
            print(f"STAGE 2 PHASE, {particles_in_cup_ratio}")
            with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
                logging_file.write(f"\nSTAGE 2 PHASE, {particles_in_cup_ratio}")
        for i in range(optim_iter_steps):
            t = time.time()
            acc_curr_params, opt_state, aux  = opt_step(acc_curr_params, opt_state)

            loss_step = aux['loss']
            if loss_step < loss_verbose:
                loss_verbose = loss_step
            #     acc_params =  acc_params.at[step:planning_end_step].set(acc_curr_params)
            #     state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:planning_end_step].set(aux['liq_position'][:, :-1])
            #     state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:planning_end_step].set(aux['mesh_position'])
            #     state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:planning_end_step].set(aux['mesh_pose'])
            #     #full traj with hist needs to be saved since mpc can be started from between.
            #     save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
            #                 'mesh_position':np.array(state_mesh_node_pos_full_traj),
            #                 'mesh_pose':np.array(state_mesh_pose_full_traj),
            #                 }
            #     filename_save = f'{out_path}/output_step_{step}_iter_{i}.npz'
            #     np.savez(filename_save, **save_data)
            if i%5 == 0:
                print("  Iter %d / %d: Loss %.6f, min so far %.6f, time %.3f" % (i, optim_iter_steps, loss_step, loss_verbose, time.time() - t))
                with open(os.path.join(out_path, 'log.txt'), 'a') as logging_file:
                    logging_file.write("\n  Iter %d / %d: Loss %.6f, min so far %.6f, time %.3f" % (i, optim_iter_steps, loss_step, loss_verbose, time.time() - t))
            if tf_flag:
                writer.add_scalar('loss', loss_step, i)

            # if step%50 == 0:
            #     #save every 100 steps
            #     acc_params =  acc_params.at[step:planning_end_step].set(acc_curr_params)
            #     state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:planning_end_step].set(aux['liq_position'][:, :-1])
            #     state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:planning_end_step].set(aux['mesh_position'])
            #     state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:planning_end_step].set(aux['mesh_pose'])
            #     save_data = {
            #                 # 'params':np.array(acc_params), 
            #                  'liq_position':np.array(state_liq_pos_full_traj[:,:,-1]), 
            #                 'mesh_position':np.array(state_mesh_node_pos_full_traj[:,:,-1]),
            #                 'mesh_pose':np.array(state_mesh_pose_full_traj[:,:,-1]),
            #                 }
            #     filename_save = f'{out_path}/output_step_{step}_iter_{i}.npz'
            #     np.savez(filename_save, **save_data)  

        jax.clear_caches()
        #to satisfy mpc
        acc_params =  acc_params.at[step:planning_end_step].set(acc_curr_params)
        state_liq_pos_full_traj =  state_liq_pos_full_traj.at[step:planning_end_step].set(aux['liq_position'][:, :-1])
        state_mesh_node_pos_full_traj =  state_mesh_node_pos_full_traj.at[step:planning_end_step].set(aux['mesh_position'])
        state_mesh_pose_full_traj =  state_mesh_pose_full_traj.at[step:planning_end_step].set(aux['mesh_pose'])

    save_data = {'params':np.array(acc_params), 'liq_position':np.array(state_liq_pos_full_traj), 
                            'mesh_position':np.array(state_mesh_node_pos_full_traj),
                            'mesh_pose':np.array(state_mesh_pose_full_traj),
                            }
    filename_save = f'{out_path}/final_output.npz'
    np.savez(filename_save, **save_data)

if tf_flag:
    writer.close()