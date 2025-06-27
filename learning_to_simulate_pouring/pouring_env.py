import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import tree
import trimesh
import numpy as np
from jax.scipy.spatial.transform import Rotation as Rotation_jax
import json, os, functools, collections, pickle, h5py




KINEMATIC_PARTICLE_ID = 3
KINEMATIC_PARTICLE_ID_1 = 4


def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  mask1 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID)
  mask2 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID_1) # separate ids for cup and jug..
  return jnp.logical_or(mask1, mask2)


def get_vel_diff(points):
    # of shape (timesteps, 3), paper Appendix B.2 
    diff = np.zeros((points.shape), dtype=np.float32)
    diff[1:] = points[1:] - points[:-1] #similar to original authors
    return diff

def get_acc_diff(points):
    # of shape (timesteps, 3), paper Appendix B.2 
    diff = np.zeros((points.shape), dtype=np.float32)
    diff[2:] = points[2:] - 2*points[1:-1] + points[:-2] #similar to original authors
    return diff

def gen_rotation(timesteps, dt_motion, max_rot):
    # Normalize the rotation axis
    rots = []
    vel = 0.
    rot = 0.
    for j in range(timesteps):
        rot = min(rot + vel*dt_motion, max_rot)
        vel = np.random.uniform(10, 60, 1)[0] # min and max angular velocity
        rots.append([-rot,0,0.])
    return np.array(rots, dtype=np.float32)

def gen_full_pose(translation_init, timesteps, dt_motion, max_rot):
    translation = np.repeat(translation_init[jnp.newaxis],timesteps, axis=0)
    rotation_u_xdir = gen_rotation(timesteps, dt_motion, max_rot)
    pose = np.concatenate((translation, rotation_u_xdir), axis=1)
    return pose

def create_step_contexts(pose_traj):
    step_context_vels = get_vel_diff(pose_traj)
    step_context_accs = get_acc_diff(pose_traj)
    step_context = jnp.concatenate([pose_traj, step_context_vels, step_context_accs], axis=-1, dtype=np.float32)
    return step_context






def get_velocity_from_acc(acc):
    # Use cumulative sum to euler integrate acceleration to velocity
    velocity = jnp.cumsum(acc, axis=0)
    # Prepend a zero entry for initial velocity
    velocity = jnp.vstack([jnp.zeros_like(velocity[0]), velocity[:-1]])
    return velocity

def get_position_from_vel(vel, pose_init):
    # Use cumulative sum to euler integrate velocity to position
    position = jnp.cumsum(vel, axis=0)
    # Prepend a zero entry for initial position
    position = jnp.vstack([jnp.zeros_like(position[0]), position[:-1]])
    #add init_pose
    position = position + pose_init
    return position.astype(np.float32)

def create_jug_control_acc(pose_init, input_acc):
    #input control is 1d(about x-axis) rotation for now.. extend TODO
    acc_full = jnp.zeros((input_acc.shape[0], 6))
    acc_full = acc_full.at[:, 3].set(input_acc)
    # acc_full = input_acc

    velocity = get_velocity_from_acc(acc_full)
    pose = get_position_from_vel(velocity, pose_init)
    return pose


def transform_mesh_to_local_coordinates(points, ref_position, ref_orientation):
    # Apply the rotation 
    rotation = Rotation_jax.from_euler('XYZ', ref_orientation, degrees=True)
    rotation_matrix = rotation.as_matrix() 
    points = points.dot(rotation_matrix.T)
    # Apply the translation
    transformed_points = points + ref_position
    return transformed_points


def get_particle_types(data_path):
  with h5py.File(f'{data_path}/simulation_1.h5', 'r') as hf:
  # with h5py.File(f'{data_path}/test/simulation_0.h5', 'r') as hf:
    graph_key = f'graph_{0}'
    graph_data = hf[graph_key]
    particle_types = jnp.array(graph_data['particle_types'][:])
  return particle_types


def build_initial_features(data_path, mesh_pt_type_list):
  with h5py.File(f'{data_path}/simulation_1.h5', 'r') as hf:
  # with h5py.File(f'{data_path}/test/simulation_0.h5', 'r') as hf:
    graph_key = f'graph_{0}'
    graph_data = hf[graph_key]
    pos_liq = graph_data['liq_position'][:]
    pos_liq = jnp.transpose(pos_liq, (1, 0, 2))[:, :-1]

    particle_types = jnp.array(graph_data['particle_types'])

    pos_mesh_nodes = graph_data['mesh_position'][:]
    pos_mesh_nodes = jnp.transpose(pos_mesh_nodes, (1, 0, 2))[:, :-1][jnp.newaxis]


    pose_mesh = graph_data['mesh_pose'][:, :-1][jnp.newaxis]  

    initial_features = {
    'liq_position': pos_liq,
    'mesh_position': pos_mesh_nodes,
    'mesh_pose': pose_mesh,
    'particle_type': particle_types,
    'particle_type_obj': jnp.array(mesh_pt_type_list),}

  return initial_features



def build_graph(nodes, max_n_node_per_graph, max_edges):
  n_node = nodes['liq_position'].shape[0]

  graph_tuple = jraph.GraphsTuple(
                        nodes=nodes,
                        edges={},
                        senders=jnp.array([]),
                        receivers=jnp.array([]),
                        globals=None,
                        n_node=jnp.array([n_node]),
                        n_edge=jnp.array([0])
            )
  padded_graph = jraph.pad_with_graphs(
                  graph_tuple,
                  n_node=max_n_node_per_graph + 1,
                  n_edge=max_edges,
                  n_graph=2
              )

  #for some reasons, padded graphs are np arrays
  data_jnp = {k: jnp.array(v) for k, v in padded_graph.nodes.items()}
  padded_graph = padded_graph._replace(nodes=data_jnp)

  return padded_graph


def forward(input_graph, jug_curr_pose, model, network_params, jug_vertices, config_dict):
    """Runs model and post-processing steps in jax, returns position sequence."""

    jug_vertices_count = config_dict['jug_vertices_count']
    @jax.jit
    def forward_fn(inputs):
      return model.apply(network_params['params'], network_params['state'], None, inputs)

    # only run for a single graph (plus one padding graph), update graph with
    assert len(input_graph.n_node) == 2, "Not a single padded graph."

    #update the mesh pose and positions
    prev_liq_position = input_graph.nodes["liq_position"]
    prev_mesh_position = input_graph.nodes["mesh_position"]
    prev_mesh_pose = input_graph.nodes["mesh_pose"]

    #update the jug vertices and pose based on external control
    jug_nodes_curr = transform_mesh_to_local_coordinates(jug_vertices, jug_curr_pose[:3], ref_orientation=jug_curr_pose[3:])
    curr_mesh_position = prev_mesh_position[0, :, -1]
    curr_mesh_position = curr_mesh_position.at[:jug_vertices_count].set(jug_nodes_curr)

    curr_mesh_pose = prev_mesh_pose[0, :, -1]
    curr_mesh_pose = curr_mesh_pose.at[0].set(jug_curr_pose)

    next_mesh_position = jnp.concatenate([prev_mesh_position[0,:, 1:], curr_mesh_position[:, None]], axis=1)
    next_mesh_position_padded = prev_mesh_position.at[0].set(next_mesh_position)

    next_mesh_pose = jnp.concatenate([prev_mesh_pose[0,:, 1:], curr_mesh_pose[:, None]], axis=1)
    next_mesh_pose_padded = prev_mesh_pose.at[0].set(next_mesh_pose)


    input_graph.nodes['mesh_position'] = next_mesh_position_padded
    input_graph.nodes['mesh_pose'] = next_mesh_pose_padded


    (output_graph,_), _  = forward_fn(input_graph)
    pred_pos = output_graph.nodes["p:position"]
    total_nodes = jnp.sum(input_graph.n_node[:-1])
    node_padding_mask = jnp.arange(prev_liq_position.shape[0]) < total_nodes

    # update history
    next_pos_seq = jnp.concatenate([prev_liq_position[:, 1:], pred_pos[:, None]], axis=1)
    next_pos_seq = jnp.where(node_padding_mask[:, None, None], next_pos_seq, prev_liq_position)


    # create new node features and update graph
    input_graph.nodes["liq_position"] = next_pos_seq
    return input_graph


def rollout(initial_graph, jug_pose, model, network_params, jug_vertices, config_dict):
  """Runs a jittable model rollout."""
  @jax.checkpoint
  def _step(graph, jug_pose):
    out_graph = forward(graph, jug_pose, model, network_params, jug_vertices, config_dict)
    out_data = dict(
        liq_position=out_graph.nodes["liq_position"], #TODO here it contains already predicted out for 1st ts, check if there are problems.
        mesh_position=out_graph.nodes["mesh_position"][0], #2nd is padded
        mesh_pose=out_graph.nodes["mesh_pose"][0],)
    return out_graph, out_data
  final_graph, trajectory = jax.lax.scan(_step, init=initial_graph,
                                         xs=jug_pose)
  return final_graph, trajectory

