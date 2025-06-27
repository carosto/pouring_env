from typing import Dict, Any
import jax
import jax.numpy as jnp
import jraph
import numpy as np

@jax.jit
def chamfer_loss_func(pred, target):
    """
    Compute the Chamfer loss between predicted and goal liquid rollout.
    """
    # For each timestep, compute pairwise distances between predicted and goal points
    dists = jnp.linalg.norm(pred[:, :, None, :] - target[:, None, :, :], axis=-1)  # Shape: (T, N, N)

    # Min distances in both directions (symmetric)
    min_pred_to_goal = jnp.min(dists, axis=2)  # Shape: (T, N) 
    min_goal_to_pred = jnp.min(dists, axis=1)  # Shape: (T, N)

    # Average across objects and time steps
    loss = jnp.mean(min_pred_to_goal) + jnp.mean(min_goal_to_pred)
    
    return loss

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


def forward(input_graph, feature_dict, model, network_params):
    """Runs model and post-processing steps in jax, returns position sequence."""

    @jax.jit
    def forward_fn(inputs):
      return model.apply(network_params['params'], network_params['state'], None, inputs)

    # only run for a single graph (plus one padding graph), update graph with
    assert len(input_graph.n_node) == 2, "Not a single padded graph."

    #update the mesh pose and positions
    prev_liq_position = input_graph.nodes["liq_position"]
    prev_mesh_position = input_graph.nodes["mesh_position"]
    prev_mesh_pose = input_graph.nodes["mesh_pose"]

    curr_mesh_position = feature_dict['mesh_position']
    curr_mesh_pose = feature_dict['mesh_pose']

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

def rollout(initial_graph, features_dict, model, network_params):
    """Runs a jittable model rollout."""
    @jax.jit
    def _step(graph, step_context):
        out_graph = forward(graph, step_context, model, network_params)
        out_data = dict(
            input_positions=out_graph.nodes["liq_position"], 
            pred_pos=out_graph.nodes["liq_position"][:-1, -1],#last node is padded
            pred_mesh_pos=out_graph.nodes["mesh_position"][0, :, -1],
            ) 
        return out_graph, out_data
    _, trajectory = jax.lax.scan(_step, init=initial_graph,
                                         xs=features_dict)

    output_dict = {
            'predicted_rollout': np.array(trajectory['pred_pos']),
            'ground_truth_rollout': np.array(features_dict['liq_position']),
            'mesh_position': np.array(trajectory['pred_mesh_pos']),
            'mesh_pose': np.array(features_dict['mesh_pose']),
        }
    # Calculate MSE
    mse = jnp.mean((trajectory['pred_pos'] - features_dict['liq_position']) ** 2)
    chamfer_loss = chamfer_loss_func(trajectory['pred_pos'], features_dict['liq_position'])
    return output_dict, mse, chamfer_loss
