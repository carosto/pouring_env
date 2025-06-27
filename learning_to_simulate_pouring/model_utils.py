# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for the LearnedSimulator model."""

import jax
import jax.numpy as jnp
import jraph
import tree
# import connectivity_utils
# from connectivity_utils import compute_fixed_radius_connectivity_jax
from . import normalizers
from . import jraph_base_models

def flatten_features(input_graph,
                     dataset_idx,
                     connectivity_radius,
                     _collision_handler_list,
                     is_padded_graph,
                     apply_normalization=False):


  #last is dummy from padding, Ignore!
  liq_position_sequence = input_graph.nodes["liq_position"]
  mesh_nodes_position_sequence = input_graph.nodes["mesh_position"]
  particle_types = input_graph.nodes["particle_type"]#for mesh nodes.. 
  particle_types_obj = input_graph.nodes["particle_type_obj"]#for mesh objs.. 
  mesh_pose_sequence = input_graph.nodes["mesh_pose"]
  epsilon = 1e-9

   #since dataset might not be completely divisible by bsize, account for the last data batch.. -1 for padded
  b_size = input_graph.nodes["mesh_position"].shape[0] - 1 

  max_edges_m_per_graph = mesh_nodes_position_sequence.shape[1]

  # for gnn handling reshape mesh pose batching. From (B, num_mesh, hist, 6) to (B*num_mesh, hist, 6) like particle batched (B*num_particles, hist, 3)
  _sh = mesh_pose_sequence.shape
  mesh_pose_sequence = mesh_pose_sequence.reshape(_sh[0]*_sh[1], _sh[2], _sh[3])
  # similarly for mesh nodes
  _sh = mesh_nodes_position_sequence.shape
  mesh_nodes_position_sequence = mesh_nodes_position_sequence.reshape(_sh[0]*_sh[1], _sh[2], _sh[3])


  most_recent_position_liq = liq_position_sequence[:, -1]
  most_recent_position_mesh_nodes = mesh_nodes_position_sequence[:, -1]
  most_recent_mesh_pose = mesh_pose_sequence[:, -1]
  radius_liq, radius_obj = connectivity_radius
  
  inputs_connectivity = [
      (most_recent_position_liq, most_recent_position_mesh_nodes, most_recent_mesh_pose),
      input_graph.n_node[:-1], # -1 because last graph is padded graph..
      particle_types,
      ## since liq-liq and liq-mesh handling is separated.
      connectivity_radius,
     ]
  #use jax.lax.switch for getting teh corresponding dataset_handler for jit and indexing
  n_edge, edges_l, edges_mo, edges_om, edges_ol_dict, closest_points_on_surface_dict = jax.lax.switch(dataset_idx, _collision_handler_list,*inputs_connectivity)
  

  senders_l, receivers_l = edges_l[:, 0], edges_l[:, 1]
  senders_mo, receivers_mo = edges_mo[:, 0], edges_mo[:, 1]
  senders_om, receivers_om = edges_om[:, 0], edges_om[:, 1]

  senders = {'e_l':senders_l, 'e_mo':senders_mo, 'e_om':senders_om  }
  receivers = {'e_l':receivers_l, 'e_mo':receivers_mo, 'e_om':receivers_om }

  #for e_ol concat from all the mesh objs
  num_mesh = int(particle_types_obj[:-1].shape[0]/b_size) #last element from jraph padding.. remove it first
  mesh_id_keys = list(range(num_mesh)) # to only iterate over actual objects in the dataset and not dummy dict entries for collision handling


  # Normalize the eleements of the graph.
  if apply_normalization:
    graph_elements_normalizer = normalizers.GraphElementsNormalizer(
        template_graph=input_graph,
        is_padded_graph=is_padded_graph)


  # Collect node features.
  node_features_v_l, node_features_v_m, node_features_v_o, = [],[],[]

  #v_l
  liq_velocity_sequence = time_diff(liq_position_sequence)  # Finite-difference.
  flat_liq_velocity_sequence = jnp.reshape(liq_velocity_sequence,
                                       [liq_velocity_sequence.shape[0], -1])
  # Normalized velocity sequence, flattening spatial axis.
  if apply_normalization:
    flat_liq_velocity_sequence = graph_elements_normalizer.normalize_node_array(
        "liq_velocity_sequence", flat_liq_velocity_sequence)
  node_features_v_l.append(flat_liq_velocity_sequence)
  node_features_v_l = jnp.concatenate(node_features_v_l, axis=-1)

  #v_m
  mesh_nodes_velocity_sequence = time_diff(mesh_nodes_position_sequence)  # Finite-difference.
  flat_mesh_nodes_velocity_sequence = jnp.reshape(mesh_nodes_velocity_sequence,
                                       [mesh_nodes_velocity_sequence.shape[0], -1])
  # Normalized velocity sequence, flattening spatial axis.
  if apply_normalization:
    #create padding based on shape..graph pad returns [True,True,False] for B=2.. from there, num_obj_nodes*B --> True, num_obj_nodes*1 --> False
    graph_pad = jraph.get_graph_padding_mask(input_graph)
    mask_padding = jnp.repeat(graph_pad, max_edges_m_per_graph)
    flat_mesh_nodes_velocity_sequence = graph_elements_normalizer.normalize_node_array(
        "mesh_nodes_velocity_sequence", flat_mesh_nodes_velocity_sequence, mask_padding)
  node_features_v_m.append(flat_mesh_nodes_velocity_sequence)

  # add dummy part for particle types..
  particle_types_m = particle_types[:-1] #last element from jraph padding.. remove it first
  particle_types_m = jnp.concatenate([particle_types_m, jnp.ones((max_edges_m_per_graph,), dtype=jnp.int32)*5], axis=0)
  node_features_v_m.append(jax.nn.one_hot(particle_types_m, 5))
  node_features_v_m = jnp.concatenate(node_features_v_m, axis=-1)

  #v_o
  mesh_pose_velocity_sequence = time_diff(mesh_pose_sequence)  # Finite-difference.
  flat_mesh_pose_velocity_sequence = jnp.reshape(mesh_pose_velocity_sequence,
                                       [mesh_pose_velocity_sequence.shape[0], -1])
  # Normalized velocity sequence, flattening spatial axis.
  if apply_normalization:
    #create padding based on shape.. num_mesh*B --> True, num_mesh*1 --> False
    mask_padding = jnp.repeat(graph_pad, num_mesh)
    flat_mesh_pose_velocity_sequence = graph_elements_normalizer.normalize_node_array(
        "mesh_pose_velocity_sequence", flat_mesh_pose_velocity_sequence, mask_padding)
  node_features_v_o.append(flat_mesh_pose_velocity_sequence)

  # Mesh type one hot encoding.. [0,1,2,0,1,2... Btimes,5,5, 5]
  particle_types_o = particle_types_obj[:-1] #last element from jraph padding.. remove it first
  #add dummy
  particle_types_o = jnp.concatenate([particle_types_o, jnp.ones((num_mesh,), dtype=jnp.int32)*5], axis=0)
  node_features_v_o.append(jax.nn.one_hot(particle_types_o, 5)) # PT TYPE SHOULD MATCH POSE SHAPE NOT NODE SHAPE
  node_features_v_o = jnp.concatenate(node_features_v_o, axis=-1)

  node_features_dict = {'v_l':node_features_v_l,'v_m':node_features_v_m, 'v_o':node_features_v_o, }


  # Collect edge features.
  edge_features_l, edge_features_mo, edge_features_om, edge_features_ol = [],[],[],[]

  # e_l
  edge_key = 'e_l'
  relative_world_position= (most_recent_position_liq[receivers[edge_key]] - most_recent_position_liq[senders[edge_key]])
  # Relative distances and norms. TODO.. for now add offset to all
  array = relative_world_position + epsilon
  relative_world_distance = jnp.linalg.norm(array, axis=-1, keepdims=True)
  if apply_normalization:
    # Scaled determined by connectivity radius.
    relative_world_position = relative_world_position / radius_liq
    relative_world_distance = relative_world_distance / radius_liq
  edge_features_l.append(relative_world_position)
  edge_features_l.append(relative_world_distance)
  edge_features_l = jnp.concatenate(edge_features_l, axis=-1)

  # e_mo
  edge_key = 'e_mo'
  #relative distance between receiver v_o and sender v_m
  relative_world_position= (most_recent_mesh_pose[:, :3][receivers[edge_key]] - most_recent_position_mesh_nodes[senders[edge_key]])
  # Relative distances and norms. TODO.. for now add offset to all
  array = relative_world_position + epsilon
  relative_world_distance = jnp.linalg.norm(array, axis=-1, keepdims=True)
  #norm i am not sure
  edge_features_mo.append(relative_world_position)
  edge_features_mo.append(relative_world_distance)
  edge_features_mo = jnp.concatenate(edge_features_mo, axis=-1)

  # e_om
  edge_key = 'e_om'
  #relative distance between receiver v_m and sender v_o
  relative_world_position= (most_recent_position_mesh_nodes[receivers[edge_key]] - most_recent_mesh_pose[:, :3][senders[edge_key]])
  # Relative distances and norms. TODO.. for now add offset to all
  array = relative_world_position + epsilon
  relative_world_distance = jnp.linalg.norm(array, axis=-1, keepdims=True)
  #norm i am not sure
  edge_features_om.append(relative_world_position)
  edge_features_om.append(relative_world_distance)
  edge_features_om = jnp.concatenate(edge_features_om, axis=-1)


  # e_ol
  edge_key = 'e_ol'
  #implementation from SDF - GNN (2024)

  senders_ol, receivers_ol = [],[]
  for mesh_id in mesh_id_keys:
    closest_pt_surface_mesh = closest_points_on_surface_dict[mesh_id]
    edges_ol = edges_ol_dict[mesh_id]
  
    if edges_ol.shape[0] > 0: #only consider when liq collides with objs
      _edge_features_ol = []
      _senders, _receivers = edges_ol[:, 0], edges_ol[:, 1]
      senders_ol.append(_senders)
      receivers_ol.append(_receivers)

      #c_ik - n_ik. senders & receivers same ids since closest_pt_surface_mesh index refers to corresponding liq particle
      relative_position_surface_to_liq= (closest_pt_surface_mesh[_receivers] -  most_recent_position_liq[_receivers])
      array = relative_position_surface_to_liq + epsilon
      relative_distance_surface_to_liq = jnp.linalg.norm(array, axis=-1, keepdims=True)
      #c_ik - o_j. 
      relative_position_surface_to_mesh= (closest_pt_surface_mesh[_receivers] - most_recent_mesh_pose[:, :3][_senders])
      array = relative_position_surface_to_mesh + epsilon
      relative_distance_surface_to_mesh = jnp.linalg.norm(array, axis=-1, keepdims=True)


      #eoj↔nk = [cj_ik − nik, cj_ik − oj, ||cj_ik − nik||, ||cj_ik − oj||]
      _edge_features_ol.append(relative_position_surface_to_liq)
      _edge_features_ol.append(relative_position_surface_to_mesh)
      _edge_features_ol.append(relative_distance_surface_to_liq)
      _edge_features_ol.append(relative_distance_surface_to_mesh)

      _edge_features_ol = jnp.concatenate(_edge_features_ol, axis=-1)
      edge_features_ol.append(_edge_features_ol)

  edge_features_ol = jnp.concatenate(edge_features_ol, axis=0)

  edge_features_dict = {'e_l':edge_features_l, 'e_mo':edge_features_mo, 'e_om':edge_features_om, 'e_ol':edge_features_ol  }

  #update senders and receivers for e_ol after removing masked parts 
  senders['e_ol'] = jnp.concatenate(senders_ol, axis=0)
  receivers ['e_ol']= jnp.concatenate(receivers_ol, axis=0)

  return jraph_base_models.MultiGraphsTuple(
            nodes=node_features_dict,
            edges=edge_features_dict,
            receivers=receivers,
            senders=senders, 
            globals=None,
            n_node=input_graph.n_node,
            n_edge=n_edge)


def time_diff(input_sequence):
  """Compute finnite time difference."""
  return input_sequence[:, 1:] - input_sequence[:, :-1]


def safe_edge_norm(array, graph, is_padded_graph, keepdims=False):
  """Compute vector norm, preventing nans in padding elements."""
  if is_padded_graph:
    padding_mask = jraph.get_edge_padding_mask(graph)
    epsilon = 1e-8
    perturb = jnp.logical_not(padding_mask) * epsilon
    array += jnp.expand_dims(perturb, range(1, len(array.shape)))
  return jnp.linalg.norm(array, axis=-1, keepdims=keepdims)


def _add_relative_distances(input_graph,
                            use_last_position_only=True):
  """Computes relative distances between particles and with walls."""

  # If these exist, there is probably something wrong.
  assert "relative_world_position" not in input_graph.edges
  assert "clipped_distance_to_walls" not in input_graph.nodes

  input_graph = tree.map_structure(lambda x: x, input_graph)  # Avoid mutating.
  particle_pos = input_graph.nodes["position"]

  if use_last_position_only:
    particle_pos = particle_pos[:, -1]

  input_graph.edges["relative_world_position"] = (
      particle_pos[input_graph.receivers] - particle_pos[input_graph.senders])

  return input_graph
