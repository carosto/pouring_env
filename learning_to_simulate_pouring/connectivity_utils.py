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

"""Tools to compute the connectivity of the graph."""
import functools

import jax
# from jax.experimental import host_callback as hcb
# from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
from sklearn import neighbors
from scipy.spatial.transform import Rotation
import open3d as o3d
import os


def get_mask_np(particle_types, id):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  return np.equal(particle_types, id)


class Connectivity_Handler:
    def __init__(self, collision_handler, max_nodes_edges_info):
        """Initialize with a collision handler instance."""
        self.collision_handler = collision_handler
        self.max_nodes_edges_info = max_nodes_edges_info

        # if max is 4 and current datast has 2 objects [0,1] --> pad with dummy keys and handle it in model_utils
        max_num_objects = max_nodes_edges_info[0]
        num_obj_dataset =  len(self.collision_handler.scene.keys())
        # self.keys_collision_objects = list(self.collision_handler.scene.keys()) + list(range(100, 100+max_num_objects-num_obj_dataset))
        self.keys_collision_objects = list(range(max_num_objects))
        
        # Create the custom JVP function with proper decoration
        self._connectivity_jax = functools.partial(
            jax.custom_jvp, nondiff_argnums=(3,))(self._connectivity_impl)
        # Define the JVP rule
        self._connectivity_jax.defjvp(self._connectivity_jvp_rule)

    def _connectivity_impl(self, positions, n_node, particle_types, radius):
        """Implementation of the connectivity computation."""
        callback_arg = (positions, particle_types, n_node, radius )
        _, max_n_liq_node_per_graph, max_edges_l_per_graph, max_edges_m_per_graph = self.max_nodes_edges_info
        batch_size = len(n_node) #if dataset split is not always divisible by set batch_size
        max_edges_ol = batch_size * max_n_liq_node_per_graph #collision with as many particles as there are per mesh
        max_edges_l = batch_size * max_edges_l_per_graph
        max_edges_m = batch_size * max_edges_m_per_graph

        out_shape = (jax.ShapeDtypeStruct((len(n_node) + 1,), jnp.int32),
                    jax.ShapeDtypeStruct((max_edges_l, 2), jnp.int32), # E_l
                    jax.ShapeDtypeStruct((max_edges_m, 2), jnp.int32), # E_om
                    jax.ShapeDtypeStruct((max_edges_m, 2), jnp.int32), # E_mo
                    {key: jax.ShapeDtypeStruct((max_edges_ol, 2), jnp.int32) for key in self.keys_collision_objects},# E_ol
                    {key: jax.ShapeDtypeStruct((max_edges_ol+1, 3), jnp.float32) for key in self.keys_collision_objects} # c_ik.. +1 for padded element to match liq_particles_shape
                    )

        n_edge, edges_l, edges_mo, edges_om, edges_ol_dict, closest_points_on_surface_dict = jax.pure_callback(self._cb_radius_query, out_shape, callback_arg,)

        return n_edge, edges_l, edges_mo, edges_om, edges_ol_dict, closest_points_on_surface_dict

    def _connectivity_jvp_rule(self, radius, primals, tangents):
        """JVP rule for the connectivity function."""
        del tangents
        primal_out = self._connectivity_jax(
            *primals, radius=radius)
        grad_out = (
              jnp.zeros_like(primal_out[0]),  
              jnp.zeros_like(primal_out[1]),  
              jnp.zeros_like(primal_out[2]), 
              jnp.zeros_like(primal_out[3]), 
              {key: jnp.zeros_like(value) for key, value in primal_out[4].items()}, # Matches the dict structure of Eol
              {key: jnp.zeros_like(value) for key, value in primal_out[5].items()}  # Matches the dict structure of collision handler
          )
        
        return primal_out, grad_out

    def __call__(self, positions, n_node, particle_types, radius):
        """Make the class callable with the same interface as before."""
        return self._connectivity_jax(positions, n_node, particle_types, radius)
    

    def _cb_radius_query(self, args):
      """Host callback function to compute connectivity."""
      _positions, particle_types, n_node, radius = args

      collision_handler = self.collision_handler


      liq_positions, mesh_node_positions, mesh_poses = _positions
      radius_liq = radius[0]
      num_mesh = len(collision_handler.scene.keys()) # get number of mesh objects

      max_num_objects, max_n_liq_node_per_graph, max_edges_l_per_graph, max_edges_m_per_graph = self.max_nodes_edges_info
      batch_size = len(n_node)
      max_edges_ol = batch_size * max_n_liq_node_per_graph #collision with as many particles as there are per mesh
      max_edges_l = batch_size * max_edges_l_per_graph
      max_edges_m = batch_size * max_edges_m_per_graph


      closest_points_on_surface_dict = {key: [] for key in collision_handler.scene.keys()} # collision for each mesh object
      edges_ol_dict = {key: [] for key in collision_handler.scene.keys()}

      edges_l, edges_om, edges_mo = [], [], []
      offset, offset_mesh_nodes, offset_mesh_pose = 0, 0, 0

      num_nodes_mesh = int(mesh_node_positions.shape[0]/(len(n_node)+1)) #to get the number of mesh nodes in each graph, (B*num_nodes_mesh, 3). len(node) = B

      for num_nodes in n_node:

        liquid_positions_graph = liq_positions[offset:offset+num_nodes]
        senders_liq, receivers_liq = compute_fixed_radius_connectivity_np(liquid_positions_graph, radius_liq, receiver_positions=liquid_positions_graph)

        # particle_type_graph = particle_types[offset_mesh_nodes:offset_mesh_nodes+num_nodes_mesh]

        offset_mesh_node = offset_mesh_nodes
        senders_Emo, receivers_Emo = [],[]
        senders_ol_dict, receivers_ol_dict = {},{}
        for mesh_id in collision_handler.scene.keys():
            #get node count for individual objects. 
            mesh_node_count = collision_handler.mesh_node_counts[mesh_id]
            senders_Emo_mesh = np.array(range(offset_mesh_node, offset_mesh_node+mesh_node_count))
            receivers_Emo_mesh = np.repeat(mesh_id+offset_mesh_pose, len(senders_Emo_mesh))
            senders_Emo.append(senders_Emo_mesh)
            receivers_Emo.append(receivers_Emo_mesh)


            colliding_particles_mask, closest_points_on_surface = collision_handler.compute_collision_connectivity(mesh_id, liquid_positions_graph, mesh_poses[mesh_id+offset_mesh_pose])
            receivers_Eol_mesh = np.where(colliding_particles_mask)[0] + offset
            senders_Eol_mesh = np.repeat(mesh_id+offset_mesh_pose, len(receivers_Eol_mesh))

            senders_ol_dict[mesh_id] = senders_Eol_mesh
            receivers_ol_dict[mesh_id] = receivers_Eol_mesh
            
            closest_points_on_surface_dict[mesh_id].append(closest_points_on_surface)

            offset_mesh_node += mesh_node_count

        senders_Emo = np.concatenate(senders_Emo, axis=0)
        receivers_Emo = np.concatenate(receivers_Emo, axis=0)

        indices_l = np.stack([senders_liq.astype(np.int32), receivers_liq.astype(np.int32)], axis=-1)
        indices_mo = np.stack([senders_Emo.astype(np.int32), receivers_Emo.astype(np.int32)], axis=-1)
        indices_om = np.stack([receivers_Emo.astype(np.int32), senders_Emo.astype(np.int32)], axis=-1) # same edges but the other way

        indices_ol_dict = {key:np.stack([senders_ol_dict[key].astype(np.int32), receivers_ol_dict[key].astype(np.int32)], axis=-1) for key in senders_ol_dict.keys()}


        edges_l.append(indices_l + offset)
        edges_mo.append(indices_mo)
        edges_om.append(indices_om)
        for key, edges_list in edges_ol_dict.items():
          edges_list.append(indices_ol_dict[key])
          
        
        offset += num_nodes
        offset_mesh_nodes += num_nodes_mesh
        offset_mesh_pose += num_mesh

      n_edge = [x.shape[0] for x in edges_l]
      total_edges = np.sum(n_edge)

      #for padding handling. last element in closest points shoulg be dummy. replace to zero
      for arr in closest_points_on_surface_dict.values():
        #here liquid particles size should be that of the dataset containing highest particles.. should pad to that value. only for jax, no implications for logic
        padding_size = batch_size*(max_n_liq_node_per_graph - arr[0].shape[0])+1 #plus 1 for padding in case arr.shape[0]==max_edges_ol
        arr.append(np.zeros((padding_size, 3,)))

      #concat them into flat array of same shape as liq for each of the mesh collisions
      closest_points_on_surface_dict = {key: np.concatenate(array_list, axis=0, dtype=np.float32) for key, array_list in closest_points_on_surface_dict.items()}
      
      if total_edges >= (max_edges_l): #max edges for l
        raise ValueError("%d edges found, max_edges for E_l: %d" % (total_edges, max_edges_l))

      #padding for E_l and E_ol. E_om, E_mo always constant
      padding_size = max_edges_l - np.sum([x.shape[0] for x in edges_l])
      padding = np.ones((padding_size, 2), dtype=np.int32) * offset
      edges_l = np.concatenate(edges_l + [padding], axis=0, dtype=np.int32)

      for key, edges_list in edges_ol_dict.items():
        padding_size = max_edges_ol - np.sum([x.shape[0] for x in edges_list])
        padding_sender = np.ones((padding_size,), dtype=np.int32) * offset_mesh_pose # sender is V_o
        padding_receiver = np.ones((padding_size,), dtype=np.int32) * offset # Receiver is V_l
        padding = np.stack((padding_sender, padding_receiver), axis=-1)
        edges_ol = np.concatenate(edges_list + [padding], axis=0, dtype=np.int32)
        edges_ol_dict[key] = edges_ol


      padding_size = max_edges_m - np.sum([x.shape[0] for x in edges_mo])
      padding_v_o = np.ones((padding_size,), dtype=np.int32) * offset_mesh_pose 
      padding_v_m = np.ones((padding_size,), dtype=np.int32) * offset_mesh_nodes

      padding = np.stack((padding_v_m, padding_v_o), axis=-1)
      edges_mo = np.concatenate(edges_mo + [padding], axis=0, dtype=np.int32)

      padding = np.stack((padding_v_o, padding_v_m), axis=-1)
      edges_om = np.concatenate(edges_om + [padding], axis=0, dtype=np.int32)

      padding_size = max_edges_l - np.sum([x.shape[0] for x in edges_l])
      n_edge = np.array(n_edge + [padding_size], dtype=np.int32) #Here last n_edge is for the dummy padding graph in the batch. 
      
      #padding for matching object meshes dict
      if num_mesh != max_num_objects:
        padding_obj_ids = list(range(num_mesh, max_num_objects))
        for pad_id in padding_obj_ids:
          closest_points_on_surface_dict[pad_id] = np.zeros((max_edges_ol+1, 3), dtype=np.float32)
          edges_ol_dict[pad_id] = np.zeros((max_edges_ol, 2), dtype=np.int32)

      return n_edge, edges_l, edges_mo, edges_om, edges_ol_dict, closest_points_on_surface_dict



def compute_fixed_radius_connectivity_np(
    positions, radius, receiver_positions=None, remove_self_edges=False):
  """Computes connectivity between positions and receiver_positions."""

  # if removing self edges, receiver positions must be none
  assert not (remove_self_edges and receiver_positions is not None)

  if receiver_positions is None:
    receiver_positions = positions

  # use kdtree for efficient calculation of pairs within radius distance
  kd_tree = neighbors.KDTree(positions)
  receivers_list = kd_tree.query_radius(receiver_positions, r=radius)
  num_nodes = len(receiver_positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)

  if remove_self_edges:
    # Remove self edges.
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

  return senders, receivers


def transform_points(points, ref_position, ref_orientation=None):
    # Apply the rotation if available
    if ref_orientation is not None:
        rotation = Rotation.from_euler('xyz', ref_orientation, degrees=True)
        points = rotation.apply(points) 
    # Apply the translation
    transformed_points = points + ref_position
    return transformed_points.astype(np.float32)

def transform_points_to_global(points, ref_position, ref_orientation=None):
    # Translate points to origin
    translated_points = points - ref_position
    
    # If rotation is provided, apply rotation
    if ref_orientation is not None:
        # Create rotation object from Euler angles
        R = Rotation.from_euler('xyz', ref_orientation, degrees=True)
        # Rotate points (use inverse rotation to change frame)
        transformed_points = R.inv().apply(translated_points)
    else:
        # If no rotation, just use the translated points
        transformed_points = translated_points
    
    return transformed_points.astype(np.float32)


class Collision_Manager:

    def __init__(self, mesh_name_list, collision_radius):
        self.collision_radius = collision_radius
        self.scene, self.mesh_node_counts = {}, {}
        # self.ray_direction = np.array([1, 0, 0], dtype=np.float32)
        for id, obj_data_list in enumerate(mesh_name_list):
          m_name, pt_type, mesh_node_count = obj_data_list
          BASE_DIR = os.path.dirname(os.path.abspath(__file__))
          meshpath = os.path.join(BASE_DIR, 'ObjectFiles', m_name)
          #meshpath = f'ObjectFiles/{m_name}'
          mesh = self.load_mesh(meshpath)
          #create a scene with the mesh for raycasting and collision detection
          _scene = o3d.t.geometry.RaycastingScene()
          mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
          _ = _scene.add_triangles(mesh_t)
          self.scene[id] = _scene
          self.mesh_node_counts[id] = mesh_node_count

    def load_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        scale_factor = 10 # isaac is 10x reality
        # scale_factor = 5 # isaac is 10x reality
        mesh.vertices = o3d.utility.Vector3dVector(vertices * scale_factor)
        return mesh

    def get_closest_surface_attr(self, id, query_points):
        # Compute closest points 
        closest_points_on_surface = self.scene[id].compute_closest_points(query_points)
        # Compute direction vectors
        directions = query_points - closest_points_on_surface['points'].numpy()
        # Get distance to the point
        unsigned_distance = self.scene[id].compute_distance(query_points)
        
        # signs = self.determine_point_inside(id, query_points)

        return closest_points_on_surface['points'].numpy(), directions, unsigned_distance.numpy()

    def compute_collision_connectivity(self, id, query_positions, mesh_pose):
        mesh_pose = np.array(mesh_pose, dtype=np.float32)
        query_positions = np.array(query_positions, dtype=np.float32)

        #if ref pose is zero, no need to transform
        positions_meshFrame = transform_points_to_global(query_positions, mesh_pose[:3], mesh_pose[3:]) if np.any(mesh_pose) else query_positions

        closest_points_on_surface, directions, unsigned_distance = self.get_closest_surface_attr(id, positions_meshFrame)

        #transform surface points back to global frame
        closest_points_on_surface = transform_points(closest_points_on_surface, mesh_pose[:3], mesh_pose[3:]) if np.any(mesh_pose) else closest_points_on_surface
        #interaction when particles are within collision radius
        colliding_particles_mask = unsigned_distance <= self.collision_radius
        # distance = signs*unsigned_distance
        return colliding_particles_mask, closest_points_on_surface

    def determine_point_inside(self, id, points):
        """
        Determines if points are inside the mesh
        """
        rays = o3d.core.Tensor(np.hstack((points, np.tile(self.ray_direction, (len(points), 1)))))
        intersections = self.scene[id].count_intersections(rays).numpy()
        signs = np.where(intersections % 2 == 1, -1, 1)
        return signs