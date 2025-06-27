import os
os.environ["DISPLAY"] = ""  # Prevent attempts to use X11
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import json
import jax

import gymnasium as gym
from gymnasium import spaces

from scipy.spatial.transform import Rotation as R

from . import learned_simulator, model_utils, pouring_env
import haiku as hk
import functools
import pickle
import open3d as o3d
import numpy as np

import jax.numpy as jnp
from  jax.nn import sigmoid
from jax import lax


from vispy import app
app.use_app('egl')  # Or 'osmesa', depending on system setup

from vispy.scene import visuals, SceneCanvas

INPUT_SEQUENCE_LENGTH = 6
dt = 1

class PouringEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
        }
    def __init__(self, gnn_model_path, data_path, clear_cache_bool=False, render_mode=None):
        super().__init__()
        self.gnn_model_path = gnn_model_path
        self.data_path = data_path

        self.metadata_model = self._read_metadata(self.data_path)
        self.collision_mesh_info_list = self.metadata_model["collision_mesh"] 
        self.mesh_pt_type_list = [ z[1] for z in  self.collision_mesh_info_list] #mesh pt type for handling in v_o

        self.connectivity_radius = self.metadata_model["default_connectivity_radius"]
        self.max_n_liq_node_per_graph = int(self.metadata_model["max_n_liq_node"]) #  can be read from position as well.. ignore for now
        self.max_edges_l_per_graph =int( self.metadata_model["max_n_edge_l"])
        self.max_edges_m_per_graph =int( self.metadata_model["max_n_edge_m"])

        self.max_nodes_edges_info = [len( self.collision_mesh_info_list), self.max_n_liq_node_per_graph,  self.max_edges_l_per_graph,  self.max_edges_m_per_graph]

        self.jug_name = self.metadata_model['collision_mesh'][0][0]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.jug_path = os.path.join(BASE_DIR, 'ObjectFiles', self.jug_name)
        #self.jug_path = f'./ObjectFiles/{self.jug_name}'

        self.jug_vertices = self._read_mesh_vertices(self.jug_path)
        self.config_dict = {'jug_vertices_count':len(self.jug_vertices),}

        self.max_time = 400 # max time steps in the environemnt

        # set up stuff from pouring_env
        self.particle_types = pouring_env.get_particle_types(self.data_path)
        self.initial_features = pouring_env.build_initial_features(self.data_path, self.mesh_pt_type_list)

        # saved trajectories for the run (0 -> max timesteps)
        self.state_liq_pos_full_traj = None 
        self.state_mesh_node_pos_full_traj = None 
        self.state_mesh_pose_full_traj = None 

        self.current_step_id = 0

        self.pt_cup_wgt = 5.
        self.pt_spill_wgt = -2
        self.pt_other_wgt = 1.
        self.action_cost = -.01
        self.jug_resting_wgt = -.000001 
        self.jug_velocity_wgt = 0#-.5
        self.alpha = 1.

        self.bb_cup_dict = {
            'Martini': [-0.59958, -0.59958,  0.7, 0.59958,  0.59958,  1.78 ]
        }
        self.bb_cup = self.bb_cup_dict['Martini'] # TODO: change when using other cups
        self.z_floor = 0.1 # exactly 0 is too low

        self.bounds_object_movement = (-5, 5, -5, 5, -5, 5) # xmin, xmax, ymin, ymax, zmin, zmax (TODO: might need to be modified)
        self.max_abs_velocities = np.array([5,5,5,5,5,5]) # (TODO: might need to be modified)

        self.model_loaded = False

        self.clear_cache_bool = clear_cache_bool # if activated, the jax cache is cleared regularly to avoid memory issues (kept running into segmentation faults)

        # set up rendering (if required)
        self.render_mode = render_mode

        if self.render_mode == "rgb_array":
            self._canvas = SceneCanvas(size=(800, 600), keys=None, show=False, bgcolor="grey")
            self._view = self._canvas.central_widget.add_view()

            self._view.camera = "turntable"
            self._view.camera.distance = 8.857
            self._view.camera.elevation = 29
            self._view.camera.azimuth = -114
            self._view.camera.center = (0.6536471000272006, 0.2559890219701245, 0.9487044121129922)

            self._markers_liquid = visuals.Markers()
            self._markers_objects = visuals.Markers()
            self._markers_rims = visuals.Markers()
            self._view.add(self._markers_liquid)
            self._view.add(self._markers_objects)
            self._view.add(self._markers_rims)

            self._canvas.render() 

        self.use_different_initial_position = False


    def reset(self, seed, options=None):
        """
        Reset the environment to an initial state.
        Args:
            seed: seed for reproducibility.
        Returns:
            state: Initial state of the environment.
            obs: Initial observation.
        """
        # randomly one of the target fill levels for a trial
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

        self.key, subkey = jax.random.split(self.key)  # Create a new key for randomness
        possible_fill_levels = [0.5]#[0.1, 0.3, 0.5]
        index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(possible_fill_levels))
        self.target_fill_level = possible_fill_levels[index] 
        print('fill level for this trial: ', self.target_fill_level)

        # load the model when the environment is set up initially (the first reset)
        if not self.model_loaded:
            self.model, self.network_params = self._load_gnn_model(self.gnn_model_path, self.connectivity_radius, 
                                                                    self.collision_mesh_info_list, self.max_nodes_edges_info)
            self._fast_apply = jax.jit(lambda p, s, g: self.model.apply(p, s, g))
            self._fast_model_step = jax.jit(lambda p, g: self._apply_model_processing_step(p, g))

            self._target_particles = jnp.load(f"/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/output/rl/test5/saved_particles_final_state.npz")['arr_0']
            self.model_loaded = True
            print('Model loaded successfully.')

        # initialize features for model input (liquid position, object positions,....)
        self.state_liq_pos_full_traj = jnp.zeros((self.max_time, self.max_n_liq_node_per_graph, INPUT_SEQUENCE_LENGTH, 3))
        self.state_mesh_node_pos_full_traj = jnp.zeros((self.max_time, self.max_edges_m_per_graph, INPUT_SEQUENCE_LENGTH, 3))
        self.state_mesh_pose_full_traj = jnp.zeros((self.max_time, 3, INPUT_SEQUENCE_LENGTH, 6)) # 3 objects

        self.state_liq_pos_full_traj = self.state_liq_pos_full_traj.at[0].set(self.initial_features["liq_position"])
        self.state_mesh_node_pos_full_traj = self.state_mesh_node_pos_full_traj.at[0].set(self.initial_features["mesh_position"][0])
        self.state_mesh_pose_full_traj = self.state_mesh_pose_full_traj.at[0].set(self.initial_features["mesh_pose"][0])

        # TODO: set back to initial_features
        self.current_jug_velocity = jnp.zeros(6)

        jug_pose = self.state_mesh_pose_full_traj[0][0, -1]

        # TODO maybe move to separate function to avoid code duplication with step()
        state_liq_pos_cur = self.state_liq_pos_full_traj[0]
        state_mesh_node_pos_cur = self.state_mesh_node_pos_full_traj[0]
        state_mesh_pose_cur = self.state_mesh_pose_full_traj[0]

        # put the current state to the model and the encoded + processed liquid values
        input_features = {'liq_position': state_liq_pos_cur,
                        'mesh_position': state_mesh_node_pos_cur[jnp.newaxis],
                        'mesh_pose': state_mesh_pose_cur[jnp.newaxis],
                        'particle_type': self.particle_types,
                        'particle_type_obj': jnp.array(self.mesh_pt_type_list),
                        }
        input_graph = pouring_env.build_graph(input_features, self.max_n_liq_node_per_graph, self.max_edges_l_per_graph)
        
        # identify rim of cup and jug (IMPORTANT: HAS TO BE DONE BEFORE ANY CHANGES TO JUG POSE)
        self.idx_rim_jug, self.idx_rim_cup = self._identify_rim_jug_cup(self.state_mesh_node_pos_full_traj[0][:, -1], self.bb_cup)

        # IF REQUIRED: MOVE JUG TO DIFFERENT INITIAL POSITION
        if self.use_different_initial_position:
            input_graph, updated_data = self._move_to_different_initial_position(jug_pose, input_graph)
            self.state_liq_pos_full_traj =  self.state_liq_pos_full_traj.at[0].set(updated_data['liq_position'][ :-1])
            self.state_mesh_node_pos_full_traj =  self.state_mesh_node_pos_full_traj.at[0].set(updated_data['mesh_position'])
            self.state_mesh_pose_full_traj =  self.state_mesh_pose_full_traj.at[0].set(updated_data['mesh_pose'])
            jug_pose = self.state_mesh_pose_full_traj[0][0, -1]

        #(_, dict_latent_graphs), _  = self.model.apply(self.network_params['params'], self.network_params['state'], None, input_graph)
        (output_graph, dict_latent_graphs), _ = self._fast_apply(self.network_params['params'], self.network_params['state'], input_graph)
        predicted_liquid_pos = output_graph.nodes["p:position"][:-1]
        latent_liquid_representation = dict_latent_graphs['latent_graph_before_decoding'].nodes['v_l'] # latent representation of liquid particles after processing step of model
        
        obs = self._get_observation(latent_liquid_representation, jug_pose, self.current_jug_velocity, particles=predicted_liquid_pos, mesh_positions=self.state_mesh_node_pos_full_traj[0][:, -1])

        self.current_step_id = 1

        self._last_particles_cup = PouringEnv._get_particles_in_cup_smooth(predicted_liquid_pos, self.bb_cup, alpha=self.alpha)
        self._last_particles_spilled = PouringEnv._get_particles_on_floor_smooth(predicted_liquid_pos, z_floor=self.z_floor, alpha=self.alpha)
        self._last_fill_level = self._last_particles_cup / len(predicted_liquid_pos) # percentage of particles in cup TODO: use actual fill level instead of just percentage of particles in cup
        return obs, {'current_fill_level': self._last_fill_level}

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            state: Current state of the environment.
            action: Action to take.
        Returns:
            obs: Observation after taking the action.
            reward: Reward for the action.
            done: Whether the episode is done.
            info: Additional information.
        """
        state_liq_pos_cur = self.state_liq_pos_full_traj[self.current_step_id-1]
        state_mesh_node_pos_cur = self.state_mesh_node_pos_full_traj[self.current_step_id-1]
        state_mesh_pose_cur = self.state_mesh_pose_full_traj[self.current_step_id-1]

        # put the current state to the model and the encoded + processed liquid values
        input_features = {'liq_position': state_liq_pos_cur,
                        'mesh_position': state_mesh_node_pos_cur[jnp.newaxis],
                        'mesh_pose': state_mesh_pose_cur[jnp.newaxis],
                        'particle_type': self.particle_types,
                        'particle_type_obj': jnp.array(self.mesh_pt_type_list),
                        }
        #print('input features: ', {key : value.shape for key, value in input_features.items()})
        #input features:  {'liq_position': (1047, 6, 3), 'mesh_position': (1, 1271, 6, 3), 'mesh_pose': (1, 3, 6, 6), 'particle_type': (1271,), 'particle_type_obj': (3,)}
        input_graph = pouring_env.build_graph(input_features, self.max_n_liq_node_per_graph, self.max_edges_l_per_graph)
        
        prev_jug_pose = state_mesh_pose_cur[0, -1]
        #print('prev_jug_pose: ', prev_jug_pose) # (6,)
        prev_jug_velocity = self.current_jug_velocity
        #print('prev_jug_velocity: ', prev_jug_velocity) # (6,)

        action_full = jnp.zeros(6)
        action_full = action_full.at[3].set(action[0]) # currently only rotation around x-axis (TODO: extend)
        new_jug_pose, new_velocity = self.update_pose_single_step(prev_jug_pose, prev_jug_velocity, action_full)
        if abs(new_velocity[3]) >= self.max_abs_velocities[3]:
            action_full = action_full.at[3].set(0)
            new_jug_pose, new_velocity = self.update_pose_single_step(prev_jug_pose, prev_jug_velocity, action_full)
            exceeding_velocity_limit = True
        else:
            exceeding_velocity_limit = False

        self.current_jug_velocity = new_velocity 
        out_graph, latent_liq_representation, predicted_liquid_pos = self._fast_model_step(new_jug_pose, input_graph)
        out_data = dict(
            liq_position=out_graph.nodes["liq_position"], 
            mesh_position=out_graph.nodes["mesh_position"][0], 
            mesh_pose=out_graph.nodes["mesh_pose"][0],)
        # update current step in trajectory #TODO cheeck index??? für welchen step ist das?
        self.state_liq_pos_full_traj =  self.state_liq_pos_full_traj.at[self.current_step_id].set(out_data['liq_position'][ :-1])
        self.state_mesh_node_pos_full_traj =  self.state_mesh_node_pos_full_traj.at[self.current_step_id].set(out_data['mesh_position'])
        self.state_mesh_pose_full_traj =  self.state_mesh_pose_full_traj.at[self.current_step_id].set(out_data['mesh_pose'])

        obs = self._get_observation(latent_liq_representation, new_jug_pose, new_velocity, particles=predicted_liquid_pos, mesh_positions=self.state_mesh_node_pos_full_traj[self.current_step_id][:, -1])

        reward = self._compute_reward(particles=predicted_liquid_pos, jug_pose=out_data['mesh_pose'][0, -1], velocity=self.current_jug_velocity, mesh_positions=self.state_mesh_node_pos_full_traj[self.current_step_id][:, -1], exceeding_velocity_limit=exceeding_velocity_limit, action=action)
        #print(self.jug_resting_wgt * out_data['mesh_pose'][0, -1][3]**2)
        terminated, truncated = self._is_done()
        
        if self.clear_cache_bool and (self.current_step_id % 500 == 0): # currently not needed anymore (after applying jiit to multiple functions)?
            jax.clear_caches()
        
        self.current_step_id += 1

        return obs, reward, terminated, truncated, {'current_fill_level': self._last_fill_level}
    
    def render(self, mode="rgb_array"):
        """
        Render the environment.
        Args:
            mode: Mode of rendering (e.g., 'human', 'rgb_array').
        Returns:
            img: Rendered image.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame_to_array()

    def close(self):
        """Clean up resources used by the environment."""
        jax.clear_caches()

        if hasattr(self, "_canvas") and self._canvas is not None:
            self._canvas.close()
            self._canvas = None
            self._view = None
            self._markers_liquid = None
            self._markers_objects = None
            self._markers_rims = None

        # Clean up Haiku/JAX model
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "network_params"):
            del self.network_params

        super().close()

        print("Environment closed.")    

    def _move_to_different_initial_position(self, initial_jug_pose, initial_graph):
        action_sequence = jnp.concatenate([
            jnp.array([-0.01] * 85),
            jnp.array([0.01] * 85), 
            jnp.array([0] * 50) # calm liquid down
        ])
        def clamp_control(u):
            return 1.*jax.nn.tanh(u)

        jug_pose_traj = pouring_env.create_jug_control_acc(initial_jug_pose, clamp_control(action_sequence))

        """@jax.checkpoint
        def _step(graph, jug_pose):
            out_graph, _, _ = self._apply_model_processing_step(graph, jug_pose)
            out_data = dict(
                liq_position=out_graph.nodes["liq_position"], 
                mesh_position=out_graph.nodes["mesh_position"][0],
                mesh_pose=out_graph.nodes["mesh_pose"][0],)
            print('test')
            return out_graph, out_data
        final_graph, trajectory = jax.lax.scan(_step, init=initial_graph,
                                                xs=jug_pose_traj)
        return final_graph[-1], trajectory[-1] # only the last step is relevant"""
        graph = initial_graph

        for t in range(len(jug_pose_traj)):
            jug_pose = jug_pose_traj[t]
            
            graph, _, _ = self._fast_model_step(jug_pose, graph)

            out_data = dict(
                liq_position=graph.nodes["liq_position"],
                mesh_position=graph.nodes["mesh_position"][0],
                mesh_pose=graph.nodes["mesh_pose"][0],
            )

        return graph, out_data

    def _identify_rim_jug_cup(self, mesh_positions, bbox_cup):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox_cup
        cup_mask = ( # mask for positions in bounding box for cup
            (mesh_positions[:, 0] >= xmin) & (mesh_positions[:, 0] <= xmax) &
            (mesh_positions[:, 1] >= ymin) & (mesh_positions[:, 1] <= ymax) &
            (mesh_positions[:, 2] >= zmin) & (mesh_positions[:, 2] <= zmax)
        ) 
        jug_mask = ~cup_mask

        jug_positions = mesh_positions[jug_mask] # this technically also includes the stem of the martini glass
        cup_positions = mesh_positions[cup_mask] # this includes only the cup potion of the martini glass, not the stem

        # jug rim is jug point with highest z
        jug_z = jug_positions[:, 2]
        max_z_v = jnp.max(jug_z)
        jug_rim_idx = jnp.where(jug_z == max_z_v)[0][12]#jnp.argmax(jug_z) # HARDCODED BECAUSE FOR SOME REASON MULTIPLE POSITIONS HAD THE SAME HIGH Z VALUE (TODO: needs to be modified if jug changes)
        jug_rim_point = jug_positions[jug_rim_idx]

        # find cup point closest to jug rim point
        diffs = cup_positions - jug_rim_point  # shape (num_cup, 3)
        distances = jnp.linalg.norm(diffs, axis=1)
        cup_rim_idx = jnp.argmin(distances)

        # convert to indices relative to original positions array
        jug_indices = jnp.nonzero(jug_mask, size=jug_mask.size)[0]
        cup_indices = jnp.nonzero(cup_mask, size=cup_mask.size)[0]

        return jug_indices[jug_rim_idx], cup_indices[cup_rim_idx]

    @staticmethod
    @jax.jit    
    def _calc_distance_jug_cup(mesh_positions, idx_rim_jug, idx_rim_cup): 
        return jnp.linalg.norm(mesh_positions[idx_rim_jug] - mesh_positions[idx_rim_cup])
        

    def _render_frame_to_array(self):
        liquid_positions = np.array(self.state_liq_pos_full_traj[self.current_step_id-1])[:, -1]
        mesh_position = np.array(self.state_mesh_node_pos_full_traj[self.current_step_id-1])[:, -1]
        rims_position = mesh_position[[self.idx_rim_jug, self.idx_rim_cup]]
        
        liquid_positions = liquid_positions.astype(np.float32)
        mesh_position = mesh_position.astype(np.float32)
        rims_position = rims_position.astype(np.float32)

        self._markers_liquid.set_data(liquid_positions, face_color=[0.1, 0.1, 0.1, 0.3], size=4)
        self._markers_objects.set_data(mesh_position, face_color=[1, 0, 0, 1], size=3)
        self._markers_rims.set_data(rims_position, face_color=[1, 0, 0, 1], size=10)

        # Render to image
        img = self._canvas.render()
        return img[:, :, :3]  # Drop alpha channel to go from (H, W, 4) RGBA to (H, W, 3) RGB

    def _get_observation(self, liquid_representation, jug_pose, jug_velocity, particles, mesh_positions):
        """
        Get the observation for the current step.
        Args:
            liquid_representation: Representation of the liquid particles.
            jug_pose: Current pose of the jug.
            jug_velocity: Current velocity of the jug.
        Returns:
            obs: Observation.
        """
        # NORMALIZATION
        # 1. normalize jug position values (to [-1,1])
        jug_position = jug_pose[:3]
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds_object_movement
        x_normalized = (2 * (jug_position[0] - xmin) / (xmax - xmin)) - 1
        y_normalized = (2 * (jug_position[1] - ymin) / (ymax - ymin)) - 1
        z_normalized = (2 * (jug_position[2] - zmin) / (zmax - zmin)) - 1
        jug_position_normalized = jnp.array([x_normalized, y_normalized, z_normalized])
        # 2. normalize jug rotation values (TODO: i hope this is correct) (currently no explicit normalization applied, rotation matrix values are already [-1,1])
        # TODO: maybe also check out 6D representation (from Zhou et al., CVPR 2019)
        rvec = jug_pose[3:]               # rotation vector
        rot = R.from_rotvec(rvec)            # convert to rotation
        rotation_matrix = rot.as_matrix().flatten()  # shape (9,)

        # TODO think about normalization for liquid representation + velocity
        #jug_velocity = abs(jug_velocity)
        min_abs_velocity = -1 * self.max_abs_velocities
        #velocity_normalized = (jug_velocity - min_abs_velocity) / (self.max_abs_velocities - min_abs_velocity)
        velocity_normalized = (2 * (jug_velocity - min_abs_velocity) / (self.max_abs_velocities - min_abs_velocity)) - 1 

        # 5. normalize timestep (to [0,1])
        ts_normalized = self.current_step_id / self.max_time

        # get distance between cup and jug + normalize
        distance = PouringEnv._calc_distance_jug_cup(mesh_positions, self.idx_rim_jug, self.idx_rim_cup)
        # TODO: these are currently only for the x rotation case!
        max_distance = 2.2347841
        min_distance = 0.26558006
        distance_normalized = (distance - min_distance) / (max_distance - min_distance)


        p_count_cup, p_count_jug = self._get_particle_counts(particles, self.bb_cup, self.z_floor)
        p_count_cup_normalized = p_count_cup / len(particles)
        p_count_jug_normalized = p_count_jug / len(particles)


        """observation = {   
                'jug_position' : jug_position_normalized,
                'jug_rotation' : rotation_matrix,
                'jug_velocity' : jug_velocity,  # jug velocity
                'timestep' : ts_normalized,  # timestep
                'liquid_data' : liquid_representation,   # latent particle representation
            } """    
        
        
        # full state
        flat_obs = jnp.concatenate([
            jnp.array(jug_position_normalized, dtype=jnp.float32),
            jnp.array(rotation_matrix, dtype=jnp.float32),
            jnp.array(velocity_normalized, dtype=jnp.float32),
            #jnp.array([ts_normalized], dtype=jnp.float32), 
            #jnp.array([self.target_fill_level], dtype=jnp.float32), # alternatively: represent target fill level as one-hot vector 
            # alternatively: use distance between target fill level and current fill level
            jnp.array(liquid_representation.flatten(), dtype=jnp.float32),
        ])
        
        """# reduced state
        flat_obs = jnp.concatenate([
            jnp.array(jug_position_normalized, dtype=jnp.float32),
            jnp.array(rotation_matrix, dtype=jnp.float32),
            jnp.array([p_count_cup_normalized], dtype=jnp.float32),
            jnp.array([p_count_jug_normalized], dtype=jnp.float32),
            jnp.array([distance_normalized], dtype=jnp.float32), 
            jnp.array(velocity_normalized, dtype=jnp.float32), 
        ])"""
        return flat_obs

    @staticmethod
    @jax.jit
    def _get_particle_counts(particles, bb_cup, z_floor):
        x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
        px, py, pz = particles[:, 0], particles[:, 1], particles[:, 2]

        in_x = (px >= x_min) & (px <= x_max)
        in_y = (py >= y_min) & (py <= y_max)
        in_z = (pz >= z_min) & (pz <= z_max)

        in_box = in_x & in_y & in_z
        count_cup = jnp.sum(in_box)

        pz = particles[:, 2]
        on_floor = pz <= z_floor
        count_spilled = jnp.sum(on_floor)

        count_jug = len(particles) - count_cup - count_spilled # currently ignoring particles currently pouring

        return count_cup, count_jug

    def _compute_reward(self, particles, jug_pose, velocity, mesh_positions, exceeding_velocity_limit, action):
        reward, new_particles_cup, new_particles_spilled, curr_fill_level = self._compute_reward_jax(particles, jug_pose, abs(velocity), mesh_positions, exceeding_velocity_limit, action,
                                       self.bb_cup, self.z_floor, self.alpha,
                                       self.pt_cup_wgt, self.pt_spill_wgt,
                                       self.action_cost, self.jug_resting_wgt, self.jug_velocity_wgt, self.idx_rim_jug, self.idx_rim_cup, self._target_particles, self._last_particles_cup, self._last_particles_spilled, self.target_fill_level)
        self._last_particles_cup = new_particles_cup
        self._last_particles_spilled = new_particles_spilled
        self._last_fill_level = curr_fill_level
        return float(reward) #/ 30.0

    @staticmethod
    @jax.jit
    def _compute_reward_jax(particles, jug_pose, velocity, mesh_positions, exceeding_velocity_limit, action,
                            bb_cup, z_floor, alpha,
                            pt_cup_wgt, pt_spill_wgt,
                            action_cost, jug_resting_wgt, jug_velocity_wgt, idx_rim_jug, idx_rim_cup, target_particle_positions, last_particles_cup, last_particles_spilled, target_fill_level):    
        """
        Compute the reward for a given transition.
        Args:
            particles: Current particle positions.
            jug_pose: Current jug pose.
            action: Action taken.
        Returns:
            reward: Reward for the transition.
        """
        # get distance between cup and jug
        distance = PouringEnv._calc_distance_jug_cup(mesh_positions, idx_rim_jug, idx_rim_cup)
        # TODO: these are currently only for the x rotation case!
        max_distance = 2.2347841
        min_distance = 0.26558006
        distance_normalized = (distance - min_distance) / (max_distance - min_distance)

        """
        # using absolute particle counts
        particles_cup = PouringEnv._get_particles_in_cup_smooth(particles, bb_cup, alpha=alpha)
        particles_spilled = PouringEnv._get_particles_on_floor_smooth(particles, z_floor=z_floor, alpha=alpha)
        # TODO: only works with real particle numbers, not with smooth values
        # to change back: remove last_particles values everywhwere, return to percentages in get_particles functions
        
        #reward = pt_cup_wgt * particles_cup + pt_spill_wgt * particles_spilled + action_cost * jnp.mean(action**2)# + jug_velocity_wgt * jnp.linalg.norm(velocity)
        particles_cup = PouringEnv._get_particles_in_cup_smooth(particles, bb_cup, alpha=alpha) / len(particles) # normalize by number of particles
        particles_spilled = PouringEnv._get_particles_on_floor_smooth(particles, z_floor=z_floor, alpha=alpha) / len(particles) # normalize by number of particles

        reward = pt_cup_wgt * particles_cup + pt_spill_wgt * particles_spilled# + action_cost * jnp.mean(action**2)# + jug_velocity_wgt * jnp.linalg.norm(velocity)
        reward = lax.cond(
            particles_cup < 0.4,
            lambda r: r,
            lambda r: r + jug_resting_wgt * jug_pose[3]**2, # try to get it to turn back
            reward
        )"""
        
        # using particle diff from last step
        particles_cup = PouringEnv._get_particles_in_cup_smooth(particles, bb_cup, alpha=alpha)
        particles_spilled = PouringEnv._get_particles_on_floor_smooth(particles, z_floor=z_floor, alpha=alpha)
        particles_cup_diff = (particles_cup - last_particles_cup) / len(particles) # normalize by number of particles
        particles_spilled_diff = (particles_spilled - last_particles_spilled) / len(particles) # normalize by number of particles
        curr_fill_level = particles_cup / len(particles) # percentage of particles in cup TODO: use actual fill level instead of just percentage of particles in cup
        #target_fill_level = 0.3 # TODO move to env attributes
        target_level_wgt = 0.1 # weight for target fill level reward # TODO move to env attributes
        #pt_cup_wgt = 0
        #pt_spill_wgt = 0
        jug_resting_wgt = -.00001 
        pt_spill_wgt = 3.5#-5
        action_cost = 0#-0.005

        #target_level_reward = -target_level_wgt * ((target_fill_level - curr_fill_level)**2 - (target_fill_level - 0)**2) # second part is to start at 0 reward when cup is empty
        #target_level_reward =  1 - ((curr_fill_level - target_fill_level) / target_fill_level) ** 2 # scaling with target fill level to make sure that it has the same scale for all target fill levels
        error_fill_level = (curr_fill_level - target_fill_level) / target_fill_level
        target_level_reward = 2 / (1 + abs(error_fill_level)**2) - 1 # bounded [-1,1] with smooth decay for extreme overshoots
        reward = pt_cup_wgt * particles_cup_diff + pt_spill_wgt * particles_spilled_diff + target_level_wgt * target_level_reward + action_cost * jnp.mean(action**2)# + jug_velocity_wgt * jnp.linalg.norm(velocity)
        reward = lax.cond(
            particles_cup/len(particles) < target_fill_level,
            lambda r: r,
            lambda r: r + jug_resting_wgt * jug_pose[3]**2, # try to get it to turn back
            reward
        )


        """reward = lax.cond(
            particles_cup < 0.4,
            lambda r: r - distance_normalized,
            lambda r: r,# + distance_normalized,# + jug_resting_wgt * jug_pose[3]**2,
            reward
        ) # remove distance punish after certain fill percentage

        # use chamfer distance as reward
        chamfer = PouringEnv.chamfer_distance(particles, target_particle_positions)
        #reward = -chamfer
        #reward = 1.0 / (chamfer + 1e-6)
        reward += jnp.exp(-6.0 * chamfer)"""

        reward = lax.cond(
            exceeding_velocity_limit,
            lambda r: r - 0.05,
            lambda r: r,
            reward
        ) # punish for trying to exceed the velocity limit

        return reward, particles_cup, particles_spilled, curr_fill_level

    def _is_done(self):
        """
        Check if the episode is terminated or truncated.
        Args:
            state: Current state.
        Returns:
            terminated: Boolean indicating if the episode is done due to reaching an end state/goal.
            truncated: Boolean indicating if the episode is done due to exceeding the maximum time steps.
        """
        terminated = False
        truncated = False
        # Define your termination condition here
        if self.current_step_id >= self.max_time:
            truncated = True
        # TODO: zweites done value is truncation (ended due to time limit e.g. -> split termination and truncation up)
        return terminated, truncated

    @staticmethod
    @jax.jit
    def chamfer_distance(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Chamfer distance between two point sets P and Q.
        
        Args:
            P: [N, D] array of points (e.g., current particle positions)
            Q: [M, D] array of points (e.g., target particle positions)
        
        Returns:
            Scalar Chamfer distance between P and Q.
        """
        # Compute pairwise distances between all points in P and Q
        # Shape: [N, M]
        dists = jnp.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)

        # For each point in P, find the nearest in Q
        min_dists_pq = jnp.min(dists, axis=1)

        # For each point in Q, find the nearest in P
        min_dists_qp = jnp.min(dists, axis=0)

        # Average the distances in both directions
        chamfer = jnp.mean(min_dists_pq) + jnp.mean(min_dists_qp)

        return chamfer

    @staticmethod
    @jax.jit  
    def _get_particles_in_cup_smooth(particles, bb_cup, alpha = 7.):
        #particles are in frame of ref of cup. simple test to check if particles in the boundingbox of cup
        """x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
        px, py, pz = particles[:, 0], particles[:, 1], particles[:, 2]
        x_in_range = sigmoid(alpha*(px - x_min)) - sigmoid(alpha*(px - x_max))
        y_in_range = sigmoid(alpha*(py - y_min)) - sigmoid(alpha*(py - y_max))
        z_in_range = sigmoid(alpha*(pz - z_min)) - sigmoid(alpha*(pz - z_max))

        comb = jnp.mean(x_in_range * y_in_range * z_in_range) + 1e-6
        return comb"""
        
        # test for checking particle numbers without smoothing
        x_min, y_min, z_min, x_max, y_max, z_max = bb_cup
        px, py, pz = particles[:, 0], particles[:, 1], particles[:, 2]

        in_x = (px >= x_min) & (px <= x_max)
        in_y = (py >= y_min) & (py <= y_max)
        in_z = (pz >= z_min) & (pz <= z_max)

        in_box = in_x & in_y & in_z
        count = jnp.sum(in_box)

        #print('Particles inside cup:', count)
        return count #/ len(particles) # TODO: return to percentage

    @staticmethod
    @jax.jit
    def _get_particles_on_floor_smooth(particles, z_floor, alpha=7.):
        #return jnp.mean(sigmoid(alpha*( z_floor - particles[:, 2]))) + 1e-6
        
        # test for checking particle numbers without smoothing
        pz = particles[:, 2]
        on_floor = pz <= z_floor
        count = jnp.sum(on_floor)

        return count #/ len(particles) #TODO: return to percentage

    def _apply_model_processing_step(self, updated_jug_pose, input_graph):
        """
        Apply the GNN model processing step to the input graph.
        Args:
            updated_jug_pose: Updated jug pose.
            input_graph: Input graph for the model.
        Returns:
            input_graph: Modified input graph after processing.
            latent_liquid_representation: Latent representation of liquid particles.
        """
        prev_liq_position = input_graph.nodes["liq_position"]
        prev_mesh_position = input_graph.nodes["mesh_position"]
        prev_mesh_pose = input_graph.nodes["mesh_pose"]

        #update the jug vertices and pose based on external control
        #print('updated jug pose: ', updated_jug_pose.shape) # (6,)
        jug_nodes_curr = pouring_env.transform_mesh_to_local_coordinates(self.jug_vertices, updated_jug_pose[:3], ref_orientation=updated_jug_pose[3:])
        curr_mesh_position = prev_mesh_position[0, :, -1]
        curr_mesh_position = curr_mesh_position.at[:self.config_dict['jug_vertices_count']].set(jug_nodes_curr)

        curr_mesh_pose = prev_mesh_pose[0, :, -1]
        curr_mesh_pose = curr_mesh_pose.at[0].set(updated_jug_pose)

        next_mesh_position = jnp.concatenate([prev_mesh_position[0,:, 1:], curr_mesh_position[:, None]], axis=1)
        next_mesh_position_padded = prev_mesh_position.at[0].set(next_mesh_position)

        next_mesh_pose = jnp.concatenate([prev_mesh_pose[0,:, 1:], curr_mesh_pose[:, None]], axis=1)
        next_mesh_pose_padded = prev_mesh_pose.at[0].set(next_mesh_pose)

        input_graph.nodes['mesh_position'] = next_mesh_position_padded
        input_graph.nodes['mesh_pose'] = next_mesh_pose_padded
        
        #(output_graph, dict_latent_graphs), _  = self.model.apply(self.network_params['params'], self.network_params['state'], None, input_graph)
        (output_graph, dict_latent_graphs), _ = self._fast_apply(self.network_params['params'], self.network_params['state'], input_graph)
        latent_liquid_representation = dict_latent_graphs['latent_graph_before_decoding'].nodes['v_l'] # latent representation of liquid particles after processing step of model
        #print('shape latent liq. representation: ', latent_liquid_representation.shape)

        ## update current state of the environment that is returned from the model
        # TODO: check if this is correct (is the liquid stuff updated? otherwise check pred_pos in pouring_env forward)
        pred_pos = output_graph.nodes["p:position"]
        total_nodes = jnp.sum(input_graph.n_node[:-1])
        node_padding_mask = jnp.arange(prev_liq_position.shape[0]) < total_nodes

        next_pos_seq = jnp.concatenate([prev_liq_position[:, 1:], pred_pos[:, None]], axis=1)
        next_pos_seq = jnp.where(node_padding_mask[:, None, None], next_pos_seq, prev_liq_position)

        # create new node features and update graph
        input_graph.nodes["liq_position"] = next_pos_seq

        return input_graph, latent_liquid_representation, pred_pos[:-1] # last particles is weird


    def _load_gnn_model(self, model_path, connectivity_radius, collision_mesh_info_list, max_nodes_edges_info):
        """
        Load the GNN model for the environment.
        Returns:
            model: Loaded GNN model.
        """
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


        model = hk.without_apply_rng(hk.transform_with_state(lambda x: haiku_model()(x)))
        network_params = self._load_network_params(model_path)['network']
        print(f"Loading gnn model at path: {model_path}")

        return model, network_params

    def _load_network_params(self, model_path):
        """
        Load the network parameters from a file.
        Args:
            model_path: Path to the model file.
        Returns:
            network_params: Loaded network parameters.
        """
        # taken from MPC file (-> see load_model)
        with open(model_path, 'rb') as f:
            numpy_params = pickle.load(f)
        return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params)     
    
    def _read_metadata(self, data_path):
        with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
            return json.loads(fp.read())
        
    def _read_mesh_vertices(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        scale_factor = 10 # isaac is 10x reality
        return jnp.array(vertices * scale_factor)
    
    def update_pose_single_step(self, pose_init, velocity_init, input_acc, dt=dt):
        """
        Update the pose and velocity of the jug for a single time step.
        Args:
            pose_init: Initial pose of the jug (position + orientation), shape (6,).
            velocity_init: Initial velocity of the jug (linear + angular), shape (6,).
            input_acc: Input accelerations (linear + angular), shape (6,).
            dt: Time step duration (default is 1.0).
        Returns:
            new_pose: Updated pose of the jug, shape (6,).
            new_velocity: Updated velocity of the jug, shape (6,).
        """
        def clamp_control(u):
            return 1.*jax.nn.tanh(u)
        acc_full = clamp_control(input_acc)  # assuming full 6D input 

        new_velocity = velocity_init + acc_full * dt
        new_pose = pose_init + new_velocity * dt

        return new_pose, new_velocity

    @property
    def action_space(self):
        """
        Define the action space of the environment.
        Returns:
            action_space: Action space specification.
        """
        #input control is 1d(about x-axis) rotation (angular velocity)
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float64)

    @property
    def observation_space(self):
        """
        Define the observation space of the environment.
        Returns:
            observation_space: Observation space specification.
        """
        #return spaces.Box(low=-1, high=1, shape=(128,), dtype=jnp.float32),  # latent particle representation
        """return spaces.Dict(
            {   
                'jug_pose' : spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64,),  # jug
                'liquid_data' : spaces.Box(low=-1, high=1, shape=(1048, 128), dtype=jnp.float32),  # latent particle representation
                'jug_velocity' : spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64,),  # jug velocity
                'timestep' : spaces.Discrete(start=0, n=self.max_time),  # timestep
            }
        )"""
        # full state
        # calculate total size
        flat_obs_size = 3 + 9 + 6 + (1048 * 128) 

        lower_bounds = [-1] * 18 + [-np.inf] * (1048 * 128) 
        upper_bounds = [1] * 18 + [np.inf] * (1048 * 128) 

        return spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), shape=(flat_obs_size,), dtype=np.float64)
        
        """# reduced state
        # calculate total size
        flat_obs_size = 3 + 9 + 1 + 1 + 1 + 6

        lower_bounds = [-1] * 18 + [0] * 3
        upper_bounds = [1] * 21

        return spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), shape=(flat_obs_size,), dtype=np.float64)"""