import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Callable
from jraph import GraphsTuple
import jraph
from reading_utils import GraphDataset 
from reading_utils_torch import GraphDataset_torch, MultiGraphDataset_torch, SingleDatasetBatchSampler
from model_utils import time_diff
import functools
import learned_simulator
import model_utils
import json
import collections
import tree

from absl import app
from absl import flags
from absl import logging
import os
import time
from tensorboardX import SummaryWriter
import pickle
import numpy as np
from torch.utils.data import DataLoader
import h5py
import rollout_evaluation

from jax import config

import determined as det
from determined import core

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
# flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_list('data_path', None, help='List of dataset directories.')
flags.DEFINE_integer('batch_size', 1, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')


FLAGS = flags.FLAGS
Stats = collections.namedtuple('Stats', ['mean', 'std'])


INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
KINEMATIC_PARTICLE_ID_1 = 4

SAVE_EVERY = 10000


def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  mask1 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID)
  mask2 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID_1) # separate ids for cup and jug..
  return jnp.logical_or(mask1, mask2)


def save_model(params, state, path):
    pickled_params = {
        'network': {'state': state, 'params': params,}
        }
    with open(path, "wb") as f:
        pickle.dump(pickled_params, f)

def load_model(path):
    with open(path, "rb") as f:
        pickled_params = pickle.loads(f.read())
        network = pickled_params['network']
        state, params = network['state'], network['params']
    return params, state

# Define an exponential decay schedule with a minimum learning rate
def exponential_decay_schedule(init_lr, min_lr, decay_steps, decay_rate):
    """
    Creates a decaying learning rate schedule that mimics TensorFlow's tf.train.exponential_decay.
    """
    # Create the exponential decay schedule from Optax
    decay_fn = optax.exponential_decay(
        init_value=init_lr - min_lr,   # Starting with the difference from min_lr
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        transition_begin=0,  # Start decay at step 0
        staircase=False      # Smooth decay rather than discrete staircase
    )

    # Ensure the learning rate doesn't go below min_lr
    def schedule_fn(step):
        return decay_fn(step) + min_lr
    
    return schedule_fn


def get_random_walk_noise_for_position_sequence(
    position_sequence, noise_std_last_step, key):
  """Returns random-walk noise in the velocity applied to the position."""
  key, subkey = jax.random.split(key)
  velocity_sequence = time_diff(position_sequence)

  # We want the noise scale in the velocity at the last step to be fixed.
  # Because we are going to compose noise at each step using a random_walk:
  # std_last_step**2 = num_velocities * std_each_step**2
  # so to keep `std_last_step` fixed, we apply at each step:
  # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
  num_velocities = velocity_sequence.shape[1]
  velocity_sequence_noise = jax.random.normal(
      subkey,
      shape=velocity_sequence.shape,
      dtype=position_sequence.dtype) * (noise_std_last_step / num_velocities ** 0.5)

  # Apply the random walk.
  velocity_sequence_noise = jnp.cumsum(velocity_sequence_noise, axis=1)

  # Integrate the noise in the velocity to the positions, assuming
  # an Euler intergrator and a dt = 1, and adding no noise to the very first
  # position (since that will only be used to calculate the first position
  # change).
  position_sequence_noise = jnp.concatenate([
      jnp.zeros_like(velocity_sequence_noise[:, 0:1]),
      jnp.cumsum(velocity_sequence_noise, axis=1)], axis=1)

  return position_sequence_noise, key

def find_latest_model(model_dir):
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pkl')]
    if not model_files:
        return None
    
    def get_global_step_from_fname(_file):
        return int(_file.split('_')[-1].split(".")[0])

    # Extract global steps and find the maximum
    global_steps = [get_global_step_from_fname(f) for f in model_files]
    latest_step = max(global_steps)
    latest_model = f'model_checkpoint_globalstep_{latest_step}.pkl'
    
    return os.path.join(model_dir, latest_model), latest_step

def init_model(forward_training, rng_key, dummy_graph, dummy_dataset_idx, dummy_targets, dummy_masked_sampled_noise, model_dir):
    output = find_latest_model(model_dir)
    if output is not None:
        latest_model_path, global_step = output
        print(f"Loading latest model: {latest_model_path.split('/')[-1]}")
        _ = forward_training.init(rng_key, dummy_graph, dummy_dataset_idx, dummy_targets, dummy_masked_sampled_noise)
        params, state = load_model(latest_model_path)
    else:
        print("No saved model found. Initializing with random parameters.")
        params, state = forward_training.init(rng_key, dummy_graph, dummy_dataset_idx, dummy_targets, dummy_masked_sampled_noise)
        global_step = 0

    return params, state, global_step


def create_training_loop(
    model: hk.Module,
    train_dataset: GraphDataset,
    val_dataset: GraphDataset,
    num_epochs: int,
    evaluate_every: int,
    writer
):
    @jax.jit
    def train_step(params, state, opt_state, batch, step, key_noisegen):
        input_graphs, dataset_idx = batch
        targets = input_graphs.nodes['target']

        def loss_fn_wrapper(params, state):
            # Use vmap to apply the model to each graph in the batch

            pred_target, new_state = forward_training.apply(params, state, rng_key, input_graphs, dataset_idx, targets, sampled_noise)
            pred_acceleration, target_acceleration = pred_target
            loss = (pred_acceleration - target_acceleration)**2
            # jax.debug.print('pred {pred_acceleration}, target {target_acceleration}', pred_acceleration=pred_acceleration[0], target_acceleration=target_acceleration[0])
            
            #take out padded nodes component of loss..
            mask_padding = jraph.get_node_padding_mask(input_graphs)
            loss = jnp.where(mask_padding[:, jnp.newaxis], loss, jnp.zeros_like(loss))

            mask_float = jnp.astype(mask_padding, jnp.float32)
            loss = jnp.sum(loss) / jnp.sum(mask_float)

            return loss, (pred_acceleration, new_state)

        sampled_noise, key_noisegen = get_random_walk_noise_for_position_sequence(
            input_graphs.nodes['liq_position'], noise_std_last_step=noise_std, key=key_noisegen)

        (loss, (predictions, new_state)), grads = jax.value_and_grad(loss_fn_wrapper, has_aux=True)(params, state)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        learning_rate = learning_rate_schedule(step)

        return params, opt_state, loss, new_state, learning_rate, key_noisegen

    @jax.jit
    def evaluate_step(params, state, batch):
        input_graphs, dataset_idx = batch
        targets = input_graphs.nodes['target']
        (output_graph,_), _  = forward_evaluation.apply(params, state, rng_key, input_graphs, dataset_idx)
        predictions = output_graph.nodes["p:position"]
        loss = jnp.mean((predictions - targets) ** 2)
        return loss
    
    def evaluate(params, state, dataset):
        window_loss = 0
        window_size = 100
        step = 0
        total_loss = 0
        num_batches = 0
        for batch in dataset:
            loss = evaluate_step(params, state, batch)
            window_loss += loss
            step += 1
            if step % window_size == 0: 
                avg_loss_in_window = window_loss / window_size
                print(f"Eval Step {step}, Average Loss in last {window_size} steps: {avg_loss_in_window:.4e}, ")
                window_loss = 0
            if step % 2000 == 0:  
                #Training GETS freaking slow over time! clearing_cache helps.
                print(f"clearing cache after {step} steps")   
                jax.clear_caches()  
            total_loss += loss
            num_batches += 1
        return total_loss / num_batches

    # Initialize model and optimizer
    noise_std = FLAGS.noise_std
    print(f'added noise is {noise_std}')
    rng_key = jax.random.PRNGKey(42)
    rng_key_noisegen = jax.random.PRNGKey(42)

    dummy_graph, dummy_dataset_idx = next(iter(train_dataset))
    dummy_targets = dummy_graph.nodes['target']
    shape_masked_sample = dummy_graph.nodes['liq_position'].shape
    dummy_masked_sampled_noise = jnp.full(shape_masked_sample, 0.0) #masked sampled noise matches position_sequence shape


    forward_training = hk.transform_with_state(lambda x, _d_idx, _tgt, _mask: model().get_predicted_and_target_normalized_accelerations(x, _d_idx, _tgt, _mask))
    forward_evaluation = hk.transform_with_state(lambda x, _d_idx: model()(x, _d_idx))

    # Use the custom method for initialization
    params, state, global_step = init_model(forward_training, rng_key, dummy_graph, dummy_dataset_idx, dummy_targets, dummy_masked_sampled_noise, FLAGS.model_path)

    optimizer = optax.adam(learning_rate=learning_rate_schedule(global_step))
    opt_state = optimizer.init(params)

    start_time = time.time()
    window_loss = 0
    window_size = 100

    print("Training start....")
    with core.init() as core_context:
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in train_dataset:
                global_step += 1
                params, opt_state, loss, state, learning_rate, rng_key_noisegen = train_step(params, state, opt_state, batch, global_step, rng_key_noisegen)
                epoch_loss += loss
                num_batches += 1

                window_loss += loss
                # Calculate average and loss time per step
                if global_step % window_size == 0: 
                    avg_loss_in_window = window_loss / window_size 
                    num_steps_per_sec = window_size/ (time.time() - start_time) 
                    print(f"Global Step {global_step}, Average Loss in last {window_size} steps: {avg_loss_in_window:.4f}, "
                    f"Num steps per second: {num_steps_per_sec:.2f}")

                    writer.add_scalar('Train/1_loss', avg_loss_in_window, global_step)
                    writer.add_scalar('Train/3_global_step', num_steps_per_sec, global_step)
                    writer.add_scalar('Train/learning_rate', learning_rate, global_step)
                    window_loss = 0
                    start_time = time.time()
                    #similar to tf, evaluate after 100 steps
                    one_step_mse = evaluate_step(params, state, batch)
                    # val_loss = evaluate(params, val_dataset)
                    writer.add_scalar('Train/2_one_step_position_mse', one_step_mse, global_step)
                    core_context.train.report_training_metrics(
                            steps_completed=global_step,
                            metrics={"1_loss": float(avg_loss_in_window)}  # Average loss across devices
                        )
                    core_context.train.report_training_metrics(
                            steps_completed=global_step,
                            metrics={"3_global_step": num_steps_per_sec}  # Average loss across devices
                        )
                    core_context.train.report_training_metrics(
                            steps_completed=global_step,
                            metrics={"4_learning_rate": float(learning_rate)}  # Average loss across devices
                        )
                    core_context.train.report_training_metrics(
                            steps_completed=global_step,
                            metrics={"2_one_step_position_mse": float(one_step_mse)}  # Average loss across devices
                        )
                    
                    if global_step % SAVE_EVERY == 0:
                        # Save the model as pickle
                        save_path = f"{FLAGS.model_path}/model_checkpoint_globalstep_{global_step}.pkl"
                        save_model(params, state, save_path)   

                if global_step % 2000 == 0:  
                    #Training GETS freaking slow over time! clearing_cache helps.
                    print(f"clearing cache after {global_step} steps")   
                    jax.clear_caches()  

            average_loss = epoch_loss / num_batches

            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
            core_context.train.report_training_metrics(
                steps_completed=global_step,
                metrics={"5_train_epoch_loss": float(average_loss)}  # Average loss across devices
            )
            writer.add_scalar('Train/epoch_loss', average_loss, epoch + 1)

            if (epoch + 1) % evaluate_every == 0:
                jax.clear_caches()  
                val_loss = evaluate(params, state, val_dataset)
                print(f"Validation Loss: {val_loss:.6e}")
                writer.add_scalar('Validation/epoch_loss', val_loss, epoch + 1)
                core_context.train.report_training_metrics(
                    steps_completed=global_step,
                    metrics={"6_val_epoch_loss": float(val_loss)}  # Average loss across devices
                )

    return params

# values similar to to TensorFlow code
init_lr = 1e-4  # Initial learning rate
min_lr = 1e-6   # Minimum learning rate
decay_steps = int(5e6)
decay_rate = 0.1
# Create the learning rate schedule
learning_rate_schedule = exponential_decay_schedule(init_lr, min_lr, decay_steps, decay_rate)

def main(_):
    """Train or evaluates the model."""


    #collect all the dataset info..
    collision_mesh_info_lists, max_nodes_edges_info_list, mesh_pt_type_lists = [],[],[]
    max_n_liq_node_dset, _max_edges_l_dset, _max_edges_m_dset, max_num_objects = 0,0,0,0
    for path in FLAGS.data_path:
        metadata = _read_metadata(path)
        
        collision_mesh_info_list = metadata["collision_mesh"] 
        mesh_pt_type_list = [ z[1] for z in collision_mesh_info_list] #mesh pt type for handling in v_o
        mesh_pt_type_lists.append(mesh_pt_type_list)

        _max_n_liq_node_per_graph = int(metadata["max_n_liq_node"]) #  can be read from position as well.. ignore for now
        _max_edges_l_per_graph =int( metadata["max_n_edge_l"])
        _max_edges_m_per_graph =int( metadata["max_n_edge_m"])
        _max_nodes_edges_info=[FLAGS.batch_size,_max_n_liq_node_per_graph, _max_edges_l_per_graph, _max_edges_m_per_graph]
        max_nodes_edges_info_list.append(_max_nodes_edges_info)

        max_n_liq_node_dset = max(max_n_liq_node_dset, _max_n_liq_node_per_graph)
        _max_edges_l_dset = max(_max_edges_l_dset, _max_edges_l_per_graph)
        _max_edges_m_dset = max(_max_edges_m_dset, _max_edges_m_per_graph)
        max_num_objects = max(max_num_objects, len(collision_mesh_info_list))

        connectivity_radius = metadata["default_connectivity_radius"] # HAS TO BE SAME FOR ALL DATASETS!!

        #list of list containing [['obj_file',particle_type, #mesh_nodes]]
        collision_mesh_info_lists.append(collision_mesh_info_list)

    #max for all datasets
    max_nodes_edges_info = [max_num_objects,max_n_liq_node_dset, _max_edges_l_dset, _max_edges_m_dset]
    print(max_nodes_edges_info)
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
                                    collision_mesh_info_lists=collision_mesh_info_lists, 
                                    max_nodes_edges_info=max_nodes_edges_info, 
                                    flatten_features_fn=flatten_fn, **model_kwargs)

    if FLAGS.mode in ['train', 'eval']:
        def collate_fn(batch):
            graphs, dataset_ids = zip(*batch)
            batched_graphs = jraph.batch(graphs)
            num_graphs = batched_graphs.n_node.shape[0]

            # All dataset_ids will be the same, return the first one
            dataset_id = dataset_ids[0]
            _max_nodes_edges_info = max_nodes_edges_info_list[dataset_id]
            _, max_n_liq_node_per_graph, max_edges_l_per_graph, _ = _max_nodes_edges_info

            padded_graphs = jraph.pad_with_graphs(
                batched_graphs,
                n_node=max_n_liq_node_per_graph * num_graphs + 1,
                n_edge=max_edges_l_per_graph * num_graphs,
                n_graph=num_graphs + 1
            )
            return padded_graphs, dataset_id


        def setup_dataloader(data_dirs, batchsize, mesh_pt_type_lists, shuffle=True):
            dataset = MultiGraphDataset_torch(data_dirs, mesh_pt_type_lists)
            
            # Use custom sampler to sample batched data from one dataset at a time.
            batch_sampler = SingleDatasetBatchSampler(
                dataset_lengths=dataset.dataset_lengths,
                batch_size=batchsize,
                shuffle=shuffle
            )
            
            train_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn
            )
            return train_loader
        
        if FLAGS.mode == 'train':
            if not os.path.exists(FLAGS.model_path):
                os.makedirs(FLAGS.model_path)
            # Set up TensorBoard writer
            log_dir = f"{FLAGS.model_path}/logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir)

            train_dirs = [f"{path}/train" for path in FLAGS.data_path]
            valid_dirs = [f"{path}/valid" for path in FLAGS.data_path]

            train_dataset = setup_dataloader(data_dirs=train_dirs, batchsize=FLAGS.batch_size, mesh_pt_type_lists=mesh_pt_type_lists, shuffle=True)
            val_dataset = setup_dataloader(data_dirs=valid_dirs, batchsize=FLAGS.batch_size, mesh_pt_type_lists=mesh_pt_type_lists, shuffle=False)
            # Train all the way through.
            trained_params = create_training_loop(
                            model=haiku_model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            num_epochs=500,
                            evaluate_every=1,
                            writer=writer
            )
            writer.close()
        else:
            pass
    elif FLAGS.mode == 'eval_rollout':
        if not FLAGS.output_path:
            raise ValueError('A rollout path must be provided.')
        total_mse = 0
        num_trials = 0
        print('Eval rollout on test set...')
        
        #init model
        model = hk.transform_with_state(lambda x: haiku_model()(x))
        #load latest model params
        output = find_latest_model(FLAGS.model_path)
        if output is not None:
            latest_model_path, _ = output
            print(f"Loading latest model: {latest_model_path.split('/')[-1]}")
            params, state = load_model(latest_model_path)
            network_params = {'state':state, 'params':params}
        else:
            raise ValueError('No saved models found')

        rng_key = jax.random.PRNGKey(42)
       #for evaluation only pass onedataset at a time
        file_list = [f for f in os.listdir(f"{FLAGS.data_path[0]}/test") if f.endswith('.h5')]

        if not os.path.exists(FLAGS.output_path):
            os.mkdir(FLAGS.output_path)
        with open(os.path.join(FLAGS.output_path, 'log.txt'), 'a') as logging_file:
            logging_file.write(f"Loading latest model: {latest_model_path.split('/')[-1]}")
        for fidx in range(len(file_list)):
            with h5py.File(os.path.join(f"{FLAGS.data_path[0]}/test", f'simulation_{fidx}.h5'), 'r') as hf:
                #iterate over the each graph index in order
                num_steps = len(hf.keys())

                #first get init positions from the first graph
                graph_key = f'graph_{0}'
                graph_data = hf[graph_key]
                pos_liq = graph_data['liq_position'][:]
                pos_liq = jnp.transpose(pos_liq, (1, 0, 2))[:, :-1]

                particle_types = jnp.array(graph_data['particle_types'])

                pos_mesh_nodes = graph_data['mesh_position'][:]
                pos_mesh_nodes = jnp.transpose(pos_mesh_nodes, (1, 0, 2))[:, :-1][jnp.newaxis]

                pose_mesh_shape = graph_data['mesh_pose'][:, -1].shape
                pose_mesh = graph_data['mesh_pose'][:, :-1][jnp.newaxis] 

                initial_features = {
                'liq_position': pos_liq,
                'mesh_position': pos_mesh_nodes,
                'mesh_pose': pose_mesh,
                'particle_type': particle_types,
                'particle_type_obj': jnp.array(mesh_pt_type_lists[0]),
                }
        
                _, max_n_liq_node_per_graph, max_edges_l_per_graph, _ = max_nodes_edges_info_list[0]

                initial_graph = rollout_evaluation.build_graph(initial_features,max_n_liq_node_per_graph, max_edges_l_per_graph)

                liq_positions_full_traj = jnp.zeros((num_steps, pos_liq.shape[0], 3))
                mesh_positions_full_traj = jnp.zeros((num_steps, pos_mesh_nodes.shape[1], 3))
                mesh_pose_full_traj = jnp.zeros((num_steps, *pose_mesh_shape))


                for idx in range(num_steps):
                    graph_key = f'graph_{idx}'
                    graph_data = hf[graph_key]

                    pos_liq = jnp.array(graph_data['liq_position'][:])[-1]
                    pos_mesh_node_pos = jnp.array(graph_data['mesh_position'][:])[-1]

                    pose_mesh = jnp.array(graph_data['mesh_pose'][:])[:, -1]


                    liq_positions_full_traj = liq_positions_full_traj.at[idx].set(pos_liq) 
                    mesh_positions_full_traj = mesh_positions_full_traj.at[idx].set(pos_mesh_node_pos) 
                    mesh_pose_full_traj = mesh_pose_full_traj.at[idx].set(pose_mesh) 

                features = {
                    'liq_position': liq_positions_full_traj,
                    'mesh_position': mesh_positions_full_traj,
                    'mesh_pose': mesh_pose_full_traj,
                }

                example_rollout, mse = rollout_evaluation.rollout(initial_graph, features, model, network_params)
                example_rollout['particle_types'] = particle_types

                total_mse += mse
                num_trials += 1
                
                filename_save = f'rollout_test_{fidx}.npz'
                filename_save = os.path.join(FLAGS.output_path, filename_save)
                np.savez(filename_save, **example_rollout)

                print(f"Trial {num_trials} (file: simulation_{fidx}.h5) - MSE: {mse}")
                with open(os.path.join(FLAGS.output_path, 'log.txt'), 'a') as logging_file:
                    logging_file.write(f"\nTrial {num_trials} (file: simulation_{fidx}.h5) - MSE: {mse}")
        average_mse = total_mse / num_trials
        print(f"\nAverage MSE across {num_trials} trials: {average_mse}")
        with open(os.path.join(FLAGS.output_path, 'log.txt'), 'a') as logging_file:
            logging_file.write(f"\nAverage MSE across {num_trials} trials: {average_mse}")




if __name__ == '__main__':
  app.run(main)