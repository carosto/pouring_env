import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Callable
from jraph import GraphsTuple
import jraph
from reading_utils import GraphDataset 
from reading_utils_torch import GraphDataset_torch 
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

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
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

SAVE_EVERY = int(10000/jax.local_device_count())


def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  mask1 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID)
  mask2 = jnp.equal(particle_types, KINEMATIC_PARTICLE_ID_1) # separate ids for cup and jug..
  return jnp.logical_or(mask1, mask2)


def _combine_std(std_x, std_y):
  return jnp.sqrt(std_x**2 + std_y**2)

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
    

def save_model(params, path):
    # Convert parameters to numpy arrays to ensure they're serializable
    numpy_params = jax.tree_util.tree_map(lambda x: np.array(x[0]), params)
    with open(path, 'wb') as f:
        pickle.dump(numpy_params, f)

def load_model(path):
    with open(path, 'rb') as f:
        numpy_params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params)

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

def init_model(forward_training, rng_key, dummy_graph, dummy_targets, dummy_masked_sampled_noise, model_dir):
    output = find_latest_model(model_dir)
    if output is not None:
        latest_model_path, global_step = output
        print(f"Loading latest model: {latest_model_path.split('/')[-1]}")
        loaded_params = load_model(latest_model_path)
        _, state = forward_training.init(rng_key, dummy_graph, dummy_targets, dummy_masked_sampled_noise)
        params = loaded_params
    else:
        print("No saved model found. Initializing with random parameters.")
        params, state = forward_training.init(rng_key, dummy_graph, dummy_targets, dummy_masked_sampled_noise)
        global_step = 0

    params = jax.device_put_replicated(params, jax.local_devices())
    state = jax.device_put_replicated(state, jax.local_devices())
    return params, state, global_step

def device_batch(graph_generator, fixed_obstacle_edges):
    """Batches a set of graphs the size of the number of devices."""
    num_devices = jax.local_device_count()
    device_batch = []
    for idx, graph in enumerate(graph_generator):
        if idx % num_devices == num_devices - 1:
            device_batch.append(graph)
            stacked_batch = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *device_batch)
            fixed_obstacle_edges_devices = jnp.repeat(fixed_obstacle_edges[jnp.newaxis], num_devices, axis=0)
            stacked_batch = stacked_batch._replace(edges={'obstacle_edges': fixed_obstacle_edges_devices})
            yield stacked_batch
            device_batch = []
        else:
            device_batch.append(graph)

def create_training_loop(
    model: hk.Module,
    train_dataset: GraphDataset,
    val_dataset: GraphDataset,
    num_epochs: int,
    evaluate_every: int,
    fixed_obstacle_edges,
    writer
):
    @functools.partial(jax.pmap, axis_name='device')
    def train_step(params, state, opt_state, batch, key_noisegen):
        input_graphs = batch
        targets = input_graphs.nodes['target']

        def loss_fn_wrapper(params, state):
            # Use vmap to apply the model to each graph in the batch

            pred_target, new_state = forward_training.apply(params, state, rng_key, input_graphs, targets, masked_sampled_noise)
            pred_acceleration, target_acceleration = pred_target
            loss = (pred_acceleration - target_acceleration)**2
            # jax.debug.print('pred {pred_acceleration}, target {target_acceleration}', pred_acceleration=pred_acceleration[0], target_acceleration=target_acceleration[0])
            
            #take out padded nodes component of loss..
            mask_padding = jraph.get_node_padding_mask(input_graphs)
            loss = jnp.where(mask_padding[:, jnp.newaxis], loss, jnp.zeros_like(loss))

            #take only losses from liq particles and not obj particles
            loss = jnp.where(non_kinematic_mask[:, jnp.newaxis], loss, jnp.zeros_like(loss))

            non_kinematic_mask_float = jnp.astype(non_kinematic_mask, jnp.float32)
            #remove masked nodes from this mask
            non_kinematic_mask_float = jnp.where(mask_padding, non_kinematic_mask_float, jnp.zeros_like(non_kinematic_mask_float))
            num_non_kinematic = jnp.sum(non_kinematic_mask_float)
            loss = jnp.sum(loss) / jnp.sum(num_non_kinematic)

            return loss, (pred_acceleration, new_state)

        sampled_noise, key_noisegen = get_random_walk_noise_for_position_sequence(
            input_graphs.nodes['position'], noise_std_last_step=noise_std, key=key_noisegen)
        non_kinematic_mask = jnp.logical_not(
            get_kinematic_mask(input_graphs.nodes['particle_type']))
        noise_mask = non_kinematic_mask[:, jnp.newaxis, jnp.newaxis].astype(sampled_noise.dtype)
        masked_sampled_noise = sampled_noise * noise_mask

        (loss, (predictions, new_state)), grads = jax.value_and_grad(loss_fn_wrapper, has_aux=True)(params, state)
        grads = jax.lax.pmean(grads, axis_name='device')
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, new_state, key_noisegen

    @functools.partial(jax.pmap, axis_name='device')
    def evaluate_step(params, batch):
        input_graphs = batch
        targets = input_graphs.nodes['target']
        output_graph, _  = forward_evaluation.apply(params, rng_key, input_graphs)
        predictions = output_graph.nodes["p:position"]
        loss = jnp.mean((predictions - targets) ** 2)
        return loss
    
    def evaluate(params, dataset):
        window_loss = 0
        window_size = int(100/jax.local_device_count())
        step = 0
        total_loss = 0
        num_batches = 0
        steps_eval = len(dataset) // (FLAGS.batch_size * jax.local_device_count())
        for idx in range(steps_eval):
            batch = next(device_batch(dataset, fixed_obstacle_edges))
            loss = evaluate_step(params, batch)
            loss = jnp.mean(jax.device_get(loss))
            window_loss += loss
            step += 1
            if step % window_size == 0: 
                avg_loss_in_window = window_loss / window_size
                print(f"Eval Step {step}, Average Loss in last {window_size} steps: {avg_loss_in_window:.4e}, ")
                window_loss = 0
            total_loss += loss
            num_batches += 1
        return total_loss / num_batches

    # Initialize model and optimizer
    noise_std = FLAGS.noise_std
    rng_key = jax.random.PRNGKey(42)
    rng_key_noisegen = jax.random.PRNGKey(42)
    rng_key_noisegen = jax.random.split(rng_key_noisegen, len(jax.local_devices()))

    dummy_graph = next(iter(train_dataset))
    dummy_graph = dummy_graph._replace(edges={'obstacle_edges': fixed_obstacle_edges})
    dummy_targets = dummy_graph.nodes['target']
    shape_masked_sample = dummy_graph.nodes['position'].shape
    dummy_masked_sampled_noise = jnp.full(shape_masked_sample, 0.0) #masked sampled noise matches position_sequence shape


    forward_training = hk.transform_with_state(lambda x, _tgt, _mask: model().get_predicted_and_target_normalized_accelerations(x, _tgt, _mask))
    forward_evaluation = hk.transform(lambda x: model()(x))

    # Use the custom method for initialization
    params, state, global_step = init_model(forward_training, rng_key, dummy_graph, dummy_targets, dummy_masked_sampled_noise, FLAGS.model_path)

    optimizer = optax.adam(learning_rate=learning_rate_schedule(global_step))
    opt_state = jax.pmap(optimizer.init)(params)

    steps_per_epoch = len(train_dataset) // (FLAGS.batch_size * jax.local_device_count())
    print(f"Steps per epoch -- {steps_per_epoch}")
    start_time = time.time()
    window_loss = 0
    window_size = int(100/jax.local_device_count())

    print("Training start....")
    with core.init(distributed=distributed) as core_context:
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for idx in range(steps_per_epoch):
                global_step += 1
                batch = next(device_batch(train_dataset, fixed_obstacle_edges))
                params, opt_state, loss, state, rng_key_noisegen = train_step(params, state, opt_state, batch, rng_key_noisegen)
                loss = jnp.mean(jax.device_get(loss))
                epoch_loss += loss
                num_batches += 1

                window_loss += loss
                # Calculate average and loss time per step
                if global_step % window_size == 0: 
                    avg_loss_in_window = window_loss / window_size 
                    num_steps_per_sec = window_size/ (time.time() - start_time) 
                    print(f"Global Step {global_step}, Average Loss in last {window_size} steps: {avg_loss_in_window:.4f}, "
                    f"Num steps per second: {num_steps_per_sec:.2f}")

                    learning_rate = learning_rate_schedule(global_step)

                    window_loss = 0
                    start_time = time.time()
                    #similar to tf, evaluate after 100 steps
                    one_step_mse = evaluate_step(params, batch)
                    one_step_mse = jnp.mean(jax.device_get(one_step_mse))

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
                    writer.add_scalar('Train/1_loss', avg_loss_in_window, global_step)
                    writer.add_scalar('Train/3_global_step', num_steps_per_sec, global_step)
                    writer.add_scalar('Train/4_learning_rate', learning_rate, global_step)
                    writer.add_scalar('Train/2_one_step_position_mse', one_step_mse, global_step)

                    if global_step % SAVE_EVERY == 0:
                        # Save the model as pickle
                        print(f"Saving model at step {global_step}")
                        save_path = f"{FLAGS.model_path}/model_checkpoint_globalstep_{global_step}.pkl"
                        save_model(params, save_path)   

                if global_step % int(2000/FLAGS.batch_size) == 0:  
                    #Training GETS freaking slow over time! clearing_cache helps.
                    print(f"clearing cache after {global_step} steps")   
                    jax.clear_caches()  

            average_loss = epoch_loss / num_batches

            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")

            core_context.train.report_training_metrics(
                steps_completed=epoch + 1,
                metrics={"5_train_epoch_loss": float(average_loss)}  # Average loss across devices
            )
            writer.add_scalar('Train/5_epoch_loss', average_loss, epoch + 1)
            if (epoch + 1) % evaluate_every == 0:
                val_loss = evaluate(params, val_dataset)
                val_loss = jnp.mean(jax.device_get(val_loss))
                print(f"Validation Loss: {val_loss:.6e}")
                core_context.train.report_training_metrics(
                    steps_completed=epoch + 1,
                    metrics={"6_val_epoch_loss": float(val_loss)}  # Average loss across devices
                )
                writer.add_scalar('Validation/epoch_loss', val_loss, epoch + 1)
                save_path = f"{FLAGS.model_path}/model_checkpoint_epoch_{epoch + 1}.pkl"
                save_model(params, save_path)   

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

    metadata = _read_metadata(FLAGS.data_path)
    connectivity_radius = metadata["default_connectivity_radius"]
    boundaries = metadata["bounds"]
    max_n_node_per_graph = int(metadata["max_n_node"]) #  can be read from position as well.. ignore for now
    max_edges =int( metadata["max_n_edge"])
    # read the fixed obstacle edges of the meshes(already offsetted) by liq postion & obj pos
    fixed_obstacle_edges = np.load(f'{FLAGS.data_path}/fixed_obstacle_edges_offseted.npy') 

    normalization_stats = get_stats(metadata, acc_noise_std=FLAGS.noise_std, vel_noise_std=FLAGS.noise_std)

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
                                    max_edges=int(max_edges*FLAGS.batch_size), 
                                    normalization_stats=normalization_stats, 
                                    flatten_features_fn=flatten_fn, **model_kwargs)

    if FLAGS.mode in ['train', 'eval']:
        print(f"\n USING pmap TO TRAIN ON {jax.local_device_count()} devices \n")
        if FLAGS.mode == 'train':
            if not os.path.exists(FLAGS.model_path):
                os.makedirs(FLAGS.model_path)
            # Set up TensorBoard writer
            log_dir = f"{FLAGS.model_path}/logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir)

            train_dataset = GraphDataset(data_dir=f"{FLAGS.data_path}/train", batch_size=FLAGS.batch_size, 
                                         max_n_node_per_graph=max_n_node_per_graph, max_edges=max_edges)
            val_dataset = GraphDataset(data_dir=f"{FLAGS.data_path}/valid", batch_size=FLAGS.batch_size, 
                                         max_n_node_per_graph=max_n_node_per_graph, max_edges=max_edges)
            # Train all the way through.
            trained_params = create_training_loop(
                            model=haiku_model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            num_epochs=100,
                            evaluate_every=1,
                            writer=writer,
                            fixed_obstacle_edges=jnp.array(fixed_obstacle_edges)
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
        model = hk.transform(lambda x: haiku_model()(x))
        #load latest model params
        output = find_latest_model(FLAGS.model_path)
        if output is not None:
            latest_model_path, _ = output
            print(f"Loading latest model: {latest_model_path.split('/')[-1]}")
            params = load_model(latest_model_path)
        else:
            raise ValueError('No saved models found')

        rng_key = jax.random.PRNGKey(42)
        graph_features = (max_n_node_per_graph, max_edges, fixed_obstacle_edges)

        file_list = [f for f in os.listdir(f"{FLAGS.data_path}/test") if f.endswith('.h5')]

        if not os.path.exists(FLAGS.output_path):
            os.mkdir(FLAGS.output_path)
        with open(os.path.join(FLAGS.output_path, 'log.txt'), 'a') as logging_file:
            logging_file.write(f"Loading latest model: {latest_model_path.split('/')[-1]}")
        for fidx in range(len(file_list)):
            with h5py.File(os.path.join(f"{FLAGS.data_path}/test", f'simulation_{fidx}.h5'), 'r') as hf:
                #iterate over the each graph index in order
                num_steps = len(hf.keys())

                #first get init positions from the first graph
                graph_key = f'graph_{0}'
                graph_data = hf[graph_key]
                positions = graph_data['positions'][:]
                positions = jnp.transpose(positions, (1, 0, 2))
                initial_positions = jnp.array(positions[:, :-1])
                particle_types = jnp.array(graph_data['particle_types'])
                step_context = jnp.array(graph_data['step_context'][-2])

                positions_full_traj = jnp.zeros((num_steps, initial_positions.shape[0], 3))
                step_context_full_traj = jnp.zeros((num_steps, step_context.shape[0]))


                for idx in range(num_steps):
                    graph_key = f'graph_{idx}'
                    graph_data = hf[graph_key]

                    pos = jnp.array(graph_data['positions'][:])

                    step_context = jnp.array(graph_data['step_context'][-2])
                    target_position = pos[-1]

                    positions_full_traj = positions_full_traj.at[idx].set(target_position) 
                    step_context_full_traj = step_context_full_traj.at[idx].set(step_context) 

                features = {
                    'position': positions_full_traj,
                    'particle_types': particle_types,
                    'step_context': step_context_full_traj,
                }

                example_rollout, mse = rollout_evaluation.evaluate_rollout(model, params, features, initial_positions, graph_features)
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
    distributed = det.core.DistributedContext( #required for logging
                rank=0,
                size=1,
                local_rank=0,
                local_size=1,
                cross_rank=0,
                cross_size=1,
                )
    app.run(main)