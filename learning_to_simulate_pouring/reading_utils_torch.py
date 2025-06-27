import os
import numpy as np
from torch.utils.data import Dataset, Sampler
from jraph import GraphsTuple
import jax.numpy as jnp
import h5py, random


# New SimulationDataset
class GraphDataset_torch(Dataset):
    def __init__(self, data_dir, metadata):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.sim_lengths = []
        
        for file in self.file_list:
            with h5py.File(os.path.join(data_dir, file), 'r') as hf:
                self.sim_lengths.append(len(hf.keys()))
        
        self.cumulative_lengths = np.cumsum([0] + self.sim_lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        file_path = os.path.join(self.data_dir, self.file_list[file_idx])
        
        with h5py.File(file_path, 'r') as hf:
            graph_key = list(hf.keys())[local_idx]
            graph_data = hf[graph_key]

            pos_liq = graph_data['liq_position']
            pos_liq = np.transpose(pos_liq, (1, 0, 2))
            target_liq_position = pos_liq[:, -1]

            pos_mesh_nodes = graph_data['mesh_position']
            pos_mesh_nodes = np.transpose(pos_mesh_nodes, (1, 0, 2))[:, :-1][np.newaxis]

            #full pose of shape (1, 3, hist,6)
            pose_mesh = graph_data['mesh_pose'][:, :-1][jnp.newaxis] 


            nodes = {
                'liq_position': jnp.array(pos_liq[:, :-1]),
                'mesh_position': jnp.array(pos_mesh_nodes),
                'mesh_pose': jnp.array(pose_mesh),
                'particle_type': jnp.array(graph_data['particle_types']),
                'target': jnp.array(target_liq_position)
            }

            graph_tuple = GraphsTuple(
                nodes=nodes,
                edges={},
                senders=jnp.array([]),
                receivers=jnp.array([]),
                globals=None,
                n_node=jnp.array([pos_liq.shape[0]]),
                n_edge=jnp.array([0])
            )

            return graph_tuple
        

class MultiGraphDataset_torch(Dataset):
    def __init__(self, data_dirs, mesh_pt_type_lists):

        self.data_dirs = data_dirs
        self.mesh_pt_type_lists = mesh_pt_type_lists # [[0,1,2], [1,2]]... list of pt_type for mesh_nodes
        
        # List of lists: Each sublist contains h5 files for one dataset
        # [[simulation_0.h5, simulation_1.h5], [simulation_0.h5, ...], ...]
        self.file_lists = []
        
        # List of lists: Each sublist contains lengths of simulations in each file
        # [[file1_len, file2_len], [file1_len, ...], ...]
        self.sim_lengths = []
        
        # Total length of each dataset [dataset1_total_len, dataset2_total_len, ...]
        self.dataset_lengths = []
        
        # Process each dataset
        for data_dir in self.data_dirs:
            # Get all h5 files in this dataset
            file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
            self.file_lists.append(file_list)
            
            # Get lengths of all files in this dataset
            dataset_sim_lengths = []
            for file in file_list:
                with h5py.File(os.path.join(data_dir, file), 'r') as hf:
                    dataset_sim_lengths.append(len(hf.keys()))
            
            self.sim_lengths.append(dataset_sim_lengths)
            self.dataset_lengths.append(sum(dataset_sim_lengths))
        
        # Cumulative sum of dataset lengths: [0, len1, len1+len2, ...]
        # Used to map global index to specific dataset
        self.cumulative_dataset_lengths = np.cumsum([0] + self.dataset_lengths)
        
        # For each dataset, cumulative sum of file lengths
        # [[0, file1_len, file1_len+file2_len, ...], [0, file1_len, ...], ...]
        self.per_dataset_cumulative_lengths = [
            np.cumsum([0] + sim_lens) for sim_lens in self.sim_lengths
        ]

    def __len__(self):
        # Total length across all datasets
        return self.cumulative_dataset_lengths[-1]

    def __getitem__(self, idx):
        # 1. Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_dataset_lengths, idx, side='right') - 1
        # Convert global index to dataset-local index
        local_idx = idx - self.cumulative_dataset_lengths[dataset_idx]
        
        # Get corresponding data_dir
        data_dir = self.data_dirs[dataset_idx]
        
        # 2. Find which file in the dataset contains this index
        file_idx = np.searchsorted(self.per_dataset_cumulative_lengths[dataset_idx], 
                                 local_idx, side='right') - 1
        # Convert dataset-local index to file-local index
        file_local_idx = local_idx - self.per_dataset_cumulative_lengths[dataset_idx][file_idx]
        
        file_path = os.path.join(data_dir, self.file_lists[dataset_idx][file_idx])

        
        with h5py.File(file_path, 'r') as hf:
            graph_key = list(hf.keys())[file_local_idx]
            graph_data = hf[graph_key]

            pos_liq = graph_data['liq_position']
            pos_liq = np.transpose(pos_liq, (1, 0, 2))
            target_liq_position = pos_liq[:, -1]
            
            pos_mesh_nodes = graph_data['mesh_position']
            pos_mesh_nodes = np.transpose(pos_mesh_nodes, (1, 0, 2))[:, :-1][np.newaxis]

            #full pose of shape (1, 3, hist,6)
            pose_mesh = graph_data['mesh_pose'][:, :-1][jnp.newaxis]

            nodes = {
                'liq_position': jnp.array(pos_liq[:, :-1]), # shape (num_particles, hist, 3)
                'mesh_position': jnp.array(pos_mesh_nodes), # shape (1, num_mesh_vertices, hist, 3)
                'mesh_pose': jnp.array(pose_mesh), # shape (1, num_mesh, hist, 6)
                'particle_type': jnp.array(graph_data['particle_types']), # shape (num_mesh_vertices,)
                'particle_type_obj': jnp.array(self.mesh_pt_type_lists[dataset_idx]), # shape (num_mesh_objects,)
                'target': jnp.array(target_liq_position) # shape (num_particles, 3)
            }

            graph_tuple = GraphsTuple(
                nodes=nodes,
                edges={},
                senders=jnp.array([]),
                receivers=jnp.array([]),
                globals=None,
                n_node=jnp.array([pos_liq.shape[0]]),
                n_edge=jnp.array([0])
            )

            # Return graph and its collision handler id
            return graph_tuple, dataset_idx


class SingleDatasetBatchSampler(Sampler):
    """
    Custom sampler to ensure each batch contains data from only one dataset
    """
    def __init__(self, dataset_lengths, batch_size, shuffle=True):
        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create index ranges for each dataset
        # assume dataset_lengths = [400, 100, 1000]
        # self.dataset_indices = [ [0,1,2,...,399],[400,401,...,499], ..]     
        self.dataset_indices = []
        start_idx = 0
        for length in dataset_lengths:
            indices = list(range(start_idx, start_idx + length))
            self.dataset_indices.append(indices)
            start_idx += length

    def __iter__(self):
        # Shuffle within each dataset
        if self.shuffle:
            for indices in self.dataset_indices:
                 # dataset1: [[45, 32, 198, ..., 3], [452, 478, 401, ..., 489]]
                random.shuffle(indices)
        
        # Create batches from each dataset
        all_batches = []
        for dataset_indices in self.dataset_indices:
        # For a given batch_size=20 creates:
        # dataset1: [[45,32,...], [198,76,...], ...]  # 20 indices per batch
        # dataset2: [[452,478,...], [423,445,...], ...]
        # dataset3: [[1200,678,...], [934,567,...], ...]
            batches = [dataset_indices[i:i + self.batch_size] 
                      for i in range(0, len(dataset_indices), self.batch_size)]
            all_batches.extend(batches)
        
        # Shuffle the order of batches
        if self.shuffle:
            #shuffle the batches
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size 
                  for indices in self.dataset_indices)