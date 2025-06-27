import h5py
import numpy as np
import jraph
import jax.numpy as jnp
from typing import Dict
import random
import os


class GraphDataset:
    def __init__(self, data_dir: str, batch_size: int, max_n_node_per_graph: int, max_edges: int):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.sim_lengths = []

        self.batch_size = batch_size
        self.max_n_node = max_n_node_per_graph * batch_size + 1
        self.max_edges = max_edges * batch_size

        for file in self.file_list:
            with h5py.File(os.path.join(data_dir, file), 'r') as hf:
                self.sim_lengths.append(len(hf.keys()))

        self.cumulative_lengths = np.cumsum([0] + self.sim_lengths)

        self.shuffle_indices()

        self._generator = self._make_generator()
        self._generator = jraph.dynamically_batch(
                        self._generator,
                        n_node=self.max_n_node,
                        n_edge=self.max_edges,
                        n_graph=self.batch_size + 1
                    )

    def __len__(self):
        return self.cumulative_lengths[-1]
    

    def shuffle_indices(self):
        self._indices = list(range(len(self)))
        random.shuffle(self._indices)

    def get_graph_by_idx(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        file_path = os.path.join(self.data_dir, self.file_list[file_idx])
        
        with h5py.File(file_path, 'r') as hf:
            graph_key = list(hf.keys())[local_idx]
            graph_data = hf[graph_key]

            pos = graph_data['positions']
            pos = np.transpose(pos, (1, 0, 2))

            target_position = pos[:, -1]

            step_context = graph_data['step_context'][-2][np.newaxis]

            nodes = {
                'position': jnp.array(pos[:, :-1]),
                'particle_type': jnp.array(graph_data['particle_types']),
                'target': jnp.array(target_position)
            }

            return jraph.GraphsTuple(
                nodes=nodes,
                edges={},
                senders=jnp.array([]),
                receivers=jnp.array([]),
                globals=jnp.array(step_context),
                n_node=jnp.array([pos.shape[0]]),
                n_edge=jnp.array([0])
            )
        
    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)

    def _make_generator(self):
        index = 0
        while True:
            if index >= len(self._indices):
                self.shuffle_indices()
                index = 0
            
            idx = self._indices[index]
            graph = self.get_graph_by_idx(idx)
            index += 1
            yield graph