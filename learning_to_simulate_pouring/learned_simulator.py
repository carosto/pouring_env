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

"""Graph Network Simulator implementation used in NeurIPS 2022 submission.

  Inverse Design for Fluid-Structure Interactions using Graph Network Simulators

  Kelsey R. Allen*, Tatiana Lopez-Guevera*, Kimberly Stachenfeld*,
  Alvaro Sanchez-Gonzalez, Peter Battaglia, Jessica Hamrick, Tobias Pfaff
"""

from typing import Any, Dict

import haiku as hk
import jraph

from . import graph_network
from . import normalizers
from .connectivity_utils import Collision_Manager, Connectivity_Handler


class LearnedSimulator(hk.Module):
  """Graph Network Simulator."""

  def __init__(self,
               connectivity_radius,
               collision_mesh_info_lists,
               max_nodes_edges_info,
               *,
               graph_network_kwargs: Dict[str, Any],
               flatten_features_fn=None,
               name="LearnedSimulator"):
    """Initialize the model.

    Args:
      connectivity_radius: Radius of connectivity within which to connect
        particles with edges.
      graph_network_kwargs: Keyword arguments to pass to the learned part of the
        graph network `model.EncodeProcessDecode`.
      flatten_features_fn: Function that takes the input graph and dataset
        metadata, and returns a graph where node and edge features are a single
        array of rank 2, and without global features. The function will be
        wrapped in a haiku module, which allows the flattening fn to instantiate
        its own variable normalizers.
      name: Name of the Haiku module.
    """
    super().__init__(name=name)
    self._connectivity_radius = connectivity_radius
    self.connectivity_handler_list = [Connectivity_Handler(Collision_Manager(collision_mesh_info_lists[index], connectivity_radius[1]), max_nodes_edges_info) for index in range(len(collision_mesh_info_lists))]
    self._graph_network_kwargs = graph_network_kwargs
    self._graph_network = None

    # Wrap flatten function in a Haiku module, so any haiku modules created
    # by the function are reused in case of multiple calls.
    self._flatten_features_fn = hk.to_module(flatten_features_fn)(
        name="flatten_features_fn")

  def _maybe_build_modules(self, input_graph):
    if self._graph_network is None:
      num_dimensions = input_graph.nodes["liq_position"].shape[-1]
      self._graph_network = graph_network.MultiEncodeProcessDecode(
          name="encode_process_decode",
          node_output_size=num_dimensions,
          **self._graph_network_kwargs)

      self.graph_elements_normalizer = normalizers.GraphElementsNormalizer(
        template_graph=input_graph,
        is_padded_graph=True)
      
      self._target_normalizer = self.graph_elements_normalizer.get_normalizer(
          name="target_normalizer")

  def __call__(self, input_graph: jraph.GraphsTuple, dataset_idx=0, padded_graph=True):
    self._maybe_build_modules(input_graph)

    flat_graphs_tuple = self._encoder_preprocessor(
        input_graph, dataset_idx, padded_graph=padded_graph)
    normalized_prediction, latent_graph_before_decoding = self._graph_network(flat_graphs_tuple)
    normalized_prediction = normalized_prediction.nodes['v_l'] # only v_l nodes to consider
    next_position, new_velocity, new_acceleration = self._decoder_postprocessor(normalized_prediction,
                                                input_graph)
    return input_graph._replace(
        nodes={"p:position": next_position},
        edges={},
        globals={},
        senders=input_graph.senders[:0],
        receivers=input_graph.receivers[:0],
        n_edge=input_graph.n_edge * 0), {'latent_graph_before_decoding': latent_graph_before_decoding, 'new_velocity': new_velocity, 'new_acceleration': new_acceleration}

  def _encoder_preprocessor(self, input_graph, dataset_idx, padded_graph):
    # Flattens the input graph
    graph_with_flat_features = self._flatten_features_fn(
        input_graph,
        dataset_idx,
        connectivity_radius=self._connectivity_radius,
        _collision_handler_list=self.connectivity_handler_list,
        is_padded_graph=padded_graph,
        apply_normalization=True
        )
    return graph_with_flat_features

  def _decoder_postprocessor(self, normalized_prediction, input_graph):
    # Un-normalize and integrate
    position_sequence = input_graph.nodes["liq_position"]

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    prediction = self._target_normalizer.inverse(normalized_prediction)

    new_position, new_velocity, new_acceleration = euler_integrate_position(position_sequence, prediction)
    return new_position, new_velocity, new_acceleration

  def get_predicted_and_target_normalized_accelerations(
      self, input_graph, dataset_idx, next_position, position_sequence_noise, padded_graph=True):  # pylint: disable=g-doc-args
    
    self._maybe_build_modules(input_graph)
    # Add noise to the input position sequence.
    noisy_position_sequence = input_graph.nodes['liq_position'] + position_sequence_noise
    input_graph.nodes['liq_position'] = noisy_position_sequence

    # Perform the forward pass with the noisy position sequence.
    input_graphs_tuple = self._encoder_preprocessor(input_graph, dataset_idx,padded_graph)
    predicted_normalized_acceleration, _ = self._graph_network(input_graphs_tuple).nodes['v_l'] # only v_l nodes to consider

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_position + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(self, next_position, position_sequence):
    """Inverse of `_decoder_postprocessor`."""

    acceleration = euler_integrate_position_inverse(position_sequence, next_position)
    
    normalized_acceleration = self.graph_elements_normalizer.normalize_node_array(
              "target_normalizer", acceleration)

    return normalized_acceleration



def euler_integrate_position(position_sequence, finite_diff_estimate):
  """Integrates finite difference estimate to position (assuming dt=1)."""
  # Uses an Euler integrator to go from acceleration to position,
  # assuming dt=1 corresponding to the size of the finite difference.
  previous_position = position_sequence[:, -1]
  previous_velocity = previous_position - position_sequence[:, -2]
  next_acceleration = finite_diff_estimate
  next_velocity = previous_velocity + next_acceleration
  next_position = previous_position + next_velocity
  return next_position, next_velocity, next_acceleration


def euler_integrate_position_inverse(position_sequence, next_position):
  """Computes a finite difference estimate from current position and history."""
  previous_position = position_sequence[:, -1]
  previous_velocity = previous_position - position_sequence[:, -2]
  next_velocity = next_position - previous_position
  acceleration = next_velocity - previous_velocity
  return acceleration
