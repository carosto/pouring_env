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

"""JAX implementation of Encode Process Decode."""

from typing import Optional, Set
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from . import jraph_base_models


class MultiEncodeProcessDecode(hk.Module):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      *,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      num_processor_repetitions: int = 1,
      node_types: Set[str],  # Set of node types to process
      edge_types: Set[str],  # Set of edge types to process
      encode_nodes: bool = True,
      encode_edges: bool = True,
      node_output_size: Optional[int] = None,
      edge_output_size: Optional[int] = None,
      include_sent_messages_in_node_update: bool = False,
      use_layer_norm: bool = True,
      name: str = "MultiEncodeProcessDecode"):
    """Inits the model.

    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of unshared message passing steps
         in the processor steps.
      num_processor_repetitions: Number of times that the same processor is
         applied sequencially.
      edge_types: Set of edge type names to process.
      encode_nodes: If False, the node encoder will be omitted.
      encode_edges: If False, the edge encoder will be omitted.
      node_output_size: Output size of the decoded node representations.
      edge_output_size: Output size of the decoded edge representations.
      include_sent_messages_in_node_update: Whether to include pooled sent
          messages from each node in the node update.
      use_layer_norm: Whether it uses layer norm or not.
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._node_types = node_types
    self._edge_types = edge_types
    self._encode_nodes = encode_nodes
    self._encode_edges = encode_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = (
        include_sent_messages_in_node_update)
    self._use_layer_norm = use_layer_norm
    self._networks_builder()

  def __call__(self, input_graph: jraph_base_models.MultiGraphsTuple) -> jraph_base_models.MultiGraphsTuple:
    """Forward pass of the learnable dynamics model."""

    # Encode the input_graph.
    latent_graph_0 = self._encode(input_graph)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)

    # Decode from the last latent graph.
    return self._decode(latent_graph_m), latent_graph_m # Return the latent graph as well for use in the RL model

  def _networks_builder(self):

    def build_mlp(name, output_size=None):
      if output_size is None:
        output_size = self._latent_size
      mlp = hk.nets.MLP(
          output_sizes=[self._mlp_hidden_size] * self._mlp_num_hidden_layers + [
              output_size], name=name + "_mlp", activation=jax.nn.relu)
      return jraph.concatenated_args(mlp)

    def build_mlp_with_maybe_layer_norm(name, output_size=None):
      network = build_mlp(name, output_size)
      if self._use_layer_norm:
        layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name=name + "_layer_norm")
        network = hk.Sequential([network, layer_norm])
      return jraph.concatenated_args(network)

    # The encoder graph network independently encodes edge and node features.
    encoder_kwargs = {}
    if self._encode_nodes:
        encoder_kwargs['embed_node_fn'] = {
            node_type: build_mlp_with_maybe_layer_norm(f"encoder_nodes_{node_type}")
            for node_type in self._node_types
        } 
    if self._encode_edges:
        encoder_kwargs['embed_edge_fn'] = {
            edge_type: build_mlp_with_maybe_layer_norm(f"encoder_edges_{edge_type}")
            for edge_type in self._edge_types
        } 
    self._encoder_network = jraph_base_models.MultiGraphMapFeatures(**encoder_kwargs)

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = []
    for step_i in range(self._num_message_passing_steps):
      # Create update functions for each edge type
      edge_fns = {
          edge_type: build_mlp_with_maybe_layer_norm(f"processor_{step_i}_edges_{edge_type}")
          for edge_type in self._edge_types
      }
      node_fns = {
          node_type: build_mlp_with_maybe_layer_norm(f"processor_{step_i}_nodes_{node_type}")
          for node_type in self._node_types
      }

      # print(f"Step {step_i} node function names:")
      # for node_type, fn in node_fns.items():
      #     print(f"{node_type}: {fn.name if hasattr(fn, 'name') else fn}")
      # print(f"Step {step_i} edge function names:")
      # for e_type, fn in edge_fns.items():
      #     print(f"{e_type}: {fn.name if hasattr(fn, 'name') else fn}")

      processor = jraph_base_models.MultiInteractionNetwork(
                update_edge_fn=edge_fns,
                update_node_fn=node_fns)
      
      self._processor_networks.append(processor)

    # The decoder MLP decodes edge/node latent features into the output sizes.
    decoder_kwargs = {}
    #decoding only for v_l nodes. everything else is identity
    decoder_kwargs['embed_node_fn'] = {
            'v_l': build_mlp("decoder_nodes_v_l", self._node_output_size) if self._node_output_size else None,
            } 

    self._decoder_network = jraph_base_models.MultiGraphMapFeatures(**decoder_kwargs)

  def _encode(
      self, input_graph: jraph_base_models.MultiGraphsTuple) -> jraph_base_models.MultiGraphsTuple:
    """Encodes the input graph features into a latent graph."""

    # Copy the globals to all of the nodes, if applicable.
    if input_graph.globals is not None:
      broadcasted_globals = jnp.repeat(
          input_graph.globals, input_graph.n_node, axis=0,
          total_repeat_length=input_graph.nodes.shape[0])
      input_graph = input_graph._replace(
          nodes=jnp.concatenate(
              [input_graph.nodes, broadcasted_globals], axis=-1),
          globals=None)

    # Encode the node and edge features.
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: jraph_base_models.MultiGraphsTuple) -> jraph_base_models.MultiGraphsTuple:
    """Processes the latent graph with several steps of message passing."""

    # Do `num_message_passing_steps` with each of the `self._processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    latent_graph = latent_graph_0
    for unused_repetition_i in range(self._num_processor_repetitions):
      for processor_network in self._processor_networks:
        latent_graph = self._process_step(processor_network, latent_graph,
                                          latent_graph_0)

    return latent_graph

  def _process_step(
      self, processor_network_k,
      latent_graph_prev_k: jraph_base_models.MultiGraphsTuple,
      latent_graph_0: jraph_base_models.MultiGraphsTuple) -> jraph_base_models.MultiGraphsTuple:
    """Single step of message passing with node/edge residual connections."""

    input_graph_k = latent_graph_prev_k

    # One step of message passing.
    latent_graph_k = processor_network_k(input_graph_k)

    # Add residuals to each node type.
    updated_nodes = {
        node_type: latent_graph_k.nodes[node_type] + latent_graph_prev_k.nodes[node_type]
        for node_type in self._node_types
    }
    # Add residuals for each edge type
    updated_edges = {
        edge_type: latent_graph_k.edges[edge_type] + latent_graph_prev_k.edges[edge_type]
        for edge_type in self._edge_types
    }
    
    latent_graph_k = latent_graph_k._replace(
        nodes=updated_nodes,
        edges=updated_edges)
    return latent_graph_k

  def _decode(self, latent_graph: jraph_base_models.MultiGraphsTuple) -> jraph_base_models.MultiGraphsTuple:
    """Decodes from the latent graph."""
    return self._decoder_network(latent_graph)
