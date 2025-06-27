"""A library of Graph Neural Network models."""

from typing import Any, Callable, Iterable, Mapping, Optional, Union, Dict, NamedTuple

import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import utils
import haiku as hk

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# Type definitions
DictTree = Dict[str, Optional[ArrayTree]]
ReceiversDict = Dict[str, jnp.ndarray]
SendersDict = Dict[str, jnp.ndarray] 
EdgeCountDict = Dict[str, jnp.ndarray]

class MultiGraphsTuple(NamedTuple):
    """GraphsTuple modified to support multiple edge sets."""
    nodes: DictTree
    edges: DictTree  # Dictionary of edge_type -> edge_features
    receivers: ReceiversDict  # Dictionary of edge_type -> receiver indices
    senders: SendersDict  # Dictionary of edge_type -> sender indices
    globals: Optional[ArrayTree]
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray   # should beDictionary of edge_type -> number of edges.. for now array



def MultiGraphNetwork(
    update_edge_fn: Dict[str, Callable],  # Dictionary of edge_type -> edge update function
    update_node_fn: Dict[str, Callable],
    update_global_fn: Optional[Callable] = None,
    aggregate_edges_for_nodes_fn: Callable = utils.segment_sum,
    aggregate_nodes_for_globals_fn: Callable = utils.segment_sum,
    aggregate_edges_for_globals_fn: Callable = utils.segment_sum):
    """Returns a method that applies a Graph Network supporting multiple edge types.

    Args:
        update_edge_fn: Dictionary mapping edge types to their update functions
        update_node_fn: Function to update node features
        update_global_fn: Function to update global features
        aggregate_edges_for_nodes_fn: Function to aggregate edges to nodes
        aggregate_nodes_for_globals_fn: Function to aggregate nodes to globals
        aggregate_edges_for_globals_fn: Function to aggregate edges to globals
    """
    
    def _ApplyMultiGraphNet(graph: MultiGraphsTuple) -> MultiGraphsTuple:
        """Applies the Graph Network to a multigraph."""
        nodes_dict, edges_dict, receivers_dict, senders_dict, globals_, n_node, n_edge = graph
        sum_n_node_vl = tree.tree_leaves(nodes_dict['v_l'])[0].shape[0]
        sum_n_node_vm = tree.tree_leaves(nodes_dict['v_m'])[0].shape[0]
        sum_n_node_vo = tree.tree_leaves(nodes_dict['v_o'])[0].shape[0]

        
        def update_edges_helper(_edge_type, sender_node_type, receiver_node_type):
            edge_update_fn = update_edge_fn[_edge_type]
            edge_features = edges_dict[_edge_type]
            sender_nodes = tree.tree_map(lambda n: n[senders_dict[_edge_type]], nodes_dict[sender_node_type])
            receiver_nodes = tree.tree_map(lambda n: n[receivers_dict[_edge_type]], nodes_dict[receiver_node_type])
            
            return edge_update_fn(edge_features, sender_nodes, receiver_nodes)
        
        # Update edges for each edge type
        updated_edges = {}
        updated_edges['e_l'] = update_edges_helper(_edge_type='e_l', sender_node_type='v_l', receiver_node_type='v_l')
        updated_edges['e_mo'] = update_edges_helper(_edge_type='e_mo', sender_node_type='v_m', receiver_node_type='v_o')
        updated_edges['e_om'] = update_edges_helper(_edge_type='e_om', sender_node_type='v_o', receiver_node_type='v_m')
        updated_edges['e_ol'] = update_edges_helper(_edge_type='e_ol', sender_node_type='v_o', receiver_node_type='v_l')


        #update nodes
        updated_nodes = {}

        #v_m
        node_update_fn_vm = update_node_fn['v_m']
        receivers = receivers_dict['e_om']
        received_features = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node_vm), updated_edges['e_om'])
        updated_nodes['v_m'] = node_update_fn_vm(nodes_dict['v_m'], received_features)

        #v_o
        node_update_fn_vo = update_node_fn['v_o']
        receivers = receivers_dict['e_mo']
        received_features = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node_vo), updated_edges['e_mo'])
        updated_nodes['v_o'] = node_update_fn_vo(nodes_dict['v_o'], received_features)

        #v_l
        node_update_fn_vl = update_node_fn['v_l']
        receivers_l, receivers_ol = receivers_dict['e_l'], receivers_dict['e_ol']
        received_features_l = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(e, receivers_l, sum_n_node_vl), updated_edges['e_l'])
        received_features_ol = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(e, receivers_ol, sum_n_node_vl), updated_edges['e_ol'])
        updated_nodes['v_l'] = node_update_fn_vl(nodes_dict['v_l'], received_features_l, received_features_ol)


        return MultiGraphsTuple(
            nodes=updated_nodes,
            edges=updated_edges,
            receivers=receivers_dict,
            senders=senders_dict,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge)

    return _ApplyMultiGraphNet


def MultiInteractionNetwork(
    update_edge_fn: Dict[str, Callable],  # Dictionary of edge_type -> edge update function
    update_node_fn: Dict[str, Callable],
    aggregate_edges_for_nodes_fn: Callable = utils.segment_sum):
    """Returns an Interaction Network supporting multiple edge types.
    
    Args:
        update_edge_fn: Dictionary mapping edge types to their update functions
        update_node_fn: Function to update node features
        aggregate_edges_for_nodes_fn: Function to aggregate edges to nodes
    """

    # Wrap edge update functions to ignore globals
    wrapped_edge_fns = {
        edge_type: lambda e, s, r: edge_fn(e, s, r)
        for edge_type, edge_fn in update_edge_fn.items()
    }

    wrapped_node_fns = {
            'v_m': lambda n, r: update_node_fn['v_m'](jnp.concatenate([n, r], axis=-1)),
            'v_o': lambda n, r: update_node_fn['v_o'](jnp.concatenate([n, r], axis=-1)),
            'v_l': lambda n, r1, r2: update_node_fn['v_l'](jnp.concatenate([n, r1, r2], axis=-1))
        }
    
    return MultiGraphNetwork(
        update_edge_fn=wrapped_edge_fns,
        update_node_fn=wrapped_node_fns,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)



def MultiGraphMapFeatures(
    embed_edge_fn: Optional[Dict[str, Callable]] = None,
    embed_node_fn: Optional[Dict[str, Callable]] = None,
    embed_global_fn: Optional[Callable] = None):
    """Returns function which embeds components of a multigraph independently.
    
    Args:
        embed_edge_fn: Dictionary mapping edge types to their embedding functions
        embed_node_fn: Function to embed node features
        embed_global_fn: Function to embed global features
    """
    identity = lambda x: x
    embed_global_fn = embed_global_fn if embed_global_fn else identity

    def Embed(graphs_tuple: MultiGraphsTuple) -> MultiGraphsTuple:
        # Embed nodes for each edge type
        if embed_node_fn is not None:
            embedded_nodes = {
                node_type: embed_node_fn.get(node_type, identity)(nodes)
                for node_type, nodes in graphs_tuple.nodes.items()
            }
        else:
            embedded_nodes = graphs_tuple.nodes

        # Embed edges for each edge type
        if embed_edge_fn is not None:
            embedded_edges = {
                edge_type: embed_edge_fn.get(edge_type, identity)(edges)
                for edge_type, edges in graphs_tuple.edges.items()
            }
        else:
            embedded_edges = graphs_tuple.edges

        return MultiGraphsTuple(
            nodes=embedded_nodes,
            edges=embedded_edges,
            receivers=graphs_tuple.receivers,
            senders=graphs_tuple.senders, 
            globals=embed_global_fn(graphs_tuple.globals),
            n_node=graphs_tuple.n_node,
            n_edge=graphs_tuple.n_edge)

    return Embed