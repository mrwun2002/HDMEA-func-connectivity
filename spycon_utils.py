import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import sys
import os
import pandas as pd
import spycon

def draw_graph_with_mapping(
        results: spycon.spycon_result.SpikeConnectivityResult,
        mapping: pd.DataFrame = None,
        graph_type: str = "binary",
        ax: plt.Axes = None,
        cax: plt.Axes = None,
    ):
    """
    Draw a NetworkX graph from the results.

    Args:
        results
        mapping (pd.DataFrame, optional): Dataframe mapping channels to positions.
        graph_type (str): Type of the graph to create:
            - 'binary': Creates an unweighted graph with the inferred connections. (Default)
            - 'stats': Creates a fully connected graph, with the decision stats as edge weights.
            - 'weighted': Creates a weighted graph, where weights are the inferred strengths.
        ax (plt.Axes, optional): Matplotlib axis in which the graph should be plotted. Default is None.
        cax (plt.Axes, optional): Matplotlib axis in which the colorbar should be plotted. Default is None.

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the connectivity.
    """
    graph = results.create_nx_graph(graph_type=graph_type)

    pos = nx.circular_layout(graph)
    for i in mapping.index:
        pos[i] = np.array(mapping.loc[i, ["x", "y"]])


    if ax is None:
        ax = plt.gca()
    if graph_type == "binary":
        nx.draw(
            graph, pos, ax=ax, with_labels=True, node_size=500, node_color="C1"
        )
    elif graph_type == "stats":
        cmap = plt.get_cmap("inferno_r")
        weights = list(nx.get_edge_attributes(graph, "weight").values())
        min_weight, max_weight = np.amin(weights), np.amax(weights)
        nx.draw(
            graph,
            pos, 
            ax=ax,
            with_labels=True,
            node_size=300,
            node_color="C1",
            edge_color=weights,
            edge_cmap=cmap,
            edge_vmin=min_weight,
            edge_vmax=max_weight,
        )
        norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            cax=cax,
            label="stats",
        )
    elif graph_type == "weighted":
        cmap = plt.get_cmap("BrBG")
        weights = list(nx.get_edge_attributes(graph, "weight").values())
        max_weight = np.amax(np.absolute(weights))
        nx.draw(
            graph,
            pos, 
            ax=ax,
            with_labels=True,
            node_size=300,
            node_color="C1",
            edge_color=weights,
            edge_vmin=-max_weight,
            edge_vmax=max_weight,
            edge_cmap=cmap,
        )
        norm = mpl.colors.Normalize(vmin=-max_weight, vmax=max_weight)
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            cax=cax,
            label="weights",
        )
    return graph