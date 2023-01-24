import os
import networkx as nx
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from paths import EDGE_PATH,GRAPH_PATH


def read_edge_data(edge_path):
    edge_df = pd.read_csv(edge_path, sep='\t', names=['source', 'edge_type', 'target']).drop_duplicates()
    edge_df.source = edge_df.source.astype(str)
    edge_df.target = edge_df.target.astype(str)
    edge_df_ = edge_df.copy()
    edge_df_.drop("edge_type", axis=1, inplace=True)
    edge_df_.source = edge_df_.source.astype(str)
    edge_df_.target = edge_df_.target.astype(str)
    edge_df_ = pd.DataFrame(np.sort(edge_df_.values, axis=1), index=edge_df_.index, columns=edge_df_.columns).drop_duplicates()
    edge_df_ = pd.merge(edge_df_, edge_df, on=["source", "target"]).drop_duplicates(subset=["source", "target"])
    return edge_df_

def create_graph():
    edge_df_ = read_edge_data(EDGE_PATH)
    print('Initiating networkx graph instance ...')
    graph = nx.Graph()
    print('Adding edges to the networkx graph instance ...')
    for index, row in tqdm(edge_df_.iterrows()):
        graph.add_edge(row['source'], row['target'], edge_type=row["edge_type"])  
    print('Saving networkx graph instance ...')
    joblib.dump(graph, GRAPH_PATH)

