import os
import pandas as pd
import numpy as np
import networkx as nx
import joblib
from tqdm import tqdm



class RAIN:
    def __init__(self, edge_path=None, graph_path=None, embedding_save_path=None):
        self.edge_path = edge_path
        self.graph_path = graph_path
        self.embedding_save_path = embedding_save_path
        
    def load_graph(self):
        if not os.path.exists(self.graph_path):
            edge_df_ = self.read_edge_data()
            print('Initiating graph instance ...')
            graph = nx.Graph()
            print('Adding edges to graph ...')
            for index, row in tqdm(edge_df_.iterrows()):
                graph.add_edge(row['source'], row['target'], edge_type=row["edge_type"])
            print('Saving graph ...')
            joblib.dump(graph, self.graph_path)
        self.graph = joblib.load(self.graph_path)
            
    def read_edge_data(self):
        edge_df = pd.read_csv(self.edge_path, sep='\t').drop_duplicates()        
        edge_df = edge_df[["source", "edge_type", "target"]]
        edge_df.source = edge_df.source.astype(str)
        edge_df.target = edge_df.target.astype(str)
        edge_df_ = edge_df.copy()
        edge_df_.drop("edge_type", axis=1, inplace=True)
        edge_df_.source = edge_df_.source.astype(str)
        edge_df_.target = edge_df_.target.astype(str)
        edge_df_ = pd.DataFrame(np.sort(edge_df_.values, axis=1), index=edge_df_.index, columns=edge_df_.columns).drop_duplicates()
        edge_df_ = pd.merge(edge_df_, edge_df, on=["source", "target"]).drop_duplicates(subset=["source", "target"])
        return edge_df_
    
    def batch_embedding(self, metadata_dict, ncores=1, nbatch=1):
        self.edge_path = metadata_dict["edge_path"]
        self.graph_path = metadata_dict["graph_path"]
        self.embedding_save_path = metadata_dict["embedding_save_path"]
        self.load_graph()
        
        
        
        
    

        
    
        

        