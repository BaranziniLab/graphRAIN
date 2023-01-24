import os
import joblib
from ._utility import read_edge_data, create_graph



class RAIN:
    def __init__(self, edge_path, graph_path):
        self.edge_path = edge_path
        self.graph_path = graph_path
        
    def load_graph(self):
        if not os.path.exists(self.graph_path):
            create_graph(self.edge_path, self.graph_path)
        self.graph = joblib.load(self.graph_path)
        
    
        

        