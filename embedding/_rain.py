import os
import pandas as pd
import numpy as np
import networkx as nx
import joblib
import multiprocessing as mp
from operator import itemgetter
import time



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
            for index, row in edge_df_.iterrows():
                graph.add_edge(row['source'], row['target'], edge_type=row["edge_type"])
            print('Saving graph ...')
            joblib.dump(graph, self.graph_path)
        self.graph = joblib.load(self.graph_path)
            
    def read_edge_data(self):
        edge_df = self.edge_df_with_attr        
        edge_df = edge_df[["source", "edge_type", "target"]]
        edge_df.source = edge_df.source.astype(str)
        edge_df.target = edge_df.target.astype(str)
#         edge_df_ = edge_df.copy()
#         edge_df_.drop("edge_type", axis=1, inplace=True)
#         edge_df_.source = edge_df_.source.astype(str)
#         edge_df_.target = edge_df_.target.astype(str)
#         edge_df_ = pd.DataFrame(np.sort(edge_df_.values, axis=1), index=edge_df_.index, columns=edge_df_.columns).drop_duplicates()
#         edge_df_ = pd.merge(edge_df_, edge_df, on=["source", "target"]).drop_duplicates(subset=["source", "target"])
        return edge_df
    
    def batch_embedding(self, metadata_dict, embedding_nodeId_list="all", ncores=1, nbatch=1, alpha=0.85):       
        time_start = time.time()
        self.edge_path = metadata_dict["edge_path"]
        self.edge_metadata_path = metadata_dict["edge_metadata_path"]
        self.graph_path = metadata_dict["graph_path"]
        self.embedding_save_path = metadata_dict["embedding_save_path"]
        self.node_metadata_path = metadata_dict["node_metadata_path"]
        self.alpha = alpha
        self.edge_df_with_attr = pd.read_csv(self.edge_path, sep='\t').drop_duplicates()
        self.load_graph()
        self.edge_metadata = pd.read_csv(self.edge_metadata_path, sep="\t")
        self.node_metadata = pd.read_csv(self.node_metadata_path, sep="\t")
        if embedding_nodeId_list == "all":            
            embedding_nodeId_list = list(self.node_metadata.Node.unique())
        embedding_nodeId_array = np.array(embedding_nodeId_list)
        embedding_nodeId_array = embedding_nodeId_array.astype(str)
        embedding_nodeId_index_batch = np.array_split(np.arange(len(embedding_nodeId_array)), nbatch)
        print("Starting batch embedding ...")
        for batch in range(nbatch):
            embedding_nodeId_index_selected = embedding_nodeId_index_batch[batch]
            embedding_nodeId_selected = embedding_nodeId_array[embedding_nodeId_index_selected]
            p = mp.Pool(ncores)
            out_dict_list = p.map(self.single_node_embedding, embedding_nodeId_selected)
            p.close()
            p.join()
            embedding_array = np.array(list(map(itemgetter('embedding'), out_dict_list)))
            print("Saving embedding array of batch {}/{}...".format(batch+1,nbatch))
            np.save(os.path.join(self.embedding_save_path, "embedding_batch_{}.npy".format(batch+1)), embedding_array, allow_pickle=False)
            del(embedding_array)
            node_list = list(map(itemgetter('node'), out_dict_list))
            feature_ids_list = list(map(itemgetter('feature_ids'), out_dict_list))[0]
            del(out_dict_list)
            embedding_map_dict = {}
            embedding_map_dict["nodes"] = node_list
            embedding_map_dict["feature_ids"] = feature_ids_list
            print("Saving embedding mapping of batch {}/{}...".format(batch+1,nbatch))
            joblib.dump(embedding_map_dict, os.path.join(self.embedding_save_path, "embedding_mapping_file_batch_{}.joblib".format(batch+1)))
            del(embedding_map_dict)
            print("Batch {}/{} completed !".format(batch+1, nbatch))
        print("Entire batch completed in {} hrs".format(round((time.time()-time_start)/(60*60), 2)))
            
                            
    def single_node_embedding(self, nodeId):
        graph_copy = self.graph
        try:
            first_nbr_of_sel_nodeId_df_ranked = self.get_first_nbr(nodeId)
            for index, row in first_nbr_of_sel_nodeId_df_ranked.iterrows():
                graph_copy[row["source"]][row["target"]]["weight"] = row["attribute_rank"]
            weight = np.zeros(self.node_metadata.shape[0],)
            node_wt_df = pd.DataFrame(list(zip(self.node_metadata.Node.values, weight)), columns=["node", "weight"])
            node_wt_df.node = node_wt_df.node.astype(str)
            node_wt_df["node_type"] = self.node_metadata.Node_Type.values
            node_wt_df.loc[node_wt_df.node==nodeId, "weight"] = 1
            personalized = pd.Series(node_wt_df.weight.values, index=node_wt_df.node).to_dict()
            personalized_pagerank = nx.pagerank(graph_copy, alpha=self.alpha, personalization=personalized, weight="weight")
            del(graph_copy)
            nodes = list(personalized_pagerank.keys())
            rank = list(personalized_pagerank.values())
            nodes_rank_df = pd.DataFrame(list(zip(nodes, rank)), columns = ["node", "page_rank"])
            nodes_rank_df = nodes_rank_df.dropna(subset=["node"])
            out_dict = {}
            out_dict["node"] = nodeId
            out_dict["embedding"] = nodes_rank_df.page_rank.values
            out_dict["feature_ids"] = nodes_rank_df.node.values
        except:
            out_dict = {}
            out_dict["node"] = nodeId
            out_dict["embedding"] = np.zeros(self.node_metadata.shape[0],)
            out_dict["feature_ids"] = None
        return out_dict
        
        
    def get_first_nbr(self, nodeId):
        first_nbr_of_sel_nodeId_df = pd.DataFrame(list(((u,v,d["edge_type"]) for u,v,d in self.graph.edges(data=True) if ((u==nodeId) | (v==nodeId)))), columns = ["source", "target","edge_type"])
        first_nbr_of_sel_nodeId_df = pd.merge(first_nbr_of_sel_nodeId_df, self.edge_metadata, on="edge_type").drop_duplicates(subset=["source", "target"])
        first_nbr_of_sel_nodeId_df_1 = pd.merge(first_nbr_of_sel_nodeId_df, self.edge_df_with_attr, on=["source", "target", "edge_type"]).rename(columns={"attribute_x":"attribute_type", "attribute_y":"attribute_value"})
        first_nbr_of_sel_nodeId_df_2 = pd.merge(first_nbr_of_sel_nodeId_df.rename(columns={"source":"target", "target":"source"}), self.edge_df_with_attr, on=["source", "target", "edge_type"]).rename(columns={"attribute_x":"attribute_type", "attribute_y":"attribute_value"})
        first_nbr_of_sel_nodeId_df_2.rename(columns={"target":"source", "source":"target"}, inplace=True)
        first_nbr_of_sel_nodeId_df = pd.concat([first_nbr_of_sel_nodeId_df_1, first_nbr_of_sel_nodeId_df_2], ignore_index=True)
        first_nbr_of_sel_nodeId_df_ranked_list = []
        unique_edge_types = first_nbr_of_sel_nodeId_df.edge_type.unique()
        for item in unique_edge_types:
            item_df = first_nbr_of_sel_nodeId_df[first_nbr_of_sel_nodeId_df.edge_type==item]
            attr_type = item_df.attribute_type.unique()[0]
            if attr_type != "unweighted":
                unique_attr_values = np.sort(item_df.attribute_value.unique())
                rank_arr = np.arange(len(unique_attr_values))
                if "pvalue" in attr_type or "reverse_weight" in attr_type:
                    unique_attr_values = unique_attr_values[::-1]
                attr_rank_df = pd.DataFrame(list(zip(unique_attr_values, rank_arr)), columns=["attribute_value", "attribute_rank"])
                item_attr_rank_df = pd.merge(item_df, attr_rank_df, on="attribute_value")
                item_attr_rank_df["attribute_rank"] = item_attr_rank_df["attribute_rank"]+1
            else:
                item_attr_rank_df = item_df
                item_attr_rank_df["attribute_rank"] = 1
            first_nbr_of_sel_nodeId_df_ranked_list.append(item_attr_rank_df)
        first_nbr_of_sel_nodeId_df_ranked = pd.concat(first_nbr_of_sel_nodeId_df_ranked_list, ignore_index=True)
        return first_nbr_of_sel_nodeId_df_ranked
            
            
        
        
        
        
    

        
    
        

        