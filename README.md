# graphRAIN

This repository holds the script for graphRAIN algorithm - algorithm that creates explainable node embeddings of a graph. 

## Introduction

GraphRAIN stands for Graph Relational Attribute Integrated Node-embedding

This generates embedding vectors for nodes in a graph by incorporating the attributes hanging from the relationship between the nodes. Since the algorithm normalizes the relational attributes, this works for both heterogeneous and homogeneous graphs.

## How to use

Following snippet shows how to use this package:

```
from embedding import RAIN

NCORES = <number of cores for parallel processing>
NBATCH = <in how many batches should the final embedding matrix to be saved>

metadata_dict = dict(
    edge_path = /path/to/edges.tsv,
    edge_metadata_path = /path/to/edge_metadata.tsv,
    node_metadata_path = /path/to/node_metadata.tsv,
    embedding_save_path = /path/to/save/output/files,
    graph_path = embedding_save_path/name_of_graph.joblib,    
)

embedding_nodeId_list = List of unique nodeIds of graph whose embedding vectors need to be computed

rain = RAIN()
rain.batch_embedding(
        metadata_dict = metadata_dict,
        embedding_nodeId_list = embedding_nodeId_list,
        ncores = NCORES,
        nbatch = NBATCH
    )
```
**Refer to [run_rain.py](https://github.com/BaranziniLab/graphRAIN/blob/main/run_rain.py) for a full example which you can run in your machine.**
