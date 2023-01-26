'''
This script shows how to use RAIN algorithm to generate explainable embedding vectors for nodes in the graph.

Before running this script:
    Step 1: Make sure to populate config.ini
    Step 2: Give number of cores as an external argument for creating embedding as parallel processing.        
Usage:
    python3 run_rain.py <ncores>
'''

from embedding import RAIN
from metadata import *
import sys

NCORES = int(sys.argv[1])

metadata_dict = dict(
    edge_path = EDGE_PATH,
    edge_metadata_path = EDGE_METADATA_PATH,
    node_metadata_path = NODE_METADATA_PATH,
    graph_path = GRAPH_PATH,
    embedding_save_path = SAVE_DIRECTORY
)

# List of nodeIds (unique identifiers of nodes), whose RAIN embeddings need to be computed.
# Option 1: Give the list explicitly as shown here
# Option 2: Load the list from an external file
# Option 3: Give embedding_nodeId_list = "all", if embeddings of all nodes need to be computed.

embedding_nodeId_list = ['DOID:3777', 'D060050', 'DOID:950', 'DOID:1272', 'DOID:3310', 'DOID:914', 'DOID:9080', 'D004172', 'D019547', 'DOID:519', 'DOID:8442', 'D005207', 'DOID:7489', 'DOID:13450', 'DOID:2527', 'DOID:2973', 'DOID:12900', 'DOID:1023', 'DOID:11406', 'DOID:3343', 'DOID:5500', 'DOID:12128', 'DOID:9120', 'DOID:1793', 'D001766']


def main():
    rain = RAIN()
    rain.batch_embedding(
        metadata_dict = metadata_dict,
        embedding_nodeId_list = embedding_nodeId_list,
        ncores = NCORES,
        nbatch = NBATCH
    )


if __name__ == "__main__":
    main()
    