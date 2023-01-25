'''
This script shows how to use RAIN algorithm to generate explainable embedding vectors for nodes in the graph.

Before running this script:
    Step 1: Make sure to populate config.ini
    Step 2: This script runs in parallel. Hence, give number of cores as external argument.    
    
Usage:
    python3 run_rain.py <ncores>
'''

from embedding import RAIN
from metadata import *
import sys

NCORES = int(sys.argv[1])

metadata_dict = dict(
    edge_path = EDGE_PATH,
    graph_path = GRAPH_PATH,
    embedding_save_path = SAVE_DIRECTORY
)


def main():
    rain = RAIN()
    rain.batch_embedding(
        metadata_dict = metadata_dict,
        ncores = NCORES,
        nbatch = NBATCH
    )


if __name__ == "__main__":
    main()
    