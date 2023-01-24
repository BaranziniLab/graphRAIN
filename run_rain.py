'''
This script shows how to use RAIN algorithm to generate explainable embedding vectors for nodes in the graph.

Before running this script:
    Step 1: Make sure to populate config.ini
    Step 2: This script runs in parallel. Hence, give number of cores as external argument.
    
Usage:
    python3 run_rain.py <ncores>
'''

from metadata import EDGE_PATH, GRAPH_PATH
from embedding import RAIN 


def main():
    rain = RAIN(EDGE_PATH, GRAPH_PATH)
    rain.load_graph()


if __name__ == "__main__":
    main()
    