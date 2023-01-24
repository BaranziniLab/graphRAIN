import os
import joblib
from backend.paths import GRAPH_PATH


def main():
    print("Loading graph ...")
    graph = load_graph(GRAPH_PATH)
    print("Graph is loaded !")

def load_graph(GRAPH_PATH):
    if not os.path.exists(GRAPH_PATH):
        from backend.create_graph import create_graph
        create_graph()
    return joblib.load(GRAPH_PATH)

if __name__ == "__main__":
    main()
    