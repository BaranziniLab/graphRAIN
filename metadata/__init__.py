from configparser import ConfigParser
import os

config_path = 'config.ini'
parser = ConfigParser()
parser.read([config_path])

input_data_path_dict = dict(parser.items("INPUT_DATA"))
EDGE_PATH = input_data_path_dict["edge_path"]
EDGE_METADATA_PATH = input_data_path_dict["edge_metadata_path"]
NODE_METADATA_PATH = input_data_path_dict["node_metadata_path"]

output_data_path_dict = dict(parser.items("OUTPUT_DATA"))
SAVE_DIRECTORY = output_data_path_dict["save_directory"]
GRAPH_NAME = output_data_path_dict["graph_name"]
GRAPH_PATH = os.path.join(SAVE_DIRECTORY, GRAPH_NAME+".joblib")
NBATCH = int(output_data_path_dict["nbatch"])


__all__ = [
    "EDGE_PATH",
    "EDGE_METADATA_PATH",
    "NODE_METADATA_PATH",
    "GRAPH_PATH",
    "SAVE_DIRECTORY",
    "NBATCH"
]