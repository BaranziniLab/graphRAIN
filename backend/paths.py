from configparser import ConfigParser
import os

config_path = os.path.join(os.path.dirname(os.getcwd()), 'config.ini')
parser = ConfigParser()
parser.read([config_path])

input_data_path_dict = dict(parser.items("INPUT_DATA"))
EDGE_PATH = input_data_path_dict["edge_path"]

output_data_path_dict = dict(parser.items("OUTPUT_DATA"))
SAVE_DIRECTORY = output_data_path_dict["save_directory"]
GRAPH_NAME = output_data_path_dict["graph_name"]
GRAPH_PATH = os.path.join(SAVE_DIRECTORY, GRAPH_NAME+".joblib")


