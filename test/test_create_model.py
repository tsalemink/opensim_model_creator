
import os
import pickle

from opensim_model_creator.Create_Model import create_model


def test(input_directory, height, weight):
    absolute_path = os.path.abspath(input_directory)

    # Read in dictionary of static marker data.
    marker_data_path = os.path.join(input_directory, "Inputs", "static.pkl")
    with open(marker_data_path, "rb") as f:
        static_marker_data = pickle.load(f)

    create_model(absolute_path, static_marker_data, height=height, weight=weight, create_muscles=True, testing=True)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05", 159.1, 40.8)
    test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01", 136.3, 32.9)
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 02", 117.9, 23)
