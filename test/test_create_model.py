
import os

from opensim_model_creator.Create_Model import create_model


def test(input_directory):
    absolute_path = os.path.abspath(input_directory)

    create_model(absolute_path)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05")
    test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01 Optimised")
