
import os

from opensim_model_creator.Create_Model import create_model


def test(input_directory):
    absolute_path = os.path.abspath(input_directory)


    # For creation of muscles, include True statement
    create_model(absolute_path, True)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01 Optimised")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 02 Optimised")