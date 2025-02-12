
import os

from opensim_model_creator.Create_Model import create_model


def test(input_directory):
    absolute_path = os.path.abspath(input_directory)


    # For creation of muscles, include True statement, for testing purposes include a true statement (sets knee joint optimisation iteration_count to 1, to speed up development)
    create_model(absolute_path, create_muscles= True, testing = True)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney integration test")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01 Optimised")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 02 Optimised")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05")