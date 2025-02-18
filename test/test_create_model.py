
import os

from opensim_model_creator.Create_Model import create_model


def test(input_directory):
    absolute_path = os.path.abspath(input_directory)


    # For creation of muscles, include True statement, for testing purposes include a true statement (sets knee joint optimisation iteration_count to 1, to speed up development)
    #included a scale_factor for the stl meshes that are brought from lauras code as currently usure if she is scaling or not
    create_model(absolute_path, height=1.4, weight=40, create_muscles=True, testing=True, scale_factor=1000)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01")
    #test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 02")
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05")
