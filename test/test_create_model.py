
import os

from opensim_model_creator.Create_Model import create_model


def test(input_directory, height, weight):
    absolute_path = os.path.abspath(input_directory)


    # For creation of muscles, include True statement, for testing purposes include a true statement (sets knee joint optimisation iteration_count to 1, to speed up development)
    #included a scale_factor for the stl meshes that are brought from lauras code as currently usure if she is scaling or not
    create_model(absolute_path, height=height, weight=weight, create_muscles=False, testing=True, scale_factor=1000)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\..\\src\\opensim_model_creator\\Participants\\Brittney 05", 159.1, 40.8)
    test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 01", 136.3, 32.9)
    test("..\\..\\src\\opensim_model_creator\\Participants\\Jinella 02", 117.9, 23)
