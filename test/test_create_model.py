
from opensim_model_creator.Create_Model import create_model


def test(input_directory):
    create_model(input_directory)


if __name__ == "__main__":
    # TODO: Move test data into test directory.
    test("..\\src\\opensim_model_creator\\Participants\\Brittney 05")
    test("..\\src\\opensim_model_creator\\Participants\\Jinella 01 Optimised")
