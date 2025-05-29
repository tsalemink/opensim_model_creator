
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import numpy as np

from opensim_model_creator.Create_Model import create_model
from opensim_model_creator.Functions.general_utils import rotate_coordinate_x
import time
def test(input_directory, height, weight):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "data", input_directory)

    static_trc = os.path.join(data_directory, "Inputs", "static.trc")
    dynamic_trc = os.path.join(data_directory, "Inputs", "kneeoptimisation.trc")
    output_directory = os.path.join(data_directory, "_output")

    # Read in dictionary of static marker data.
    marker_data_path = os.path.join(data_directory, "Inputs", "static.pkl")
    with open(marker_data_path, "rb") as f:
        static_marker_data = pickle.load(f)

    ## rotate static marker data to match opensim coordinate system (remove later)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    for lm in static_marker_data:
        static_marker_data[lm] = np.dot(rotation_matrix, static_marker_data[lm])

    create_model(static_trc, dynamic_trc, output_directory, static_marker_data, weight, height, create_muscles=False, testing=True)


if __name__ == "__main__":
    start_time = time.time()
    #test("Sydney 01", 1.634, 53.5)
    #test("Brittney 05", 1.591, 40.8),
    test("Jinella 01", 1.363, 32.9),
    #test("Jinella 02", 1.179, 23)
    end_time = time.time()
    runtime_seconds = end_time - start_time
    print(f"Optimization completed in {runtime_seconds:.2f} seconds.")

    # # Define test cases as (directory, height, weight) tuples
    # test_cases = [
    #     ("Sydney 01", 1.634, 53.5),
    #     ("Brittney 05", 159.1, 40.8),
    #     ("Jinella 01", 136.3, 32.9),
    #     ("Jinella 02", 117.9, 23)
    # ]
    #
    # # Use ProcessPoolExecutor for parallel execution
    # max_workers = os.cpu_count() or 4  # Default to the number of CPU cores
    # print(f"Starting with {max_workers} parallel processes...")
    #
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     # Submit all test cases to the executor
    #     futures = {executor.submit(test, *case): case for case in test_cases}
    #
    #     # As each future completes, log the result
    #     for future in as_completed(futures):
    #         case = futures[future]
    #         try:
    #             future.result()  # Will raise any exceptions that occurred
    #             print(f"\033[92m✅ Test completed successfully for {case[0]}\033[0m")
    #         except Exception as e:
    #             print(f"\033[91m❌ Test failed for {case[0]}: {e}\033[0m")
