from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import numpy as np
import pandas as pd

from opensim_model_creator.Create_Model import create_model
from opensim_model_creator.Functions.general_utils import rotate_coordinate_x
import time


def test(input_directory, subject_info):
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

    create_model(static_trc, dynamic_trc, output_directory, static_marker_data, subject_info, 9)


if __name__ == "__main__":
    start_time = time.time()
    #test("Sydney 01", 1.634, 53.5)
    #test("Brittney 05", 159.1, 40.8),
    test("ID_problems", subject_info=pd.DataFrame({
        "Age": [12],
        "Height": [109.8],
        "Mass": [16.2],
        "Sex": [1],
        "ASIS_width": [146.8],
        "left_epicon_width": [64.2],
        "left_malleolar_width": [47.8],
        "right_epicon_width": [65.4],
        "right_malleolar_width": [47.0],
    }))
    #test("RCH000010", subject_info=pd.DataFrame({
    #    "Age": [18],
    #    "Height": [175.8],
    #    "Mass": [79],
    #    "Sex": [1],
    #   "ASIS_width": [240],
    #    "left_epicon_width": [101],
    #    "left_malleolar_width": [65],
    #    "right_epicon_width": [101],
    #    "right_malleolar_width": [65],
    #}))
    #test("Jinella 01", 1.363, 32.9),
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
