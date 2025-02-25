
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle

from opensim_model_creator.Create_Model import create_model


def test(input_directory, height, weight):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "data", input_directory)

    # Read in dictionary of static marker data.
    marker_data_path = os.path.join(data_directory, "Inputs", "static.pkl")
    with open(marker_data_path, "rb") as f:
        static_marker_data = pickle.load(f)

    create_model(data_directory, static_marker_data, height=height, weight=weight, create_muscles=True, testing=True)


if __name__ == "__main__":
    # Define test cases as (directory, height, weight) tuples
    test_cases = [
        ("Brittney 05", 159.1, 40.8),
        ("Jinella 01", 136.3, 32.9),
        ("Jinella 02", 117.9, 23)
    ]

    # Use ProcessPoolExecutor for parallel execution
    max_workers = os.cpu_count() or 4  # Default to the number of CPU cores
    print(f"Starting with {max_workers} parallel processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all test cases to the executor
        futures = {executor.submit(test, *case): case for case in test_cases}

        # As each future completes, log the result
        for future in as_completed(futures):
            case = futures[future]
            try:
                future.result()  # Will raise any exceptions that occurred
                print(f"\033[92m✅ Test completed successfully for {case[0]}\033[0m")
            except Exception as e:
                print(f"\033[91m❌ Test failed for {case[0]}: {e}\033[0m")
