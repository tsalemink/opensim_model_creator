# file_utils.py contains helper functions designed for the retrieval, manipulation or saving of files for the opensim_model_creator
import os
import sys
import shutil


def search_files_by_keywords(folder_path, keywords):
    """
    Searches for files in a given folder that contain all the specified keywords in their names.

    Args:
        folder_path (str): Path to the folder where the search is performed.
        keywords (str): A space-separated string of keywords to match in filenames.

    Returns:
        list: A list of filenames that match all the keywords.
    """
    # Split the keywords into a list of words and convert to lowercase
    keywords_list = keywords.lower().split()

    # Get all files in the folder
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
        return []

    # Find files that match all keywords
    matching_files = [
        file for file in files
        if all(keyword in file.lower() for keyword in keywords_list)
    ]
    matching_files[0] = folder_path + "/" + matching_files[0]
    return matching_files


def clear_directory(folder_path):
    """
    `shutil.rmtree` fails for OneDrive directories.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except PermissionError as e:
                print(f"Error: {e}")


def get_results_dir():
    if getattr(sys, 'frozen', False):
        results_directory = os.path.join(os.getenv("LOCALAPPDATA"), "MusculoSkeletal", "C3D-Parser")
    else:
        results_directory = os.path.join(os.getcwd(), "opensim_results")
    os.makedirs(results_directory, exist_ok=True)

    return results_directory
