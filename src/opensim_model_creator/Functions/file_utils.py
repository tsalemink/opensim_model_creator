# file_utils.py contains helper functions designed for the retrieval, manipulation or saving of files for the opensim_model_creator
import os
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


def reset_folder(folder_path):
    """
    Deletes the folder if it exists and recreates it as an empty directory.

    Parameters:
    - folder_path (str): Path to the folder to reset.
    """
    if os.path.exists(folder_path):  # Check if folder exists
        shutil.rmtree(folder_path)  # Delete the entire folder
    os.makedirs(folder_path, exist_ok=True)  # Recreate the empty folder