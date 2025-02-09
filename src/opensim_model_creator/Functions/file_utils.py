# file_utils.py contains helper functions designed for the retrieval, manipulation or saving of files for the opensim_model_creator
import os
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd

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

def select_directory():
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the directory selection dialog
    selected_directory = filedialog.askdirectory(title="Select a Participent Directory to produce osim model")

    # Return the selected directory
    return selected_directory

def get_participant_directories():
    """
    Prompts the user to select a participant directory and returns the path
    to the corresponding "Inputs" subdirectory.

    Returns:
        str: Path to the "Inputs" subdirectory if a directory is selected, otherwise None.
    """
    participant_folder = select_directory()
    if participant_folder:
        print(f"Selected directory: {participant_folder}")
        return participant_folder, os.path.join(participant_folder, "Inputs"), os.path.join(participant_folder, "Models"), os.path.join(participant_folder, "Meshes")
    else:
        print("No directory selected.")
        return None



