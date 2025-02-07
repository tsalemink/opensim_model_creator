#Import packages
import numpy as np
import tkinter as tk
from tkinter import filedialog
import re
import pandas as pd

def rotate_coordinate_x(coord, angle_degrees):
    """
    Rotates a 3D coordinate around the X-axis by a given angle.

    Args:
        coord (list or tuple): The 3D coordinate (x, y, z) to rotate.
        angle_degrees (float): The angle in degrees to rotate.

    Returns:
        np.ndarray: The rotated 3D coordinate as a NumPy array.
    """
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix for X-axis
    rotation_matrix = np.array([
        [-1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Rotate the coordinate
    rotated_coord = np.dot(rotation_matrix, coord)
    return rotated_coord

def midpoint_3d(coord1, coord2):
    """
    Calculate the midpoint between two 3D coordinates.

    Args:
        coord1 (tuple or list or np.ndarray): The first 3D coordinate (x1, y1, z1).
        coord2 (tuple or list or np.ndarray): The second 3D coordinate (x2, y2, z2).

    Returns:
        np.ndarray: Midpoint (x, y, z) as a numpy array.
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    return (coord1 + coord2) / 2

def vector_between_points(coord1, coord2, normalize=False):
    """
    Calculate the vector between two 3D coordinates and optionally normalize it.

    Args:
        coord1 (tuple or list or np.ndarray): The first 3D coordinate (x1, y1, z1).
        coord2 (tuple or list or np.ndarray): The second 3D coordinate (x2, y2, z2).
        normalize (bool): Whether to normalize the resulting vector. Default is True.

    Returns:
        np.ndarray: Vector as a numpy array (normalized if specified).
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    vector = coord2 - coord1

    if normalize:
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            raise ValueError("Cannot normalize a zero vector.")
        vector = vector / magnitude

    return vector

def select_directory():
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the directory selection dialog
    selected_directory = filedialog.askdirectory(title="Select a Participent Directory to produce osim model")

    # Return the selected directory
    return selected_directory

def read_trc_file_as_dict(file_path, include_times=False):
    """
    Reads a .trc file and parses marker data into a dictionary where each marker
    has its X, Y, and Z coordinates as arrays.

    Args:
        file_path (str): Path to the .trc file.
        include_times (bool): If True, returns the start and end times.

    Returns:
        dict: A dictionary containing the average marker positions.
        tuple (optional): (start_time, end_time) if include_times is True.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse metadata
    metadata = {}
    metadata["FileType"] = lines[0].strip()
    header_info = lines[1].strip().split("\t")
    metadata_values = lines[2].strip().split("\t")
    for key, value in zip(header_info, metadata_values):
        try:
            metadata[key] = float(value)
        except ValueError:
            metadata[key] = value

    # Parse column headers
    column_headers = lines[3].strip().split("\t")
    column_headers = [header for header in column_headers if header]  # Remove empty entries
    column_headers_additional = column_headers[:2]  # First two entries are Frame# and Time
    column_headers = column_headers[2:]  # Remaining headers for markers

    # Parse sub headers
    sub_headers = lines[4].strip().split("\t")

    # Combine headers to form unique names
    full_headers = []
    sub_header_index = 0  # Tracks the position in the sub_headers list
    for main in column_headers:
        for _ in range(3):  # Loop through three subheaders for each main header (X, Y, Z)
            if sub_header_index < len(sub_headers):
                sub = sub_headers[sub_header_index]

                # Remove the number attached to the sub-header using regex
                sub = re.sub(r'\d+$', '', sub)  # Removes trailing digits

                full_headers.append(f"{main}_{sub}")
                sub_header_index += 1
            else:
                break  # Stop if sub_headers are exhausted

    # Add the additional headers (Frame# and Time) back to the full headers
    full_headers = column_headers_additional + full_headers

    # Parse marker data
    data_start_idx = 5  # Row index where actual data begins
    marker_data = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=data_start_idx,
        names=full_headers,
    )

    marker_data_cols = list(marker_data.columns[1:])
    marker_data.drop(marker_data.columns[-1], axis=1, inplace=True)
    marker_data.columns = marker_data_cols

    # Extract time information if requested
    start_time, end_time = None, None
    if include_times and "Time" in marker_data:
        start_time = marker_data["Time"].iloc[0]
        end_time = marker_data["Time"].iloc[-1]

    # Transform marker data into a dictionary
    markers_dict = {}
    for marker in set(col.split("_")[0] for col in full_headers if "_" in col):
        markers_dict[marker] = {
            "X": marker_data.get(f"{marker}_X", pd.Series()).values,
            "Y": marker_data.get(f"{marker}_Y", pd.Series()).values,
            "Z": marker_data.get(f"{marker}_Z", pd.Series()).values,
        }

    # Compute the average of each marker's X, Y, and Z values
    marker_static_avg = {}
    for marker, coords in markers_dict.items():
        avg_x = np.nanmean(coords["X"]) if len(coords["X"]) > 0 else None
        avg_y = np.nanmean(coords["Y"]) if len(coords["Y"]) > 0 else None
        avg_z = np.nanmean(coords["Z"]) if len(coords["Z"]) > 0 else None

        # Store the averages as a tuple
        marker_static_avg[marker] = (avg_x / 1000, avg_y / 1000, avg_z / 1000)

    if include_times:
        return marker_static_avg, (start_time, end_time),markers_dict
    return marker_static_avg,markers_dict