#Import packages
import numpy as np
import re
import pandas as pd
import os
import trimesh
from tkinter import Tk
from tkinter.filedialog import askdirectory
import shutil

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

def compute_marker_midpoint(model, marker1_name, marker2_name):
    """
    Computes the midpoint between two markers in an OpenSim model.

    Parameters:
        model (osim.Model): The OpenSim model containing the markers.
        marker1_name (str): Name of the first marker.
        marker2_name (str): Name of the second marker.

    Returns:
        np.array: The 3D midpoint of the two marker locations.
    """
    # Retrieve the marker set from the model
    marker_set = model.getMarkerSet()

    # Check if both markers exist in the model
    if not marker_set.contains(marker1_name) or not marker_set.contains(marker2_name):
        raise ValueError(f"Markers '{marker1_name}' or '{marker2_name}' not found in the model.")

    # Retrieve the markers
    marker1 = marker_set.get(marker1_name)
    marker2 = marker_set.get(marker2_name)

    # Get their local positions and convert to NumPy arrays
    marker1_position = np.array([marker1.get_location().get(i) for i in range(3)])
    marker2_position = np.array([marker2.get_location().get(i) for i in range(3)])

    # Compute the midpoint using the existing function
    midpoint = midpoint_3d(marker1_position, marker2_position)

    return midpoint

def convert_and_scale_mesh(input_ply, output_stl, scale_factor=1000):
    """
    Converts a .ply mesh to .stl format and scales it down by a given factor.

    Parameters:
    - input_ply (str): Path to the input .ply file.
    - output_stl (str): Path to save the output .stl file.
    - scale_factor (float): The factor by which to scale the mesh (default: 1000).

    Returns:
    - None
    """
    # Load the .ply file
    mesh = trimesh.load(input_ply)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file is not a valid mesh.")

    # Scale the mesh
    mesh.apply_scale(1 / scale_factor)

    # Export the scaled mesh to .stl format
    mesh.export(output_stl, file_type="stl")
    print(f"Converted and scaled mesh saved to: {output_stl}")

def batch_convert_and_scale(scale_factor=1000, input_dir=None):
    """
    Converts all .ply files in the specified or selected directory to .stl format while scaling them.
    Deletes any existing .stl files in the input directory before processing.

    Parameters:
    - scale_factor (float): The factor by which to scale the meshes (default: 1000).
    - input_dir (str): Optional path to the input directory containing .ply files.
                       If not provided, a file dialog will prompt the user to select a directory.

    Returns:
    - None
    """
    if input_dir is None:
        # Prompt the user to select the input directory
        Tk().withdraw()  # Hide the root Tk window
        input_dir = askdirectory(title="Select Directory Containing .ply Files")
        if not input_dir:
            print("No directory selected. Exiting.")
            return

    # Check if the directory exists
    if not os.path.isdir(input_dir):
        print(f"The directory '{input_dir}' does not exist. Exiting.")
        return

    # Delete all existing .stl files in the directory
    stl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".stl")]
    for stl_file in stl_files:
        try:
            os.remove(os.path.join(input_dir, stl_file))
            print(f"Deleted existing .stl file: {stl_file}")
        except Exception as e:
            print(f"Failed to delete {stl_file}: {e}")

    # Process each .ply file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".ply"):
            input_ply = os.path.join(input_dir, file_name)
            output_stl = os.path.join(input_dir, file_name.replace(".ply", ".stl"))
            try:
                convert_and_scale_mesh(input_ply, output_stl, scale_factor)
                print(f"Converted and scaled {file_name} to {output_stl}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

def combine_pelvis_meshes(input_dir, output_path=None):
    """
    Searches an input directory for two STL files containing the word 'pelvis',
    combines them into a single mesh, saves the combined mesh to the specified output file,
    and deletes the original STL files.

    Parameters:
    - input_dir (str): Directory to search for the STL files.
    - output_path (str, optional): Path to save the combined mesh. Defaults to 'combined_pelvis_mesh.stl' in the input directory.

    Returns:
    - None
    """
    # Find all STL files containing "pelvis" in their names
    pelvis_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".stl") and "pelvis" in f.lower()
    ]

    # Check if exactly two files are found
    if len(pelvis_files) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 STL files containing 'pelvis' in their names, but found {len(pelvis_files)}."
        )

    # Load the meshes
    mesh1 = trimesh.load(pelvis_files[0])
    mesh2 = trimesh.load(pelvis_files[1])

    # Ensure both are valid meshes
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("One or both of the loaded files are not valid meshes.")

    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([mesh1, mesh2])

    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(input_dir, "combined_pelvis_mesh.stl")

    # Save the combined mesh
    combined_mesh.export(output_path, file_type="stl")
    print(f"Combined pelvis mesh saved to: {output_path}")

    # Delete the original files
    for file_path in pelvis_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def move_stl_meshes(input_dir, output_dir):
    """
    Moves all .stl files from the input directory to the output directory.
    If a file with the same name already exists in the output directory, it is replaced.

    Parameters:
    - input_dir (str): Directory to search for .stl files.
    - output_dir (str): Directory to move the .stl files to.

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .stl files in the input directory
    stl_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".stl")
    ]

    # Move each .stl file to the output directory
    for stl_file in stl_files:
        dest_file = os.path.join(output_dir, os.path.basename(stl_file))
        try:
            # Remove the existing file if it exists
            if os.path.exists(dest_file):
                os.remove(dest_file)

            # Move the file to the output directory
            shutil.move(stl_file, output_dir)
            print(f"Moved: {stl_file} -> {output_dir}")
        except Exception as e:
            print(f"Failed to move {stl_file}: {e}")

