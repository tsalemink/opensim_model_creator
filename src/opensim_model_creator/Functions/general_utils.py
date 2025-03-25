#Import packages
import numpy as np
import re
import pandas as pd
import os
import trimesh
from tkinter import Tk
from tkinter.filedialog import askdirectory
import shutil
from gias3.mesh import vtktools, simplemesh

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

        # Store the averages as a tuple (divided by 1000 to match opensim)
        #TODO: Just a reminder placed here incase trc files are ever converted to mm to come back to this
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


def scale_marker_data(marker_data, scale_factor) -> None:
    for values in marker_data.values():
        for i in range(3):
            values[i] *= scale_factor


def scale_stl_mesh(input_stl, output_stl, scale_factor=1000):
    """
    Scales an STL mesh by a given factor and saves it.

    Parameters:
    - input_stl (str): Path to the input STL file.
    - output_stl (str): Path to save the scaled STL file.
    - scale_factor (float): Scaling factor (default: 1000).

    Returns:
    - None
    """
    # Load the STL file
    mesh = vtktools.loadpoly(input_stl)

    # Scale the mesh
    mesh.v = mesh.v*(1/scale_factor)

    # Export the scaled STL
    vtktools.savepoly(mesh, output_stl)
    print(f"Scaled and saved: {output_stl}")


def process_participant_meshes(participant_inputs, meshes_folder, scale_factor=1000):
    """
    Reads STL files from participant_inputs, scales them, saves to meshes_folder,
    and combines left and right pelvis meshes into a single file.

    Parameters:
    - participant_inputs (str): Directory containing the participant's STL files.
    - meshes_folder (str): Directory to save processed STL files.
    - scale_factor (float): Scaling factor for the meshes (default: 1000).

    Returns:
    - None
    """
    # Ensure output directory exists
    os.makedirs(meshes_folder, exist_ok=True)

    # Find all STL files in the participant inputs folder
    stl_files = [
        f for f in os.listdir(participant_inputs)
        if f.lower().endswith(".stl")
    ]

    if not stl_files:
        print("No STL files found in the participant inputs folder.")
        return

    # Process each STL file: Scale and move to the meshes folder
    for file_name in stl_files:
        input_stl = os.path.join(participant_inputs, file_name)
        output_stl = os.path.join(meshes_folder, file_name)

        try:
            scale_stl_mesh(input_stl, output_stl, scale_factor)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Combine pelvis STL files
    combine_pelvis_meshes(meshes_folder)


def combine_pelvis_meshes(meshes_folder):
    """
    Combines the left and right pelvis STL files in the specified directory into one.

    Parameters:
    - meshes_folder (str): Directory where the pelvis STL files are stored.

    Returns:
    - None
    """
    # Find pelvis STL files
    pelvis_files = [
        os.path.join(meshes_folder, f)
        for f in os.listdir(meshes_folder)
        if f.lower().endswith(".stl") and "pelvis" in f.lower()
    ]

    # Ensure exactly two pelvis files are found
    if len(pelvis_files) != 2:
        print(f"Expected 2 pelvis STL files, but found {len(pelvis_files)}.")
        return

    try:
        # Load the meshes
        mesh1 = vtktools.loadpoly(pelvis_files[0])
        mesh2 = vtktools.loadpoly(pelvis_files[1])

        # create a new merged mesh object
        merged_mesh = simplemesh.SimpleMesh()
        # merge vertices of all meshes
        merged_mesh_vert = np.concatenate((mesh1.v, mesh2.v), axis=0)
        # merge faces of all meshes, need to add number of vertices for all previous meshes so the face numbers are correct
        a = len(mesh1.v)

        merged_mesh_faces = np.concatenate((mesh1.f, mesh2.f + a), axis=0)
        # assign vertices and faces to new mesh
        merged_mesh.v = merged_mesh_vert
        merged_mesh.f = merged_mesh_faces

        # Save combined pelvis mesh
        output_path = os.path.join(meshes_folder, "combined_pelvis_mesh.stl")
        vtktools.savepoly(merged_mesh,output_path)

        print(f"Combined pelvis mesh saved to: {output_path}")

    except Exception as e:
        print(f"Failed to combine pelvis meshes: {e}")


def copy_mesh_files(input_dir, output_dir, extensions=(".stl", ".vtp")):
    """
    Copies all .stl and .vtp files from the input directory to the output directory.
    If a file with the same name already exists in the output directory, it is replaced.

    Parameters:
    - input_dir (str): Directory to search for .stl and .vtp files.
    - output_dir (str): Directory to move the files to.
    - extensions (tuple): File extensions to look for (default: (".stl", ".vtp")).

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching files in the input directory
    mesh_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(extensions)
    ]

    # Copy each file to the output directory
    for mesh_file in mesh_files:
        dest_file = os.path.join(output_dir, os.path.basename(mesh_file))
        try:
            # Remove the existing file if it exists
            if os.path.exists(dest_file):
                os.remove(dest_file)

            # Copy the file to the output directory
            shutil.copy(mesh_file, output_dir)
            print(f"Moved: {mesh_file} -> {output_dir}")
        except Exception as e:
            print(f"Failed to move {mesh_file}: {e}")

