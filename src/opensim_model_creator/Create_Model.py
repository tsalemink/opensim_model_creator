import opensim as osim
import xml.etree.ElementTree as ET
from idlelib.autocomplete import FORCE
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import trimesh
import numpy as np
#import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from numpy.ma.core import argmax


#%% Functions
def print_bodies(model):
    # List all bodies in the model
    print("Bodies in the model:")
    for i in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(i)
        print(f"- {body.getName()}")

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

def save_model_to_folder(model, output_folder, filename="model_with_geometry.osim"):
    """
    Saves the model to a specified folder with the given filename.

    Args:
        model (opensim.Model): The OpenSim model to save.
        output_folder (str): The folder where the model file will be saved.
        filename (str): The name of the output model file. Defaults to "model_with_geometry.osim".
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine the folder path and filename
    output_path = os.path.join(output_folder, filename)

    # Save the model to the specified location
    model.printToXML(output_path)
    print(f"Model saved to: {output_path}")

def update_model_name(model, new_name):
    """
    Updates the name of the OpenSim model.

    Args:
        model (opensim.Model): The OpenSim model to update.
        new_name (str): The new name for the model.
    """
    # Set the new name for the model
    model.setName(new_name)
    print(f"Model name updated to: {new_name}")

def add_mesh_to_body(model, body_name, mesh_filename, offset_translation=(0, 0, 0), offset_orientation=(0, 0, 0)):
    """
    Adds a mesh geometry to a specified body in the OpenSim model.

    Args:
        model (opensim.Model): The OpenSim model.
        body_name (str): The name of the body to attach the mesh to.
        mesh_filename (str): The path to the mesh file.
        offset_translation (tuple): (x, y, z) translation offset for the mesh relative to the body.
        offset_orientation (tuple): (x, y, z) orientation offset for the mesh relative to the body.

    Raises:
        ValueError: If the specified body is not found in the model.
    """
    # Extract the file name without the directory path
    geometry_name = os.path.basename(mesh_filename).split('.')[0]

    # Get the body from the model
    try:
        body = model.getBodySet().get(body_name)
    except Exception as e:
        raise ValueError(f"Body '{body_name}' not found in the model.") from e

    # Create a new Mesh geometry
    mesh_geometry = osim.Mesh(mesh_filename)
    mesh_geometry.setName(geometry_name)

    # Set the offset frame for the mesh
    offset_frame = osim.PhysicalOffsetFrame()
    offset_frame.setName(f"{geometry_name}_offset")
    offset_frame.setParentFrame(body)
    offset_frame.set_translation(osim.Vec3(*offset_translation))
    offset_frame.set_orientation(osim.Vec3(*offset_orientation))

    # Add the offset frame to the body
    body.addComponent(offset_frame)

    # Attach the mesh to the offset frame
    offset_frame.attachGeometry(mesh_geometry)

    print(f"Added mesh '{geometry_name}' to body '{body_name}' with translation {offset_translation} and orientation {offset_orientation}.")


def extract_mesh_info_trimesh(file_path):
    """
    Extracts size, position, and volume information from a mesh file (STL or VTP) using trimesh.
    If the file is a VTP, it converts it to an STL beforehand.

    Args:
        file_path (str): Path to the mesh file (STL or VTP).

    Returns:
        dict: A dictionary containing the bounding box size, center, and volume.
    """
    # Check if the file is a VTP
    if file_path.lower().endswith('.vtp'):
        print(f"Converting VTP file to STL: {file_path}")
        # Load the VTP file with pyvista
        mesh = pv.read(file_path)

        # Temporary STL filename
        stl_temp_file = file_path.replace('.vtp', '.stl')

        # Save the mesh as an STL file
        mesh.save(stl_temp_file)
        print(f"Converted to STL: {stl_temp_file}")

        # Update the file path to point to the STL file
        file_path = stl_temp_file

    # Load the mesh with trimesh
    mesh = trimesh.load(file_path)

    # Get bounding box size
    bounding_box_size = mesh.bounding_box.extents

    # Get the center of the mesh
    center = mesh.bounding_box.centroid

    # Get the volume
    volume = mesh.volume

    # Optionally remove the temporary STL file
    if file_path.endswith('.stl') and '_temp' in file_path:
        os.remove(file_path)

    return {
        "bounding_box_size": bounding_box_size,
        "center": center,
        "volume": volume,
    }

def load_landmarks(file_path):
    """
    Loads landmarks from a file where each line contains a landmark name
    followed by its x, y, and z coordinates.

    Args:
        file_path (str): Path to the file containing landmarks.

    Returns:
        dict: A dictionary where keys are landmark names and values are numpy arrays of coordinates.
    """
    landmarks = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line != "\n":
                # Split the line into parts
                parts = line.strip().split()
                name = parts[0]  # The first part is the name
                coordinates = list(map(float, parts[1:]))  # Remaining parts are coordinates
                coordinates = [num / 1000 for num in coordinates]  # Convert from mm to meters
                landmarks[name] = np.array(coordinates)
    return landmarks



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

def determine_transform_child_to_parent(parent_rotated_centre, child_rotated_centre, initial_parent_landmark,initial_child_landmark):

    #Computing the initial rotated vector from the child to parent landmarks (i.e lateral epicondyle to ASIS)
    initial_vector = initial_child_landmark - initial_parent_landmark
    initial_vector_rot = rotate_coordinate_x(initial_vector, 90)

    #rotate landmarks to match opensim configuration
    parent_landmark_rot = rotate_coordinate_x(initial_parent_landmark, 90)
    child_landmark_rot = rotate_coordinate_x(initial_child_landmark, 90)

    #remove rotated centres from landmarks to get their actual positions on a rotated mesh
    parent_landmark_global = parent_rotated_centre - parent_landmark_rot
    child_landmark_global = child_rotated_centre - child_landmark_rot

    #compute the current vector between landmarks in their current configurations
    current_vector_global = child_landmark_global - parent_landmark_global


    update_vector = initial_vector_rot + current_vector_global
    return -update_vector


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


#FUNCTIONS ABOVE###############################################################################################################


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


import tkinter as tk
from tkinter import filedialog


def select_directory():
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the directory selection dialog
    selected_directory = filedialog.askdirectory(title="Select a Participent Directory to produce osim model")

    # Return the selected directory
    return selected_directory



def update_subtalar_joint_range(input_file, output_file, joint_name, range_min, range_max):
    """
    Updates the range of the subtalar joint's coordinate in an OpenSim .osim file.

    Parameters:
    - input_file (str): Path to the input .osim file.
    - output_file (str): Path to save the updated .osim file.
    - joint_name (str): Name of the subtalar joint (e.g., "calcn_l_to_talus_l").
    - range_min (float): New minimum range value.
    - range_max (float): New maximum range value.

    Returns:
    - None
    """
    # Parse the .osim file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Update <Coordinate> section
    coordinate_updated = False
    for coordinate in root.findall(".//Coordinate"):
        if coordinate.get("name") == joint_name:
            range_element = coordinate.find("range")
            if range_element is not None:
                range_element.text = f"{range_min} {range_max}"
                coordinate_updated = True
                print(f"Updated range for {joint_name} to [{range_min}, {range_max}].")
            else:
                print(f"No <range> element found for {joint_name}.")
                return

    if not coordinate_updated:
        print(f"Coordinate '{joint_name}' not found in the .osim file.")
        return

    # Update <SpatialTransform> if necessary
    custom_joint = root.find(f".//CustomJoint[@name='{joint_name}']")
    if custom_joint is not None:
        spatial_transform = custom_joint.find("SpatialTransform")
        if spatial_transform is not None:
            for transform_axis in spatial_transform.findall("TransformAxis"):
                coordinates = transform_axis.find("coordinates")
                if coordinates is not None and joint_name in coordinates.text:
                    print(f"Verified <SpatialTransform> alignment for {joint_name}.")
        else:
            print(f"No <SpatialTransform> found for joint '{joint_name}'.")
    else:
        print(f"No <CustomJoint> found for joint '{joint_name}'.")

    # Save the updated .osim file
    tree.write(output_file)
    print(f"Updated .osim file saved to: {output_file}")

def update_rx_coordinates(input_file, output_file, updates):
    """
    Updates 'rx' coordinate names in both <Coordinate> and <SpatialTransform> sections.

    Parameters:
    - input_file (str): Path to the input .osim file.
    - output_file (str): Path to save the updated .osim file.
    - updates (list of tuples): List of (joint_name, new_name) tuples specifying the updates.

    Returns:
    - None
    """
    # Parse the .osim file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Update <Coordinate> section
    for joint_name, new_name in updates:
        coordinate = root.find(f".//Coordinate[@name='rx']")
        if coordinate is not None:
            coordinate.set("name", new_name)
            print(f"Updated <Coordinate> name to '{new_name}' for joint '{joint_name}'.")
        else:
            print(f"<Coordinate> 'rx' not found for joint '{joint_name}'.")

    # Update <SpatialTransform> section
    for joint_name, new_name in updates:
        custom_joint = root.find(f".//CustomJoint[@name='{joint_name}']")
        if custom_joint is not None:
            spatial_transform = custom_joint.find("SpatialTransform")
            if spatial_transform is not None:
                for transform_axis in spatial_transform.findall("TransformAxis"):
                    coordinates = transform_axis.find("coordinates")
                    if coordinates is not None and coordinates.text and "rx" in coordinates.text:
                        coordinates.text = coordinates.text.replace("rx", new_name)
                        print(f"Updated 'rx' to '{new_name}' in <SpatialTransform> for joint '{joint_name}'.")
            else:
                print(f"<SpatialTransform> not found for joint '{joint_name}'.")
        else:
            print(f"CustomJoint '{joint_name}' not found.")

    # Save the updated .osim file
    tree.write(output_file)
    print(f"Updated .osim file saved to: {output_file}")


# Function to prompt user to select a file
def select_osim_file():
    # Hide the root window
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an OpenSim Model (.osim) File",
        filetypes=[("OpenSim Model Files", "*.osim")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected. Please select an .osim file.")
    return file_path

def update_rotation_axes(file_path, output_path, joint_names, new_axes):
    """
    Updates the rotation axes of specified CustomJoints in an OpenSim .osim file.

    Parameters:
    - file_path (str): Path to the input .osim file.
    - output_path (str): Path to save the updated .osim file.
    - joint_names (list of str): List of joint names to modify.
    - new_axes (list of tuple): New rotation axes for each TransformAxis.

    Returns:
    - None
    """
    # Load and parse the .osim file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Function to modify a specific joint
    def modify_joint(joint_name):
        # Locate the joint
        custom_joint = root.find(f".//CustomJoint[@name='{joint_name}']")
        if custom_joint is not None:
            print(f"Found CustomJoint: {joint_name}")
            spatial_transform = custom_joint.find("SpatialTransform")

            # Update the rotation axes
            for i, axis_values in enumerate(new_axes):  # new_axes is a list of (x, y, z) tuples
                transform_axis = spatial_transform.find(f"TransformAxis[@name='rotation{i + 1}']")
                if transform_axis is not None:
                    axis_element = transform_axis.find("axis")
                    axis_element.text = f"{axis_values[0]} {axis_values[1]} {axis_values[2]}"
                    print(f"Updated {joint_name} rotation{i + 1} axis to: {axis_element.text}")
                else:
                    print(f"TransformAxis rotation{i + 1} not found for {joint_name}.")
        else:
            print(f"CustomJoint '{joint_name}' not found.")

    # Modify each joint
    for joint_name in joint_names:
        modify_joint(joint_name)

    # Save the updated .osim file
    tree.write(output_path)
    print(f"Updated .osim file saved to: {output_path}")


def move_rx_to_first_rotation(file_path, output_path, joint_names):
    """
    Moves the 'rx' coordinate from the third rotation (rotation3) to the first rotation (rotation1)
    for specified CustomJoints in an OpenSim .osim file.

    Parameters:
    - file_path (str): Path to the input .osim file.
    - output_path (str): Path to save the updated .osim file.
    - joint_names (list of str): List of joint names to modify.

    Returns:
    - None
    """
    # Load and parse the .osim file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Function to modify a specific joint
    def modify_joint(joint_name):
        # Locate the joint
        custom_joint = root.find(f".//CustomJoint[@name='{joint_name}']")
        if custom_joint is not None:
            print(f"Found CustomJoint: {joint_name}")
            spatial_transform = custom_joint.find("SpatialTransform")

            # Get the current coordinates for rotation3
            rotation3 = spatial_transform.find("TransformAxis[@name='rotation3']")
            rotation1 = spatial_transform.find("TransformAxis[@name='rotation1']")
            if rotation3 is not None and rotation1 is not None:
                coordinates_element = rotation3.find("coordinates")
                if coordinates_element is not None and "rx" in coordinates_element.text:
                    # Move 'rx' from rotation3 to rotation1
                    coordinates_element.text = coordinates_element.text.replace("rx", "").strip()
                    rotation1_coordinates = rotation1.find("coordinates")
                    if rotation1_coordinates is None:
                        rotation1_coordinates = ET.SubElement(rotation1, "coordinates")
                    rotation1_coordinates.text = "rx"
                    print(f"Moved 'rx' from rotation3 to rotation1 for {joint_name}.")
                else:
                    print(f"'rx' not found in rotation3 for {joint_name}.")
            else:
                print(f"Missing TransformAxis for {joint_name}.")
        else:
            print(f"CustomJoint '{joint_name}' not found.")

    # Modify each joint
    for joint_name in joint_names:
        modify_joint(joint_name)

    # Save the updated .osim file
    tree.write(output_path)
    print(f"Updated .osim file saved to: {output_path}")

def update_subtalar_joint(file_path, output_path, joint_name):
    """
    Updates the SpatialTransform of the left subtalar joint:
    - Ensures 'rx' controls rotation1 with a LinearFunction.
    - Removes the LinearFunction from rotation3.

    Parameters:
    - file_path (str): Path to the input .osim file.
    - output_path (str): Path to save the updated .osim file.
    - joint_name (str): Name of the left subtalar joint.

    Returns:
    - None
    """
    # Load and parse the .osim file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Locate the CustomJoint
    custom_joint = root.find(f".//CustomJoint[@name='{joint_name}']")
    if custom_joint is None:
        print(f"CustomJoint '{joint_name}' not found.")
        return

    print(f"Updating SpatialTransform for CustomJoint: {joint_name}")
    spatial_transform = custom_joint.find("SpatialTransform")
    if spatial_transform is None:
        print(f"SpatialTransform not found for CustomJoint: {joint_name}")
        return

    # Update rotation1 to include rx with a LinearFunction
    rotation1 = spatial_transform.find("TransformAxis[@name='rotation1']")
    if rotation1 is not None:
        # Ensure 'rx' is the coordinate for rotation1
        coordinates = rotation1.find("coordinates")
        if coordinates is None:
            coordinates = ET.SubElement(rotation1, "coordinates")
        coordinates.text = "rx"

        # Add a LinearFunction with coefficients 1 0
        linear_function = rotation1.find("LinearFunction")
        if linear_function is None:
            linear_function = ET.SubElement(rotation1, "LinearFunction", name="function")
        coefficients = linear_function.find("coefficients")
        if coefficients is None:
            coefficients = ET.SubElement(linear_function, "coefficients")
        coefficients.text = "1 0"

        print(f"Updated rotation1: coordinate='rx', function='1 0'")

    else:
        print(f"TransformAxis rotation1 not found for CustomJoint: {joint_name}")

    # Remove the LinearFunction from rotation3
    rotation3 = spatial_transform.find("TransformAxis[@name='rotation3']")
    if rotation3 is not None:
        linear_function = rotation3.find("LinearFunction")
        if linear_function is not None:
            rotation3.remove(linear_function)
            print("Removed LinearFunction from rotation3.")
        else:
            print("No LinearFunction found for rotation3.")
    else:
        print(f"TransformAxis rotation3 not found for CustomJoint: {joint_name}")

    # Save the updated .osim file
    tree.write(output_path)
    print(f"Updated .osim file saved to: {output_path}")


# Example usage
participant_folder = select_directory()
if participant_folder:
    print(f"Selected directory: {participant_folder}")
else:
    print("No directory selected.")

from scipy.spatial.transform import Rotation as R

def add_markers_to_body(model, body_name, marker_names, mocap_file, center, custom_names=None):
    """
    Adds multiple markers to a specified body in an OpenSim model with optional custom names.

    Args:
        model (osim.Model): The OpenSim model to which the markers will be added.
        body_name (str): The name of the body to which the markers will be attached.
        marker_names (list): A list of marker names to be added.
        mocap_file (dict): A dictionary where keys are marker names and values are their (x, y, z) coordinates.
        center (tuple): The reference center point for calculating marker positions.
        custom_names (list, optional): A list of custom names for the markers. If None, use `marker_names`.

    """
    try:
        # Get the specified body from the model
        body = model.getBodySet().get(body_name)

        # Ensure custom_names matches marker_names if provided
        if custom_names and len(custom_names) != len(marker_names):
            raise ValueError("Length of custom_names must match the length of marker_names.")

        for i, marker_name in enumerate(marker_names):
            # Ensure the marker name exists in the mocap file dictionary
            if marker_name not in mocap_file:
                print(f"Marker '{marker_name}' not found in mocap file. Skipping.")
                continue

            # Get the marker location
            location = mocap_file[marker_name]
            landmark_position = vector_between_points(location, center)
            landmark_position = rotate_coordinate_x(landmark_position, 90)
            marker_location = osim.Vec3(*landmark_position)

            # Determine the marker's name
            final_name = custom_names[i] if custom_names else marker_name

            # Create and add the marker
            marker = osim.Marker(final_name, body, marker_location)
            model.addMarker(marker)

            print(f"Marker '{final_name}' added to body '{body_name}' at location {location}.")

    except Exception as e:
        print(f"Error adding markers to body '{body_name}': {e}")

import re

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

def compute_axis_from_euler(euler_angles, original_axis, convention='xyz'):
    """
    Computes the transformed axis vector using Euler angles.

    Args:
        euler_angles (np.array): Euler angles (roll, pitch, yaw) in radians.
        original_axis (np.array): A 3D vector representing the original axis.
        convention (str): The convention used for Euler angles (e.g., 'xyz', 'zyx').

    Returns:
        np.array: The transformed axis vector.
    """
    # Create the rotation from Euler angles
    rotation = R.from_euler(convention, euler_angles, degrees=False)

    # Apply the rotation to the original axis
    transformed_axis = rotation.apply(original_axis)

    return transformed_axis

from scipy.optimize import minimize

def calculate_euler_to_align_axis_with_optimization(target_vector, secondary_vector, align_axis='z'):
    """
    Calculates Euler angles to align a single axis of a coordinate system to a target vector,
    and optimizes to minimize the difference between the secondary vector and its rotated version.

    Args:
        target_vector (np.array): A 3D vector to align the specified axis to.
        secondary_vector (np.array): A 3D vector to be optimized for alignment.
        align_axis (str): The axis to align ('x', 'y', or 'z').

    Returns:
        tuple: (Optimized Euler angles, new secondary vector).
    """
    # Normalize the input vectors
    target_vector = target_vector / np.linalg.norm(target_vector)
    secondary_vector = secondary_vector / np.linalg.norm(secondary_vector)

    # Define the original coordinate system axis to align
    if align_axis == 'x':
        original_axis = np.array([1, 0, 0])
        euler_index = 0  # Roll
    elif align_axis == 'y':
        original_axis = np.array([0, 1, 0])
        euler_index = 1  # Pitch
    elif align_axis == 'z':
        original_axis = np.array([0, 0, 1])
        euler_index = 2  # Yaw
    else:
        raise ValueError("align_axis must be one of 'x', 'y', or 'z'.")

    # Calculate the rotation axis and angle
    rotation_axis = np.cross(original_axis, target_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(original_axis, target_vector))

    # Construct the rotation vector and initial Euler angles
    rotation_vector = rotation_axis * rotation_angle
    rotation = R.from_rotvec(rotation_vector)
    initial_euler_angles = rotation.as_euler('xyz', degrees=False)

    # Optimization: Adjust the Euler angle for the specified axis
    def objective(euler_angle):
        # Update the Euler angle for the specified axis
        euler_angles = initial_euler_angles.copy()
        euler_angles[euler_index] = euler_angle[0]

        # Compute the new secondary vector after applying the rotation
        rotation = R.from_euler('xyz', euler_angles, degrees=False)
        new_secondary_vector = rotation.apply(secondary_vector)

        # Calculate the error (minimize the angle between vectors)
        dot_product = np.dot(new_secondary_vector, secondary_vector)
        return 1 - dot_product  # Maximize alignment (dot product close to 1)

    # Perform optimization
    result = minimize(objective, [initial_euler_angles[euler_index]], bounds=[(-1.3*np.pi, 2*np.pi)],options={"disp": False, "maxiter": 10000, "gtol": 1e-15, "ftol": 1e-15})
    optimized_angle = result.x[0]

    # Update the Euler angles with the optimized value
    optimized_euler_angles = initial_euler_angles.copy()
    optimized_euler_angles[euler_index] = optimized_angle

    # Compute the final rotated secondary vector
    rotation = R.from_euler('xyz', optimized_euler_angles, degrees=False)
    optimized_secondary_vector = rotation.apply(secondary_vector)

    return optimized_euler_angles

def align_y_axis_with_vector_and_z_axis_to_plane(vector, plane_points):
    """
    Calculates Euler angles to align the y-axis to the provided vector and the z-axis parallel to a plane.

    Args:
        vector (np.array): A 3D vector to align the y-axis to.
        plane_points (list of np.array): Three points (3D) defining the plane.

    Returns:
        np.array: Euler angles (roll, pitch, yaw) in radians to achieve the alignment.
    """
    # Normalize the vector to align with y-axis
    vector = vector / np.linalg.norm(vector)

    # Calculate the normal vector of the plane
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal /= np.linalg.norm(plane_normal)

    # Ensure orthogonal alignment:
    # 1. y-axis aligned to the provided vector
    y_axis = vector

    # 2. z-axis should be parallel to the plane (orthogonal to the plane normal)
    #    Compute the projection of the original z-axis ([0, 0, 1]) onto the plane
    original_z = np.array([0, 0, 1])
    z_axis = original_z - np.dot(original_z, plane_normal) * plane_normal
    z_axis /= np.linalg.norm(z_axis)

    # 3. x-axis is computed as orthogonal to both y-axis and z-axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Construct the rotation matrix
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

    # Convert the rotation matrix to Euler angles
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)

    return euler_angles

def align_y_axis_with_vector_and_x_axis_to_plane(vector, plane_points, anterior_direction=np.array([1, 0, 0])):
    """
    Calculates Euler angles to align the y-axis to the provided vector
    and the x-axis perpendicular to a plane, ensuring the x-axis points anteriorly.

    Args:
        vector (np.array): A 3D vector to align the y-axis to.
        plane_points (list of np.array): Three points (3D) defining the plane.
        anterior_direction (np.array): A reference direction to ensure the x-axis points anteriorly.

    Returns:
        np.array: Euler angles (roll, pitch, yaw) in radians to achieve the alignment.
    """
    # Normalize the vector to align with y-axis
    vector = vector / np.linalg.norm(vector)

    # Calculate the normal vector of the plane
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal /= np.linalg.norm(plane_normal)

    # Ensure orthogonal alignment:
    # 1. y-axis aligned to the provided vector
    y_axis = vector

    # 2. x-axis should be perpendicular to the plane normal
    x_axis = plane_normal

    # Ensure the x-axis points in the anterior direction
    anterior_direction = anterior_direction / np.linalg.norm(anterior_direction)
    if np.dot(x_axis, anterior_direction) < 0:
        x_axis = -x_axis

    # 3. z-axis is computed as orthogonal to both y-axis and x-axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Recompute x-axis to ensure strict orthogonality
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Construct the rotation matrix
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

    # Convert the rotation matrix to Euler angles
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)

    return euler_angles

def compute_euler_angles_from_vectors(from_vector, to_vector, order='xyz'):
    """
    Computes the Euler angles required to rotate one vector to align with another.

    Args:
        from_vector (np.array): The initial vector.
        to_vector (np.array): The target vector to align with.
        order (str): The Euler angle order (default: 'xyz').

    Returns:
        np.array: Euler angles (in radians) for the specified rotation order.
    """
    # Normalize both vectors
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    # Calculate the rotation axis (cross product)
    rotation_axis = np.cross(from_vector, to_vector)
    axis_norm = np.linalg.norm(rotation_axis)

    if axis_norm < 1e-6:  # If vectors are nearly aligned
        if np.allclose(from_vector, to_vector):
            return np.array([0.0, 0.0, 0.0])  # No rotation needed
        else:
            # Opposite vectors: Rotate by 180 degrees
            orthogonal_axis = np.array([1.0, 0.0, 0.0]) if not np.allclose(from_vector, [1, 0, 0]) else np.array([0, 1, 0])
            rotation_axis = np.cross(from_vector, orthogonal_axis)
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.pi
    else:
        # Calculate the angle between the vectors
        angle = np.arccos(np.clip(np.dot(from_vector, to_vector), -2.0, 2.0))
        rotation_axis /= axis_norm

    # Create the rotation object using axis-angle
    rotation_vector = rotation_axis * angle
    rotation = R.from_rotvec(rotation_vector)

    # Convert to Euler angles
    euler_angles = rotation.as_euler(order, degrees=False)
    return euler_angles

import plotly.graph_objects as go

def plot_3d_marker_positions_interactive(marker_positions, time_step, title="3D Marker Positions"):
    """
    Plot marker positions in a 3D interactive space for a given time step, with markers grouped by categories and colored differently.

    Args:
        marker_positions (dict): Dictionary containing marker positions for all time steps.
        time_step (int): The time step to visualize (index in the marker position arrays).
        title (str): Title of the plot.

    Returns:
        None: Displays an interactive 3D plot.
    """
    # Define marker categories and their colors
    marker_categories = {
        "pelvis": ["RASI", "LASI", "LPSI", "RPSI"],  # Example pelvis markers
        "left_leg": ["LTHI", "LTIB", "LKNE", "LANK","LPAT"],  # Example left leg markers
        "right_leg": ["RTHI", "RTIB", "RKNE", "RANK","RPAT"],  # Example right leg markers
        "left_foot": ["LTOE", "LHEE"],  # Example left foot markers
        "right_foot": ["RTOE", "RHEE"],  # Example right foot markers
    }

    category_colors = {
        "pelvis": "blue",
        "left_leg": "green",
        "right_leg": "red",
        "left_foot": "purple",
        "right_foot": "orange",
    }

    # Initialize data for Plotly
    data = []
    for category, color in category_colors.items():
        category_x, category_y, category_z, category_names = [], [], [], []

        for marker_name in marker_categories.get(category, []):
            if marker_name in marker_positions:
                pos = marker_positions[marker_name][time_step]
                category_x.append(pos[0])
                category_y.append(pos[1])
                category_z.append(pos[2])
                category_names.append(marker_name)

        # Add category scatter plot
        if category_x:  # Ensure the category has markers
            data.append(go.Scatter3d(
                x=category_x,
                y=category_y,
                z=category_z,
                mode='markers+text',
                marker=dict(size=6, color=color),
                text=category_names,
                textposition="top center",
                name=category
            ))

    # Plot uncategorized markers in gray
    uncategorized_x, uncategorized_y, uncategorized_z, uncategorized_names = [], [], [], []
    for marker_name, positions in marker_positions.items():
        if all(marker_name not in markers for markers in marker_categories.values()):
            pos = positions[time_step]
            uncategorized_x.append(pos[0])
            uncategorized_y.append(pos[1])
            uncategorized_z.append(pos[2])
            uncategorized_names.append(marker_name)

    if uncategorized_x:
        data.append(go.Scatter3d(
            x=uncategorized_x,
            y=uncategorized_y,
            z=uncategorized_z,
            mode='markers+text',
            marker=dict(size=6, color="gray"),
            text=uncategorized_names,
            textposition="top center",
            name="Uncategorized"
        ))

    # Create the layout
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
            aspectmode='data'  # Ensures consistent scaling of axes
        ),
        legend=dict(
            x=1,
            y=0.5,
            title="Marker Categories"
        )
    )

    # Create the figure and show it
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def adjust_model_markers(model_path, output_model_path, marker_differences):
    """
    Adjust the model marker positions based on the given differences.

    Args:
        model_path (str): Path to the OpenSim model file.
        output_model_path (str): Path to save the adjusted model.
        marker_differences (dict): Dictionary containing the average differences for each marker
                                   in the format {'marker_name': [dx, dy, dz]}.
    """
    # Load the OpenSim model
    model = osim.Model(model_path)
    model.finalizeConnections()
    state = model.initSystem()
    # Iterate over the marker differences
    for marker_name, difference in marker_differences.items():
        try:
            # Get the marker
            marker = model.getMarkerSet().get(marker_name)

            # Get the parent body of the marker
            parent_body = marker.getParentFrame()

            # Get the current location offset (relative to the parent frame)
            current_offset = marker.get_location()

            #%% Trying to convert the average marker distnace to be representative to the parent frame

            # Convert the global difference to an OpenSim Vec3
            global_difference = osim.Vec3(*difference)

            # Find the current marker position in the global frame
            current_location_in_ground = parent_body.findStationLocationInGround(state, marker.get_location())

            current_local_offset = parent_body.findStationLocationInAnotherFrame(state, current_location_in_ground, parent_body)


            # Compute the new marker position in the global frame
            new_location_in_ground = osim.Vec3(
                current_location_in_ground.get(0) + global_difference.get(0),
                current_location_in_ground.get(1) + global_difference.get(1),
                current_location_in_ground.get(2) + global_difference.get(2)
            )

            # Transform the new global position back to the local frame of the parent body
            new_local_offset = model.getGround().findStationLocationInAnotherFrame(state, new_location_in_ground, parent_body)

            print(new_location_in_ground)
            print(new_local_offset)
            print(current_offset)
            # Update the marker's local offset
            marker.set_location(new_local_offset)

            print(f"Adjusted marker '{marker_name}' to new local offset: {new_local_offset}")

        except Exception as e:
            print(f"Error adjusting marker '{marker_name}': {e}")

    # Save the updated model
    model.setName("Optimised_Knee_Moved_Markers")
    model.printToXML(output_model_path)
    print(f"Model updated and saved to: {output_model_path}")


def parse_model_marker_locations(sto_file_path):
    """
    Parse the _ik_model_marker_locations.sto file to extract marker positions.

    Args:
        sto_file_path (str): Path to the .sto file.

    Returns:
        dict: Dictionary of model marker positions with marker names as keys.
        list: List of time values.
    """
    with open(sto_file_path, 'r') as file:
        lines = file.readlines()

    # Identify the header line with column labels
    header_index = None
    for idx, line in enumerate(lines):
        if line.startswith("time"):
            header_index = idx
            break

    if header_index is None:
        raise ValueError("Header line with column labels not found in the .sto file.")

    # Extract marker names from the header
    headers = lines[header_index].strip().split("\t")
    marker_names = headers[1:]  # Skip "time" column

    # Parse data rows
    data = np.loadtxt(lines[header_index + 1:], dtype=float)

    # Organize data into a dictionary
    marker_positions = {name: [] for name in marker_names}


    time_list = data[:, 0]
    for i, marker_name in enumerate(marker_positions.keys()):
        marker_positions[marker_name] = data[:, 1 + i]

    # Initialize the combined marker positions dictionary
    combined_marker_positions = {}

    # Group marker components into XYZ coordinates
    for marker_name in marker_positions.keys():
        # Extract base name by removing '_tx', '_ty', '_tz'
        base_name = marker_name.rsplit('_', 1)[0]

        # Initialize an entry for the base name if it doesn't already exist
        if base_name not in combined_marker_positions:
            combined_marker_positions[base_name] = []

    # Populate the combined dictionary with XYZ coordinates for each time step
    for base_name in combined_marker_positions.keys():
        x = marker_positions[f"{base_name}_tx"]
        y = marker_positions[f"{base_name}_ty"]
        z = marker_positions[f"{base_name}_tz"]

        # Stack X, Y, Z into a single array for each time step
        xyz_coordinates = np.column_stack((x, y, z))
        combined_marker_positions[base_name] = xyz_coordinates

    return combined_marker_positions, time_list


def optimize_knee_axis(model_path, trc_file, start_time, end_time, marker_weights, initial_params, temp_model_path_1, temp_model_path_2,final_output_model):
    """
    Optimize the knee joint orientation to minimize IK errors.

    Args:
        model_path (str): Path to the OpenSim model file.
        trc_file (str): Path to the TRC file.
        start_time (float): Start time for IK analysis.
        end_time (float): End time for IK analysis.
        marker_weights (dict): Marker weights for IK analysis.
        initial_params (list): Initial joint orientations for optimization.
        temp_model_path_1 (str): Path for temporary model file 1.
        temp_model_path_2 (str): Path for temporary model file 2.
        final_output_model (str): Path to save the final model.

    Returns:
        OptimizeResult: Results of the optimization process.
    """
    def objective(params):
        left_knee_x, left_knee_y, right_knee_x, right_knee_y = params*1000

        # Adjust left knee
        adjust_joint_orientation(
            model_path=model_path,
            joint_name="tibfib_l_to_femur_l",
            rotation_adjustment=osim.Vec3(left_knee_x, left_knee_y, 0.0),
            output_model_path=temp_model_path_1
        )

        # Adjust right knee
        adjust_joint_orientation(
            model_path=temp_model_path_1,
            joint_name="tibfib_r_to_femur_r",
            rotation_adjustment=osim.Vec3(right_knee_x, right_knee_y, 0.0),
            output_model_path=temp_model_path_2
        )
        print([left_knee_x, left_knee_y, right_knee_x, right_knee_y])
        # Perform IK and compute error
        errors = perform_IK(temp_model_path_2, trc_file, start_time, end_time, marker_weights)
        print(errors["Average RMS Error"])
        return errors["Average RMS Error"]*1e4 if errors else float("inf")

    bounds = [(-0.001, 0.001)] * 4
    result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds, options={"disp": True, "maxiter": 1})
    model = osim.Model(temp_model_path_2)
    model_name_here = final_output_model.split("/")[-1].split(".")[0]
    model.setName(model_name_here)
    model.printToXML(final_output_model)
    return result


def perform_IK(model_file, trc_file, start_time, end_time, marker_weights, output_errors_file="_ik_marker_errors.sto"):
    """
    Perform Inverse Kinematics analysis using OpenSim.

    Args:
        model_file (str): Path to the OpenSim model file.
        trc_file (str): Path to the TRC file.
        start_time (float): Start time for IK.
        end_time (float): End time for IK.
        marker_weights (dict): Marker weights for IK analysis.
        output_errors_file (str): Path to save marker errors.

    Returns:
        dict: Dictionary containing average RMS error and max error.
    """
    try:
        model = osim.Model(model_file)
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(model)
        ik_tool.setMarkerDataFileName(trc_file)
        ik_tool.setStartTime(start_time)
        ik_tool.setEndTime(end_time)
        ik_tool.setOutputMotionFileName("ik_output.mot")
        ik_tool.set_report_marker_locations(True)

        # Configure marker weights
        ik_task_set = osim.IKTaskSet()
        for marker_name, weight in marker_weights.items():
            task = osim.IKMarkerTask()
            task.setName(marker_name)
            task.setWeight(weight)
            task.setApply(True)
            ik_task_set.adoptAndAppend(task)

        ik_tool.set_IKTaskSet(ik_task_set)
        ik_tool.run()

        return extract_ik_errors(output_errors_file)

    except Exception as e:
        print(f"Error during IK: {e}")
        return None


def extract_ik_errors(error_file_path):
    """
    Extract RMS and maximum marker errors from an IK error file.

    Args:
        error_file_path (str): Path to the IK error file (.sto).

    Returns:
        dict: Dictionary with the average RMS error and maximum error.
    """
    try:
        with open(error_file_path, 'r') as file:
            lines = file.readlines()

        # Find header and data rows
        data_start_idx = None
        headers = []
        for idx, line in enumerate(lines):
            if line.startswith("endheader"):
                data_start_idx = idx + 2
                headers = lines[idx + 1].strip().split()
                break

        data = np.loadtxt(lines[data_start_idx:], dtype=float)

        # Extract errors
        rms_idx = headers.index("marker_error_RMS")
        max_idx = headers.index("marker_error_max")

        rms_error = np.sqrt(np.mean(data[:, rms_idx] ** 2))
        max_error = np.max(data[:, max_idx])

        return {"Average RMS Error": rms_error, "Max Error": max_error}

    except Exception as e:
        print(f"Error reading IK error file: {e}")
        return None


def adjust_joint_orientation(model_path, joint_name, rotation_adjustment, output_model_path):
    """
    Adjust the orientation of a joint's child frame in an OpenSim model.

    Args:
        model_path (str): Path to the OpenSim model file (.osim).
        joint_name (str): Name of the joint to adjust.
        rotation_adjustment (osim.Vec3): Adjustments to the joint's orientation in radians.
        output_model_path (str): Path to save the updated model.

    Returns:
        None: Saves the updated model with the joint orientation adjusted.
    """
    try:
        # Load the model
        model = osim.Model(model_path)
        state = model.initSystem()

        # Access the joint
        joint = model.getJointSet().get(joint_name)

        # Access the child frame
        child_frame = joint.upd_frames(1)
        current_orientation1 = np.array([child_frame.get_orientation().get(i) for i in range(3)])

        # Apply rotation adjustments
        new_orientation1 = current_orientation1 + np.array([rotation_adjustment.get(i) for i in range(3)])
        child_frame.set_orientation(osim.Vec3(*new_orientation1))


        # Access the parent frame
        parent_frame = joint.upd_frames(0)
        current_orientation2 = np.array([parent_frame.get_orientation().get(i) for i in range(3)])

        # Apply rotation adjustments
        new_orientation2 = current_orientation2 + np.array([rotation_adjustment.get(i) for i in range(3)])
        parent_frame.set_orientation(osim.Vec3(*new_orientation2))




        # Save the updated model
        model.printToXML(output_model_path)
        print(f"Joint '{joint_name}' updated and saved to: {output_model_path}")

    except Exception as e:
        print(f"Error updating joint '{joint_name}': {e}")


def run_knee_joint_optimisation(
    source_file_path1,
    knee_optimisation_trc_file,
    start_time,
    end_time,
    temp_model_path_1,
    temp_model_path_2,
    marker_weights,
    final_output_model_path,
    initial_params=None
):
    """
    Run knee joint optimization for an OpenSim model.

    Args:
        source_file_path1 (str): Path to the source OpenSim model file.
        knee_optimisation_trc_file (str): Path to the TRC file for optimization.
        start_time (float): Start time for IK analysis.
        end_time (float): End time for IK analysis.
        marker_weights (dict, optional): Marker weights for IK analysis.
        initial_params (list, optional): Initial joint rotations for x and y.
        temp_model_path_1 (str, optional): Temporary model file path 1.
        temp_model_path_2 (str, optional): Temporary model file path 2.

    Returns:
        None
    """
    # Default initial parameters
    if initial_params is None:
        initial_params = [0, 0, 0, 0]



    # Suppress OpenSim logging
    osim.Logger.setLevelString("Off")

    # Run optimization
    result = optimize_knee_axis(
        model_path=source_file_path1,
        trc_file=knee_optimisation_trc_file,
        start_time=start_time,
        end_time=end_time,
        marker_weights=marker_weights,
        initial_params=initial_params,
        temp_model_path_1=temp_model_path_1,
        temp_model_path_2=temp_model_path_2,
        final_output_model = final_output_model_path
    )

    print(f"Optimized Joint Orientations: {result.x}")


def parse_muscle_node_files_recursive(root_directory):
    """
    Recursively parses muscle node files in a root directory and constructs a nested dictionary.

    Args:
    - root_directory (str): Path to the root directory containing subfolders for bones.

    Returns:
    - dict: Nested dictionary where the first level keys are bone names (e.g., 'Pelvis'),
            the second level keys are muscle numbers (strings), and values are lists of node numbers.
    """
    muscle_nodes = {}

    # Walk through the root directory and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Get the current folder name (e.g., 'Pelvis', 'Femur')
        bone_name = os.path.basename(dirpath)
        if bone_name not in muscle_nodes:
            muscle_nodes[bone_name] = {}

        for file_name in filenames:
            if file_name.endswith("_NodeNo.txt"):
                # Extract the muscle number(s) by splitting on the last underscore
                muscle_number = file_name.rsplit("_", 1)[0]

                # Read the node numbers from the file
                with open(os.path.join(dirpath, file_name), 'r') as file:
                    nodes = [int(line.strip()) for line in file if line.strip().isdigit()]

                # Add to dictionary under the corresponding bone name
                muscle_nodes[bone_name][muscle_number] = nodes

    # Remove empty entries (for folders without valid files)
    muscle_nodes = {k: v for k, v in muscle_nodes.items() if v}
    return muscle_nodes


def parse_ply_files_by_side_and_bone(directory, bones):
    """
    Parses .ply files by side ('left', 'right') and bone names to extract node coordinates.

    Args:
        directory (str): The directory containing the .ply files.
        bones (list of str): List of bone names to process (e.g., ['tibfib', 'pelvis', 'femur']).

    Returns:
        dict: A nested dictionary with structure {side: {bone: {node_number: [x, y, z], ...}, ...}, ...}.
    """
    sides = ['Left', 'Right']
    parsed_data = {side: {} for side in sides}

    for side in sides:
        for bone in bones:
            # Search for the specific file using the provided function
            ply_files = search_files_by_keywords(directory, side+" "+bone)

            if len(ply_files) == 0:
                print(f"No .ply file found for {side} {bone}. Skipping.")
                continue
            if len(ply_files) > 1:
                print(f"Multiple .ply files found for {side} {bone}: {ply_files}. Using the first one.")

            ply_path = ply_files[0]  # Use the first match

            # Process the .ply file to extract node coordinates
            node_dict = {}
            with open(ply_path, "r") as file:
                lines = file.readlines()

            # Locate the "end_header" and extract vertex data
            vertex_count = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line.strip() == "end_header":
                    header_end = i
                    break

            vertex_data = lines[header_end + 1: header_end + 1 + vertex_count]

            # Parse vertex data
            for i, line in enumerate(vertex_data):
                if line.strip():  # Avoid empty lines
                    coords = list(map(float, line.split()))
                    node_dict[i] = coords  # Node numbers start from 0

            # Store the parsed nodes in the dictionary
            parsed_data[side][bone] = node_dict
            print(f"Processed {side} {bone}: {len(node_dict)} nodes.")

    return parsed_data

def map_muscle_nodes_to_coordinates(
    muscle_linkages, muscle_number_to_nodes_key, node_to_coordinate
):
    """
    Maps muscle numbers to their corresponding mean node coordinates and updates muscle_linkages.

    Parameters:
    - muscle_linkages (dict): Dictionary defining muscle linkages with body parts and muscle numbers.
    - muscle_number_to_nodes_key (dict): Dictionary mapping muscle numbers to nodes by body part.
    - node_to_coordinate (dict): Dictionary mapping node numbers to their coordinates.

    Returns:
    - None: Updates the muscle_linkages dictionary in place with mean coordinates for each attachment.
    """
    sides = ["Left", "Right"]

    for muscle, attachment_types in muscle_linkages.items():
        for attachment_type, attachments in attachment_types.items():
            for attachment in attachments:
                body_part = attachment[0]
                muscle_number = attachment[1]

                # Retrieve nodes for the muscle number from muscle_number_to_nodes_key
                nodes = muscle_number_to_nodes_key.get(body_part, {}).get(muscle_number, [])

                # Retrieve coordinates for these nodes from node_to_coordinate
                for side in sides:
                    current_coordinates = []
                    for node in nodes:
                        try:
                            # Append scaled coordinates
                            current_coordinates.append(
                                [coord / 1000 for coord in node_to_coordinate[side][body_part][node]]
                            )
                        except KeyError:
                            print(f"Node {node} not found in {side}/{body_part}. Skipping.")

                    # Ensure there are coordinates to compute mean
                    if current_coordinates:
                        # Convert the list of coordinates to a NumPy array
                        coordinates_array = np.array(current_coordinates)

                        # Compute the mean along each axis (x, y, z)
                        mean_coordinates = np.mean(coordinates_array, axis=0)

                        # Append mean_coordinates to the attachment list
                        attachment.append(mean_coordinates)

    return muscle_linkages


def add_all_muscle_attachment_markers(model, muscle_linkages, centers):
    sides = ["_l", "_r"]
    for muscle in muscle_linkages.keys():
        for attachment_type in muscle_linkages[muscle].keys():
            if len(muscle_linkages[muscle].keys()) < 2:
                continue
            # Iterate through each attachment for the current muscle and attachment type
            i = 1
            for attachment in muscle_linkages[muscle][attachment_type]:
                if attachment[0] == "Tibia" or attachment[0] == "Fibula":
                    continue
                muscle_name_storage = [0,0]
                for side in sides:
                    # Extract the body name from the attachment and convert to lowercase
                    body_name = attachment[0].lower()
                    if body_name != "pelvis":
                        body_name = body_name + side
                        if side == "_l":
                            center = centers[attachment[0]][0]
                        elif side == "_r":
                            center = centers[attachment[0]][1]
                    else:
                        center = centers[attachment[0]]
                    body_name = body_name + "_b"

                    if len(muscle_linkages[muscle][attachment_type]) > 1:
                        muscle_name = attachment_type + side+ "_" + muscle.lower().replace(" ", "_") + "_" + str(i)
                    else:
                        muscle_name = attachment_type + side + "_" + muscle.lower().replace(" ", "_")
                    muscle_name = [muscle_name]
                    if side == "_l":
                        location = {muscle_name[0]:attachment[2]}
                        muscle_name_storage[0] = muscle_name[0]
                    elif side == "_r":
                        location = {muscle_name[0]:attachment[3]}
                        muscle_name_storage[1] = muscle_name[0]
                    add_markers_to_body(model, body_name, muscle_name,location, center)
                attachment.append(muscle_name_storage[0])
                attachment.append(muscle_name_storage[1])
                i = i+1
    return model, muscle_linkages


def add_all_muscles_to_model_with_simple_names(model, local_muscle_positions, muscle_linkages):
    """
    Adds all muscles to the model based on muscle_linkages and local_muscle_positions.
    Names muscles using a simple convention like 'l_gem_1' or 'r_gem_2'.

    Args:
        model (osim.Model): OpenSim model to which the muscles will be added.
        muscle_linkages (dict): Dictionary defining muscle linkages with body parts and muscle numbers.
        local_muscle_positions (dict): Dictionary mapping muscle marker names to positions and parent bodies.
    """
    for muscle_name, attachments in muscle_linkages.items():
        # Ensure both origin ('ori') and insertion ('ins') exist for the muscle
        if 'ori' not in attachments or 'ins' not in attachments:
            print(f"Skipping {muscle_name}: Missing origin or insertion data.")
            continue



        origins = attachments['ori']
        insertions = attachments['ins']


        if origins[0][0] == "Tibia" or origins[0][0] == "Fibula" or insertions[0][0] == "Fibula" or insertions[0][0] == "Tibia":
            continue
        # Iterate through all combinations of origins and insertions
        for origin_index, origin in enumerate(origins):
            for insertion_index, insertion in enumerate(insertions):
                # Generate left muscle name
                origin_marker_name_l = origin[4]  # Example: 'ori_l_gem_1'
                insertion_marker_name_l = insertion[4]  # Example: 'ins_l_gem'
                origin_position_l = local_muscle_positions.get(origin_marker_name_l)
                insertion_position_l = local_muscle_positions.get(insertion_marker_name_l)

                if origin_position_l and insertion_position_l:
                    simple_name_l = f"l_{muscle_name.lower().replace(' ', '_')}_{origin_index + 1}_{insertion_index + 1}"
                    add_muscle_to_model(
                        model=model,
                        muscle_name=simple_name_l,
                        origin_point=origin_position_l,
                        insertion_point=insertion_position_l,
                    )

                # Generate right muscle name
                origin_marker_name_r = origin[5]  # Example: 'ori_r_gem_1'
                insertion_marker_name_r = insertion[5]  # Example: 'ins_r_gem'
                origin_position_r = local_muscle_positions.get(origin_marker_name_r)
                insertion_position_r = local_muscle_positions.get(insertion_marker_name_r)

                if origin_position_r and insertion_position_r:
                    simple_name_r = f"r_{muscle_name.lower().replace(' ', '_')}_{origin_index + 1}_{insertion_index + 1}"
                    add_muscle_to_model(
                        model=model,
                        muscle_name=simple_name_r,
                        origin_point=origin_position_r,
                        insertion_point=insertion_position_r,
                    )


def add_wrapping_objects_to_model(model, wrapping_objects):
    """
    Adds wrapping objects to the model and assigns them to the corresponding muscles.

    Parameters:
    - model (osim.Model): The OpenSim model.
    - wrapping_objects (dict): Dictionary containing wrapping object details for each muscle.

    Returns:
    - None: Updates the model in place.
    """
    # Get all forces (muscles) in the model
    force_set = model.updForceSet()

    for muscle_name, wrap_objects in wrapping_objects.items():
        for wrap_data in wrap_objects:
            try:
                # Extract wrapping object details
                wrap_name = wrap_data["name"]
                body_name = wrap_data["body"]
                wrap_type = wrap_data["type"]
                translation = wrap_data["translation"]
                rotation = wrap_data["rotation"]
                radius = wrap_data["radius"]
                length = wrap_data.get("length", None)  # Length only for cylinders
                quadrant = wrap_data.get("quadrant")

                # Get the body to attach the wrapping object
                body = model.getBodySet().get(body_name)

                # Create the appropriate wrapping object
                if wrap_type == "cylinder":
                    wrap_object = osim.WrapCylinder()
                    wrap_object.set_radius(radius)
                    wrap_object.set_length(length)
                elif wrap_type == "sphere":
                    wrap_object = osim.WrapSphere()
                    wrap_object.set_radius(radius)
                else:
                    print(f"Unknown wrapping object type '{wrap_type}' for {wrap_name}. Skipping.")
                    continue

                # Set properties of the wrapping object
                wrap_object.setName(wrap_name)
                wrap_object.set_translation(osim.Vec3(*translation))
                wrap_object.set_xyz_body_rotation(osim.Vec3(*rotation))
                wrap_object.set_quadrant(quadrant)
                # Attach to the correct body
                body.addWrapObject(wrap_object)

                print(f"Added wrapping object '{wrap_name}' to '{body_name}'.")

                # **Assign Wrapping Object to the Muscle**
                force = force_set.get(muscle_name)  # Find the muscle by name
                # Cast To Muscle
                muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(force)
                muscle.updGeometryPath().addPathWrap(wrap_object)
                print(f"Assigned wrapping object '{wrap_name}' to muscle '{muscle_name}'.")


            except Exception as e:
                print(f"Error processing wrapping object '{wrap_name}': {e}")
    return model


def add_muscle_to_model(
    model,
    muscle_name,
    origin_point,
    insertion_point,
    via_points=None,
    max_isometric_force=500.0,
    optimal_fiber_length=0.04,
    tendon_slack_length=0.2,
    pennation_angle_at_optimal=0.1,
    max_contraction_velocity=10.0,
):
    """
    Adds a Millard2012EquilibriumMuscle to an OpenSim model.

    Parameters:
    - model (osim.Model): The OpenSim model to which the muscle will be added.
    - muscle_name (str): Name of the muscle.
    - limb_side (str): Specify 'left' or 'right' for naming consistency.
    - origin_point (tuple): (body_name, Vec3) for the muscle's origin.
    - insertion_point (tuple): (body_name, Vec3) for the muscle's insertion.
    - via_points (list of tuples, optional): [(body_name, Vec3), ...] for intermediate points.
    - max_isometric_force (float): Maximum isometric force in Newtons. Default is 500 N.
    - optimal_fiber_length (float): Optimal fiber length in meters. Default is 0.04 m.
    - tendon_slack_length (float): Tendon slack length in meters. Default is 0.2 m.
    - pennation_angle_at_optimal (float): Pennation angle at optimal fiber length in radians. Default is 0.1 rad.
    - max_contraction_velocity (float): Maximum contraction velocity in fiber lengths per second. Default is 10.

    Returns:
    - None: The muscle is added directly to the model.
    """
    try:
        # Create the muscle
        muscle = osim.Millard2012EquilibriumMuscle()
        muscle.setName(muscle_name)

        # Set muscle properties
        muscle.set_max_isometric_force(max_isometric_force)
        muscle.set_optimal_fiber_length(optimal_fiber_length)
        muscle.set_tendon_slack_length(tendon_slack_length)
        muscle.set_pennation_angle_at_optimal(pennation_angle_at_optimal)
        muscle.set_max_contraction_velocity(max_contraction_velocity)

        # Set origin point
        origin_body, origin_vec = origin_point
        muscle.addNewPathPoint("origin", model.getBodySet().get(origin_body), osim.Vec3(origin_vec))

        # Set insertion point
        insertion_body, insertion_vec = insertion_point
        muscle.addNewPathPoint("insertion", model.getBodySet().get(insertion_body), osim.Vec3(insertion_vec))

        # Add via points if provided
        if via_points:
            for i, (via_body, via_vec) in enumerate(via_points):
                muscle.addNewPathPoint(f"via_{i+1}", model.getBodySet().get(via_body), osim.vec3(via_vec))

        # Add the muscle to the model
        model.addForce(muscle)
        print(f"Muscle '{muscle_name}' added to the model.")

    except Exception as e:
        print(f"Error adding muscle '{muscle_name}': {e}")


def compute_and_adjust_markers(
    model_path,
    ik_output_mot_path,
    model_marker_locations_path,
    actual_marker_positions_dict,
    output_model_path
):
    """
    Compute marker differences and adjust markers in the model.

    Args:
        model_path (str): Path to the OpenSim model file.
        ik_output_mot_path (str): Path to the IK motion file (.mot).
        model_marker_locations_path (str): Path to the model marker locations file (.sto).
        actual_marker_positions_dict (dict): Dictionary of actual marker positions.
        output_model_path (str): Path to save the updated model.

    Returns:
        None
    """
    # Load the model
    model = osim.Model(model_path)
    state = model.initSystem()

    # Initialize dictionaries
    marker_differences = {}
    model_marker_positions = {}
    actual_marker_positions = {}

    # Initialize the dictionary with empty lists for each marker
    for marker in model.getMarkerSet():
        marker_name = marker.getName()
        marker_differences[marker_name] = []
        actual_marker_positions[marker_name] = []

    # Load the time list from the IK motion file
    motion_storage = osim.Storage(ik_output_mot_path)
    time_array = osim.ArrayDouble()
    motion_storage.getTimeColumn(time_array)
    time_list = [time_array.get(i) for i in range(time_array.size())]

    # Load model marker positions from the .sto file
    model_marker_positions, model_time_list = parse_model_marker_locations(model_marker_locations_path)

    # Ensure the time lists match between IK output and model marker locations
    if not np.allclose(time_list, model_time_list):
        raise ValueError("Mismatch in time lists between IK output and model marker locations.")

    # Loop through time steps and compute marker positions
    for index, time in enumerate(time_list):
        markers = model.getMarkerSet()
        for marker in markers:
            marker_name = marker.getName()

            # Get the actual marker position
            actual_x = actual_marker_positions_dict[marker_name]['X'][index] / 1000
            actual_y = actual_marker_positions_dict[marker_name]['Y'][index] / 1000
            actual_z = actual_marker_positions_dict[marker_name]['Z'][index] / 1000
            actual_marker_position = np.array([actual_x, actual_y, actual_z])

            # Store the positions and differences
            actual_marker_positions[marker_name].append(actual_marker_position)

            if marker_name == 'LPAT' or marker_name == 'RPAT':
                continue

            model_marker_position = model_marker_positions[marker_name][index]

            # Compute marker difference
            marker_difference = np.array(actual_marker_position - model_marker_position)
            marker_differences[marker_name].append(marker_difference)

    # Remove empty marker differences
    marker_differences = {key: value for key, value in marker_differences.items() if value}

    # Compute average marker differences
    average_marker_differences = {
        marker_name: np.mean(positions, axis=0)
        for marker_name, positions in marker_differences.items()
    }

    # Adjust model markers
    adjust_model_markers(model_path, output_model_path, average_marker_differences)

#Participant inputs folder location
participant_inputs = participant_folder + "/Inputs"


#Necessary muscle dictionaries

#Correlation betwen muscle names and the relevant origins/insertion positions
muscle_linkages = {
    "Extobl": {
        "ins": [["Pelvis", "58"]],
    },
    "Intobl": {
        "ins": [["Pelvis", "59"]],
    },
    "Ercspn": {
        "ins": [["Pelvis", "105c"]],
    },
    "Glut max": {
        "ori": [["Pelvis", "106"]],
        "ins": [["Femur", "106"]],
    },
    "Glut min": {
        "ori": [["Pelvis", "108"]],
        "ins": [["Femur", "108"]],
    },
    "Tfl": {
        "ori": [["Pelvis", "109"]], #        "ins":[["Tibia", "109"]], - doesn't exist in the tibial node number file
    },
    "Obt int": {
        "ori": [["Pelvis", "111"]],
        "ins": [["Femur", "111_112_113"]],
    },
    "Obt ext": {
        "ori": [["Pelvis", "123"]],
        "ins": [["Femur", "123"]],
    },
    "Gem": {
        "ori": [["Pelvis", "112"],["Pelvis","113"]],
        "ins": [["Femur", "111_112_113"]],
    },
    "Quad fem": {
        "ori": [["Pelvis", "114"]],
        "ins": [["Femur", "114"]],
    },
    "Sar": {
        "ori": [["Pelvis", "115"]],
        "ins": [["Tibia", "115"]],
    },
    "Rect fem": {
        "ori": [["Pelvis", "116a"],["Pelvis","116a_1"]],
        "ins": [["Tibia", "116"]],
    },
    "Pect": {
        "ori": [["Pelvis", "118"]],
        "ins": [["Femur", "118"]],
    },
    "Add long": {
        "ori": [["Pelvis", "119"]],
        "ins": [["Femur", "119"]],
    },
    "Add brev": {
        "ori": [["Pelvis", "120"]],
        "ins": [["Femur", "120"]],
    },
    "Grac": {
        "ori": [["Pelvis", "122"]],
        "ins": [["Tibia", "122"]],
    },
    "Bifemlh": {
        "ori": [["Pelvis", "124a"]],
        "ins": [["Fibula", "124"]],
    },
    "Bifemsh": {
        "ori": [["Femur", "124b"]],
        "ins": [["Fibula", "124"]],
    },
    "Semimem": {
        "ori": [["Pelvis", "126"]],
        "ins": [["Tibia", "126"]],
    },
    "Iliacus": {
        "ori": [["Pelvis", "105a"]],
        "ins": [["Femur", "105"]], #just saying insertion is the same as psoas inseriton
    },
    "Glut med": {
        "ori": [["Pelvis", "107"]],
        "ins": [["Femur", "107"]],
    },
    "Add mag": {
        "ori": [["Pelvis", "121"]],
        "ins": [["Femur", "121"],["Femur","121_1"]],
    },
    "Semiten": {
        "ori": [["Pelvis", "125"]],
        "ins": [["Tibia", "125"]],
    },
    "Psoas": {

        "ins": [["Femur", "105"]],
    },
    "Peri": {
        "ins": [["Femur", "110"]], #        "ori": [["Sacrum", "110"]], - i dont know how to include this as theres no sacrum shapemodel component
    },
    "Vas lat": {
        "ori": [["Femur", "116b"]],
        "ins": [["Tibia", "116"]],
    },
    "Vas int/ articularis genus": {
        "ori": [["Femur", "117"]], #i dont know where it inserts (probably same as vas med and rext fem)

    },
    "Vas int": {
        "ori": [["Femur", "116c"]],#i dont know where it inserts (porbably saame as vas med and rect fem)

    },
    "Med gas": {
        "ori": [["Femur", "132a"]],# inserts on the foot (there is a med gas in the tibia section, unsure as to why)

    },
    "Lat head of gastrocnemius": {
        "ori": [["Femur", "133"]], #inseriton on the foot

    },
    "Popliteus m.": {
        "ori": [["Femur", "134"]], #insertion on the foot

    },
    "Tib ant": {
        "ori": [["Tibia", "127"]],#insertion on the foot

    },
    "Tib_post": {
        "ori": [["Tibia", "135"],["Fibula","135"]], #i assume theres 2 origins from the tib and fib that go to the foot

    },
    "Soleus": {
        "ori": [["Fibula", "132b"]], #inserts on the foot

    },
    "Obturator interus/gemellus": {

        "ins": [["Femur", "123"]],#i dont know where the origin is
    },
    "Ext dig": {
        "ori": [["Tibia", "128"],["Tibia","128_1"]], #inserts on foot, 2 origins?

    },
    "Flex dig": {
        "ori": [["Tibia", "136"]], #inserts on foot

    },
    "Flex hal": {
        "ori": [["Fibula", "137"]],#inserts on foot

    },
    "Per long": {
        "ori": [["Fibula", "130"],["Fibula","130_1"]],#insrets on foot

    },
    "Per brev": {
        "ori": [["Fibula", "131"]],#inserts on foot

    },
    "Ext hal": {
        "ori": [["Fibula", "129"]], #inserts on foot

    },
}



#Relates the muscle number to the nodes that make up the insertion/origin
muscle_number_to_nodes_key = parse_muscle_node_files_recursive("High_Level_Inputs/final_node_numbers")

#relates the node number to a set of coordinates
node_to_coordinate = parse_ply_files_by_side_and_bone(participant_inputs, ["Femur","Tibfib","Pelvis"])

# Call the function with your dictionaries
muscle_linkages = map_muscle_nodes_to_coordinates(
    muscle_linkages, muscle_number_to_nodes_key, node_to_coordinate
)


#Creating the average muscle positions for each muscle (might need to find a way to make a function to do this...)
muscle_positions = {}
muscle_bone_groups = {}

#muscle_names = ["Glut med", "Glut max","Glut min","Pect","Quad fem","Add long","Add brev"] #Use only the names of the origins
muscle_names = ["Add mag"]
body_names = ["pelvis_b","femur_l_b","femur_r_b","tibfib_l_b","tibfib_r_b"]




#%%Extraction of meshes from the ply files
#Import necessary functions
from Extract_Scale_Meshes import *


#Runs the conversion process
batch_convert_and_scale(input_dir = participant_inputs)


# Combine the meshes
combine_pelvis_meshes(input_dir = participant_inputs)

# Cuts the stls to the meshes folder
move_stl_meshes(input_dir = participant_inputs, output_dir = participant_folder + "/Meshes")


# %% Initialisation of models and extraction of relevant landmarks/marker placements

# Define folder paths
output_folder = participant_folder + "\Models"  # Folder to save or load models
meshes = participant_folder + "\Meshes"  # Folder containing mesh files and related data

# Initialise the OpenSim model
empty_model = osim.Model("High_Level_Inputs\Feet.osim")  # Load the base model file (Feet.osim)
state = empty_model.initSystem()  # Initialise the system for the model (necessary for modifications)

# Load and extract landmarks for left and right limbs
left_landmarks_file = search_files_by_keywords(participant_inputs, "left lms predicted")[0]  # Find the left limb landmarks file
right_landmarks_file = search_files_by_keywords(participant_inputs, "right lms predicted")[0]  # Find the right limb landmarks file
left_landmarks = load_landmarks(left_landmarks_file)  # Load left limb landmarks as a dictionary
right_landmarks = load_landmarks(right_landmarks_file)  # Load right limb landmarks as a dictionary

# Load the TRC file and extract marker placements
mocap_trc_file = search_files_by_keywords(participant_inputs, "static")[0]  # Find the TRC file containing marker data
mocap_static_trc,dontcare = read_trc_file_as_dict(mocap_trc_file)  # Read the TRC file into a dictionary for marker placements


# %% Creation of the pelvis body and pelvis joint (to ground)

# Create the pelvis body
# Define the pelvis body with a mass of 1.0 kg, a center of mass at the origin, and no moments of inertia
pelvis = osim.Body("pelvis_b", 1.0, osim.Vec3(0, 0, 0), osim.Inertia(0, 0, 0))

# Add the pelvis body to the model
empty_model.addBody(pelvis)


LASIS = rotate_coordinate_x(left_landmarks["ASIS"], 90)
RASIS = rotate_coordinate_x(right_landmarks["ASIS"], 90)
RANK = rotate_coordinate_x(right_landmarks["malleolus_med"], 90)
pelvis_sideways_vector = vector_between_points(LASIS, RASIS)  # Flexion occurs about the alignment of the pelvis ASIS landmarks
alignment_to_axis = (0,0,1)

pelvis_realignment = compute_euler_angles_from_vectors(pelvis_sideways_vector, alignment_to_axis)
pelvis_realignment[0] = 0
#pelvis_realignment[1] = 0
pelvis_realignment[2] = 0





#Height ground offset (for visualisation purposes only when opened in Opensim)
RASIS_to_RANK = np.linalg.norm(vector_between_points(RASIS, RANK))
height_offset = RASIS_to_RANK + 0.035
#height_offset = 0


# Attach the pelvis body to the ground using a FreeJoint
# A FreeJoint allows 6 degrees of freedom (translation and rotation) between the pelvis and the ground
pelvis_joint = osim.FreeJoint(
    "pelvis_to_ground",                # Name of the joint
    empty_model.getGround(),           # Parent frame (ground)
    osim.Vec3(0, height_offset, 0),                # Location of the joint in the parent frame
    osim.Vec3(0,0,0),                # Orientation of the joint in the parent frame
    pelvis,                            # Child body (pelvis)
    osim.Vec3(0, 0, 0),                # Location of the joint in the child frame
    osim.Vec3(pelvis_realignment)                 # Orientation of the joint in the child frame
)

# Add the pelvis joint to the model
empty_model.addJoint(pelvis_joint)



# %% Attaching mesh and markers to pelvis body

# Attach the mesh for the pelvis to the pelvis body
# Search the mesh files using the keyword "pelvis" and retrieve the first result
mesh_filename = search_files_by_keywords(meshes, "pelvis")[0]

# Extract the center of the pelvis mesh using trimesh and rotate it to match the coordinate system
info = extract_mesh_info_trimesh(mesh_filename)
pelvis_center = info['center']
rotated_pelvis_center = rotate_coordinate_x(pelvis_center, 90)

# Add the pelvis mesh to the body with an orientation offset to match OpenSim's axis alignment
add_mesh_to_body(empty_model, "pelvis_b", mesh_filename, offset_orientation=(-1.5708, 0, 0),
                 offset_translation=(rotated_pelvis_center[0], rotated_pelvis_center[1], rotated_pelvis_center[2]))

# Add mocap markers to the pelvis body
add_markers_to_body(empty_model, "pelvis_b", ["RASI", "LASI", "RPSI", "LPSI"], mocap_static_trc, pelvis_center)

# Add anatomical landmarks to the pelvis body with custom marker names
add_markers_to_body(empty_model, "pelvis_b", ["ASIS", "PSIS", "SAC"], left_landmarks, pelvis_center,
                    ["lms_LASI", "lms_LPSI", "lms_SAC"])
add_markers_to_body(empty_model, "pelvis_b", ["ASIS", "PSIS"], right_landmarks, pelvis_center,
                    ["lms_RASI", "lms_RPSI"])




# %% Creation of femur bodies and attachment of meshes, markers, and landmarks

# Define the femur body properties (common for both left and right femurs)
femur_mass = 8.0  # Mass of the femur in kg
femur_mass_center = osim.Vec3(0, -0.2, 0)  # Center of mass location in the femur frame
femur_inertia = osim.Inertia(0.1, 0.1, 0.01)  # Moments of inertia

# Create the left and right femur bodies
left_femur = osim.Body("femur_l_b", femur_mass, femur_mass_center, femur_inertia)
right_femur = osim.Body("femur_r_b", femur_mass, femur_mass_center, femur_inertia)

# Add the femur bodies to the model
empty_model.addBody(left_femur)
empty_model.addBody(right_femur)

# Attach the mesh for the right femur
mesh_filename = search_files_by_keywords(meshes, "right femur")[0]
info = extract_mesh_info_trimesh(mesh_filename)
femur_r_center = info['center']  # Extract center of the right femur
rotated_r_femur_center = rotate_coordinate_x(femur_r_center, 90)  # Rotate to match coordinate system
add_mesh_to_body(empty_model, "femur_r_b", mesh_filename, offset_orientation=(-1.5708, 0, 0),
                 offset_translation=(rotated_r_femur_center[0], rotated_r_femur_center[1], rotated_r_femur_center[2]))

# Attach the mesh for the left femur
mesh_filename = search_files_by_keywords(meshes, "left femur")[0]
info = extract_mesh_info_trimesh(mesh_filename)
femur_l_center = info['center']  # Extract center of the left femur
rotated_l_femur_center = rotate_coordinate_x(femur_l_center, 90)  # Rotate to match coordinate system
add_mesh_to_body(empty_model, "femur_l_b", mesh_filename, offset_orientation=(-1.5708, 0, 0),
                 offset_translation=(rotated_l_femur_center[0], rotated_l_femur_center[1], rotated_l_femur_center[2]))

# Add mocap markers to the femur bodies
add_markers_to_body(empty_model, "femur_l_b", ["LTHI", "LPAT", "LKNE"], mocap_static_trc, femur_l_center)
add_markers_to_body(empty_model, "femur_r_b", ["RTHI", "RPAT", "RKNE"], mocap_static_trc, femur_r_center)

# Add anatomical landmarks to the femur bodies with custom marker names
add_markers_to_body(empty_model, "femur_l_b", ["LEC", "MEC"], left_landmarks, femur_l_center, ["lms_LLEC", "lms_LMEC"])
add_markers_to_body(empty_model, "femur_r_b", ["LEC", "MEC"], right_landmarks, femur_r_center, ["lms_RLEC", "lms_RMEC"])


# %% Creation of the left hip joint coordinate system

# Extract landmarks required to position the joint coordinate systems of the left hip joint
LASIS = rotate_coordinate_x(left_landmarks["ASIS"], 90)
RASIS = rotate_coordinate_x(right_landmarks["ASIS"], 90)
l_LEC = rotate_coordinate_x(left_landmarks["LEC"], 90)
l_MEC = rotate_coordinate_x(left_landmarks["MEC"], 90)
l_HJC = rotate_coordinate_x(left_landmarks["hjc"], 90)
l_EC_midpoint = midpoint_3d(l_LEC, l_MEC)

# Compute the vectors for flexion and rotation of the left hip joint
flexion_vector = vector_between_points(RASIS, LASIS)  # Flexion occurs about the alignment of the pelvis ASIS landmarks
rotation_vector = vector_between_points(l_HJC, l_EC_midpoint)  # Rotation about the HJC and epicondylar midpoint

# Compute the Euler rotation angles to adjust the coordinate system for the left hip joint
l_hip_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')



LKNE_sideways_vector = vector_between_points(l_LEC, l_MEC)
alignment_to_axis = (0,0,1)

LKNE_alignment_angles = compute_euler_angles_from_vectors(LKNE_sideways_vector, alignment_to_axis)
LKNE_alignment_angles[0] = 0
LKNE_alignment_angles[2] = 0

LHIP_vertical_vector = vector_between_points(l_EC_midpoint, l_HJC)
alignment_to_axis = (0,-1,0)
LHIP_vert_alignment_angles = compute_euler_angles_from_vectors(LHIP_vertical_vector, alignment_to_axis)
LHIP_vert_alignment_angles[0] = 0
LHIP_vert_alignment_angles[1] = 0


# %% Positioning of the left hip joint

# Compute the absolute translation required to position the left femur
l_hip_position_total = determine_transform_child_to_parent(rotated_pelvis_center, rotated_l_femur_center,
                                                            left_landmarks["ASIS"], left_landmarks["LEC"])

# Extract the left hip joint center
l_hjc = left_landmarks['hjc']

# Compute the child position as the vector between the femur center and the joint center
l_hip_child_position = rotate_coordinate_x(vector_between_points(l_hjc, femur_l_center), 90)

# Compute the parent position by combining the absolute and child positions
l_hip_parent_position = l_hip_position_total + l_hip_child_position

# Create the spatial transform for the custom left hip joint
spatial_transform_left = osim.SpatialTransform()

# First rotation (Flexion/Extension) along X-axis
flexion_axis_left = spatial_transform_left.updTransformAxis(0)
flexion_axis_left.setCoordinateNames(osim.ArrayStr("hip_flexion_l", 1))
flexion_axis_left.setAxis(osim.Vec3(0, 0, 1))  # X-axis
flexion_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Second rotation (Adduction/Abduction) along Z-axis
adduction_axis_left = spatial_transform_left.updTransformAxis(1)
adduction_axis_left.setCoordinateNames(osim.ArrayStr("hip_adduction_l", 1))
adduction_axis_left.setAxis(osim.Vec3(1, 0, 0))  # Z-axis
adduction_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Third rotation (Internal/External Rotation) along Y-axis
rotation_axis_left = spatial_transform_left.updTransformAxis(2)
rotation_axis_left.setCoordinateNames(osim.ArrayStr("hip_rotation_l", 1))
rotation_axis_left.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
rotation_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Restore your original orientation adjustments for the femur frame
adjusted_femur_orientation_left = (
    l_hip_angles + LKNE_alignment_angles - pelvis_realignment - LHIP_vert_alignment_angles
)

# Create the custom left hip joint with all restored parameters
left_hip_joint = osim.CustomJoint(
    "femur_l_to_pelvis",              # Joint name
    pelvis,                           # Parent frame (Pelvis)
    osim.Vec3(l_hip_parent_position), # Location in parent frame
    osim.Vec3(l_hip_angles),          # Orientation in parent frame
    left_femur,                       # Child frame (Femur)
    osim.Vec3(l_hip_child_position),  # Location in child frame
    osim.Vec3(adjusted_femur_orientation_left),  # Adjusted orientation in child frame
    spatial_transform_left             # The defined spatial transform
)


# %% Creation of the right hip joint coordinate system

# Extract landmarks required to position the joint coordinate systems of the right hip joint
LASIS = rotate_coordinate_x(left_landmarks["ASIS"], 90)
RASIS = rotate_coordinate_x(right_landmarks["ASIS"], 90)
r_LEC = rotate_coordinate_x(right_landmarks["LEC"], 90)
r_MEC = rotate_coordinate_x(right_landmarks["MEC"], 90)
r_HJC = rotate_coordinate_x(right_landmarks["hjc"], 90)
r_EC_midpoint = midpoint_3d(r_LEC, r_MEC)

# Compute the vectors for flexion and rotation of the right hip joint
flexion_vector = vector_between_points(RASIS, LASIS)  # Flexion occurs about the alignment of the pelvis ASIS landmarks
rotation_vector = vector_between_points(r_HJC, r_EC_midpoint)  # Rotation about the HJC and epicondylar midpoint

# Compute the Euler rotation angles to adjust the coordinate system for the right hip joint
r_hip_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')

# Potential reorientation of the right femur/body to align with ISB coordinate definitions
r_femur_angles = align_y_axis_with_vector_and_z_axis_to_plane(rotation_vector, [r_HJC, r_LEC, r_MEC])


RKNE_sideways_vector = vector_between_points(r_MEC, r_LEC)
alignment_to_axis = (0,0,1)

RKNE_alignment_angles = compute_euler_angles_from_vectors(RKNE_sideways_vector, alignment_to_axis)
RKNE_alignment_angles[0] = 0
RKNE_alignment_angles[2] = 0

RHIP_vertical_vector = vector_between_points(r_EC_midpoint, r_HJC)
alignment_to_axis = (0,-1,0)
RHIP_vert_alignment_angles = compute_euler_angles_from_vectors(RHIP_vertical_vector, alignment_to_axis)
RHIP_vert_alignment_angles[0] = 0
RHIP_vert_alignment_angles[1] = 0



# %% Positioning of the right hip joint

# Compute the absolute translation required to position the right femur
r_hip_position_total = determine_transform_child_to_parent(rotated_pelvis_center, rotated_r_femur_center,
                                                            right_landmarks["ASIS"], right_landmarks["LEC"])

# Extract the right hip joint center
r_hjc = right_landmarks['hjc']

# Compute the child position as the vector between the femur center and the joint center
r_hip_child_position = rotate_coordinate_x(vector_between_points(r_hjc, femur_r_center), 90)

# Compute the parent position by combining the absolute and child positions
r_hip_parent_position = r_hip_position_total + r_hip_child_position

################################################

# Create the spatial transform for the custom joint
spatial_transform = osim.SpatialTransform()

# First rotation (Flexion/Extension) along X-axis
flexion_axis = spatial_transform.updTransformAxis(0)
flexion_axis.setCoordinateNames(osim.ArrayStr("hip_flexion_r", 1))
flexion_axis.setAxis(osim.Vec3(0, 0, 1))  # X-axis
flexion_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Second rotation (Adduction/Abduction) along Z-axis
adduction_axis = spatial_transform.updTransformAxis(1)
adduction_axis.setCoordinateNames(osim.ArrayStr("hip_adduction_r", 1))
adduction_axis.setAxis(osim.Vec3(1, 0, 0))  # Z-axis
adduction_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Third rotation (Internal/External Rotation) along Y-axis
rotation_axis = spatial_transform.updTransformAxis(2)
rotation_axis.setCoordinateNames(osim.ArrayStr("hip_rotation_r", 1))
rotation_axis.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
rotation_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

# Restore your original orientation adjustments for the femur frame
adjusted_femur_orientation = (
    r_hip_angles + RKNE_alignment_angles - pelvis_realignment - RHIP_vert_alignment_angles
)

# Create the custom hip joint with all restored parameters
right_hip_joint = osim.CustomJoint(
    "femur_r_to_pelvis",              # Joint name
    pelvis,                           # Parent frame (Pelvis)
    osim.Vec3(r_hip_parent_position), # Location in parent frame
    osim.Vec3(r_hip_angles),          # Orientation in parent frame
    right_femur,                      # Child frame (Femur)
    osim.Vec3(r_hip_child_position),  # Location in child frame
    osim.Vec3(adjusted_femur_orientation),  # Adjusted orientation in child frame
    spatial_transform                  # The defined spatial transform
)

########################################################################################

# Add the hip joints to the model
empty_model.addJoint(left_hip_joint)
empty_model.addJoint(right_hip_joint)



# %% Creation of the Tibia/Fibula (TibFib) Bodies

# Define the tibfib body properties
tibfib_mass = 5.0  # Mass of the tibfib body in kilograms
tibfib_mass_center = osim.Vec3(0, -0.3, 0)  # Center of mass location relative to the tibfib frame
tibfib_inertia = osim.Inertia(0.08, 0.08, 0.005)  # Moments of inertia for the tibfib body

# Create the left and right tibfib bodies
left_tibfib = osim.Body("tibfib_l_b", tibfib_mass, tibfib_mass_center, tibfib_inertia)  # Left tibfib
right_tibfib = osim.Body("tibfib_r_b", tibfib_mass, tibfib_mass_center, tibfib_inertia)  # Right tibfib

# Add the tibfib bodies to the model
empty_model.addBody(left_tibfib)  # Add the left tibfib body to the model
empty_model.addBody(right_tibfib)  # Add the right tibfib body to the model


# Attach the mesh for the right tibfib body
# Search for the mesh file corresponding to the right tibfib
mesh_filename = search_files_by_keywords(meshes, "right tibfib")[0]
info = extract_mesh_info_trimesh(mesh_filename)  # Extract mesh information using trimesh
tibfib_r_center = info['center']  # Get the center of the mesh
rotated_r_tibfib_center = rotate_coordinate_x(tibfib_r_center, 90)  # Rotate the center to align with OpenSim's coordinate system

# Add the mesh to the right tibfib body with an orientation offset to align axes
add_mesh_to_body(empty_model, "tibfib_r_b", mesh_filename,
                 offset_orientation=(-1.5708, 0, 0),  # Align the mesh orientation with OpenSim axes
                 offset_translation=(rotated_r_tibfib_center[0], rotated_r_tibfib_center[1], rotated_r_tibfib_center[2]))


# Attach the mesh for the left tibfib body
# Search for the mesh file corresponding to the left tibfib
mesh_filename = search_files_by_keywords(meshes, "left tibfib")[0]
info = extract_mesh_info_trimesh(mesh_filename)  # Extract mesh information using trimesh
tibfib_l_center = info['center']  # Get the center of the mesh
rotated_l_tibfib_center = rotate_coordinate_x(tibfib_l_center, 90)  # Rotate the center to align with OpenSim's coordinate system

# Add the mesh to the left tibfib body with an orientation offset to align axes
add_mesh_to_body(empty_model, "tibfib_l_b", mesh_filename,
                 offset_orientation=(-1.5708, 0, 0),  # Align the mesh orientation with OpenSim axes
                 offset_translation=(rotated_l_tibfib_center[0], rotated_l_tibfib_center[1], rotated_l_tibfib_center[2]))


# Add mocap markers to the tibfib bodies
# Add mocap markers for the left tibfib body
add_markers_to_body(empty_model, "tibfib_l_b", ["LANK", "LTIB","LTOE","LHEE"], mocap_static_trc, tibfib_l_center)

#Add landmark markers for the left tibfib body
add_markers_to_body(empty_model, "tibfib_l_b", ["malleolus_med", "malleolus_lat"], left_landmarks, tibfib_l_center,["lms_LMMAL","lms_LLMAL"])

# Add mocap markers for the right tibfib body
add_markers_to_body(empty_model, "tibfib_r_b", ["RANK", "RTIB","RTOE","RHEE"], mocap_static_trc, tibfib_r_center)

#Add landmark markers for the right tibfib body
add_markers_to_body(empty_model, "tibfib_r_b", ["malleolus_med", "malleolus_lat"], right_landmarks, tibfib_r_center,["lms_RMMAL","lms_RLMAL"])


# %% Creation of the left knee joint coordinate system

# Compute the flexion vector
# Flexion and extension occur about the line connecting the medial and lateral epicondyles
flexion_vector = vector_between_points(l_MEC, l_LEC)

# Compute the rotation vector
# Rotation occurs about the line connecting the hip joint center (HJC) and the midpoint of the femoral epicondyles
rotation_vector = vector_between_points(l_HJC, l_EC_midpoint)

# Calculate the Euler rotation angles (radians) to align the knee joint coordinate system
# The z-axis is aligned with the flexion vector, and other axes are orthogonalized appropriately
l_knee_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')


# %% Positioning of the left knee joint

# Compute the total transformation required to align the tibfib with the femur
# This determines the relative movement required to transition from the femur's frame to the tibfib's frame
l_knee_position_total = determine_transform_child_to_parent(
    rotated_l_femur_center, rotated_l_tibfib_center, left_landmarks["LEC"], left_landmarks["malleolus_lat"]
)

# Extract the medial and lateral epicondyle landmarks
lec = left_landmarks["LEC"]  # Lateral epicondyle landmark
mec = left_landmarks["MEC"]  # Medial epicondyle landmark

# Compute the midpoint between the lateral and medial epicondyles
EC_midpoint = midpoint_3d(lec, mec)

# Compute the child position (location of the joint in the tibfib frame)
# This is the vector between the midpoint of the epicondyles and the center of the tibfib body
l_knee_child_position = rotate_coordinate_x(vector_between_points(EC_midpoint, tibfib_l_center), 90)

# Compute the parent position (location of the joint in the femur frame)
# This is the total transformation combined with the child position
l_knee_parent_position = l_knee_position_total + l_knee_child_position



# Find the midpoint of the medial and lateral malleoli
malleolus_med = rotate_coordinate_x(left_landmarks["malleolus_med"],90) # Medial malleolus landmark
LKNE_vertical_vector = vector_between_points(malleolus_med, rotate_coordinate_x(mec,90))
alignment_to_axis = (0,-1,0)
LKNE_vert_alignment_angles = compute_euler_angles_from_vectors(LKNE_vertical_vector, alignment_to_axis)
LKNE_vert_alignment_angles[0] = 0
LKNE_vert_alignment_angles[1] = 0
#LKNE_vert_alignment_angles[2] = 0 #Comment out for neutralising of the left tibia




# %% Define the left knee joint

# Define the knee joint connecting the left tibfib to the left femur
# A PinJoint allows rotation about a single axis (flexion/extension in this case)
left_knee_joint = osim.PinJoint(
    "tibfib_l_to_femur_l",          # Name of the joint
    left_femur,                     # Parent body (femur)
    osim.Vec3(l_knee_parent_position),  # Location of the joint in the femur frame
    osim.Vec3(l_knee_angles),           # Orientation of the joint in the femur frame
    left_tibfib,                        # Child body (tibfib)
    osim.Vec3(l_knee_child_position),   # Location of the joint in the tibfib frame
    osim.Vec3(l_knee_angles-LKNE_vert_alignment_angles+LHIP_vert_alignment_angles)            # Orientation of the joint in the tibfib frame
)


# %% Positioning of the right knee joint

# Compute the total transformation required to align the tibfib with the femur
# This determines the relative movement required to transition from the femur's frame to the tibfib's frame
r_knee_position_total = determine_transform_child_to_parent(
    rotated_r_femur_center, rotated_r_tibfib_center, right_landmarks["LEC"], right_landmarks["malleolus_lat"]
)

# Extract the medial and lateral epicondyle landmarks
lec = right_landmarks["LEC"]  # Lateral epicondyle landmark
mec = right_landmarks["MEC"]  # Medial epicondyle landmark

# Compute the midpoint between the lateral and medial epicondyles
EC_midpoint = midpoint_3d(lec, mec)

# Compute the child position (location of the joint in the tibfib frame)
# This is the vector between the midpoint of the epicondyles and the center of the tibfib body
r_knee_child_position = rotate_coordinate_x(vector_between_points(EC_midpoint, tibfib_r_center), 90)

# Compute the parent position (location of the joint in the femur frame)
# This is the total transformation combined with the child position
r_knee_parent_position = r_knee_position_total + r_knee_child_position


# %% Creation of the right knee joint coordinate system

# Compute the flexion vector
# Flexion and extension occur about the line connecting the medial and lateral epicondyles
flexion_vector = vector_between_points(r_LEC, r_MEC)

# Compute the rotation vector
# Rotation occurs about the line connecting the hip joint center (HJC) and the midpoint of the femoral epicondyles
rotation_vector = vector_between_points(r_HJC, r_EC_midpoint)

# Calculate the Euler rotation angles (radians) to align the knee joint coordinate system
# The z-axis is aligned with the flexion vector, and other axes are orthogonalized appropriately
r_knee_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')


# Find the midpoint of the medial and lateral malleoli
malleolus_med = rotate_coordinate_x(right_landmarks["malleolus_med"],90) # Medial malleolus landmark
RKNE_vertical_vector = vector_between_points(malleolus_med, rotate_coordinate_x(mec,90))
alignment_to_axis = (0,-1,0)
RKNE_vert_alignment_angles = compute_euler_angles_from_vectors(RKNE_vertical_vector, alignment_to_axis)
RKNE_vert_alignment_angles[0] = 0
RKNE_vert_alignment_angles[1] = 0
#RKNE_vert_alignment_angles[2] = 0 #Comment out for neutralising of the right tibia



# %% Define the right knee joint

# Define the knee joint connecting the right tibfib to the right femur
# A PinJoint allows rotation about a single axis (flexion/extension in this case)
right_knee_joint = osim.PinJoint(
    "tibfib_r_to_femur_r",          # Name of the joint
    right_femur,                    # Parent body (femur)
    osim.Vec3(r_knee_parent_position),  # Location of the joint in the femur frame
    osim.Vec3(r_knee_angles),           # Orientation of the joint in the femur frame
    right_tibfib,                        # Child body (tibfib)
    osim.Vec3(r_knee_child_position),   # Location of the joint in the tibfib frame
    osim.Vec3(r_knee_angles-RKNE_vert_alignment_angles+RHIP_vert_alignment_angles)            # Orientation of the joint in the tibfib frame
)

# %% Adding the knee joints to the model

# Add the left knee joint to the OpenSim model
# This connects the left tibfib to the left femur, allowing flexion/extension motion
empty_model.addJoint(left_knee_joint)

# Add the right knee joint to the OpenSim model
# This connects the right tibfib to the right femur, allowing flexion/extension motion
empty_model.addJoint(right_knee_joint)


#%% Access the feet bodies, as they already exist as part of the base model.

# Access the body named "talus_l_b"
left_talus = empty_model.getBodySet().get("talus_l_b")
# Access the body named "talus_r_b"
right_talus = empty_model.getBodySet().get("talus_r_b")


#%% Remove the pre-existing joints present for these feet bodies

# Locate the joint by name in the model's JointSet
joint_name_to_remove = "talus_l_b_to_ground"  # Replace with the actual joint name
if empty_model.getJointSet().contains(joint_name_to_remove):
    joint_to_remove = empty_model.getJointSet().get(joint_name_to_remove)
    empty_model.updJointSet().remove(joint_to_remove)
    print(f"Joint '{joint_name_to_remove}' has been removed.")
else:
    print(f"Joint '{joint_name_to_remove}' not found in the model.")

joint_name_to_remove = "talus_r_b_to_ground"  # Repeat for the right side if needed
if empty_model.getJointSet().contains(joint_name_to_remove):
    joint_to_remove = empty_model.getJointSet().get(joint_name_to_remove)
    empty_model.updJointSet().remove(joint_to_remove)
    print(f"Joint '{joint_name_to_remove}' has been removed.")
else:
    print(f"Joint '{joint_name_to_remove}' not found in the model.")

# %% Creation of the left ankle joint coordinate system and positioning

# Define manual adjustments for the left and right talus positions in the child frame
manual__l_talus_positioning_child = (-0.001, 0.017, -0.0025)  # Manual adjustment for left talus
manual_r_talus_positioning_child = (-0.001, 0.017, 0.0025)   # Manual adjustment for right talus


# Calculate the left ankle joint center
# Find the midpoint of the medial and lateral malleoli
malleolus_lat = left_landmarks["malleolus_lat"]  # Lateral malleolus landmark
malleolus_med = left_landmarks["malleolus_med"]  # Medial malleolus landmark
mal_midpoint = midpoint_3d(malleolus_lat, malleolus_med)  # Midpoint between malleoli

# Compute the shift of the talus relative to the tibfib in the rotated coordinate system
l_talus_shift = vector_between_points(rotate_coordinate_x(mal_midpoint, 90), rotated_l_tibfib_center)

# Rotate malleolus landmarks to align with OpenSim's coordinate system
l_LMAL = rotate_coordinate_x(malleolus_lat, 90)  # Rotated lateral malleolus
l_MMAL = rotate_coordinate_x(malleolus_med, 90)  # Rotated medial malleolus


# %% Reorientation of the left ankle joint coordinate system

# Compute the flexion vector
# Flexion and extension occur about the line connecting the medial and lateral malleoli
flexion_vector = vector_between_points(l_MMAL, l_LMAL)

# Compute the midpoint of the malleoli
l_MAL_midpoint = midpoint_3d(l_MMAL, l_LMAL)

# Compute the rotation vector
# Rotation occurs about the line connecting the femoral epicondyle midpoint (l_EC_midpoint)
# and the midpoint of the malleoli (l_MAL_midpoint)
rotation_vector = vector_between_points(l_EC_midpoint, l_MAL_midpoint)

# Calculate the Euler rotation angles (radians) to align the left ankle joint coordinate system
# The z-axis is aligned with the flexion vector, and other axes are orthogonalized appropriately
l_ankle_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')


# %% Reorientation of the left foot relative to the ankle joint  - work in prgress






# %% Definition of the left ankle joint

# Define the ankle joint connecting the left talus to the left tibfib
# A PinJoint allows rotation about a single axis (flexion/extension in this case)
left_ankle_joint = osim.PinJoint(
    "talus_l_to_tibfib_l",               # Name of the joint
    left_tibfib,                         # Parent body (tibfib)
    osim.Vec3(l_talus_shift),            # Location of the joint in the tibfib frame
    osim.Vec3(l_ankle_angles),           # Orientation of the joint in the tibfib frame
    left_talus,                          # Child body (talus)
    osim.Vec3(manual__l_talus_positioning_child),  # Manually adjusted location of the joint in the talus frame
    osim.Vec3(l_ankle_angles)            # Orientation of the joint in the talus frame
)

# %% Creation of the right ankle joint coordinate system and positioning

# Calculate the right ankle joint center
# Find the midpoint of the medial and lateral malleoli
malleolus_lat = right_landmarks["malleolus_lat"]  # Lateral malleolus landmark
malleolus_med = right_landmarks["malleolus_med"]  # Medial malleolus landmark
mal_midpoint = midpoint_3d(malleolus_lat, malleolus_med)  # Midpoint between malleoli

# Compute the shift of the talus relative to the tibfib in the rotated coordinate system
r_talus_shift = vector_between_points(rotate_coordinate_x(mal_midpoint, 90), rotated_r_tibfib_center)

# Rotate malleolus landmarks to align with OpenSim's coordinate system
r_LMAL = rotate_coordinate_x(malleolus_lat, 90)  # Rotated lateral malleolus
r_MMAL = rotate_coordinate_x(malleolus_med, 90)  # Rotated medial malleolus


# %% Reorientation of the right ankle joint coordinate system

# Compute the flexion vector
# Flexion and extension occur about the line connecting the medial and lateral malleoli
flexion_vector = vector_between_points(r_LMAL, r_MMAL)

# Compute the midpoint of the malleoli
r_MAL_midpoint = midpoint_3d(r_MMAL, r_LMAL)

# Compute the rotation vector
# Rotation occurs about the line connecting the femoral epicondyle midpoint (r_EC_midpoint)
# and the midpoint of the malleoli (r_MAL_midpoint)
rotation_vector = vector_between_points(r_EC_midpoint, r_MAL_midpoint)

# Calculate the Euler rotation angles (radians) to align the right ankle joint coordinate system
# The z-axis is aligned with the flexion vector, and other axes are orthogonalized appropriately
r_ankle_angles = calculate_euler_to_align_axis_with_optimization(flexion_vector, rotation_vector, 'z')


# %% Reorientation of the right foot relative to the ankle joint





# %% Definition of the right ankle joint

# Define the ankle joint connecting the right talus to the right tibfib
# A PinJoint allows rotation about a single axis (flexion/extension in this case)
right_ankle_joint = osim.PinJoint(
    "talus_r_to_tibfib_r",               # Name of the joint
    right_tibfib,                        # Parent body (tibfib)
    osim.Vec3(r_talus_shift),            # Location of the joint in the tibfib frame
    osim.Vec3(r_ankle_angles),           # Orientation of the joint in the tibfib frame
    right_talus,                         # Child body (talus)
    osim.Vec3(manual_r_talus_positioning_child),  # Manually adjusted location of the joint in the talus frame
    osim.Vec3(r_ankle_angles))


# %% Add the ankle joints to the model

# Add the left ankle joint to the OpenSim model
empty_model.addJoint(left_ankle_joint)

# Add the right ankle joint to the OpenSim model
empty_model.addJoint(right_ankle_joint)

center_info = {
    "Pelvis": pelvis_center,
    "Femur": [femur_l_center,femur_r_center],
    "Tibfib": [tibfib_l_center,tibfib_r_center],
}

empty_model, muscle_linkages = add_all_muscle_attachment_markers(empty_model,muscle_linkages,center_info)


# Finalise the connections of the model
empty_model.finalizeConnections()

if participant_folder:
    # Extract the directory name as the model name and replace spaces with underscores
    model_name = os.path.basename(participant_folder).replace(" ", "_")

    # Update the model name
    update_model_name(empty_model, model_name)

    # Save the model with the same name
    save_model_to_folder(empty_model, output_folder, f"{model_name}.osim")
else:
    print("No directory selected.")






perform_updates = True

if perform_updates == True:

    output_file = output_folder +"/"f"{model_name}.osim"

    # Load the selected model
    model = empty_model

    # Print out all the body names in the model
    bodySet = model.getBodySet()
    #for i in range(bodySet.getSize()):
        #print(bodySet.get(i).getName())

    # Locate hip joints
    l_hip_joint = model.getJointSet().get('femur_l_to_pelvis')
    r_hip_joint = model.getJointSet().get('femur_r_to_pelvis')

    # Locate knee joints
    l_knee_joint = model.getJointSet().get('tibfib_l_to_femur_l')
    r_knee_joint = model.getJointSet().get('tibfib_r_to_femur_r')

    # Locate Ankle joints
    l_ankle_joint = model.getJointSet().get('talus_l_to_tibfib_l')
    r_ankle_joint = model.getJointSet().get('talus_r_to_tibfib_r')

    pelvis_joint = model.getJointSet().get('pelvis_to_ground')

    pelvis_tilt = pelvis_joint.upd_coordinates(2)
    pelvis_obliquity = pelvis_joint.upd_coordinates(0)
    pelvis_rotation = pelvis_joint.upd_coordinates(1)

    pelvis_tilt.setName("pelvis_tilt")
    pelvis_obliquity.setName("pelvis_list")
    pelvis_rotation.setName("pelvis_rotation")

    # Access and rename the translational coordinates
    pelvis_translation_x = pelvis_joint.upd_coordinates(3)  # Translation along x-axis
    pelvis_translation_y = pelvis_joint.upd_coordinates(4)  # Translation along y-axis
    pelvis_translation_z = pelvis_joint.upd_coordinates(5)  # Translation along z-axis

    pelvis_translation_x.setName("pelvis_tx")
    pelvis_translation_y.setName("pelvis_ty")
    pelvis_translation_z.setName("pelvis_tz")




    # Set coordinates range for left hip joint
    l_hip_flexion = l_hip_joint.upd_coordinates(0)
    l_hip_abduction = l_hip_joint.upd_coordinates(1)
    l_hip_rotation = l_hip_joint.upd_coordinates(2)

    l_hip_flexion.setRangeMin(-1.5)
    l_hip_flexion.setRangeMax(1.8)

    l_hip_abduction.setRangeMin(-0.8)
    l_hip_abduction.setRangeMax(1.2)

    l_hip_rotation.setRangeMin(-0.8)
    l_hip_rotation.setRangeMax(0.8)


    # Set coordinates range for right hip joint
    r_hip_flexion = r_hip_joint.upd_coordinates(0)
    r_hip_abduction = r_hip_joint.upd_coordinates(1)
    r_hip_rotation = r_hip_joint.upd_coordinates(2)

    r_hip_flexion.setRangeMin(-1.5)
    r_hip_flexion.setRangeMax(1.8)

    r_hip_abduction.setRangeMin(-1.2)
    r_hip_abduction.setRangeMax(0.8)

    r_hip_rotation.setRangeMin(-0.8)
    r_hip_rotation.setRangeMax(0.8)

    # Set coordinates range and names for left knee joint
    l_knee_flexion = l_knee_joint.upd_coordinates(0)
    l_knee_flexion.setName("knee_flexion_l")
    l_knee_flexion.setRangeMin(-2.2)
    l_knee_flexion.setRangeMax(0.0)

    # Set coordinates range and names for right knee joint
    r_knee_flexion = r_knee_joint.upd_coordinates(0)
    r_knee_flexion.setName("knee_flexion_r")
    r_knee_flexion.setRangeMin(-2.2)
    r_knee_flexion.setRangeMax(0.0)

    # Set coordinates range and names for right ankle joint
    r_ankle_flexion = r_ankle_joint.upd_coordinates(0)
    r_ankle_flexion.setName("ankle_angle_r")
    r_ankle_flexion.setRangeMin(-1)
    r_ankle_flexion.setRangeMax(0.8)
    # Set coordinates range and names for right ankle joint
    l_ankle_flexion = l_ankle_joint.upd_coordinates(0)
    l_ankle_flexion.setName("ankle_angle_l")
    l_ankle_flexion.setRangeMin(-1)
    l_ankle_flexion.setRangeMax(0.8)

    # Locate body segments based on printed names
    pelvis = model.getBodySet().get('pelvis_b')
    femur_l = model.getBodySet().get('femur_l_b')
    femur_r = model.getBodySet().get('femur_r_b')
    tibfib_l = model.getBodySet().get('tibfib_l_b')
    tibfib_r = model.getBodySet().get('tibfib_r_b')







    # Function to set mass, center of mass, and inertia
    def set_mass_com_inertia(body, mass, com, inertia):
        body.setMass(mass)
        body.setMassCenter(osim.Vec3(*com))
        body.setInertia(osim.Inertia(inertia[0], inertia[1], inertia[2], inertia[3], inertia[4], inertia[5]))


    # Set mass for each body (kg)
    masses = {
        'pelvis_b': 6.5,
        'femur_l_b': 6.0,
        'femur_r_b': 6.0,
        'tibfib_l_b': 2.5,
        'tibfib_r_b': 2.5}

    # Set centre of mass for each body
    coms = {
        'pelvis_b': [-0.01, 0.0, 0.0],
        'femur_l_b': [0.0, -0.125, 0.0],
        'femur_r_b': [0.0, -0.125, 0.0],
        'tibfib_l_b': [0.0, -0.12, 0.0],
        'tibfib_r_b': [0.0, -0.12, 0.0]}

    # Set inertia for each body
    inertias = {
        'pelvis_b': [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        'femur_l_b': [0.1, 0.025, 0.1, 0.0, 0.0, 0.0],
        'femur_r_b': [0.1, 0.025, 0.1, 0.0, 0.0, 0.0],
        'tibfib_l_b': [0.025, 0.0025, 0.025, 0.0, 0.0, 0.0],
        'tibfib_r_b': [0.025, 0.0025, 0.025, 0.0, 0.0, 0.0]}

    # Apply mass, center of mass, and inertia to each body segment
    set_mass_com_inertia(pelvis, masses['pelvis_b'], coms['pelvis_b'], inertias['pelvis_b'])
    set_mass_com_inertia(femur_l, masses['femur_l_b'], coms['femur_l_b'], inertias['femur_l_b'])
    set_mass_com_inertia(femur_r, masses['femur_r_b'], coms['femur_r_b'], inertias['femur_r_b'])
    set_mass_com_inertia(tibfib_l, masses['tibfib_l_b'], coms['tibfib_l_b'], inertias['tibfib_l_b'])
    set_mass_com_inertia(tibfib_r, masses['tibfib_r_b'], coms['tibfib_r_b'], inertias['tibfib_r_b'])

    # Finalise the initial iteration of model
    model.finalizeConnections()
    model.printToXML(output_file)

    input_file = output_file

    ##########################################################
    # Joint names and new rotation axes
    joints_to_update = ["calcn_l_to_talus_l"]
    new_rotation_axes = [(-0.78718, -0.604747, -0.120949), (0, 1, 0),
                         (-0.120949, 0, 0.78718)]  # Example: standard X, Y, Z axes
    # Update rotation axes
    update_rotation_axes(input_file, output_file, joints_to_update, new_rotation_axes)

    # Joint names and new rotation axes
    joints_to_update = ["calcn_r_to_talus_r"]
    new_rotation_axes = [(0.78718, 0.604747, -0.120949), (0, 1, 0),
                         (-0.120949, 0, -0.78718)]  # Example: standard X, Y, Z axes
    # Update rotation axes
    update_rotation_axes(input_file, output_file, joints_to_update, new_rotation_axes)
    ################################################################

    # utilise second function (move_rx...) to move the coordinate system
    input_file = output_file  # Replace with your input .osim file
    output_file = output_file  # Replace with your desired output file name

    # List of joints to modify
    joints_to_modify = ["calcn_l_to_talus_l", "calcn_r_to_talus_r"]
    # Call the function
    move_rx_to_first_rotation(input_file, output_file, joints_to_modify)

    # Use final function to update the subtalar joints
    input_file = output_file  # Replace with the path to your current .osim file
    output_file = output_file  # Replace with the desired output path
    update_subtalar_joint(input_file, output_file, "calcn_l_to_talus_l")
    update_subtalar_joint(input_file, output_file, "calcn_r_to_talus_r")

    input_file = output_file  # Path to input .osim file
    output_file = output_file  # Path to save the updated .osim file

    # List of joint updates (joint_name, new_coordinate_name)
    updates = [
        ("calcn_l_to_talus_l", "subtalar_angle_l"),
        ("calcn_r_to_talus_r", "subtalar_angle_r"),
    ]

    # Call the function
    update_rx_coordinates(input_file, output_file, updates)

    # Update the range for the left and right subtalar joints
    update_subtalar_joint_range(input_file, output_file, "subtalar_angle_l", -1, 1)
    update_subtalar_joint_range(input_file, output_file, "subtalar_angle_r", -1, 1)



#%% Reinitialise the model for feet adjustments

# Reload the model
empty_model = osim.Model(output_file)

# Initialize the model's system
state = empty_model.initSystem()
#%% === Adjust Orientation of the Left Foot ===

# Access the markers again after reinitialization
toe_marker = empty_model.getMarkerSet().get("LTOE")  # Left toe marker
heel_marker = empty_model.getMarkerSet().get("LHEE")  # Left heel marker

# Get marker positions in their local body frames
toe_local_position = toe_marker.get_location()  # Marker position relative to toes_l_b
heel_local_position = heel_marker.get_location()  # Marker position relative to calcn_l_b

# Get the transform between toes_l_b and calcn_l_b
toes_body = empty_model.getBodySet().get("toes_l_b")
calcn_body = empty_model.getBodySet().get("calcn_l_b")
toes_to_calcn_transform = toes_body.findTransformBetween(state, calcn_body)

# Extract translation vector from the Transform
translation = toes_to_calcn_transform.p()

# Convert translation to a NumPy array
translation_vector = np.array([translation[0], translation[1], translation[2]])

# Convert toe_local_position (Vec3) to a NumPy array for matrix operations
toe_local_array = np.array([toe_local_position.get(0), toe_local_position.get(1), toe_local_position.get(2)])

# Calculate the toe marker's position in the calcn_l_b frame
toe_position_in_calcn = toe_local_array + translation_vector

# The heel marker is already in the calcn_l_b frame
heel_position_in_calcn = np.array([heel_local_position.get(0), heel_local_position.get(1), heel_local_position.get(2)])

# Compute the initial foot vector (heel to toe, normalized)
left_foot_vector_initial = vector_between_points(heel_position_in_calcn, toe_position_in_calcn, True)

# Compute the actual foot vector from mocap data (heel to toe, normalized)
# Mocap data is rotated and negated to align with the model's coordinate system
left_foot_vector_actual = vector_between_points(
    -rotate_coordinate_x(mocap_static_trc["LHEE"], 90),
    -rotate_coordinate_x(mocap_static_trc["LTOE"], 90),
    True
)

# Plot the two vectors for visualization
#plot_3d_vectors(left_foot_vector_initial, left_foot_vector_actual)

# Compute the Euler angles to align the initial vector with the actual vector
l_foot_update_to_match_actual_rotation = compute_euler_angles_from_vectors(
    left_foot_vector_initial, left_foot_vector_actual
)

# Set unnecessary rotations (x and z axes) to zero
l_foot_update_to_match_actual_rotation[0] = 0
#l_foot_update_to_match_actual_rotation[2] = 0

# Access the left ankle joint by name
left_ankle_joint = empty_model.getJointSet().get("talus_l_to_tibfib_l")

# Access the current orientation of the child frame (talus)
current_orientation = left_ankle_joint.get_frames(1).get_orientation()

# Extract the current orientation values as a NumPy array
current_orientation_values = np.array([
    current_orientation.get(0),
    current_orientation.get(1),
    current_orientation.get(2)
])

# Subtract the calculated Euler angles to adjust the orientation
new_orientation_values = current_orientation_values - np.array(l_foot_update_to_match_actual_rotation)

# Update the child frame's orientation with the new values
left_ankle_joint.upd_frames(1).set_orientation(osim.Vec3(*new_orientation_values))


# Initialize the model's system
state = empty_model.initSystem()

#Attempting to make the foot be flat with the ground
left_foot_transform_in_ground = toes_body.getTransformInGround(state)
# Extract the rotation matrix from the transform
rotation_matrix = left_foot_transform_in_ground.R().asMat33()
# Convert the rotation matrix to Euler angles
rotation = osim.Rotation(rotation_matrix)
euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

# Set unnecessary rotations (x and z axes) to zero
euler_angles[0] = 0
euler_angles[1] = 0
#euler_angles[2] = 0


# Extract the components of the Vec3 and negate them
inverse_euler_angles = osim.Vec3(
    -euler_angles.get(0),  # Negate X angle
    -euler_angles.get(1),  # Negate Y angle
    -euler_angles.get(2)   # Negate Z angle
)
# Convert osim.Vec3 to NumPy array
inverse_euler_angles_array = np.array([
    inverse_euler_angles.get(0),
    inverse_euler_angles.get(1),
    inverse_euler_angles.get(2)
])
# Access the joint connecting talus_l_b to its parent (e.g., tibfib_l_b)
left_ankle_joint = empty_model.getJointSet().get("talus_l_to_tibfib_l")
# Access the child frame's current orientation
current_orientation = left_ankle_joint.get_frames(1).get_orientation()
current_orientation_values = np.array([current_orientation.get(0),
                                        current_orientation.get(1),
                                        current_orientation.get(2)])

# Apply the inverse rotation to the current orientation
new_orientation_values = current_orientation_values - inverse_euler_angles_array

# Update the child frame's orientation
left_ankle_joint.upd_frames(1).set_orientation(osim.Vec3(*new_orientation_values))




#%% === Adjust Orientation of the Right Foot ===

# Access the markers again after reinitialization
toe_marker = empty_model.getMarkerSet().get("RTOE")  # Right toe marker
heel_marker = empty_model.getMarkerSet().get("RHEE")  # Right heel marker

# Get marker positions in their local body frames
toe_local_position = toe_marker.get_location()  # Marker position relative to toes_r_b
heel_local_position = heel_marker.get_location()  # Marker position relative to calcn_r_b

# Get the transform between toes_r_b and calcn_r_b
toes_body = empty_model.getBodySet().get("toes_r_b")
calcn_body = empty_model.getBodySet().get("calcn_r_b")
toes_to_calcn_transform = toes_body.findTransformBetween(state, calcn_body)

# Extract translation vector from the Transform
translation = toes_to_calcn_transform.p()

# Convert translation to a NumPy array
translation_vector = np.array([translation[0], translation[1], translation[2]])

# Convert toe_local_position (Vec3) to a NumPy array for matrix operations
toe_local_array = np.array([toe_local_position.get(0), toe_local_position.get(1), toe_local_position.get(2)])

# Calculate the toe marker's position in the calcn_r_b frame
toe_position_in_calcn_r = toe_local_array + translation_vector

# The heel marker is already in the calcn_r_b frame
heel_position_in_calcn_r = np.array([heel_local_position.get(0), heel_local_position.get(1), heel_local_position.get(2)])

# Compute the initial foot vector (heel to toe, normalized)
right_foot_vector_initial = vector_between_points(heel_position_in_calcn_r, toe_position_in_calcn_r, True)

# Compute the actual foot vector from mocap data (heel to toe, normalized)
# Mocap data is rotated and negated to align with the model's coordinate system
right_foot_vector_actual = vector_between_points(
    -rotate_coordinate_x(mocap_static_trc["RHEE"], 90),
    -rotate_coordinate_x(mocap_static_trc["RTOE"], 90),
    True
)

# Plot the two vectors for visualization
#plot_3d_vectors(right_foot_vector_initial, right_foot_vector_actual)

# Compute the Euler angles to align the initial vector with the actual vector
r_foot_update_to_match_actual_rotation = compute_euler_angles_from_vectors(
    right_foot_vector_initial, right_foot_vector_actual
)

# Set unnecessary rotations (x and z axes) to zero
r_foot_update_to_match_actual_rotation[0] = 0
#r_foot_update_to_match_actual_rotation[2] = 0

# Access the right ankle joint by name
right_ankle_joint = empty_model.getJointSet().get("talus_r_to_tibfib_r")

# Access the current orientation of the child frame (talus)
current_orientation = right_ankle_joint.get_frames(1).get_orientation()

# Extract the current orientation values as a NumPy array
current_orientation_values = np.array([
    current_orientation.get(0),
    current_orientation.get(1),
    current_orientation.get(2)
])

# Subtract the calculated Euler angles to adjust the orientation
new_orientation_values = current_orientation_values - np.array(r_foot_update_to_match_actual_rotation)

# Update the child frame's orientation with the new values
right_ankle_joint.upd_frames(1).set_orientation(osim.Vec3(*new_orientation_values))

# Initialize the model's system
state = empty_model.initSystem()

#Attempting to make the foot be flat with the ground
right_foot_transform_in_ground = toes_body.getTransformInGround(state)
# Extract the rotation matrix from the transform
rotation_matrix = right_foot_transform_in_ground.R().asMat33()
# Convert the rotation matrix to Euler angles
rotation = osim.Rotation(rotation_matrix)
euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

# Set unnecessary rotations (x and z axes) to zero
euler_angles[0] = 0
euler_angles[1] = 0
#euler_angles[2] = 0


# Extract the components of the Vec3 and negate them
inverse_euler_angles = osim.Vec3(
    -euler_angles.get(0),  # Negate X angle
    -euler_angles.get(1),  # Negate Y angle
    -euler_angles.get(2)   # Negate Z angle
)
# Convert osim.Vec3 to NumPy array
inverse_euler_angles_array = np.array([
    inverse_euler_angles.get(0),
    inverse_euler_angles.get(1),
    inverse_euler_angles.get(2)
])
# Access the joint connecting talus_l_b to its parent (e.g., tibfib_l_b)
right_ankle_joint = empty_model.getJointSet().get("talus_r_to_tibfib_r")
# Access the child frame's current orientation
current_orientation = right_ankle_joint.get_frames(1).get_orientation()
current_orientation_values = np.array([current_orientation.get(0),
                                        current_orientation.get(1),
                                        current_orientation.get(2)])

# Apply the inverse rotation to the current orientation
new_orientation_values = current_orientation_values - inverse_euler_angles_array

# Update the child frame's orientation
right_ankle_joint.upd_frames(1).set_orientation(osim.Vec3(*new_orientation_values))

# Finalise the initial iteration of model
empty_model.finalizeConnections()
empty_model.printToXML(output_file)

#Code here to extract the muscle positions in the local coordinate system

def extract_local_muscle_positions(model_path):
    """
    Extracts the local positions of markers relative to their parent frames and includes the body they are attached to.

    Args:
        model_path (str): Path to the OpenSim model file containing the markers.

    Returns:
        dict: Dictionary with marker names as keys and a tuple (body_name, local_position) as values.
    """
    # Load the model
    model = osim.Model(model_path)
    state = model.initSystem()

    # Dictionary to store local positions and attached body
    local_positions = {}

    # Iterate through the markers in the MarkerSet
    markerset = model.getMarkerSet()
    for i in range(markerset.getSize()):
        marker = markerset.get(i)
        marker_name = marker.getName()

        # Check if the marker name starts with "ins_" or "ori_"
        if marker_name.startswith("ins_") or marker_name.startswith("ori_"):
            try:
                # Get the marker's local position
                local_vec = marker.get_location()

                # Get the body the marker is attached to
                body_name = marker.getParentFrame().getName()

                # Store the local position and body name in the dictionary
                local_positions[marker_name] = (
                    body_name,
                    (local_vec.get(0), local_vec.get(1), local_vec.get(2)),
                )
            except Exception as e:
                print(f"Error processing marker '{marker_name}': {e}")

    return local_positions

local_muscle_positions = extract_local_muscle_positions(empty_model)




#%% Look to scale the size of the feet automatically and move the markers to appropriate positions

scaling_file = search_files_by_keywords("High_Level_Inputs", "ScaleSettings")[0]
scale_tool = osim.ScaleTool(scaling_file)
scale_tool.setPathToSubject(participant_folder)


# Set the model file
scale_tool.getGenericModelMaker().setModelFileName(output_file)  # Replace with your model file

banter,(start_time, end_time),dontcare = read_trc_file_as_dict(mocap_trc_file,True)
# Create an OpenSim ArrayDouble and populate it with start_time and end_time
time_range = osim.ArrayDouble()
time_range.append(start_time)
time_range.append(end_time)



# Set the output file for the MarkerPlacer and MarkerPlacer settings
#Do u want to move markers to match the static file? - causes the feet to be poor currently
scale_tool.getMarkerPlacer().setApply(True)
scale_tool.getMarkerPlacer().setOutputModelFileName("/Models/scaled_foot.osim")
scale_tool.getMarkerPlacer().setMarkerFileName("/Inputs/"+mocap_trc_file.split('/')[-1])
scale_tool.getMarkerPlacer().setTimeRange(time_range)

scale_tool.getModelScaler().setOutputModelFileName("Models/scaled_foot.osim")
scale_tool.getModelScaler().setMarkerFileName("/Inputs/"+mocap_trc_file.split('/')[-1])
scale_tool.getModelScaler().setTimeRange(time_range)


scaled_output_file = "Participants/" + participant_folder.split('/')[-1] + "/Models/scaling_tool_settings.xml"

# Verify the loaded scaling settings (optional)
scale_tool.printToXML(scaled_output_file)  # Outputs a copy of the loaded settings



# Run the scaling process
scale_tool.run()




#%% Create the JMP settings files and move trcs & model automatically

# Define file paths
default_file_path = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\JMP_Default.xml"
output_file_path = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\JMP_Actual.xml"

knee_optimisation_trc_file = search_files_by_keywords(participant_inputs, "optimisation")[0]  # Find the TRC file containing marker data
banter,(start_time, end_time),banter1 = read_trc_file_as_dict(knee_optimisation_trc_file,True)
#copy the scaled_feet_variant.osim from the directory above to the JMP Code direcotry
import shutil
# Define source and destination file paths
source_file_path1 = os.path.join(participant_folder, "Models", "scaled_foot.osim")  # Source path
destination_file_path1 = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\scaled_foot.osim"  # Destination path
destination_file_path2 = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\trcfile.trc"

# Copy the file
try:
    shutil.copy(source_file_path1, destination_file_path1)
    shutil.copy(knee_optimisation_trc_file, destination_file_path2)
    print(f"File copied successfully to {destination_file_path1}")
except Exception as e:
    print(f"Error while copying the file: {e}")

#Replacement words
replacements = {
    "INSERTMODELINPUT": "scaled_foot.osim",
    "INSERTMODELOUTPUT": "JMP_optimised_knee.osim",
    "INSERTTRCFILE": "trcfile.trc",
}

# Read the default file, replace placeholders, and save to the output file
try:
    with open(default_file_path, 'r') as file:
        content = file.read()

    # Perform replacements
    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    # Save the updated content to a new file
    with open(output_file_path, 'w') as file:
        file.write(content)

    print(f"File updated successfully and saved as: {output_file_path}")

except FileNotFoundError:
    print(f"File not found: {default_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")


#%%Adjusting & Optimising the Knee Joint Orientations

# Default temporary model paths
temp_model_path_1 = output_folder+ "/temp1.osim"
temp_model_path_2 = output_folder+ "/temp2.osim"
optimised_knee_model = output_folder+ "/Optimised_Knee_Axes.osim"


marker_weights = {
            "RASI": 5, "LASI": 5, "RTHI": 1, "RTIB": 1,
            "RANK": 10, "LTHI": 1, "LTIB": 1, "LANK": 10,
            "RPSI": 1, "LPSI": 1, "RHEE": 1, "LHEE": 1,
            "RTOE": 1, "LTOE": 1, "RKNE": 2.5, "LKNE": 2.5
        }

#This runs the knee joint optimisation
run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2,marker_weights,optimised_knee_model)




#%% Run IK, and extract the model marker positions and compare to those of the actual marker positions across the entire time trial.


optimised_knee_moved_marker_model = output_folder+"/Optimised_Knee_Axes_Moved_Markers.osim"

compute_and_adjust_markers(optimised_knee_model,"ik_output.mot","_ik_model_marker_locations.sto",banter1,optimised_knee_moved_marker_model)

re_optimised_knee_moved_marker_model = output_folder+"/Re_Optimised_Knee_Axes_Moved_Markers.osim"

#This runs the knee joint optimisation for the second time
#run_knee_joint_optimisation(optimised_knee_moved_marker_model, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2,marker_weights,re_optimised_knee_moved_marker_model)


print("\n")
print("Prior to Knee Alignment & Marker movement")
print(perform_IK(source_file_path1,knee_optimisation_trc_file,start_time, end_time, marker_weights))
print("\n")
print("Following Knee Alignment but Prior to Marker Adjustment")
print(perform_IK(optimised_knee_model,knee_optimisation_trc_file,start_time, end_time, marker_weights))
print("\n")
print("Following Both Knee Alignment & Marker Adjustment")
print(perform_IK(optimised_knee_moved_marker_model,knee_optimisation_trc_file,start_time, end_time, marker_weights))
print("\n")


#print("Following Both Knee Alignment & Marker Adjustment (Then Re-Aligned Again)")
#print(perform_IK(re_optimised_knee_moved_marker_model,knee_optimisation_trc_file,start_time, end_time, marker_weights))

#Lets try to add some muscles to the model
#function to compute the position of a mesh node in the local coordinate system of a bone (can try it on the landmarks first), im assuming we can create a new model, add the body/ mesh, add a "marker" to represent this point, then use the marker.getlocation attribute thing to get the position of the marker in the local bone coordinate system (hopefully)

model = osim.Model(optimised_knee_moved_marker_model)

add_all_muscles_to_model_with_simple_names(model, local_muscle_positions,muscle_linkages)

muscle_model_name = os.path.basename(participant_folder).replace(" ", "_")

muscle_model = output_folder+"/Muscle_" +muscle_model_name+ ".osim"
model.setName("Muscle_"+muscle_model_name)
model.finalizeConnections()
model.printToXML(muscle_model)



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

#begin attempt at adding wrapping objects to muscles

#get the marker set of the model and find some markers
#compute the midpoint between the LASI and LPSI markers using the midpoint_3d function




marker_model = osim.Model(empty_model)
state = marker_model.initSystem()
marker_set = marker_model.getMarkerSet()



#%% Setting translations for glute max 1 wrapping objects (and determining acceptable radii of cylinders)


#%% Pelvis Wrap Objects
# Get the reference frame (Pelvis)
pelvis_frame = empty_model.getBodySet().get("pelvis_b")
# Ratio = SIS_x_distance / radii
desired_glut_radii_ratio = 3.15

# Left side
l_obt_ext_marker = marker_set.get("ins_l_iliacus")
l_obt_ext_global = l_obt_ext_marker.getLocationInGround(state)  # Get the marker's position in global coordinates

l_glut_wrap_position_pelvis = compute_marker_midpoint(marker_model, "ori_l_rect_fem_1", "ori_l_gem_1")
l_glut_wrap_global = pelvis_frame.findStationLocationInAnotherFrame(state, osim.Vec3(l_glut_wrap_position_pelvis), empty_model.getGround()) # Convert to OpenSim Vec3

# Set the desired global forward-backward (anterior-posterior) position
l_glut_wrap_global[0] = l_obt_ext_global.get(0)  # Modify the x position in the global frame

# Convert the updated position back to the pelvis's local frame
l_glut_wrap_local = empty_model.getGround().findStationLocationInAnotherFrame(state, l_glut_wrap_global,pelvis_frame)

# Update the wrap object position
l_glut_wrap_position_pelvis = np.array([l_glut_wrap_local.get(i) for i in range(3)])


# Right side
r_obt_ext_marker = marker_set.get("ins_r_iliacus")
r_obt_ext_global = r_obt_ext_marker.getLocationInGround(state)  # Get the marker's position in global coordinates

r_glut_wrap_position_pelvis = compute_marker_midpoint(marker_model, "ori_r_rect_fem_1", "ori_r_gem_1")
r_glut_wrap_global = pelvis_frame.findStationLocationInAnotherFrame(state, osim.Vec3(r_glut_wrap_position_pelvis), empty_model.getGround()) # Convert to OpenSim Vec3

# Set the desired global forward-backward (anterior-posterior) position
r_glut_wrap_global[0] = r_obt_ext_global.get(0)  # Modify the x position in the global frame

# Convert the modified position back to the pelvis's local frame
r_glut_wrap_local = empty_model.getGround().findStationLocationInAnotherFrame(state, r_glut_wrap_global,pelvis_frame)

# Update the wrap object position
r_glut_wrap_position_pelvis = np.array([r_glut_wrap_local.get(i) for i in range(3)])


#radius
# Get global positions of the markers
l_asis_global = marker_set.get("lms_LASI").getLocationInGround(state)
l_psis_global = marker_set.get("lms_LPSI").getLocationInGround(state)

# Compute the global X-distance
SIS_x_dist = l_asis_global.get(0) - l_psis_global.get(0)

# Compute the radius
radius_1 = SIS_x_dist / desired_glut_radii_ratio




#%% Femur Wrap Objects

l_femur_frame = empty_model.getBodySet().get("femur_l_b")


# Left side
l_obt_ext_marker = marker_set.get("ins_l_iliacus")
l_obt_ext_global = l_obt_ext_marker.getLocationInGround(state)  # Get the marker's position in global coordinates

l_glut_wrap_position_femur = compute_marker_midpoint(marker_model, "ins_l_glut_med", "ins_l_obt_ext")
l_glut_wrap_global = l_femur_frame.findStationLocationInAnotherFrame(state, osim.Vec3(l_glut_wrap_position_femur), empty_model.getGround()) # Convert to OpenSim Vec3

# Set the desired global forward-backward (anterior-posterior) position
l_glut_wrap_global[2] = l_obt_ext_global.get(2)  # Modify the x position in the global frame

# Convert the updated position back to the pelvis's local frame
l_glut_wrap_local = empty_model.getGround().findStationLocationInAnotherFrame(state, l_glut_wrap_global,l_femur_frame)

# Update the wrap object position
l_glut_wrap_position_femur = np.array([l_glut_wrap_local.get(i) for i in range(3)])


r_femur_frame = empty_model.getBodySet().get("femur_r_b")

# Right side
r_obt_ext_marker = marker_set.get("ins_r_iliacus")
r_obt_ext_global = r_obt_ext_marker.getLocationInGround(state)  # Get the marker's position in global coordinates

r_glut_wrap_position_femur = compute_marker_midpoint(marker_model, "ins_r_glut_med", "ins_r_obt_ext")
r_glut_wrap_global = r_femur_frame.findStationLocationInAnotherFrame(state, osim.Vec3(r_glut_wrap_position_femur), empty_model.getGround()) # Convert to OpenSim Vec3

# Set the desired global forward-backward (anterior-posterior) position
r_glut_wrap_global[2] = r_obt_ext_global.get(2)  # Modify the z position in the global frame

# Convert the modified position back to the pelvis's local frame
r_glut_wrap_local = empty_model.getGround().findStationLocationInAnotherFrame(state, r_glut_wrap_global,r_femur_frame)

# Update the wrap object position
r_glut_wrap_position_femur = np.array([r_glut_wrap_local.get(i) for i in range(3)])





wrapping_objects = {
    "l_glut_max_1_1": [  # Muscle name (key), list of wrapping objects (values)
        {   # Wrapping object 1 (Pelvis)
            "name": "l_glut_max_1_1_pelvis_wrap",  # Unique name
            "body": "pelvis_b",
            "type": "cylinder",
            "translation": tuple(l_glut_wrap_position_pelvis),
            "rotation": (0.75, -0.390000, 0),
            "radius": radius_1,
            "length": 0.1,
            "quadrant": "-x"
        },
        {  # Wrapping object 2 (Femur)
            "name": "l_glut_max_1_1_femur_wrap",  # Unique name
            "body": "femur_l_b",
            "type": "cylinder",
            "translation": tuple(l_glut_wrap_position_femur),
            "rotation": (-0.143263, -0.123715, 0.421776),
            "radius": radius_1*0.45,
            "length": 0.1,
            "quadrant": "-x"
        }

    ],
    "r_glut_max_1_1": [  # Muscle name (key), list of wrapping objects (values)
        {  # Wrapping object 1 (Pelvis)
            "name": "r_glut_max_1_1_pelvis_wrap",  # Unique name
            "body": "pelvis_b",
            "type": "cylinder",
            "translation": tuple(r_glut_wrap_position_pelvis),
            "rotation": (-0.750000, 0.390000, 0),
            "radius": radius_1,
            "length": 0.1,
            "quadrant": "-x"
        },
    ]
}



model = add_wrapping_objects_to_model(model, wrapping_objects)
model.finalizeConnections()
model.printToXML(muscle_model)

