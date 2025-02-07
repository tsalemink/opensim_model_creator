#Import Packages
import opensim as osim
import os
import trimesh
import pyvista as pv
import numpy as np
import xml.etree.ElementTree as ET

#Import required functions
from Functions.general_utils import rotate_coordinate_x, vector_between_points

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