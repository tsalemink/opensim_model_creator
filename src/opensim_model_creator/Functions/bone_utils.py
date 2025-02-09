#Import Packages
import opensim as osim
import os
import trimesh
import pyvista as pv
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

#Import required functions
from Functions.general_utils import rotate_coordinate_x, vector_between_points, read_trc_file_as_dict
from Functions.file_utils import search_files_by_keywords

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

def run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2, marker_weights, final_output_model_path, initial_params=None):
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

def compute_and_adjust_markers(model_path, ik_output_mot_path, model_marker_locations_path, actual_marker_positions_dict, output_model_path):
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

def initialize_model_and_extract_landmarks(participant_inputs):
    """
    Initializes the OpenSim model and extracts relevant landmarks and marker placements.

    Parameters:
        participant_inputs (str): Path to the participant's input directory.

    Returns:
        tuple: A tuple containing:
            - empty_model (osim.Model): The initialized OpenSim model.
            - state (osim.State): The system state of the model.
            - left_landmarks (dict): Dictionary of extracted left limb landmarks.
            - right_landmarks (dict): Dictionary of extracted right limb landmarks.
            - mocap_static_trc (dict): Dictionary containing marker placements from TRC file.
    """
    # Initialise the OpenSim model
    empty_model = osim.Model("High_Level_Inputs/Feet.osim")  # Load the base model file
    state = empty_model.initSystem()  # Initialise the system

    # Load and extract landmarks for left and right limbs
    left_landmarks_file = search_files_by_keywords(participant_inputs, "left lms predicted")[0]
    right_landmarks_file = search_files_by_keywords(participant_inputs, "right lms predicted")[0]
    left_landmarks = load_landmarks(left_landmarks_file)
    right_landmarks = load_landmarks(right_landmarks_file)

    # Load the TRC file and extract marker placements
    mocap_trc_file = search_files_by_keywords(participant_inputs, "static")[0]
    mocap_static_trc, _ = read_trc_file_as_dict(mocap_trc_file)

    return empty_model, state, left_landmarks, right_landmarks, mocap_static_trc, mocap_trc_file

def create_pelvis_body_and_joint(model, left_landmarks, right_landmarks, meshes, mocap_static_trc, realign_pelvis=True):
    """
    Creates the pelvis body, attaches it to the ground with a FreeJoint, and adds a mesh and markers.

    Parameters:
        model (osim.Model): The OpenSim model.
        left_landmarks (dict): Dictionary of extracted left limb landmarks.
        right_landmarks (dict): Dictionary of extracted right limb landmarks.
        meshes (str): Path to the directory containing mesh files.
        mocap_static_trc (dict): Dictionary containing marker placements from TRC file.
        realign_pelvis (bool): Whether to apply pelvis realignment (default: True).

    Returns:
        tuple:
            - pelvis (osim.Body): The created pelvis body.
            - pelvis_joint (osim.FreeJoint): The created pelvis joint.
            - rotated_pelvis_center (np.array): The rotated center of the pelvis mesh.
    """
    # Create the pelvis body
    pelvis = osim.Body("pelvis_b", 1.0, osim.Vec3(0, 0, 0), osim.Inertia(0, 0, 0))
    model.addBody(pelvis)

    # Compute pelvis alignment
    LASIS = rotate_coordinate_x(left_landmarks["ASIS"], 90)
    RASIS = rotate_coordinate_x(right_landmarks["ASIS"], 90)
    RANK = rotate_coordinate_x(right_landmarks["malleolus_med"], 90)

    pelvis_sideways_vector = vector_between_points(LASIS, RASIS)
    alignment_to_axis = (0, 0, 1)
    pelvis_realignment = compute_euler_angles_from_vectors(pelvis_sideways_vector, alignment_to_axis)
    pelvis_realignment[0] = 0
    pelvis_realignment[2] = 0  # Keep necessary rotations

    # Apply realignment conditionally
    if not realign_pelvis:
        pelvis_realignment[1] = 0  # Set to 0 if realign_pelvis is False

    # Compute ground height offset (for visualization)
    RASIS_to_RANK = np.linalg.norm(vector_between_points(RASIS, RANK))
    height_offset = RASIS_to_RANK + 0.035

    # Attach the pelvis body to the ground using a FreeJoint
    pelvis_joint = osim.FreeJoint(
        "pelvis_to_ground",
        model.getGround(),
        osim.Vec3(0, height_offset, 0),
        osim.Vec3(0, 0, 0),
        pelvis,
        osim.Vec3(0, 0, 0),
        osim.Vec3(pelvis_realignment)
    )
    model.addJoint(pelvis_joint)

    # Attach the mesh for the pelvis
    mesh_filename = search_files_by_keywords(meshes, "pelvis")[0]
    info = extract_mesh_info_trimesh(mesh_filename)
    pelvis_center = info['center']
    rotated_pelvis_center = rotate_coordinate_x(pelvis_center, 90)

    add_mesh_to_body(model, "pelvis_b", mesh_filename, offset_orientation=(-1.5708, 0, 0),
                     offset_translation=(rotated_pelvis_center[0], rotated_pelvis_center[1], rotated_pelvis_center[2]))

    # Add mocap markers
    add_markers_to_body(model, "pelvis_b", ["RASI", "LASI", "RPSI", "LPSI"], mocap_static_trc, pelvis_center)

    # Add anatomical landmarks
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS", "SAC"], left_landmarks, pelvis_center,
                        ["lms_LASI", "lms_LPSI", "lms_SAC"])
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS"], right_landmarks, pelvis_center,
                        ["lms_RASI", "lms_RPSI"])

    return pelvis, pelvis_joint, rotated_pelvis_center, pelvis_realignment, pelvis_center