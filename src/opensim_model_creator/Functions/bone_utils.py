# Import Packages
import opensim as osim
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from opensim_model_creator.Functions.file_utils import search_files_by_keywords, get_results_dir
# Import required functions
from opensim_model_creator.Functions.general_utils import rotate_coordinate_x, vector_between_points, \
    read_trc_file_as_dict, midpoint_3d
from gias3.musculoskeletal import model_alignment

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
high_level_inputs = os.path.join(root_directory, "High_Level_Inputs")


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

    print(
        f"Added mesh '{geometry_name}' to body '{body_name}' with translation {offset_translation} and orientation {offset_orientation}.")


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


def load_x_opt(file_path):
    """
    Loads optimisation parameters from a file where each line contains a name and a different number of variables for
    pelvis: translation (x,y,z) and rotation (x(list), y(rotation), z(tilt))
    hip: flexion, adduction, rotation
    knee: flexion, adduction

    Args:
        file_path (str): Path to the file containing x_opt parameters.

    Returns:
        dict: A dictionary where keys are joint names and values are numpy arrays of coordinates.
    """
    x_opt = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line != "\n":
                # Split the line into parts
                parts = line.strip().split()
                name = parts[0]  # The first part is the name
                coordinates = list(map(float, parts[1:]))  # Remaining parts are coordinates
                x_opt[name] = np.array(coordinates)
    return x_opt


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
        center (np.array): The reference center point for calculating marker positions.
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
            landmark_position = vector_between_points(center, location)
            # landmark_position = rotate_coordinate_x(landmark_position, 90)
            marker_location = osim.Vec3(*location)

            # Determine the marker's name
            final_name = custom_names[i] if custom_names else marker_name

            # Create and add the marker
            marker = osim.Marker(final_name, body, marker_location)
            model.addMarker(marker)

            print(f"Marker '{final_name}' added to body '{body_name}' at location {location}.")

    except Exception as e:
        print(f"Error adding markers to body '{body_name}': {e}")


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
            orthogonal_axis = np.array([1.0, 0.0, 0.0]) if not np.allclose(from_vector, [1, 0, 0]) else np.array(
                [0, 1, 0])
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


def optimize_knee_axis(model_path, trc_file, start_time, end_time, marker_weights, initial_params, temp_model_path_1,
                       temp_model_path_2, final_output_model, iteration_count):
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
        iteration_count: how many iterations to perform in the optimisation
    Returns:
        OptimizeResult: Results of the optimization process.
    """

    def objective(params):
        left_knee_x, right_knee_x, left_knee_y, right_knee_y = params

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
        return errors["Average RMS Error"] if errors else float("inf")

    # Sets bounds for knee joint optimisation
    bounds = [(-0.1, 0.1)] * 4
    result = minimize(objective, np.array(initial_params), method="Powell", bounds=bounds,
                      options={"disp": True, "maxiter": 3, "xtol": 0.1, "ftol": 0.01})
    model = osim.Model(temp_model_path_2)
    model_name_here = os.path.basename(final_output_model)
    model.setName(model_name_here)
    model.printToXML(final_output_model)
    return result


def perform_IK(model_file, trc_file, start_time, end_time, marker_weights):
    """
    Perform Inverse Kinematics analysis using OpenSim.

    Args:
        model_file (str): Path to the OpenSim model file.
        trc_file (str): Path to the TRC file.
        start_time (float): Start time for IK.
        end_time (float): End time for IK.
        marker_weights (dict): Marker weights for IK analysis.

    Returns:
        dict: Dictionary containing average RMS error and max error.
    """
    try:
        results_directory = get_results_dir()
        model = osim.Model(model_file)
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(model)
        ik_tool.setMarkerDataFileName(trc_file)
        ik_tool.setStartTime(start_time)
        ik_tool.setEndTime(end_time)
        ik_output = os.path.join(results_directory, "ik_output.mot")
        ik_tool.setOutputMotionFileName(ik_output)
        ik_tool.set_report_marker_locations(True)
        ik_tool.setResultsDir(results_directory)

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

        output_errors_file = os.path.join(results_directory, "_ik_marker_errors.sto")

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


def zero_joint_orientation(model, output_file, x_opt_left, x_opt_right, l_hip_joint, r_hip_joint, l_knee_joint,
                           r_knee_joint,
                           pelvis_joint, left_lm, right_lm):
    '''
    Zeros all of the joint orientations so that default values can be set. The joints are zeroed to the articulated
    shape model zero pose using x_opt which contains the 'pose' for the predicted model
    Args:
        x_opt_left (dict): joint angles left
        x_opt_right (dict): joint angles right
        l_hip_joint: osim joint instance for the left hip
        r_hip_joint:
        l_knee_joint:
        r_knee_joint:
        pelvis_joint:

    '''

    # Access the child frame for the pelvis ground joint
    child_frame = pelvis_joint.upd_frames(1)
    new_translation = osim.Vec3(((x_opt_left['pelvis_rigid'][0] + x_opt_right['pelvis_rigid'][0]) / 2000),
                                ((x_opt_left['pelvis_rigid'][1] + x_opt_right['pelvis_rigid'][1]) / 2000),
                                ((x_opt_left['pelvis_rigid'][2] + x_opt_right['pelvis_rigid'][2]) / 2000))
    #child_frame.set_translation(new_translation)
    # Apply rotation adjustments from x_opt
    new_orientation_pel = osim.Vec3(((x_opt_left['pelvis_rigid'][3] + x_opt_right['pelvis_rigid'][3]) / 2),
                                    ((x_opt_left['pelvis_rigid'][4] + x_opt_right['pelvis_rigid'][4]) / 2),
                                    ((x_opt_left['pelvis_rigid'][5] + x_opt_right['pelvis_rigid'][5]) / 2))
    child_frame.set_orientation(new_orientation_pel)
    model.finalizeConnections()

    # adjust adduction angles for the hip joint
    z_axis = right_lm['ASIS'] - left_lm['ASIS']
    o, x, y_axis, z_axis_fem_l = model_alignment.createFemurACSISB(left_lm['hjc'], left_lm['MEC'], left_lm['LEC'], 'left')
    hip_add_l = project_rotation(x_opt_left['hip_rot'][2], y_axis, z_axis, x)
    print(x_opt_left['hip_rot'][2])
    print(hip_add_l)

    o, x, y_axis, z_axis_fem_r = model_alignment.createFemurACSISB(right_lm['hjc'], right_lm['MEC'], right_lm['LEC'], 'right')
    hip_add_r = project_rotation(x_opt_right['hip_rot'][2], y_axis, z_axis, x)
    print(x_opt_right['hip_rot'][2])
    print(hip_add_r)

    # Access the child frame for the left hip joint
    child_frame = l_hip_joint.upd_frames(1)
    # Apply rotation adjustments from x_opt
    new_orientation = osim.Vec3(hip_add_l, x_opt_left['hip_rot'][1],
                                x_opt_left['hip_rot'][0])
    child_frame.set_orientation(new_orientation)
    model.finalizeConnections()

    # Access the child frame for the right hip joint
    child_frame = r_hip_joint.upd_frames(1)
    # Apply rotation adjustments from x_opt
    new_orientation = osim.Vec3(hip_add_r, x_opt_right['hip_rot'][1],
                                x_opt_right['hip_rot'][0])
    child_frame.set_orientation(new_orientation)
    model.finalizeConnections()

    # adjust adduction angles for the knee joint
    #o, x, y_axis, z = model_alignment.createTibiaFibulaACSISB_2()
    #knee_add_l = project_rotation(x_opt_left['knee_rot'][1], y_axis, z_axis_fem_l, x)
    #print(x_opt_left['knee_rot'][1])
    #print(knee_add_l)

    #o, x, y_axis, z = model_alignment.createFemurACSISB(right_lm['hjc'], right_lm['MEC'], right_lm['LEC'], 'right')
    #hip_add_r = project_rotation(x_opt_right['hip_rot'][2], y_axis, z_axis, x)
    #print(x_opt_right['hip_rot'][2])
    #print(hip_add_r)
    # Access the child frame of the left knee joint
    child_frame = l_knee_joint.upd_frames(1)
    # Apply rotation adjustments from x_opt
    #new_orientation = osim.Vec3(0.0, 0.0, x_opt_left['knee_rot'][0] * -1)
    # child_frame.set_orientation(new_orientation)
    new_orientation = osim.Vec3(x_opt_left['knee_rot'][1], 0.0, x_opt_left['knee_rot'][0] * -1)
    #child_frame.set_orientation(new_orientation)
    model.finalizeConnections()

    # Access the child frame of the right knee joint
    child_frame = r_knee_joint.upd_frames(1)
    # Apply rotation adjustments from x_opt
    #new_orientation = osim.Vec3(0.0, 0.0, x_opt_right['knee_rot'][0] * -1)
    # child_frame.set_orientation(new_orientation)
    new_orientation = osim.Vec3(x_opt_right['knee_rot'][1], 0.0, x_opt_right['knee_rot'][0] * -1)
    # child_frame.set_orientation(new_orientation)
    #child_frame.set_orientation(new_orientation)
    #model.printToXML(output_file)
    #input()
    model.finalizeConnections()
    model.printToXML(output_file)
    return hip_add_l, hip_add_r


def project_rotation(theta, y_axis, z_axis, x):
    # Normalize all input vectors
    y = y_axis / np.linalg.norm(y_axis)
    z = z_axis / np.linalg.norm(z_axis)
    x = x / np.linalg.norm(x)

    # Rotation axis = cross product of y and z
    x_cross = np.cross(y, z)
    x_cross = x_cross / np.linalg.norm(x_cross)

    # Project rotation onto x
    theta_x = theta * np.dot(x_cross, x)
    return theta_x


def run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1,
                                temp_model_path_2, marker_weights, final_output_model_path, initial_params=None,
                                iteration_count=3):
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
        final_output_model=final_output_model_path,
        iteration_count=iteration_count
    )

    print(f"Optimized Joint Orientations: {result.x}")


def initialize_model_and_extract_landmarks(asm_directory):
    """
    Initializes the OpenSim model and extracts relevant landmarks and marker placements.

    Parameters:
        asm_directory (str): Path to the directory containing the mesh and landmarks produced by the ASM fit.

    Returns:
        tuple: A tuple containing:
            - empty_model (osim.Model): The initialized OpenSim model.
            - state (osim.State): The system state of the model.
            - left_landmarks (dict): Dictionary of extracted left limb landmarks.
            - right_landmarks (dict): Dictionary of extracted right limb landmarks.
            - mocap_static_trc (dict): Dictionary containing marker placements from TRC file.
    """
    # Initialise the OpenSim model
    empty_model = osim.Model(os.path.join(high_level_inputs, "Feet.osim"))  # Load the base model file
    state = empty_model.initSystem()  # Initialise the system

    # Load and extract landmarks for left and right limbs
    left_landmarks_file = search_files_by_keywords(asm_directory, "left lms predicted")[0]
    right_landmarks_file = search_files_by_keywords(asm_directory, "right lms predicted")[0]
    left_landmarks = load_landmarks(left_landmarks_file)
    right_landmarks = load_landmarks(right_landmarks_file)
    x_opt_left_file = search_files_by_keywords(asm_directory, "x opt left")[0]
    x_opt_right_file = search_files_by_keywords(asm_directory, "x opt right")[0]
    x_opt_left = load_x_opt(x_opt_left_file)
    x_opt_right = load_x_opt(x_opt_right_file)

    return empty_model, state, left_landmarks, right_landmarks, x_opt_left, x_opt_right


def create_pelvis_body_and_joint(model, left_landmarks, right_landmarks, meshes, mocap_static_trc):
    """
    Creates the pelvis body, attaches it to the ground with a FreeJoint, and adds a mesh and markers.

    Parameters:
        model (osim.Model): The OpenSim model.
        left_landmarks (dict): Dictionary of extracted left limb landmarks.
        right_landmarks (dict): Dictionary of extracted right limb landmarks.
        meshes (str): Path to the directory containing mesh files.
        mocap_static_trc (dict): Dictionary containing marker placements from TRC file.

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
    LASIS = left_landmarks["ASIS"]
    RASIS = right_landmarks["ASIS"]
    LPSIS = np.array(left_landmarks["PSIS"])
    RPSIS = np.array(right_landmarks["PSIS"])

    # Define the pelvis anatomical coordinate system from the articulated shape model (this needs to be aligned with the opensim global coordinate system)
    pelvis_origin, x_axis, y_axis, z_axis = model_alignment.createPelvisACSISB(LASIS, RASIS, LPSIS, RPSIS)

    # Create an OpenSim Rotation from pelvis axes
    rot = osim.Mat33()
    rot.set(0, 0, x_axis[0])
    rot.set(1, 0, x_axis[1])
    rot.set(2, 0, x_axis[2])

    rot.set(0, 1, y_axis[0])
    rot.set(1, 1, y_axis[1])
    rot.set(2, 1, y_axis[2])

    rot.set(0, 2, z_axis[0])
    rot.set(1, 2, z_axis[1])
    rot.set(2, 2, z_axis[2])
    pelvis_rotation_osim = osim.Rotation(rot)

    RANK = right_landmarks["malleolus_med"]
    pelvis_center = midpoint_3d(RASIS, LASIS)

    # Compute ground height offset (for visualization)
    RASIS_to_RANK = np.linalg.norm(vector_between_points(RASIS, RANK))
    height_offset = np.array([0.0, RASIS_to_RANK + 0.035, 0.0])
    pelvis_translation = osim.Vec3(pelvis_origin-height_offset)
    # Attach the pelvis body to the ground using a FreeJoint
    pelvis_joint = osim.FreeJoint(
        "pelvis_to_ground",
        model.getGround(),
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        pelvis,
        pelvis_translation,
        pelvis_rotation_osim.convertRotationToBodyFixedXYZ()
    )
    model.addJoint(pelvis_joint)

    # Attach the mesh for the pelvis
    mesh_path = os.path.join(meshes, "combined_pelvis_mesh.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    add_mesh_to_body(model, "pelvis_b", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=(0, 0, 0))

    # Add mocap markers
    add_markers_to_body(model, "pelvis_b", ["RASI", "LASI", "RPSI", "LPSI"], mocap_static_trc, pelvis_origin)

    # Add anatomical landmarks
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS", "SAC"], left_landmarks, pelvis_origin,
                        ["LASI_ssm", "LPSI_ssm", "SAC_ssm"])
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS"], right_landmarks, pelvis_origin,
                        ["RASI_ssm", "RPSI_ssm"])

    return pelvis, pelvis_joint, pelvis_center


def create_femur_bodies_and_hip_joints(empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc, pelvis, x_opt_left, x_opt_right):
    """
    Creates the left and right femur bodies and attaches custom hip joints to the OpenSim model.

    Args:
        empty_model (osim.Model): The OpenSim model to which femur bodies and hip joints will be added.
        left_landmarks (dict): Dictionary containing the anatomical landmarks for the left side.
        right_landmarks (dict): Dictionary containing the anatomical landmarks for the right side.
        meshes (str): Directory containing the mesh files for the left and right femurs.
        mocap_static_trc (dict): Motion capture static marker data used to position markers.
        pelvis (osim.Body): The pelvis body in the OpenSim model.

    Returns:
        tuple: A tuple containing:
            - l_LEC (np.array): Rotated coordinates of the left lateral epicondyle.
            - l_MEC (np.array): Rotated coordinates of the left medial epicondyle.
            - l_HJC (np.array): Rotated coordinates of the left hip joint center.
            - l_EC_midpoint (np.array): Midpoint of the left lateral and medial epicondyles.
            - left_femur (osim.Body): The left femur body added to the model.
            - femur_l_center (np.array): Original center of the left femur.
            - rotated_l_femur_center (np.array): Rotated center of the left femur.
            - LKNE_alignment_angles (np.array): Alignment angles for the left knee.
            - LHIP_vert_alignment_angles (np.array): Vertical alignment angles for the left hip.
            - r_LEC (np.array): Rotated coordinates of the right lateral epicondyle.
            - r_MEC (np.array): Rotated coordinates of the right medial epicondyle.
            - r_HJC (np.array): Rotated coordinates of the right hip joint center.
            - r_EC_midpoint (np.array): Midpoint of the right lateral and medial epicondyles.
            - right_femur (osim.Body): The right femur body added to the model.
            - femur_r_center (np.array): Original center of the right femur.
            - rotated_r_femur_center (np.array): Rotated center of the right femur.
            - RKNE_alignment_angles (np.array): Alignment angles for the right knee.
            - RHIP_vert_alignment_angles (np.array): Vertical alignment angles for the right hip.
    """
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

    # Extract landmarks required to position the joint coordinate systems of the left hip joint

    r_hjc = right_landmarks["hjc"]
    l_hjc = left_landmarks["hjc"]

    # Attach the mesh for the right femur
    mesh_path = os.path.join(meshes, "predicted_mesh_right_femur.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    femur_r_center = r_hjc  # Extract center of the right femur
    add_mesh_to_body(empty_model, "femur_r_b", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=(0, 0, 0))

    # Attach the mesh for the left femur
    mesh_path = os.path.join(meshes, "predicted_mesh_left_femur.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    femur_l_center = l_hjc  # Extract center of the right femur
    add_mesh_to_body(empty_model, "femur_l_b", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=(0, 0, 0))

    # Add mocap markers to the femur bodies, taken from static trial for tracking markers
    add_markers_to_body(empty_model, "femur_l_b", ["LTHI", "LKNE", "LKNEM"], mocap_static_trc, femur_l_center)
    add_markers_to_body(empty_model, "femur_r_b", ["RTHI", "RKNE", "RKNEM"], mocap_static_trc, femur_r_center)

    # Add anatomical landmarks to the femur bodies with custom marker names, taken from shape model prediction for anatomical markers
    add_markers_to_body(empty_model, "femur_l_b", ["LEC", "MEC"], left_landmarks, femur_l_center,
                        ["LKNE_ssm", "LKNEM_ssm"])
    add_markers_to_body(empty_model, "femur_r_b", ["LEC", "MEC"], right_landmarks, femur_r_center,
                        ["RKNE_ssm", "RKNEM_ssm"])

    # find midpoint of knee
    l_LEC = left_landmarks["LEC"]
    l_MEC = left_landmarks["MEC"]
    l_EC_midpoint = midpoint_3d(l_LEC, l_MEC)

    # Create the spatial transform for the custom left hip joint ###need to modify with values from shape model
    spatial_transform_left = osim.SpatialTransform()

    # First rotation (Flexion/Extension) along X-axis
    flexion_axis_left = spatial_transform_left.updTransformAxis(0)
    flexion_axis_left.setCoordinateNames(osim.ArrayStr("hip_flexion_l", 1))
    flexion_axis_left.setAxis(osim.Vec3(0, 0, 1))  # X-axis
    flexion_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Second rotation (Adduction/Abduction) along Z-axis
    adduction_axis_left = spatial_transform_left.updTransformAxis(1)
    adduction_axis_left.setCoordinateNames(osim.ArrayStr("hip_adduction_l", 1))
    adduction_axis_left.setAxis(osim.Vec3(-1, 0, 0))  # Z-axis
    adduction_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Third rotation (Internal/External Rotation) along Y-axis
    rotation_axis_left = spatial_transform_left.updTransformAxis(2)
    rotation_axis_left.setCoordinateNames(osim.ArrayStr("hip_rotation_l", 1))
    rotation_axis_left.setAxis(osim.Vec3(0, -1, 0))  # Y-axis
    rotation_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Create the custom left hip joint with all restored parameters
    left_hip_joint = osim.CustomJoint(
        "femur_l_to_pelvis",  # Joint name
        pelvis,  # Parent frame (Pelvis)
        osim.Vec3(l_hjc),  # Location in parent frame
        osim.Vec3(0, 0, 0),  # Orientation in parent frame
        left_femur,  # Child frame (Femur)
        osim.Vec3(l_hjc),  # Location in child frame
        osim.Vec3(x_opt_left[2]*-0.1, x_opt_left[1]*-0.1, x_opt_left[0]),  # Adjusted orientation in child frame
        spatial_transform_left  # The defined spatial transform
    )

    # Creation of the right hip joint coordinate system

    # find midpoint of right knee
    r_LEC = right_landmarks["LEC"]
    r_MEC = right_landmarks["MEC"]
    r_EC_midpoint = midpoint_3d(r_LEC, r_MEC)

    ################################################

    # Create the spatial transform for the custom joint ###Need to update
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

    # Create the custom hip joint with all restored parameters
    right_hip_joint = osim.CustomJoint(
        "femur_r_to_pelvis",  # Joint name
        pelvis,  # Parent frame (Pelvis)
        osim.Vec3(r_hjc),  # Location in parent frame
        osim.Vec3(0, 0, 0),  # Orientation in parent frame
        right_femur,  # Child frame (Femur)
        osim.Vec3(r_hjc),  # Location in child frame
        osim.Vec3(x_opt_right[2]*0.1, x_opt_right[1]*0.1, x_opt_right[0]),  # Adjusted orientation in child frame
        spatial_transform  # The defined spatial transform
    )

    ########################################################################################

    # Add the hip joints to the model
    empty_model.addJoint(left_hip_joint)
    empty_model.addJoint(right_hip_joint)

    return l_LEC, l_MEC, l_hjc, l_EC_midpoint, left_femur, femur_l_center, r_LEC, r_MEC, r_hjc, r_EC_midpoint, right_femur, femur_r_center


def create_tibfib_bodies_and_knee_joints(
        empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc,
        left_femur, right_femur, x_opt_left, x_opt_right):
    """
    Creates tibia and fibula (tibfib) bodies and defines the knee joints within an OpenSim model.

    Args:
        empty_model (osim.Model): The OpenSim model to which the tibfib bodies and knee joints will be added.
        left_landmarks (dict): Landmark coordinates for the left side.
        right_landmarks (dict): Landmark coordinates for the right side.
        meshes (str): Path to the folder containing mesh files.
        mocap_static_trc (dict): Motion capture data for static trials.
        left_femur (osim.Body): The left femur body in the model.
        right_femur (osim.Body): The right femur body in the model.

    Returns:
        tuple:
            - rotated_l_tibia_center (np.ndarray): Rotated center of the left tibia.
            - rotated_r_tibia_center (np.ndarray): Rotated center of the right tibia.
            - tibia_l_center (np.ndarray): Center of the left tibia.
            - tibia_r_center (np.ndarray): Center of the right tibia.
            - left_tibfib (osim.Body): Created left tibfib body.
            - right_tibfib (osim.Body): Created right tibfib body.
    """
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

    # Attach the mesh for the right tibia body
    # Search for the mesh file corresponding to the right tibfib
    mesh_path = os.path.join(meshes, "predicted_mesh_right_tibia.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    r_LMAL = right_landmarks['malleolus_lat']
    r_MMAL = right_landmarks['malleolus_med']
    r_mid_mal = midpoint_3d(r_LMAL, r_MMAL)
    tibia_r_center = r_mid_mal

    # Add the mesh to the right tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_r_b", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=(0, 0, 0))

    # Attach the mesh for the right fibula body
    # Search for the mesh file corresponding to the right fibula
    mesh_path = os.path.join(meshes, "predicted_mesh_right_fibula.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    # Add the mesh to the right tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_r_b", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=(0, 0, 0))

    # Attach the mesh for the left tibia body
    # Search for the mesh file corresponding to the left tibfib
    mesh_path = os.path.join(meshes, "predicted_mesh_left_tibia.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    l_LMAL = left_landmarks['malleolus_lat']
    l_MMAL = left_landmarks['malleolus_med']
    l_mid_mal = midpoint_3d(l_LMAL, l_MMAL)
    tibia_l_center = l_mid_mal

    # Add the mesh to the left tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_l_b", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=(0, 0, 0))

    # Attach the mesh for the left fibula body
    # Search for the mesh file corresponding to the left tibfib
    mesh_path = os.path.join(meshes, "predicted_mesh_left_fibula.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    # Add the mesh to the left tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_l_b", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=(0, 0, 0))

    # Add mocap markers to the tibfib bodies
    # Add mocap markers for the left tibfib body
    add_markers_to_body(empty_model, "tibfib_l_b", ["LTIB", "LTOE", "LHEE", "LMED", "LANK"], mocap_static_trc,
                        tibia_l_center)

    # Add landmark markers for the left tibfib body
    add_markers_to_body(empty_model, "tibfib_l_b", ["malleolus_med", "malleolus_lat"], left_landmarks, tibia_l_center,
                        ["LMED_ssm", "LANK_ssm"])

    # Add mocap markers for the right tibfib body
    add_markers_to_body(empty_model, "tibfib_r_b", ["RTIB", "RTOE", "RHEE", "RMED", "RANK"], mocap_static_trc,
                        tibia_r_center)

    # Add landmark markers for the right tibfib body
    add_markers_to_body(empty_model, "tibfib_r_b", ["malleolus_med", "malleolus_lat"], right_landmarks, tibia_r_center,
                        ["RMED_ssm", "RANK_ssm"])

    # Extract the medial and lateral epicondyle landmarks
    l_lec = left_landmarks["LEC"]  # Lateral epicondyle landmark
    l_mec = left_landmarks["MEC"]  # Medial epicondyle landmark

    # Compute the midpoint between the lateral and medial epicondyles
    l_EC_midpoint = midpoint_3d(l_lec, l_mec)

    # %% Define the left knee joint
    # Create the spatial transform for the custom knee joint
    spatial_transform = osim.SpatialTransform()

    # First rotation (Flexion/Extension) along X-axis
    flexion_axis = spatial_transform.updTransformAxis(0)
    flexion_axis.setCoordinateNames(osim.ArrayStr("knee_flexion_l", 1))
    flexion_axis.setAxis(osim.Vec3(0, 0, -1))  # X-axis
    flexion_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Second rotation (Adduction/Abduction) along Z-axis
    adduction_axis = spatial_transform.updTransformAxis(1)
    adduction_axis.setCoordinateNames(osim.ArrayStr("knee_adduction_l", 1))
    adduction_axis.setAxis(osim.Vec3(-1, 0, 0))  # Z-axis
    adduction_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Third rotation (Internal/External Rotation) along Y-axis
    rotation_axis = spatial_transform.updTransformAxis(2)
    rotation_axis.setCoordinateNames(osim.ArrayStr("knee_rotation_l", 1))
    rotation_axis.setAxis(osim.Vec3(0, -1, 0))  # Y-axis
    rotation_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # add spline to knee movement for translation1
    translation1 = spatial_transform.updTransformAxis(3)
    translation1.setCoordinateNames(osim.ArrayStr("knee_flexion_l", 1))
    translation1.setAxis(osim.Vec3(1, 0, 0))  # X-axis
    # Create SimmSpline for translation1
    x1 = [-2.0944, -1.74533, -1.39626, -1.0472, -0.698132, -0.349066, -0.174533, 0.197344, 0.337395, 0.490178, 1.52146,
          2.0944]
    y1 = [-0.0032, 0.00179, 0.00411, 0.0041, 0.00212, -0.001, -0.0031, -0.005227, -0.005435, -0.005574, -0.005435,
          -0.00525]
    offset1 = np.interp(0.0, x1, y1)  # Linear interpolation to find y at x=0
    spline1_offset = osim.SimmSpline()
    for xi, yi in zip(x1, y1):
        spline1_offset.addPoint(xi, yi - offset1)
    translation1.set_function(spline1_offset)

    # add spline to knee movement for translation2
    translation2 = spatial_transform.updTransformAxis(4)
    translation2.setCoordinateNames(osim.ArrayStr("knee_flexion_l", 1))
    translation2.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
    x2 = [-2.0944, -1.22173, -0.523599, -0.349066, -0.174533, 0.159149, 2.0944]
    y2 = [-0.4226, -0.4082, -0.399, -0.3976, -0.3966, -0.395264, -0.396]
    offset2 = np.interp(0.0, x2, y2)
    spline2_offset = osim.SimmSpline()
    for xi, yi in zip(x2, y2):
        spline2_offset.addPoint(xi, yi - offset2)
    translation2.set_function(spline2_offset)

    # Create TransformAxis translation3
    translation3 = spatial_transform.updTransformAxis(5)
    translation3.setAxis(osim.Vec3(0, 0, 1))  # Z-axis
    translation3.set_function(osim.Constant(0))

    # Define the knee joint connecting the left tibfib to the left femur
    left_knee_joint = osim.CustomJoint(
        "tibfib_l_to_femur_l",  # Name of the joint
        left_femur,  # Parent body (femur)
        osim.Vec3(l_EC_midpoint),  # Location of the joint in the femur frame
        osim.Vec3(0, 0, 0),  # Orientation of the joint in the femur frame
        left_tibfib,  # Child body (tibfib)
        osim.Vec3(l_EC_midpoint),  # Location of the joint in the tibfib frame
        osim.Vec3(x_opt_left[1]*-0.1, 0, x_opt_left[0]*-1),
        spatial_transform
        # Orientation of the joint in the tibfib frame
    )

    # %% Positioning of the right knee joint

    # Extract the medial and lateral epicondyle landmarks
    r_lec = right_landmarks["LEC"]  # Lateral epicondyle landmark
    r_mec = right_landmarks["MEC"]  # Medial epicondyle landmark

    # Compute the midpoint between the lateral and medial epicondyles
    r_EC_midpoint = midpoint_3d(r_lec, r_mec)

    # %% Define the right knee joint
    # Create the spatial transform for the custom knee joint
    spatial_transform = osim.SpatialTransform()

    # First rotation (Flexion/Extension) along X-axis
    flexion_axis = spatial_transform.updTransformAxis(0)
    flexion_axis.setCoordinateNames(osim.ArrayStr("knee_flexion_r", 1))
    flexion_axis.setAxis(osim.Vec3(0, 0, -1))  # X-axis
    flexion_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Second rotation (Adduction/Abduction) along Z-axis
    adduction_axis = spatial_transform.updTransformAxis(1)
    adduction_axis.setCoordinateNames(osim.ArrayStr("knee_adduction_r", 1))
    adduction_axis.setAxis(osim.Vec3(1, 0, 0))  # Z-axis
    adduction_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Third rotation (Internal/External Rotation) along Y-axis
    rotation_axis = spatial_transform.updTransformAxis(2)
    rotation_axis.setCoordinateNames(osim.ArrayStr("knee_rotation_r", 1))
    rotation_axis.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
    rotation_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # add spline to knee movement for translation1
    translation1 = spatial_transform.updTransformAxis(3)
    translation1.setCoordinateNames(osim.ArrayStr("knee_flexion_r", 1))
    translation1.setAxis(osim.Vec3(1, 0, 0))  # X-axis
    # Create SimmSpline for translation1
    x1 = [-2.0944, -1.74533, -1.39626, -1.0472, -0.698132, -0.349066, -0.174533, 0.197344, 0.337395, 0.490178, 1.52146,
          2.0944]
    y1 = [-0.0032, 0.00179, 0.00411, 0.0041, 0.00212, -0.001, -0.0031, -0.005227, -0.005435, -0.005574, -0.005435,
          -0.00525]
    offset1 = np.interp(0.0, x1, y1)  # Linear interpolation to find y at x=0
    spline1_offset = osim.SimmSpline()
    for xi, yi in zip(x1, y1):
        spline1_offset.addPoint(xi, yi - offset1)
    translation1.set_function(spline1_offset)

    # add spline to knee movement for translation2
    translation2 = spatial_transform.updTransformAxis(4)
    translation2.setCoordinateNames(osim.ArrayStr("knee_flexion_r", 1))
    translation2.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
    x2 = [-2.0944, -1.22173, -0.523599, -0.349066, -0.174533, 0.159149, 2.0944]
    y2 = [-0.4226, -0.4082, -0.399, -0.3976, -0.3966, -0.395264, -0.396]
    offset2 = np.interp(0.0, x2, y2)
    spline2_offset = osim.SimmSpline()
    for xi, yi in zip(x2, y2):
        spline2_offset.addPoint(xi, yi - offset2)
    translation2.set_function(spline2_offset)

    # Create TransformAxis translation3
    translation3 = spatial_transform.updTransformAxis(5)
    translation3.setAxis(osim.Vec3(0, 0, 1))  # Z-axis
    translation3.set_function(osim.Constant(0))

    # Define the knee joint connecting the right tibfib to the right femur
    right_knee_joint = osim.CustomJoint(
        "tibfib_r_to_femur_r",  # Name of the joint
        right_femur,  # Parent body (femur)
        osim.Vec3(r_EC_midpoint),  # Location of the joint in the femur frame
        osim.Vec3(0, 0, 0),  # Orientation of the joint in the femur frame
        right_tibfib,  # Child body (tibfib)
        osim.Vec3(r_EC_midpoint),  # Location of the joint in the tibfib frame
        osim.Vec3(x_opt_right[1]*0.1, 0, x_opt_right[0]*-1),
        spatial_transform
        # Orientation of the joint in the tibfib frame
    )

    # %% Adding the knee joints to the model

    # Add the left knee joint to the OpenSim model
    # This connects the left tibfib to the left femur, allowing flexion/extension motion
    empty_model.addJoint(left_knee_joint)

    # Add the right knee joint to the OpenSim model
    # This connects the right tibfib to the right femur, allowing flexion/extension motion
    empty_model.addJoint(right_knee_joint)

    return tibia_l_center, tibia_r_center, left_tibfib, right_tibfib


def repurpose_feet_bodies_and_create_joints(empty_model, left_landmarks, right_landmarks, tibfib_l_center,
                                            tibfib_r_center, left_tibfib, right_tibfib):
    """
      Repurposes the foot bodies (talus) in the OpenSim model and creates ankle joints
      (PinJoint) connecting the talus to the tibia/fibula (tibfib) segments.

      Args:
          empty_model (osim.Model): The OpenSim model where the joints and bodies are added.
          left_landmarks (dict): Dictionary of left leg anatomical landmarks with their 3D coordinates.
          right_landmarks (dict): Dictionary of right leg anatomical landmarks with their 3D coordinates.
          tibfib_l_center (np.array): Center of the left tibfib segment in the rotated coordinate system.
          tibfib_r_center (np.array): Center of the right tibfib segment in the rotated coordinate system.
          left_tibfib (osim.Body): The left tibfib body in the OpenSim model.
          right_tibfib (osim.Body): The right tibfib body in the OpenSim model.

      Returns:
          None: The function modifies the OpenSim model in place by adding new joints.
      """
    # Access the body named "talus_l_b"
    left_talus = empty_model.getBodySet().get("talus_l_b")
    # Access the body named "talus_r_b"
    right_talus = empty_model.getBodySet().get("talus_r_b")

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

    # Define manual adjustments for the left and right talus positions in the child frame
    manual__l_talus_positioning_child = (-0.001, 0.017, -0.0025)  # Manual adjustment for left talus
    manual_r_talus_positioning_child = (-0.001, 0.017, 0.0025)  # Manual adjustment for right talus

    # Define the ankle joint connecting the left talus to the left tibfib
    # A PinJoint allows rotation about a single axis (flexion/extension in this case)
    left_ankle_joint = osim.PinJoint(
        "talus_l_to_tibfib_l",  # Name of the joint
        left_tibfib,  # Parent body (tibfib)
        osim.Vec3(tibfib_l_center),  # Location of the joint in the tibfib frame
        osim.Vec3(0, 0, 0),  # Orientation of the joint in the tibfib frame
        left_talus,  # Child body (talus)
        osim.Vec3(manual__l_talus_positioning_child),  # Manually adjusted location of the joint in the talus frame
        osim.Vec3(0, 0, 0)  # Orientation of the joint in the talus frame
    )

    # Define the ankle joint connecting the right talus to the right tibfib
    # A PinJoint allows rotation about a single axis (flexion/extension in this case)
    right_ankle_joint = osim.PinJoint(
        "talus_r_to_tibfib_r",  # Name of the joint
        right_tibfib,  # Parent body (tibfib)
        osim.Vec3(tibfib_r_center),  # Location of the joint in the tibfib frame
        osim.Vec3(0, 0, 0),  # Orientation of the joint in the tibfib frame
        right_talus,  # Child body (talus)
        osim.Vec3(manual_r_talus_positioning_child),  # Manually adjusted location of the joint in the talus frame
        osim.Vec3(0, 0, 0))

    # Add the left ankle joint to the OpenSim model
    empty_model.addJoint(left_ankle_joint)

    # Add the right ankle joint to the OpenSim model
    empty_model.addJoint(right_ankle_joint)


def update_mesh_file_paths(input_osim, output_osim, mesh_directory, foot_mesh_files):
    """
    Updates the paths of <mesh_file> elements in an OpenSim .osim file to relative paths
    based on the path from the .osim file to the mesh directory.

    Parameters:
    - input_osim (str): Path to the input .osim file.
    - output_osim (str): Path to save the updated .osim file.
    - foot_mesh_files (list of str): List of mesh filenames (e.g., ["l_talus.vtp", "r_talus.vtp"]).

    Returns:
    - None
    """

    # Parse the .osim file
    tree = ET.parse(input_osim)
    root = tree.getroot()

    # Track updated files
    updated_count = 0

    # Find and update <mesh_file> elements
    for mesh_file_element in root.findall(".//mesh_file"):
        current_file = mesh_file_element.text.strip()

        # Check if the current mesh file matches one in the provided list
        for foot_mesh in foot_mesh_files:
            if current_file.endswith(foot_mesh):  # Ensure we match the filename regardless of the path
                mesh_path = os.path.join(mesh_directory, foot_mesh)
                relative_path = os.path.relpath(mesh_path, os.path.dirname(output_osim))

                # Update the XML with the new absolute path
                mesh_file_element.text = relative_path
                updated_count += 1
                break  # Stop checking once a match is found

    # Save the updated .osim file
    if updated_count > 0:
        tree.write(output_osim)
        print(f"Updated {updated_count} mesh file references.")
    else:
        print("No matching <mesh_file> elements found to update.")


def estimate_body_segment_parameters(height, weight):
    """
    Estimates the segment masses and inertial properties of the body based on height and weight.

    Args:
        height (float): Height of the participant in meters.
        weight (float): Weight of the participant in kg.

    Returns:
        dict: A dictionary containing segment masses and inertial properties.
    """
    # Segment mass as percentage of body mass (Winter, 2009 Biomechanics & ASCM)
    segment_mass_percentages = {
        "pelvis": 0.111,  # 11.1% of body mass
        "femur": 0.146,  # 14.6% of body mass (each)
        "tibfib": 0.0465,  # 4.65% of body mass (each)
    }

    # Estimated segment lengths as a percentage of body height (Winter, 2009)
    segment_length_percentages = {
        "pelvis": 0.24,  # 24% of height
        "femur": 0.245,  # 24.5% of height
        "tibfib": 0.246,  # 24.6% of height
    }

    # Approximate segment radii (based on height & segment length)
    segment_radii_percentages = {
        "pelvis": 0.14,  # Pelvis is wider
        "femur": 0.12,  # Femur is narrower
        "tibfib": 0.09,  # Tibia/Fibula is thinnest
    }

    # Compute segment masses
    masses = {key: weight * value for key, value in segment_mass_percentages.items()}

    # Estimate segment lengths
    segment_lengths = {key: height * value for key, value in segment_length_percentages.items()}

    # Estimate segment radii
    segment_radii = {key: segment_lengths[key] * segment_radii_percentages[key] for key in segment_radii_percentages}

    # Compute inertia using appropriate models
    inertias = {}

    for segment in segment_mass_percentages.keys():
        mass = masses[segment]

        if segment == "pelvis":
            # Pelvis modeled as an ellipsoid
            a = 0.20 * height / 2  # Half of pelvic width
            b = 0.14 * height / 2  # Half of pelvic depth
            c = 0.24 * height / 2  # Half of pelvic height

            I_x = (1 / 5) * mass * (b ** 2 + c ** 2)  # Rotation around X (Forward-Backward)
            I_y = (1 / 5) * mass * (a ** 2 + c ** 2)  # Rotation around Y (Vertical)
            I_z = (1 / 5) * mass * (a ** 2 + b ** 2)  # Rotation around Z (Left-Right)

            inertias[segment] = [I_x, I_y, I_z, 0, 0, 0]  # OpenSim inertia format

        else:
            # Long bones modeled as cylinders
            length = segment_lengths[segment]
            radius = segment_radii[segment]

            I1 = (1 / 12) * mass * (3 * radius ** 2 + length ** 2)  # Transverse moment (X)
            I2 = (1 / 12) * mass * (3 * radius ** 2 + length ** 2)  # Transverse moment (Z)
            I3 = (1 / 2) * mass * radius ** 2  # Longitudinal moment (Y)

            # Assign inertia with correct OpenSim axes
            inertias[segment] = [I1, I3, I2, 0, 0, 0]  # Y-axis gets I3 (smallest)

    return {
        "masses": masses,
        "inertias": inertias
    }


def perform_updates(empty_model, output_folder, mesh_directory, model_name, weight, height, x_opt_left, x_opt_right,
                    left_lm, right_lm):
    """
    Performs a series of updates on an OpenSim model including setting joint ranges,
    renaming coordinates, updating body segment properties, modifying joint rotation axes,
    and ensuring proper subtalar joint configuration.

    Args:
        empty_model (osim.Model): The OpenSim model to be updated.
        output_folder (str): Path to the directory where the updated model will be saved.
        mesh_directory (str): Path to the directory containing the mesh files.
        model_name (str): Name of the model, used for output file naming.
        weight (float): Participant's weight in kilograms.
        height (float): Participant's height in meters.

    Returns:
        str: The path to the final updated .osim model file.

    Steps:
        1. Load and configure the initial OpenSim model.
        2. Set joint coordinate names and ranges for the pelvis, hip, knee, and ankle joints.
        3. Configure body segment properties including mass, center of mass, and inertia.
        4. Update rotation axes for subtalar joints ('calcn_l_to_talus_l' and 'calcn_r_to_talus_r').
        5. Move 'rx' coordinate from rotation3 to rotation1 for subtalar joints.
        6. Apply specific updates to the left and right subtalar joints, including renaming and range setting.
        7. Update mesh file paths for foot and talus models.
    """
    output_file = output_folder + "/"f"{model_name}.osim"

    # Load the selected model
    model = empty_model
    state = model.initSystem()

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

    #hip_add_l, hip_add_r = zero_joint_orientation(model, output_file, x_opt_left, x_opt_right, l_hip_joint, r_hip_joint,
    #                                              l_knee_joint, r_knee_joint, pelvis_joint, left_lm, right_lm)

    pelvis_obliquity = pelvis_joint.upd_coordinates(0)
    pelvis_rotation = pelvis_joint.upd_coordinates(1)
    pelvis_tilt = pelvis_joint.upd_coordinates(2)
    # rename pelvis rotations and set default values from x_opt

    pelvis_obliquity.setName("pelvis_list")
    pelvis_obliquity.setDefaultValue(((x_opt_left['pelvis_rigid'][3] + x_opt_right['pelvis_rigid'][3]) / 2))
    pelvis_rotation.setName("pelvis_rotation")
    pelvis_rotation.setDefaultValue(((x_opt_left['pelvis_rigid'][4] + x_opt_right['pelvis_rigid'][4]) / 2))
    pelvis_tilt.setName("pelvis_tilt")
    pelvis_tilt.setDefaultValue(((x_opt_left['pelvis_rigid'][5] + x_opt_right['pelvis_rigid'][5]) / 2))
    model.finalizeConnections()
    # Access and rename the translational coordinates
    pelvis_translation_x = pelvis_joint.upd_coordinates(3)  # Translation along x-axis
    pelvis_translation_y = pelvis_joint.upd_coordinates(4)  # Translation along y-axis
    pelvis_translation_z = pelvis_joint.upd_coordinates(5)  # Translation along z-axis

    pelvis_translation_x.setName("pelvis_tx")
    #pelvis_translation_x.setDefaultValue(((x_opt_left['pelvis_rigid'][0] + x_opt_right['pelvis_rigid'][0]) / 2000))
    pelvis_translation_y.setName("pelvis_ty")
    #pelvis_translation_y.setDefaultValue(((x_opt_left['pelvis_rigid'][1] + x_opt_right['pelvis_rigid'][1]) / 2000))
    pelvis_translation_z.setName("pelvis_tz")
    #pelvis_translation_z.setDefaultValue(((x_opt_left['pelvis_rigid'][2] + x_opt_right['pelvis_rigid'][2]) / 2000))

    # Set coordinates range for left hip joint
    l_hip_flexion = l_hip_joint.upd_coordinates(0)
    l_hip_abduction = l_hip_joint.upd_coordinates(1)
    l_hip_rotation = l_hip_joint.upd_coordinates(2)

    l_hip_flexion.setRangeMin(-1.5)
    l_hip_flexion.setRangeMax(1.8)
    l_hip_flexion.setDefaultValue(x_opt_left['hip_rot'][0])

    l_hip_rotation.setRangeMin(-0.8)
    l_hip_rotation.setRangeMax(0.8)
    l_hip_rotation.setDefaultValue(x_opt_left['hip_rot'][1]*-0.1)

    l_hip_abduction.setRangeMin(-0.8)
    l_hip_abduction.setRangeMax(1.2)
    l_hip_abduction.setDefaultValue(x_opt_left['hip_rot'][2]*-0.1)

    model.finalizeConnections()

    # Set coordinates range for right hip joint
    r_hip_flexion = r_hip_joint.upd_coordinates(0)
    r_hip_abduction = r_hip_joint.upd_coordinates(1)
    r_hip_rotation = r_hip_joint.upd_coordinates(2)

    r_hip_flexion.setRangeMin(-1.5)
    r_hip_flexion.setRangeMax(1.8)
    r_hip_flexion.setDefaultValue(x_opt_right['hip_rot'][0])

    r_hip_rotation.setRangeMin(-0.8)
    r_hip_rotation.setRangeMax(0.8)
    r_hip_rotation.setDefaultValue(x_opt_right['hip_rot'][1]*0.1)

    r_hip_abduction.setRangeMin(-1.2)
    r_hip_abduction.setRangeMax(0.8)
    r_hip_abduction.setDefaultValue(x_opt_right['hip_rot'][2]*0.1)

    model.finalizeConnections()
    # Set coordinates range and names for left knee joint

    l_knee_flexion = l_knee_joint.upd_coordinates(0)
    l_knee_flexion.setName("knee_flexion_l")
    l_knee_flexion.setRangeMin(-0.2)
    l_knee_flexion.setRangeMax(2.2)
    l_knee_flexion.setDefaultValue(x_opt_left['knee_rot'][0])
    model.finalizeConnections()

    l_knee_add = l_knee_joint.upd_coordinates(1)
    l_knee_add.setName("knee_adduction_l")
    l_knee_add.setRangeMin(x_opt_left['knee_rot'][1]*-0.1)
    l_knee_add.setRangeMax(x_opt_left['knee_rot'][1]*-0.1)
    l_knee_add.setDefaultValue(x_opt_left['knee_rot'][1]*-0.1)
    l_knee_add.setDefaultLocked(True)
    model.finalizeConnections()

    l_knee_rot = l_knee_joint.upd_coordinates(2)
    l_knee_rot.setName("knee_rotation_l")
    l_knee_rot.setRangeMin(0.0)
    l_knee_rot.setRangeMax(0.0)
    l_knee_rot.setDefaultValue(0.0)
    l_knee_rot.setDefaultLocked(True)
    model.finalizeConnections()

    # Set coordinates range and names for right knee joint
    r_knee_flexion = r_knee_joint.upd_coordinates(0)
    r_knee_flexion.setName("knee_flexion_r")
    r_knee_flexion.setRangeMin(-0.2)
    r_knee_flexion.setRangeMax(2.2)
    r_knee_flexion.setDefaultValue(x_opt_right['knee_rot'][0])
    model.finalizeConnections()

    r_knee_add = r_knee_joint.upd_coordinates(1)
    r_knee_add.setName("knee_adduction_r")
    r_knee_add.setRangeMin(x_opt_right['knee_rot'][1]*0.1)
    r_knee_add.setRangeMax(x_opt_right['knee_rot'][1]*0.1)
    r_knee_add.setDefaultValue(x_opt_right['knee_rot'][1]*0.1)
    r_knee_add.setDefaultLocked(True)
    model.finalizeConnections()

    r_knee_rot = r_knee_joint.upd_coordinates(2)
    r_knee_rot.setName("knee_rotation_r")
    r_knee_rot.setRangeMin(0.0)
    r_knee_rot.setRangeMax(0.0)
    r_knee_rot.setDefaultValue(0.0)
    r_knee_rot.setDefaultLocked(True)
    model.finalizeConnections()

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

    def set_mass_com_inertia(body, mass, com, inertia):
        """
        Sets mass, center of mass, and inertia for an OpenSim body.

        Args:
            body (osim.Body): OpenSim body segment.
            mass (float): Mass in kg.
            com (list): Center of mass [x, y, z] in meters.
            inertia (list): Inertia tensor [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].
        """
        body.setMass(mass)
        body.setMassCenter(osim.Vec3(*com))
        body.setInertia(osim.Inertia(*inertia))

    # Compute body segment parameters
    params = estimate_body_segment_parameters(height, weight)
    masses = params["masses"]
    inertias = params["inertias"]

    # Set all COMs at [0,0,0] (assuming mesh centroids)
    coms = {key: [0, 0, 0] for key in masses.keys()}

    # Apply mass, center of mass, and inertia
    set_mass_com_inertia(pelvis, masses["pelvis"], coms["pelvis"], inertias["pelvis"])
    set_mass_com_inertia(femur_l, masses["femur"], coms["femur"], inertias["femur"])
    set_mass_com_inertia(femur_r, masses["femur"], coms["femur"], inertias["femur"])
    set_mass_com_inertia(tibfib_l, masses["tibfib"], coms["tibfib"], inertias["tibfib"])
    set_mass_com_inertia(tibfib_r, masses["tibfib"], coms["tibfib"], inertias["tibfib"])

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

    # updates the path to feet mesh files
    update_mesh_file_paths(input_file, output_file, mesh_directory,
                           ["l_bofoot.vtp", "r_bofoot.vtp", "l_foot.vtp", "r_foot.vtp", "l_talus.vtp", "r_talus.vtp"])

    return output_file


def feet_adjustments(output_file, empty_model, mocap_static_trc, realign_feet=True):
    """
      Adjusts the orientation of the left and right feet in an OpenSim model to align with mocap (motion capture) data.

      The function computes the appropriate rotations for both feet to ensure they are aligned with the ground
      and match the positions indicated by the mocap static trial. This is particularly useful for preparing the
      model for inverse kinematics or other biomechanical analyses.

      Args:
          output_file (str): Path to save the updated OpenSim model file.
          empty_model (osim.Model): The OpenSim model with foot components to be aligned.
          mocap_static_trc (dict): Dictionary containing motion capture marker data with marker names as keys
                                   and (x, y, z) coordinates as values.
          realign_feet (bool, optional): Whether to fully realign the feet.
                                         If False, only minimal adjustments are made. Default is True.

      Returns:
          None. The function modifies the OpenSim model in place and saves the updated model to the specified file.

      Key Steps:
      1. Initialize the model's system.
      2. Calculate the foot vectors (heel to toe) for both feet in the model's coordinate system.
      3. Compute the target foot vectors from the mocap data.
      4. Calculate the Euler angles required to align the model foot vectors with the mocap vectors.
      5. Apply these rotations to the ankle joints of both feet.
      6. Further adjust the foot orientation to ensure they are flat with the ground.

      Example Usage:
          feet_adjustments("updated_model.osim", model, mocap_data, realign_feet=True)

      Note:
          This function primarily handles the left and right feet independently and uses Euler angles for
          rotation adjustments. It focuses on aligning the feet both forward-facing and flat to the ground.
      """
    # Initialize the model's system
    state = empty_model.initSystem()
    # === Adjust Orientation of the Left Foot ===

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
    heel_position_in_calcn = np.array(
        [heel_local_position.get(0), heel_local_position.get(1), heel_local_position.get(2)])

    # Compute the initial foot vector (heel to toe, normalized)
    left_foot_vector_initial = vector_between_points(heel_position_in_calcn, toe_position_in_calcn, True)

    # Compute the actual foot vector from mocap data (heel to toe, normalized)
    # Mocap data is rotated and negated to align with the model's coordinate system
    left_foot_vector_actual = vector_between_points(mocap_static_trc["LHEE"], mocap_static_trc["LTOE"],
                                                    True
                                                    )

    # Compute the Euler angles to align the initial vector with the actual vector
    l_foot_update_to_match_actual_rotation = compute_euler_angles_from_vectors(
        left_foot_vector_initial, left_foot_vector_actual
    )

    # Set unnecessary rotations (x and z axes) to zero
    l_foot_update_to_match_actual_rotation[0] = 0

    if not realign_feet:
        l_foot_update_to_match_actual_rotation[1] = 0
        l_foot_update_to_match_actual_rotation[2] = 0

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

    # Attempting to make the foot be flat with the ground
    left_foot_transform_in_ground = toes_body.getTransformInGround(state)
    # Extract the rotation matrix from the transform
    rotation_matrix = left_foot_transform_in_ground.R().asMat33()
    # Convert the rotation matrix to Euler angles
    rotation = osim.Rotation(rotation_matrix)
    euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

    # Set unnecessary rotations (x and z axes) to zero
    euler_angles[0] = 0
    euler_angles[1] = 0
    if not realign_feet:
        euler_angles[2] = 0

    # Extract the components of the Vec3 and negate them
    inverse_euler_angles = osim.Vec3(
        -euler_angles.get(0),  # Negate X angle
        -euler_angles.get(1),  # Negate Y angle
        -euler_angles.get(2)  # Negate Z angle
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

    # === Adjust Orientation of the Right Foot ===

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
    heel_position_in_calcn_r = np.array(
        [heel_local_position.get(0), heel_local_position.get(1), heel_local_position.get(2)])

    # Compute the initial foot vector (heel to toe, normalized)
    right_foot_vector_initial = vector_between_points(heel_position_in_calcn_r, toe_position_in_calcn_r, True)

    # Compute the actual foot vector from mocap data (heel to toe, normalized)
    # Mocap data is rotated and negated to align with the model's coordinate system
    right_foot_vector_actual = vector_between_points(mocap_static_trc["RHEE"], mocap_static_trc["RTOE"],
                                                     True
                                                     )

    # Plot the two vectors for visualization
    # plot_3d_vectors(right_foot_vector_initial, right_foot_vector_actual)

    # Compute the Euler angles to align the initial vector with the actual vector
    r_foot_update_to_match_actual_rotation = compute_euler_angles_from_vectors(
        right_foot_vector_initial, right_foot_vector_actual
    )

    # Set unnecessary rotations (x and z axes) to zero
    r_foot_update_to_match_actual_rotation[0] = 0

    if not realign_feet:
        r_foot_update_to_match_actual_rotation[1] = 0
        r_foot_update_to_match_actual_rotation[2] = 0

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

    # Attempting to make the foot be flat with the ground
    right_foot_transform_in_ground = toes_body.getTransformInGround(state)
    # Extract the rotation matrix from the transform
    rotation_matrix = right_foot_transform_in_ground.R().asMat33()
    # Convert the rotation matrix to Euler angles
    rotation = osim.Rotation(rotation_matrix)
    euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

    # Set unnecessary rotations (x and z axes) to zero
    euler_angles[0] = 0
    euler_angles[1] = 0

    if not realign_feet:
        euler_angles[2] = 0

    # Extract the components of the Vec3 and negate them
    inverse_euler_angles = osim.Vec3(
        -euler_angles.get(0),  # Negate X angle
        -euler_angles.get(1),  # Negate Y angle
        -euler_angles.get(2)  # Negate Z angle
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


def perform_scaling(output_directory, output_file, static_trc_file):
    """
    Performs scaling of an OpenSim model using a scaling tool with marker-based calibration.

    This function uses an OpenSim ScaleTool to adjust the size and marker positions of a musculoskeletal model
    based on a participant's motion capture data. The scaling process uses a predefined ScaleSettings XML file
    to guide the scaling and marker placement process.

    Args:
        output_directory (str): Path to the directory where the outputs should be written.
        output_file (str): Path to the OpenSim model (.osim) file to be scaled.
        static_trc_file (str): Path to the motion capture (.trc) file containing static marker data.

    Returns:
        None
    """

    scaling_file = os.path.join(high_level_inputs, "ScaleSettings.xml")
    scale_tool = osim.ScaleTool(scaling_file)
    scale_tool.setPathToSubject(os.path.join(output_directory, ""))

    # Set the model file
    scale_tool.getGenericModelMaker().setModelFileName(output_file)  # Replace with your model file

    ignore, (start_time, end_time), dontcare = read_trc_file_as_dict(static_trc_file, True)
    # Create an OpenSim ArrayDouble and populate it with start_time and end_time
    time_range = osim.ArrayDouble()
    time_range.append(start_time)
    time_range.append(end_time)

    # Set the output file for the MarkerPlacer and MarkerPlacer settings
    # Do u want to move markers to match the static file? - causes the feet to be poor currently
    relative_path = os.path.relpath(static_trc_file, output_directory)

    scale_tool.getMarkerPlacer().setApply(True)
    scale_tool.getMarkerPlacer().setOutputModelFileName("scaled_foot.osim")
    scale_tool.getMarkerPlacer().setMarkerFileName(relative_path)
    scale_tool.getMarkerPlacer().setTimeRange(time_range)

    scale_tool.getModelScaler().setOutputModelFileName("scaled_foot.osim")
    scale_tool.getModelScaler().setMarkerFileName(relative_path)
    scale_tool.getModelScaler().setTimeRange(time_range)

    scaled_output_file = os.path.join(output_directory, "scaling_tool_settings.xml")

    # Verify the loaded scaling settings (optional)
    scale_tool.printToXML(scaled_output_file)  # Outputs a copy of the loaded settings

    # Run the scaling process
    scale_tool.run()
