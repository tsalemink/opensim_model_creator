# Import Packages
import opensim as osim
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from opensim_model_creator.Functions.file_utils import search_files_by_keywords
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
            marker_location = osim.Vec3(*landmark_position)

            # Determine the marker's name
            final_name = custom_names[i] if custom_names else marker_name

            # Create and add the marker
            marker = osim.Marker(final_name, body, marker_location)
            model.addMarker(marker)

            print(f"Marker '{final_name}' added to body '{body_name}' at location {location}.")

    except Exception as e:
        print(f"Error adding markers to body '{body_name}': {e}")


def optimize_knee_axis(model_path, trc_file, start_time, end_time, marker_weights, initial_params, temp_model_path_1,
                       temp_model_path_2, final_output_model):
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
    output_directory = os.path.dirname(final_output_model)
    osim_results_directory = os.path.join(output_directory, "opensim_results")

    def objective(params):
        left_knee_x, right_knee_x, left_knee_y, right_knee_y = params

        # Adjust left knee
        adjust_joint_orientation(
            model_path=model_path,
            joint_name="knee_l",
            rotation_adjustment=osim.Vec3(left_knee_x, left_knee_y, 0.0),
            output_model_path=temp_model_path_1
        )

        # Adjust right knee
        adjust_joint_orientation(
            model_path=temp_model_path_1,
            joint_name="knee_r",
            rotation_adjustment=osim.Vec3(right_knee_x, right_knee_y, 0.0),
            output_model_path=temp_model_path_2
        )
        # Perform IK and compute error
        errors = perform_IK(temp_model_path_2, trc_file, osim_results_directory, start_time, end_time, marker_weights)
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


def perform_IK(model_file, trc_file, results_directory, start_time, end_time, marker_weights):
    """
    Perform Inverse Kinematics analysis using OpenSim.

    Args:
        model_file (str): Path to the OpenSim model file.
        trc_file (str): Path to the TRC file.
        results_directory (str): Path to output OpenSim processing results to.
        start_time (float): Start time for IK.
        end_time (float): End time for IK.
        marker_weights (dict): Marker weights for IK analysis.

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


def run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1,
                                temp_model_path_2, marker_weights, final_output_model_path, initial_params=None):
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
        final_output_model_path: where to save the optimised knee joint model

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
        final_output_model=final_output_model_path
    )

    print(f"Optimized Joint Orientations: {result.x}")


def initialize_model_and_extract_landmarks(asm_directory, model_directory):
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
    empty_model = osim.Model(os.path.join(model_directory, "Feet_scaled.osim"))  # Load the base model file
    state = empty_model.initSystem()  # Initialise the system
    marker_set = empty_model.updMarkerSet()
    marker_set.remove(marker_set.getIndex("RASI"))
    marker_set.remove(marker_set.getIndex('LASI'))
    marker_set.remove(marker_set.getIndex('RKNE'))
    marker_set.remove(marker_set.getIndex('LKNE'))
    marker_set.remove(marker_set.getIndex('RANK'))
    marker_set.remove(marker_set.getIndex('LANK'))
    print(f"Extra markers removed from foot model")

    # Load and extract landmarks for left and right limbs
    left_landmarks_file = search_files_by_keywords(asm_directory, "left lms predicted")[0]
    right_landmarks_file = search_files_by_keywords(asm_directory, "right lms predicted")[0]
    left_landmarks = load_landmarks(left_landmarks_file)
    right_landmarks = load_landmarks(right_landmarks_file)
    x_opt_left_file = search_files_by_keywords(asm_directory, "x opt left")[0]
    x_opt_right_file = search_files_by_keywords(asm_directory, "x opt right")[0]
    x_opt_left = load_x_opt(x_opt_left_file)
    x_opt_right = load_x_opt(x_opt_right_file)

    # initialise units and gravity
    empty_model.set_gravity(osim.Vec3(0, -9.80665, 0))
    empty_model.set_length_units('meters')
    empty_model.set_force_units('N')

    # placeholders for publications and credits
    empty_model.set_credits('Carman et. al., 2025')
    empty_model.set_publications('Carman et. al., 2025')

    return empty_model, state, left_landmarks, right_landmarks, x_opt_left, x_opt_right


def create_pelvis_body_and_joint(model, left_landmarks, right_landmarks, meshes, mocap_static_trc, sex):
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
            - rotated_pelvis_center (np.array): The rotated center of the pelvis mesh.
    """
    # Create the pelvis body
    pelvis = osim.Body("pelvis_b", 1.0, osim.Vec3(0, 0, 0), osim.Inertia(0, 0, 0))
    model.addBody(pelvis)
    body_axes = {}

    # Compute pelvis alignment, this is to align the bone meshes to the opensim global coordinate frame
    LASIS = left_landmarks["ASIS"]
    RASIS = right_landmarks["ASIS"]
    SACR = (left_landmarks['SAC'] + right_landmarks['SAC']) / 2
    r_hjc = right_landmarks["hjc"]
    l_hjc = left_landmarks["hjc"]

    # define the length of the pelvis (for centre of mass calculations)
    asis_mid = (RASIS + LASIS) / 2
    asis_width = np.sqrt(np.sum((RASIS - LASIS) ** 2, axis=0))
    if sex == 1:
        # find lumbar joint centre based on regression equations of (Dumas et al., 2018, 2007)
        lumbar_joint_centre = asis_mid + np.array([-0.34 * asis_width, 0.049 * asis_width, 0])
    elif sex == 2:
        lumbar_joint_centre = asis_mid + np.array([-0.335 * asis_width, -0.032 * asis_width, 0])

    centre_of_hjc = (r_hjc + l_hjc) / 2
    pelvis_length = np.sqrt(np.sum((lumbar_joint_centre - centre_of_hjc) ** 2, axis=0))

    # Define the pelvis anatomical coordinate system from the articulated shape model (this needs to be aligned with
    # the opensim global coordinate system)
    pelvis_origin, x_axis, y_axis, z_axis = model_alignment.createPelvisACSISB_sacr(LASIS, RASIS, SACR)
    body_axes['pelvis'] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis
    }

    # Create an OpenSim Rotation from pelvis axes
    rot = create_osim_rot(x_axis, y_axis, z_axis)

    # set opensim rotation object
    pelvis_rotation_osim = osim.Rotation(rot)

    # Compute ground height offset (for visualization)
    RANK = right_landmarks["malleolus_med"]
    pelvis_center = midpoint_3d(RASIS, LASIS)
    RASIS_to_RANK = np.linalg.norm(vector_between_points(RASIS, RANK))
    height_offset = np.array([0.0, RASIS_to_RANK + 0.035, 0.0])

    # compute pelvis translation as an osim vector
    pelvis_translation = osim.Vec3(pelvis_origin - height_offset)

    # Create the spatial transform for the custom pelvis to ground joint
    spatial_transform_pelvis = osim.SpatialTransform()

    # First rotation (pelvis rotation) along Y-axis
    pelvis_rotation = spatial_transform_pelvis.updTransformAxis(0)
    pelvis_rotation.setCoordinateNames(osim.ArrayStr("pelvis_rotation", 1))
    pelvis_rotation.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
    pelvis_rotation.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Second rotation (pelvis list/obliquity) along X-axis
    pelvis_list = spatial_transform_pelvis.updTransformAxis(1)
    pelvis_list.setCoordinateNames(osim.ArrayStr("pelvis_list", 1))
    pelvis_list.setAxis(osim.Vec3(1, 0, 0))  # Z-axis
    pelvis_list.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Third rotation (pelvis tilt) along Z-axis
    pelvis_tilt = spatial_transform_pelvis.updTransformAxis(2)
    pelvis_tilt.setCoordinateNames(osim.ArrayStr("pelvis_tilt", 1))
    pelvis_tilt.setAxis(osim.Vec3(0, 0, 1))  # X-axis
    pelvis_tilt.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Fourth transform (pelvis tx) along X-axis
    pelvis_tx = spatial_transform_pelvis.updTransformAxis(3)
    pelvis_tx.setCoordinateNames(osim.ArrayStr("pelvis_tx", 1))
    pelvis_tx.setAxis(osim.Vec3(1, 0, 0))  # X-axis
    pelvis_tx.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Fifth transform (pelvis ty) along Y-axis
    pelvis_ty = spatial_transform_pelvis.updTransformAxis(4)
    pelvis_ty.setCoordinateNames(osim.ArrayStr("pelvis_ty", 1))
    pelvis_ty.setAxis(osim.Vec3(0, 1, 0))  # Z-axis
    pelvis_ty.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Fourth transform (pelvis tz) along Z-axis
    pelvis_tz = spatial_transform_pelvis.updTransformAxis(5)
    pelvis_tz.setCoordinateNames(osim.ArrayStr("pelvis_tz", 1))
    pelvis_tz.setAxis(osim.Vec3(0, 0, 1))  # Y-axis
    pelvis_tz.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Attach the pelvis body to the ground using a FreeJoint, set the rotation and translation for the pelvis relative
    # to the global CS
    pelvis_joint = osim.CustomJoint(
        "pelvis_to_ground",
        model.getGround(),
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        pelvis,
        osim.Vec3(-pelvis_origin),
        pelvis_rotation_osim.convertRotationToBodyFixedXYZ(),
        spatial_transform_pelvis
    )
    model.addJoint(pelvis_joint)

    # Attach the mesh for the pelvis
    mesh_path = os.path.join(meshes, "combined_pelvis_mesh.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    add_mesh_to_body(model, "pelvis_b", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=-pelvis_origin)

    # Add mocap markers
    add_markers_to_body(model, "pelvis_b", ["RASI", "LASI", "RPSI", "LPSI"], mocap_static_trc, pelvis_origin)

    # Add anatomical landmarks
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS", "SAC"], left_landmarks, pelvis_origin,
                        ["LASI_ssm", "LPSI_ssm", "SAC_ssm"])
    add_markers_to_body(model, "pelvis_b", ["ASIS", "PSIS"], right_landmarks, pelvis_origin,
                        ["RASI_ssm", "RPSI_ssm"])

    return pelvis, pelvis_origin, pelvis_length, lumbar_joint_centre, body_axes, pelvis_rotation_osim


def create_femur_bodies_and_hip_joints(empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc, pelvis,
                                       pelvis_centre, body_axes, pelvis_rot):
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
            - left_femur (osim.Body): The left femur body added to the model.
            - femur_l_center (np.array): Original center of the left femur.
            - right_femur (osim.Body): The right femur body added to the model.
            - femur_r_center (np.array): Original center of the right femur.
    """
    # Define the femur body properties (common for both left and right femurs)
    femur_mass = 8.0  # Mass of the femur in kg
    femur_mass_center = osim.Vec3(0, -0.2, 0)  # Center of mass location in the femur frame
    femur_inertia = osim.Inertia(0.1, 0.1, 0.01)  # Moments of inertia

    # Create the left and right femur bodies
    left_femur = osim.Body("femur_l", femur_mass, femur_mass_center, femur_inertia)
    right_femur = osim.Body("femur_r", femur_mass, femur_mass_center, femur_inertia)

    # Add the femur bodies to the model
    empty_model.addBody(left_femur)
    empty_model.addBody(right_femur)

    # Extract landmarks required to position the joint coordinate systems of the left hip joint
    r_hjc = right_landmarks["hjc"]
    l_hjc = left_landmarks["hjc"]
    l_ecc = (left_landmarks["LEC"] + left_landmarks["MEC"]) / 2  # left epicondylar centre
    r_ecc = (right_landmarks["LEC"] + right_landmarks["MEC"]) / 2  # right epicondylar centre

    # calculate femur length (needed for centre of mass calculation
    l_femur_length = np.sqrt(np.sum((l_hjc - l_ecc) ** 2, axis=0))
    r_femur_length = np.sqrt(np.sum((r_hjc - r_ecc) ** 2, axis=0))

    # Attach the mesh for the right femur
    mesh_path = os.path.join(meshes, "predicted_mesh_right_femur.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    femur_r_center = r_hjc  # Extract center of the right femur
    add_mesh_to_body(empty_model, "femur_r", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=-femur_r_center)

    # Attach the mesh for the left femur
    mesh_path = os.path.join(meshes, "predicted_mesh_left_femur.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    femur_l_center = l_hjc  # Extract center of the right femur
    add_mesh_to_body(empty_model, "femur_l", relative_path, offset_orientation=(0, 0, 0),
                     offset_translation=-femur_l_center)

    # Add mocap markers to the femur bodies, taken from static trial for tracking markers
    add_markers_to_body(empty_model, "femur_l", ["LTHI", "LKNE", "LKNEM"], mocap_static_trc, femur_l_center)
    add_markers_to_body(empty_model, "femur_r", ["RTHI", "RKNE", "RKNEM"], mocap_static_trc, femur_r_center)

    # Add anatomical landmarks to the femur bodies with custom marker names, taken from shape model prediction for anatomical markers
    add_markers_to_body(empty_model, "femur_l", ["LEC", "MEC"], left_landmarks, femur_l_center,
                        ["LKNE_ssm", "LKNEM_ssm"])
    add_markers_to_body(empty_model, "femur_r", ["LEC", "MEC"], right_landmarks, femur_r_center,
                        ["RKNE_ssm", "RKNEM_ssm"])

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
    rotation_axis_left.set_function(osim.LinearFunction(1, 0))  # Ensures movement.

    # set opensim rotation object
    femur_l_origin, x_axis, y_axis, z_axis = model_alignment.createFemurACSISB(femur_l_center, left_landmarks['MEC'],
                                                                               left_landmarks['LEC'], side='left')
    body_axes['femur_l'] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis
    }
    rot_l = osim.Rotation(create_osim_rot_bodies(x_axis, y_axis, z_axis, body_axes['pelvis']['x'], body_axes['pelvis']['y'], body_axes['pelvis']['z'], inverse=True))
    femur_r_origin, x_axis, y_axis, z_axis = model_alignment.createFemurACSISB(femur_r_center, right_landmarks['MEC'],
                                                                               right_landmarks['LEC'], side='right')
    body_axes['femur_r'] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis
    }
    rot_r = osim.Rotation(create_osim_rot_bodies(x_axis, y_axis, z_axis, body_axes['pelvis']['x'], body_axes['pelvis']['y'], body_axes['pelvis']['z'], inverse=True))

    # Create the custom left hip joint with all restored parameters, femur orientation defined from x_opt
    left_hip_joint = osim.CustomJoint(
        "hip_l",  # Joint name
        pelvis,  # Parent frame (Pelvis)
        osim.Vec3(l_hjc - pelvis_centre),  # Location in parent frame
        osim.Vec3(0, 0, 0),  # Orientation in parent frame
        left_femur,  # Child frame (Femur)
        osim.Vec3(0, 0, 0),  # Location in child frame
        rot_l.convertRotationToBodyFixedXYZ(),
        spatial_transform_left  # The defined spatial transform
    )
    ################################################
    # Creation of the right hip joint coordinate system
    # Create the spatial transform for the custom joint ###Need to update
    spatial_transform = osim.SpatialTransform()

    # First rotation (Flexion/Extension) along X-axis
    flexion_axis = spatial_transform.updTransformAxis(0)
    flexion_axis.setCoordinateNames(osim.ArrayStr("hip_flexion_r", 1))
    flexion_axis.setAxis(osim.Vec3(0, 0, 1))  # X-axis
    flexion_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Second rotation (Adduction/Abduction) along Z-axis (opposite to left)
    adduction_axis = spatial_transform.updTransformAxis(1)
    adduction_axis.setCoordinateNames(osim.ArrayStr("hip_adduction_r", 1))
    adduction_axis.setAxis(osim.Vec3(1, 0, 0))  # Z-axis
    adduction_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Third rotation (Internal/External Rotation) along Y-axis (opposite to left)
    rotation_axis = spatial_transform.updTransformAxis(2)
    rotation_axis.setCoordinateNames(osim.ArrayStr("hip_rotation_r", 1))
    rotation_axis.setAxis(osim.Vec3(0, 1, 0))  # Y-axis
    rotation_axis.set_function(osim.LinearFunction(1, 0))  # Ensures movement

    # Create the custom hip joint with all restored parameters, femur orientation defined from x_opt
    right_hip_joint = osim.CustomJoint(
        "hip_r",  # Joint name
        pelvis,  # Parent frame (Pelvis)
        osim.Vec3(r_hjc - pelvis_centre),  # Location in parent frame
        osim.Vec3(0, 0, 0),  # Orientation in parent frame
        right_femur,  # Child frame (Femur)
        osim.Vec3(0, 0, 0),  # Location in child frame
        rot_r.convertRotationToBodyFixedXYZ(),
        spatial_transform  # The defined spatial transform
    )
    ########################################################################################

    # Add the hip joints to the model
    empty_model.addJoint(left_hip_joint)
    empty_model.addJoint(right_hip_joint)

    return left_femur, femur_l_center, right_femur, femur_r_center, l_femur_length, r_femur_length, body_axes


def create_tibfib_bodies_and_knee_joints(
        empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc,
        left_femur, right_femur, femur_l_center, femur_r_center, body_axes):
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
        x_opt_right (dict): default joint orientation angles for the right side
        x_opt_left (dict): default joint orientation angles for the left side

    Returns:
        tuple:
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
    left_tibfib = osim.Body("tibfib_l", tibfib_mass, tibfib_mass_center, tibfib_inertia)  # Left tibfib
    right_tibfib = osim.Body("tibfib_r", tibfib_mass, tibfib_mass_center, tibfib_inertia)  # Right tibfib

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
    add_mesh_to_body(empty_model, "tibfib_r", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=-tibia_r_center)

    # Attach the mesh for the right fibula body
    # Search for the mesh file corresponding to the right fibula
    mesh_path = os.path.join(meshes, "predicted_mesh_right_fibula.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    # Add the mesh to the right tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_r", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=-tibia_r_center)

    # Attach the mesh for the left tibia body
    # Search for the mesh file corresponding to the left tibfib
    mesh_path = os.path.join(meshes, "predicted_mesh_left_tibia.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))
    l_LMAL = left_landmarks['malleolus_lat']
    l_MMAL = left_landmarks['malleolus_med']
    l_mid_mal = midpoint_3d(l_LMAL, l_MMAL)
    tibia_l_center = l_mid_mal

    tibfib_l_origin, x_axis, y_axis, z_axis = model_alignment.createTibiaFibulaACSISB_2(l_LMAL, l_MMAL,
                                                                                        left_landmarks['condyle_med'],
                                                                                        left_landmarks['condyle_lat'],
                                                                                        side='left')
    body_axes['tibfib_l'] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis
    }
    rot_fem_l = osim.Rotation(create_osim_rot(body_axes['femur_l']['x'], body_axes['femur_l']['y'], body_axes['femur_l']['z'], inverse=False))
    rot_l = osim.Rotation(create_osim_rot_bodies(x_axis, y_axis, z_axis, body_axes['femur_l']['x'], body_axes['femur_l']['y'], body_axes['femur_l']['z'], inverse=True))
    tibfib_r_origin, x_axis, y_axis, z_axis = model_alignment.createTibiaFibulaACSISB_2(r_LMAL, r_MMAL,
                                                                                        right_landmarks['condyle_med'],
                                                                                        right_landmarks['condyle_lat'],
                                                                                        side='right')
    body_axes['tibfib_r'] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis
    }
    rot_fem_r = osim.Rotation(
        create_osim_rot(body_axes['femur_r']['x'], body_axes['femur_r']['y'], body_axes['femur_r']['z'], inverse=False))
    rot_r = osim.Rotation(create_osim_rot_bodies(x_axis, y_axis, z_axis, body_axes['femur_r']['x'], body_axes['femur_r']['y'], body_axes['femur_r']['z'], inverse=True))

    # Add the mesh to the left tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_l", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=-tibia_l_center)

    # Attach the mesh for the left fibula body
    # Search for the mesh file corresponding to the left tibfib
    mesh_path = os.path.join(meshes, "predicted_mesh_left_fibula.stl")
    relative_path = os.path.relpath(mesh_path, os.path.dirname(meshes))

    # Add the mesh to the left tibfib body with an orientation offset to align axes
    add_mesh_to_body(empty_model, "tibfib_l", relative_path,
                     offset_orientation=(0, 0, 0),  # Align the mesh orientation with OpenSim axes
                     offset_translation=-tibia_l_center)

    # Add mocap markers to the tibfib bodies
    # Add mocap markers for the left tibfib body
    add_markers_to_body(empty_model, "tibfib_l", ["LTIB", "LTOE", "LHEE", "LMED", "LANK"], mocap_static_trc,
                        tibia_l_center)

    # Add landmark markers for the left tibfib body
    add_markers_to_body(empty_model, "tibfib_l", ["malleolus_med", "malleolus_lat"], left_landmarks, tibia_l_center,
                        ["LMED_ssm", "LANK_ssm"])

    # Add mocap markers for the right tibfib body
    add_markers_to_body(empty_model, "tibfib_r", ["RTIB", "RTOE", "RHEE", "RMED", "RANK"], mocap_static_trc,
                        tibia_r_center)

    # Add landmark markers for the right tibfib body
    add_markers_to_body(empty_model, "tibfib_r", ["malleolus_med", "malleolus_lat"], right_landmarks, tibia_r_center,
                        ["RMED_ssm", "RANK_ssm"])

    # Extract the medial and lateral epicondyle landmarks
    l_lec = left_landmarks["LEC"]  # Lateral epicondyle landmark
    l_mec = left_landmarks["MEC"]  # Medial epicondyle landmark

    # Compute the midpoint between the lateral and medial epicondyles
    l_EC_midpoint = midpoint_3d(l_lec, l_mec)

    # calculate tibfib length (needed for CoM calculations)
    l_tibfib_length = np.sqrt(np.sum((l_EC_midpoint - tibia_l_center) ** 2, axis=0))

    # %% Define the left knee joint
    # Create the spatial transform for the custom knee joint
    spatial_transform = osim.SpatialTransform()

    # First rotation (Flexion/Extension) along X-axis, positive flexion is -Xx
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
        "knee_l",  # Name of the joint
        left_femur,  # Parent body (femur)
        osim.Vec3(l_EC_midpoint - femur_l_center),  # Location of the joint in the femur frame
        rot_fem_l.convertRotationToBodyFixedXYZ(),  # Orientation of the joint in the femur frame
        left_tibfib,  # Child body (tibfib)
        osim.Vec3(l_EC_midpoint - tibia_l_center),  # Location of the joint in the tibfib frame
        rot_l.convertRotationToBodyFixedXYZ(),
        spatial_transform
    )
    # %% Positioning of the right knee joint

    # Extract the medial and lateral epicondyle landmarks
    r_lec = right_landmarks["LEC"]  # Lateral epicondyle landmark
    r_mec = right_landmarks["MEC"]  # Medial epicondyle landmark

    # Compute the midpoint between the lateral and medial epicondyles
    r_EC_midpoint = midpoint_3d(r_lec, r_mec)

    # calculate tibfib length (needed for CoM calculations)
    r_tibfib_length = np.sqrt(np.sum((r_EC_midpoint - tibia_r_center) ** 2, axis=0))

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
        "knee_r",  # Name of the joint
        right_femur,  # Parent body (femur)
        osim.Vec3(r_EC_midpoint - femur_r_center),  # Location of the joint in the femur frame
        rot_fem_r.convertRotationToBodyFixedXYZ(),  # Orientation of the joint in the femur frame
        right_tibfib,  # Child body (tibfib)
        osim.Vec3(r_EC_midpoint - tibia_r_center),  # Location of the joint in the tibfib frame
        rot_r.convertRotationToBodyFixedXYZ(),
        spatial_transform
    )
    # %% Adding the knee joints to the model

    # Add the left knee joint to the OpenSim model
    # This connects the left tibfib to the left femur, allowing flexion/extension motion
    empty_model.addJoint(left_knee_joint)

    # Add the right knee joint to the OpenSim model
    # This connects the right tibfib to the right femur, allowing flexion/extension motion
    empty_model.addJoint(right_knee_joint)

    return tibia_l_center, tibia_r_center, left_tibfib, right_tibfib, l_tibfib_length, r_tibfib_length, l_EC_midpoint, r_EC_midpoint


def repurpose_feet_bodies_and_create_joints(empty_model, left_tibfib, right_tibfib, body_axes):
    """
      Repurposes the foot bodies (talus) in the OpenSim model and creates ankle joints
      (PinJoint) connecting the talus to the tibia/fibula (tibfib) segments.

      Args:
          empty_model (osim.Model): The OpenSim model where the joints and bodies are added.
          tibfib_l_center (np.array): Center of the left tibfib segment in the rotated coordinate system.
          tibfib_r_center (np.array): Center of the right tibfib segment in the rotated coordinate system.
          left_tibfib (osim.Body): The left tibfib body in the OpenSim model.
          right_tibfib (osim.Body): The right tibfib body in the OpenSim model.

      Returns:
          None: The function modifies the OpenSim model in place by adding new joints.
      """
    # Access the body named "talus_l_b"
    left_talus = empty_model.getBodySet().get("talus_l")
    # Access the body named "talus_r_b"
    right_talus = empty_model.getBodySet().get("talus_r")

    # Locate the joint by name in the model's JointSet
    joint_name_to_remove = "talus_l_to_ground"  # Replace with the actual joint name
    if empty_model.getJointSet().contains(joint_name_to_remove):
        joint_to_remove = empty_model.getJointSet().get(joint_name_to_remove)
        empty_model.updJointSet().remove(joint_to_remove)
        print(f"Joint '{joint_name_to_remove}' has been removed.")
    else:
        print(f"Joint '{joint_name_to_remove}' not found in the model.")

    joint_name_to_remove = "talus_r_to_ground"  # Repeat for the right side if needed
    if empty_model.getJointSet().contains(joint_name_to_remove):
        joint_to_remove = empty_model.getJointSet().get(joint_name_to_remove)
        empty_model.updJointSet().remove(joint_to_remove)
        print(f"Joint '{joint_name_to_remove}' has been removed.")
    else:
        print(f"Joint '{joint_name_to_remove}' not found in the model.")

    rot_tib_l = osim.Rotation(
        create_osim_rot([0,0,0], body_axes['tibfib_l']['y'],[0,0,0],  inverse=False))

    rot_tib_r = osim.Rotation(
        create_osim_rot([0,0,0], body_axes['tibfib_r']['y'],[0,0,0],  inverse=False))

    # Define the ankle joint connecting the left talus to the left tibfib
    # A PinJoint allows rotation about a single axis (flexion/extension in this case)
    left_ankle_joint = osim.PinJoint(
        "ankle_l",  # Name of the joint
        left_tibfib,  # Parent body (tibfib)
        osim.Vec3(-0.0, -0.01, 0),  # Location of the joint in the tibfib frame
        rot_tib_l.convertRotationToBodyFixedXYZ(),  # Orientation of the joint in the tibfib frame
        left_talus,  # Child body (talus)
        osim.Vec3(0, 0, 0),  # Manually adjusted location of the joint in the talus frame
        osim.Vec3(-0.175895, 0.105208, 0.0186622)  # Orientation of the joint in the talus frame
    )

    # Define the ankle joint connecting the right talus to the right tibfib
    # A PinJoint allows rotation about a single axis (flexion/extension in this case)
    right_ankle_joint = osim.PinJoint(
        "ankle_r",  # Name of the joint
        right_tibfib,  # Parent body (tibfib)
        osim.Vec3(-0.0, -0.01, 0),  # Location of the joint in the tibfib frame
        rot_tib_r.convertRotationToBodyFixedXYZ(),  # Orientation of the joint in the tibfib frame
        right_talus,  # Child body (talus)
        osim.Vec3(0, 0, 0),  # Manually adjusted location of the joint in the talus frame
        osim.Vec3(0.175895, -0.105208, 0.0186622))

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


def estimate_body_segment_parameters(weight, age, sex, segment_lengths, segment_centres, joint_centres):
    """
    Estimates the segment masses and inertial properties of the body based on height and weight.

    Args:
        weight (float): Weight of the participant in kg.
        age (float): age of participant in years
        sex: 1 = Female, 2 = Male
        segment_lengths (dict): lengths of each segment in metres
        segment_centres (dict): location of the segment centres required for com calculation
        joint_centres (dict): the segment joint centres as defined by the osim model

    Returns:
        dict: A dictionary containing segment masses, segment coms, and inertial properties.
    """

    # calculate offset for pelvis and tibfib
    ljc = segment_centres['pelvis']
    asis_mid = joint_centres['pelvis']
    l_knee_c = segment_centres['l_tibfib']
    l_ankle_c = joint_centres['l_tibfib']
    r_knee_c = segment_centres['r_tibfib']
    r_ankle_c = joint_centres['r_tibfib']

    # extract bone lengths
    pel_l = segment_lengths['pelvis']
    l_fem_l = segment_lengths['l_femur']
    r_fem_l = segment_lengths['r_femur']
    l_tib_l = segment_lengths['l_tibfib']
    r_tib_l = segment_lengths['r_tibfib']

    if age < 14:
        # use coefficients for children (Lahkar et al., 2025) ages 3 - 13 years
        if sex == 1:
            # female coefficients
            masses = {
                "pelvis": 0.1562 * weight,  # 15.62% of body mass
                "l_femur": (0.0875 + 0.0036 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "r_femur": (0.0875 + 0.0036 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "l_tibfib": (0.0375 + 0.0011 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "r_tibfib": (0.0375 + 0.0011 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "l_talus": 0.064 * 0.0133 * weight,
                # foot is 1.33% of body mass, need to distribute across talus, calcaneus, and toes
                "r_talus": 0.064 * 0.0133 * weight,
                "l_calcn": 0.796 * 0.0133 * weight,
                "r_calcn": 0.796 * 0.0133 * weight,
                "l_toes": 0.14 * 0.0133 * weight,
                "r_toes": 0.14 * 0.0133 * weight
            }

            segment_coms = {
                "pelvis": (ljc + np.array(
                    [0.0209 * pel_l, (-0.6194 + -0.0154 * age) * pel_l, 0.0029 * pel_l])) - asis_mid,
                "l_femur": np.array([(-0.0694 + 0.0024 * age) * l_fem_l, -0.4454 * l_fem_l, -0.0157 * l_fem_l]),
                "r_femur": np.array([(-0.0694 + 0.0024 * age) * r_fem_l, -0.4454 * r_fem_l, 0.0157 * r_fem_l]),
                "l_tibfib": (l_knee_c + np.array([-0.0293 * l_tib_l, (-0.4358 + 0.0022 * age) * l_tib_l,
                                                  -(0.0436 + -0.001 * age) * l_tib_l])) - l_ankle_c,
                "r_tibfib": (r_knee_c + np.array([-0.0293 * r_tib_l, (-0.4358 + 0.0022 * age) * r_tib_l,
                                                  (0.0436 + -0.001 * age) * r_tib_l])) - r_ankle_c
            }

            segment_radii_percentages = {
                "pelvis": [0.9116, 0.9453, 0.9193],
                "l_femur": [0.2926 + -0.0014 * age, 0.1672 + - 0.0032 * age, 0.3004 + -0.0016 * age],
                "r_femur": [0.2926 + -0.0014 * age, 0.1672 + - 0.0032 * age, 0.3004 + -0.0016 * age],
                "l_tibfib": [0.2981 + -0.0009 * age, 0.1268 + -0.0028 * age, 0.2992 + -0.001 * age],
                "r_tibfib": [0.2981 + -0.0009 * age, 0.1268 + -0.0028 * age, 0.2992 + -0.001 * age]
            }
        elif sex == 2:
            # male coefficients
            masses = {
                "pelvis": 0.1515 * weight,  # 15.15% of body mass
                "l_femur": (0.0779 + 0.0041 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "r_femur": (0.0779 + 0.0041 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "l_tibfib": (0.0376 + 0.0011 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "r_tibfib": (0.0376 + 0.0011 * age) * weight,  # age dependent, where percentage = a0 + a1 * age
                "l_talus": 0.064 * 0.0144 * weight,
                # foot is 1.44% of body mass, need to distribute across talus, calcaneus, and toes
                "r_talus": 0.064 * 0.0144 * weight,
                "l_calcn": 0.796 * 0.0144 * weight,
                "r_calcn": 0.796 * 0.0144 * weight,
                "l_toes": 0.14 * 0.0144 * weight,
                "r_toes": 0.14 * 0.0144 * weight
            }

            segment_coms = {
                "pelvis": (ljc + np.array(
                    [-0.0128 * pel_l, (-0.5079 + -0.0126 * age) * pel_l, -0.0046 * pel_l])) - asis_mid,
                "l_femur": np.array([(-0.0843 + 0.0027 * age) * l_fem_l, -0.4446 * l_fem_l, -0.0184 * l_fem_l]),
                "r_femur": np.array([(-0.0843 + 0.0027 * age) * r_fem_l, -0.4446 * r_fem_l, 0.0184 * r_fem_l]),
                "l_tibfib": (l_knee_c + np.array([-0.0267 * l_tib_l, (-0.4397 + 0.0023 * age) * l_tib_l,
                                                  -(0.0462 + -0.0011 * age) * l_tib_l])) - l_ankle_c,
                "r_tibfib": (r_knee_c + np.array([-0.0267 * r_tib_l, (-0.4397 + 0.0023 * age) * r_tib_l,
                                                  (0.0462 + -0.0011 * age) * r_tib_l])) - r_ankle_c
            }

            segment_radii_percentages = {
                "pelvis": [0.9673, 0.9903, 0.9787],
                "l_femur": [0.2972 + -0.0015 * age, 0.1626 + - 0.0024 * age, 0.3042 + -0.0016 * age],
                "r_femur": [0.2972 + -0.0015 * age, 0.1626 + - 0.0024 * age, 0.3042 + -0.0016 * age],
                "l_tibfib": [0.3020 + -0.0011 * age, 0.1222 + -0.0022 * age, 0.3018 + -0.0011 * age],
                "r_tibfib": [0.3020 + -0.0011 * age, 0.1222 + -0.0022 * age, 0.3018 + -0.0011 * age]
            }
    elif age > 13:
        # use coefficients for adults (Dumas et al., 2018, 2007)
        if sex == 1:
            # female coefficients
            masses = {
                "pelvis": 0.147 * weight,  # 15.62% of body mass
                "l_femur": 0.146 * weight,
                "r_femur": 0.146 * weight,
                "l_tibfib": 0.045 * weight,
                "r_tibfib": 0.045 * weight,
                "l_talus": 0.064 * 0.01 * weight,
                # foot is 1% of body mass, need to distribute across talus, calcaneus, and toes
                "r_talus": 0.064 * 0.01 * weight,
                "l_calcn": 0.796 * 0.01 * weight,
                "r_calcn": 0.796 * 0.01 * weight,
                "l_toes": 0.14 * 0.01 * weight,
                "r_toes": 0.14 * 0.01 * weight

            }

            segment_coms = {
                "pelvis": (ljc + np.array([-0.072 * pel_l, -0.228 * pel_l, 0.002 * pel_l])) - asis_mid,
                "l_femur": np.array([-0.077 * l_fem_l, -0.377 * l_fem_l, -0.008 * l_fem_l]),
                "r_femur": np.array([-0.077 * r_fem_l, -0.377 * r_fem_l, 0.008 * r_fem_l]),
                "l_tibfib": (l_knee_c + np.array([-0.049 * l_tib_l, -0.404 * l_tib_l, -0.031 * l_tib_l])) - l_ankle_c,
                "r_tibfib": (r_knee_c + np.array([-0.049 * r_tib_l, -0.404 * r_tib_l, 0.031 * r_tib_l])) - r_ankle_c
            }

            segment_radii_percentages = {
                "pelvis": [0.95, 1.05, 0.82],
                "l_femur": [0.31, 0.19, 0.32],
                "r_femur": [0.31, 0.19, 0.32],
                "l_tibfib": [0.28, 0.1, 0.28],
                "r_tibfib": [0.28, 0.1, 0.28]
            }
        elif sex == 2:
            # male coefficients
            masses = {
                "pelvis": 0.142 * weight,
                "l_femur": 0.123 * weight,
                "r_femur": 0.123 * weight,
                "l_tibfib": 0.048 * weight,
                "r_tibfib": 0.048 * weight,
                "l_talus": 0.064 * 0.012 * weight,
                # foot is 1% of body mass, need to distribute across talus, calcaneus, and toes
                "r_talus": 0.064 * 0.012 * weight,
                "l_calcn": 0.796 * 0.012 * weight,
                "r_calcn": 0.796 * 0.012 * weight,
                "l_toes": 0.14 * 0.012 * weight,
                "r_toes": 0.14 * 0.012 * weight
            }

            segment_coms = {
                "pelvis": ljc + np.array([-0.002 * pel_l, -0.282 * pel_l, -0.006 * pel_l]) - asis_mid,
                "l_femur": np.array([-0.041 * l_fem_l, -0.429 * l_fem_l, -0.033 * l_fem_l]),
                "r_femur": np.array([-0.041 * r_fem_l, -0.429 * r_fem_l, 0.033 * r_fem_l]),
                "l_tibfib": (l_knee_c + np.array([-0.048 * l_tib_l, -0.41 * l_tib_l, -0.007 * l_tib_l])) - l_ankle_c,
                "r_tibfib": (r_knee_c + np.array([-0.048 * r_tib_l, -0.41 * r_tib_l, 0.007 * r_tib_l])) - r_ankle_c
            }

            segment_radii_percentages = {
                "pelvis": [1.02, 1.06, 0.96],
                "l_femur": [0.29, 0.15, 0.3],
                "r_femur": [0.29, 0.15, 0.3],
                "l_tibfib": [0.28, 0.1, 0.28],
                "r_tibfib": [0.28, 0.1, 0.28]
            }

    # Compute inertia using radius of gyration
    inertias = {}

    def compute_principal_inertia(m, L, r_percent):
        r = np.array(r_percent)
        rg_m = r * L  # radii in metres
        I = m * (rg_m ** 2)  # gives [Ixx, Iyy, Izz]
        return np.append(I, [0, 0, 0])  # array [Ixx, Iyy, Izz]

    for segment in segment_radii_percentages.keys():
        inertias[segment] = compute_principal_inertia(masses[segment], segment_coms[segment],
                                                      segment_radii_percentages[segment])

    return {
        "masses": masses,
        "inertias": inertias,
        "coms": segment_coms
    }


def perform_updates(empty_model, output_folder, mesh_directory, model_name, weight, x_opt_left, x_opt_right,
                    age, sex, segment_lengths, segment_centres, joint_centres):
    """
    Performs a series of updates on an OpenSim model including setting joint ranges, default values,
    renaming coordinates, updating body segment properties, modifying joint rotation axes,
    and ensuring proper subtalar joint configuration.

    Args:
        empty_model (osim.Model): The OpenSim model to be updated.
        output_folder (str): Path to the directory where the updated model will be saved.
        mesh_directory (str): Path to the directory containing the mesh files.
        model_name (str): Name of the model, used for output file naming.
        weight (float): Participant's weight in kilograms.
        height (float): Participant's height in meters.
        x_opt_right (dict): default values for joint orientations
        x_opt_left (dict): default values for joint orientations

    Returns:
        str: The path to the final updated .osim model file.

    Steps:
        1. Load and configure the initial OpenSim model.
        2. Set joint coordinate names, default values, and ranges for the pelvis, hip, knee, and ankle joints.
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
    l_hip_joint = model.getJointSet().get('hip_l')
    r_hip_joint = model.getJointSet().get('hip_r')

    # Locate knee joints
    l_knee_joint = model.getJointSet().get('knee_l')
    r_knee_joint = model.getJointSet().get('knee_r')

    # Locate Ankle joints
    l_ankle_joint = model.getJointSet().get('ankle_l')
    r_ankle_joint = model.getJointSet().get('ankle_r')

    # Locate pelvis joint
    pelvis_joint = model.getJointSet().get('pelvis_to_ground')

    pelvis_rotation = pelvis_joint.upd_coordinates(0)
    pelvis_list = pelvis_joint.upd_coordinates(1)
    pelvis_tilt = pelvis_joint.upd_coordinates(2)
    pelvis_tx = pelvis_joint.upd_coordinates(3)
    pelvis_ty = pelvis_joint.upd_coordinates(4)
    pelvis_tz = pelvis_joint.upd_coordinates(5)

    # rename pelvis rotations and set default values from x_opt as an average value from the left and right side
    pelvis_rotation.setRangeMin(-6.2831853071795862)
    pelvis_rotation.setRangeMax(6.2831853071795862)
    pelvis_rotation.setDefaultValue(((x_opt_left['pelvis_rigid'][4] + x_opt_right['pelvis_rigid'][4]) / 2))
    pelvis_rotation.setDefaultClamped(True)
    pelvis_rotation.setDefaultLocked(False)

    pelvis_list.setRangeMin(-1.5707963300000001)
    pelvis_list.setRangeMax(1.5707963300000001)
    pelvis_list.setDefaultValue(((x_opt_left['pelvis_rigid'][3] + x_opt_right['pelvis_rigid'][3]) / 2))
    pelvis_list.setDefaultClamped(True)
    pelvis_list.setDefaultLocked(False)

    pelvis_tilt.setRangeMin(-1.5707963300000001)
    pelvis_tilt.setRangeMax(1.5707963300000001)
    pelvis_tilt.setDefaultValue(((x_opt_left['pelvis_rigid'][5] + x_opt_right['pelvis_rigid'][5]) / 2))
    pelvis_tilt.setDefaultClamped(True)
    pelvis_tilt.setDefaultLocked(False)

    pelvis_tx.setRangeMin(-50)
    pelvis_tx.setRangeMax(50)
    pelvis_tx.setDefaultValue(0.0)
    pelvis_tx.setDefaultClamped(True)
    pelvis_tx.setDefaultLocked(False)

    pelvis_ty.setRangeMin(-50)
    pelvis_ty.setRangeMax(50)
    pelvis_ty.setDefaultValue(0.0)
    pelvis_ty.setDefaultClamped(True)
    pelvis_ty.setDefaultLocked(False)

    pelvis_tz.setRangeMin(-3)
    pelvis_tz.setRangeMax(3)
    pelvis_tz.setDefaultValue(0.0)
    pelvis_tz.setDefaultClamped(True)
    pelvis_tz.setDefaultLocked(False)

    # Set coordinates ranges and default values for left hip joint
    l_hip_flexion = l_hip_joint.upd_coordinates(0)
    l_hip_abduction = l_hip_joint.upd_coordinates(1)
    l_hip_rotation = l_hip_joint.upd_coordinates(2)

    l_hip_flexion.setRangeMin(-1.5)
    l_hip_flexion.setRangeMax(1.8)
    l_hip_flexion.setDefaultValue(x_opt_left['hip_rot'][0])
    l_hip_flexion.setDefaultClamped(True)
    l_hip_flexion.setDefaultLocked(False)

    l_hip_rotation.setRangeMin(-0.8)
    l_hip_rotation.setRangeMax(0.8)
    l_hip_rotation.setDefaultValue(x_opt_left['hip_rot'][1] * -0.1)
    l_hip_rotation.setDefaultClamped(True)
    l_hip_rotation.setDefaultLocked(False)

    l_hip_abduction.setRangeMin(-0.8)
    l_hip_abduction.setRangeMax(1.2)
    l_hip_abduction.setDefaultValue(x_opt_left['hip_rot'][2] * -0.1)
    l_hip_abduction.setDefaultClamped(True)
    l_hip_abduction.setDefaultLocked(False)

    # Set coordinates ranges and default values for right hip joint
    r_hip_flexion = r_hip_joint.upd_coordinates(0)
    r_hip_abduction = r_hip_joint.upd_coordinates(1)
    r_hip_rotation = r_hip_joint.upd_coordinates(2)

    r_hip_flexion.setRangeMin(-1.5)
    r_hip_flexion.setRangeMax(1.8)
    r_hip_flexion.setDefaultValue(x_opt_right['hip_rot'][0])
    r_hip_flexion.setDefaultClamped(True)
    r_hip_flexion.setDefaultLocked(False)

    r_hip_rotation.setRangeMin(-0.8)
    r_hip_rotation.setRangeMax(0.8)
    r_hip_rotation.setDefaultValue(x_opt_right['hip_rot'][1] * 0.1)
    r_hip_rotation.setDefaultClamped(True)
    r_hip_rotation.setDefaultLocked(False)

    r_hip_abduction.setRangeMin(-1.2)
    r_hip_abduction.setRangeMax(0.8)
    r_hip_abduction.setDefaultValue(x_opt_right['hip_rot'][2] * 0.1)
    r_hip_abduction.setDefaultClamped(True)
    r_hip_abduction.setDefaultLocked(False)

    # Set coordinates ranges, default values, and names for left knee joint
    l_knee_flexion = l_knee_joint.upd_coordinates(0)
    l_knee_flexion.setName("knee_flexion_l")
    l_knee_flexion.setRangeMin(-0.2)
    l_knee_flexion.setRangeMax(2.2)
    l_knee_flexion.setDefaultValue(x_opt_left['knee_rot'][0])
    l_knee_flexion.setDefaultClamped(True)
    l_knee_flexion.setDefaultLocked(False)

    l_knee_add = l_knee_joint.upd_coordinates(1)
    l_knee_add.setName("knee_adduction_l")
    l_knee_add.setRangeMin(x_opt_left['knee_rot'][1] * -0.1)
    l_knee_add.setRangeMax(x_opt_left['knee_rot'][1] * -0.1)
    l_knee_add.setDefaultValue(x_opt_left['knee_rot'][1] * -0.1)
    l_knee_add.setDefaultLocked(True)  # lock add/abduction in the knee joint
    l_knee_add.setDefaultClamped(True)

    l_knee_rot = l_knee_joint.upd_coordinates(2)
    l_knee_rot.setName("knee_rotation_l")
    l_knee_rot.setRangeMin(0.0)
    l_knee_rot.setRangeMax(0.0)
    l_knee_rot.setDefaultValue(0.0)
    l_knee_rot.setDefaultLocked(True)  # lock i/e rotation in the knee joint
    l_knee_rot.setDefaultClamped(True)

    # Set coordinates ranges, default values, and names for right knee joint
    r_knee_flexion = r_knee_joint.upd_coordinates(0)
    r_knee_flexion.setName("knee_flexion_r")
    r_knee_flexion.setRangeMin(-0.2)
    r_knee_flexion.setRangeMax(2.2)
    r_knee_flexion.setDefaultValue(x_opt_right['knee_rot'][0])
    r_knee_flexion.setDefaultClamped(True)
    r_knee_flexion.setDefaultLocked(False)

    r_knee_add = r_knee_joint.upd_coordinates(1)
    r_knee_add.setName("knee_adduction_r")
    r_knee_add.setRangeMin(x_opt_right['knee_rot'][1] * 0.1)
    r_knee_add.setRangeMax(x_opt_right['knee_rot'][1] * 0.1)
    r_knee_add.setDefaultValue(x_opt_right['knee_rot'][1] * 0.1)
    r_knee_add.setDefaultLocked(True)  # lock add/abduction in the knee joint
    r_knee_add.setDefaultClamped(True)

    r_knee_rot = r_knee_joint.upd_coordinates(2)
    r_knee_rot.setName("knee_rotation_r")
    r_knee_rot.setRangeMin(0.0)
    r_knee_rot.setRangeMax(0.0)
    r_knee_rot.setDefaultValue(0.0)
    r_knee_rot.setDefaultLocked(True)  # lock i/e rotation in the knee joint
    r_knee_rot.setDefaultClamped(True)

    # Set coordinates range and names for right ankle joint
    r_ankle_flexion = r_ankle_joint.upd_coordinates(0)
    r_ankle_flexion.setName("ankle_angle_r")
    r_ankle_flexion.setRangeMin(-1.5707963267948966)
    r_ankle_flexion.setRangeMax(0.87266462599716477)
    r_ankle_flexion.setDefaultValue(0.0)
    r_ankle_flexion.setDefaultClamped(True)
    r_ankle_flexion.setDefaultLocked(False)
    # Set coordinates range and names for right ankle joint
    l_ankle_flexion = l_ankle_joint.upd_coordinates(0)
    l_ankle_flexion.setName("ankle_angle_l")
    l_ankle_flexion.setRangeMin(-1.5707963267948966)
    l_ankle_flexion.setRangeMax(0.87266462599716477)
    l_ankle_flexion.setDefaultValue(0.0)
    l_ankle_flexion.setDefaultClamped(True)
    l_ankle_flexion.setDefaultLocked(False)

    # Locate body segments based on printed names
    pelvis = model.getBodySet().get('pelvis_b')
    femur_l = model.getBodySet().get('femur_l')
    femur_r = model.getBodySet().get('femur_r')
    tibfib_l = model.getBodySet().get('tibfib_l')
    tibfib_r = model.getBodySet().get('tibfib_r')
    calcn_l = model.getBodySet().get('calcn_l')
    calcn_r = model.getBodySet().get('calcn_r')
    talus_l = model.getBodySet().get('talus_l')
    talus_r = model.getBodySet().get('talus_r')
    toes_l = model.getBodySet().get('toes_l')
    toes_r = model.getBodySet().get('toes_r')

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
    params = estimate_body_segment_parameters(weight, age, sex, segment_lengths, segment_centres, joint_centres)
    masses = params["masses"]
    inertias = params["inertias"]
    coms = params["coms"]

    # Apply mass, center of mass, and inertia
    set_mass_com_inertia(pelvis, masses["pelvis"], coms["pelvis"], inertias["pelvis"])
    set_mass_com_inertia(femur_l, masses["l_femur"], coms["l_femur"], inertias["l_femur"])
    set_mass_com_inertia(femur_r, masses["r_femur"], coms["r_femur"], inertias["r_femur"])
    set_mass_com_inertia(tibfib_l, masses["l_tibfib"], coms["l_tibfib"], inertias["l_tibfib"])
    set_mass_com_inertia(tibfib_r, masses["r_tibfib"], coms["r_tibfib"], inertias["r_tibfib"])

    # update masses of the foot bones
    calcn_l.setMass(masses['l_calcn'])
    calcn_r.setMass(masses['r_calcn'])
    talus_l.setMass(masses['l_talus'])
    talus_r.setMass(masses['r_talus'])
    toes_l.setMass(masses['l_toes'])
    toes_r.setMass(masses['r_toes'])

    # Finalise the initial iteration of model
    model.finalizeConnections()
    model.printToXML(output_file)

    input_file = output_file
    # updates the path to feet mesh files
    update_mesh_file_paths(input_file, output_file, mesh_directory,
                           ["l_bofoot.vtp", "r_bofoot.vtp", "l_foot.vtp", "r_foot.vtp", "l_talus.vtp", "r_talus.vtp"])

    return output_file


def feet_adjustments(empty_model, mocap_static_trc, realign_feet=False, left_foot_flat=False, right_foot_flat=False):
    """
    Adjusts the orientation of the left and right feet in an OpenSim model to align with mocap (motion capture) data.

    The function computes the appropriate rotations for both feet to ensure they are aligned with the ground
    and match the positions indicated by the mocap static trial. This is particularly useful for preparing the
    model for inverse kinematics or other biomechanical analyses.

    Args:
        empty_model (osim.Model): The OpenSim model with foot components to be aligned.
        mocap_static_trc (dict): Dictionary containing motion capture marker data with marker names as keys and
            (x, y, z) coordinates as values.
        realign_feet (bool, optional): Whether to fully realign the feet, around y and z axes. If align_foot_to_ground
            selected, this will override the z axes rotation. If False, no adjustments are made and the ankle remains at
            the default 0.0 angle defined by the alignment of the talus and tibfib coordinate frames. Default is False.
        left_foot_flat (bool, optional): Whether to align the left foot to be parallel with the ground. Uses the foot
            bone geometry to align. This only modifies rotation about the z axis and sets a default angle for the ankle.
        right_foot_flat (bool, optional): Whether to align the right foot to be parallel with the ground. Uses the foot
            bone geometry to align. This only modifies rotation about the z axis and sets a default angle for the ankle.

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
        This function primarily handles the left and right feet independently and uses Euler angles for rotation
        adjustments. It focuses on aligning the feet both forward-facing and flat to the ground.
    """
    # Initialize the model's system
    state = empty_model.initSystem()
    if realign_feet:
        # === Adjust Orientation of the Left Foot ===

        # --- Get marker positions in GROUND ---
        toe_marker = empty_model.getMarkerSet().get("LTOE")
        heel_marker = empty_model.getMarkerSet().get("LHEE")

        toe_ground = vec3_to_numpy(toe_marker.getLocationInGround(state))
        heel_ground = vec3_to_numpy(heel_marker.getLocationInGround(state))

        # Model foot vector (ground)
        model_vec = toe_ground - heel_ground
        model_vec /= np.linalg.norm(model_vec)

        # Mocap foot vector (already in ground)
        mocap_vec = mocap_static_trc["LTOE"] - mocap_static_trc["LHEE"]
        mocap_vec /= np.linalg.norm(mocap_vec)

        # Z rotation (transverse plane / yaw)
        theta_z = np.arctan2(
            model_vec[0] * mocap_vec[1] - model_vec[1] * mocap_vec[0],
            model_vec[0] * mocap_vec[0] + model_vec[1] * mocap_vec[1]
        )

        # Apply Z rotation to model vector
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        model_vec_rot = Rz @ model_vec

        # Y rotation (internal/external)
        theta_y = np.arctan2(model_vec_rot[2], model_vec_rot[0]) - \
                  np.arctan2(mocap_vec[2], mocap_vec[0])

        #APPLY TO OPENSIM
        left_ankle_joint = empty_model.getJointSet().get("ankle_l")

        # --- Z → ankle flexion default ---
        if not left_foot_flat:
            l_ankle_flexion = left_ankle_joint.upd_coordinates(0)
            l_ankle_flexion.setDefaultValue(theta_z)

        # --- Update child frame orientation (Y only) ---
        child_frame = left_ankle_joint.upd_frames(1)
        current_orient = child_frame.get_orientation()

        current = np.array([
            current_orient.get(0),
            current_orient.get(1),
            current_orient.get(2)
        ])

        # Only adjust Y
        new_orient = current.copy()
        new_orient[1] -= theta_y

        child_frame.set_orientation(osim.Vec3(*new_orient))

        # === Adjust Orientation of the Right Foot ===

        # --- Get marker positions in GROUND ---
        toe_marker = empty_model.getMarkerSet().get("RTOE")
        heel_marker = empty_model.getMarkerSet().get("RHEE")

        toe_ground = vec3_to_numpy(toe_marker.getLocationInGround(state))
        heel_ground = vec3_to_numpy(heel_marker.getLocationInGround(state))

        # Model foot vector (ground)
        model_vec = toe_ground - heel_ground
        model_vec /= np.linalg.norm(model_vec)

        # Mocap foot vector (already in ground)
        mocap_vec = mocap_static_trc["RTOE"] - mocap_static_trc["RHEE"]
        mocap_vec /= np.linalg.norm(mocap_vec)

        # Z rotation (transverse plane / yaw)
        theta_z = np.arctan2(
            model_vec[0] * mocap_vec[1] - model_vec[1] * mocap_vec[0],
            model_vec[0] * mocap_vec[0] + model_vec[1] * mocap_vec[1]
        )

        # Apply Z rotation to model vector
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        model_vec_rot = Rz @ model_vec

        # Y rotation (internal/external)
        theta_y = np.arctan2(model_vec_rot[2], model_vec_rot[0]) - \
                  np.arctan2(mocap_vec[2], mocap_vec[0])

        # APPLY TO OPENSIM
        right_ankle_joint = empty_model.getJointSet().get("ankle_r")

        # --- Z → ankle flexion default ---
        if not right_foot_flat:
            r_ankle_flexion = right_ankle_joint.upd_coordinates(0)
            r_ankle_flexion.setDefaultValue(theta_z)

        # --- Update child frame orientation (Y only) ---
        child_frame = right_ankle_joint.upd_frames(1)
        current_orient = child_frame.get_orientation()

        current = np.array([
            current_orient.get(0),
            current_orient.get(1),
            current_orient.get(2)
        ])

        # Only adjust Y
        new_orient = current.copy()
        new_orient[1] -= theta_y

        child_frame.set_orientation(osim.Vec3(*new_orient))

    state = empty_model.initSystem()
    if left_foot_flat:
        ## left side ##
        toes_body_l = empty_model.getBodySet().get("toes_l")
        left_foot_transform_in_ground = toes_body_l.getTransformInGround(state)
        # Extract the rotation matrix from the transform
        rotation_matrix = left_foot_transform_in_ground.R().asMat33()
        # Convert the rotation matrix to Euler angles
        rotation = osim.Rotation(rotation_matrix)
        euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

        # Set unnecessary rotations (x and y axes) to zero
        euler_angles[0] = 0
        euler_angles[1] = 0

        # assign rotation about z as a default angle
        default_angle_l = -euler_angles[2]
        left_ankle_joint = empty_model.getJointSet().get("ankle_l")
        l_ankle_flexion = left_ankle_joint.upd_coordinates(0)
        l_ankle_flexion.setDefaultValue(default_angle_l)

    if right_foot_flat:
        ## right side ##
        toes_body_r = empty_model.getBodySet().get("toes_r")
        right_foot_transform_in_ground = toes_body_r.getTransformInGround(state)
        # Extract the rotation matrix from the transform
        rotation_matrix = right_foot_transform_in_ground.R().asMat33()
        # Convert the rotation matrix to Euler angles
        rotation = osim.Rotation(rotation_matrix)
        euler_angles = rotation.convertRotationToBodyFixedXYZ()  # Angles in radians

        # Set unnecessary rotations (x and y axes) to zero
        euler_angles[0] = 0
        euler_angles[1] = 0

        # assign rotation about z as default angle
        default_angle_r = -euler_angles[2]
        right_ankle_joint = empty_model.getJointSet().get("ankle_r")
        r_ankle_flexion = right_ankle_joint.upd_coordinates(0)
        r_ankle_flexion.setDefaultValue(default_angle_r)

    move_marker_to_body(empty_model, state, "RTOE_0", "toes_r")
    move_marker_to_body(empty_model, state, "RHEE_0", "calcn_r")
    move_marker_to_body(empty_model, state, "LTOE_0", "toes_l")
    move_marker_to_body(empty_model, state, "LHEE_0", "calcn_l")

    # set model markers from static trial
    marker_set = empty_model.getMarkerSet()
    marker_set.get("RTOE").setName("RTOE_model")
    marker_set.get("RHEE").setName("RHEE_model")
    marker_set.get("LTOE").setName("LTOE_model")
    marker_set.get("LHEE").setName("LHEE_model")

    marker_set.get("RTOE_0").setName("RTOE")
    marker_set.get("RHEE_0").setName("RHEE")
    marker_set.get("LTOE_0").setName("LTOE")
    marker_set.get("LHEE_0").setName("LHEE")


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

    # scale_tool.getMarkerPlacer().setApply(True)
    # scale_tool.getMarkerPlacer().setOutputModelFileName("Lower_Limb.osim")
    # scale_tool.getMarkerPlacer().setMarkerFileName(relative_path)
    # scale_tool.getMarkerPlacer().setTimeRange(time_range)

    scale_tool.getModelScaler().setOutputModelFileName("Feet_scaled.osim")
    scale_tool.getModelScaler().setMarkerFileName(relative_path)
    scale_tool.getModelScaler().setTimeRange(time_range)

    scaled_output_file = os.path.join(output_directory, "scaling_tool_settings.xml")

    # Verify the loaded scaling settings (optional)
    scale_tool.printToXML(scaled_output_file)  # Outputs a copy of the loaded settings

    # Run the scaling process
    scale_tool.run()


def vec3_to_numpy(v):
    return np.array([v.get(i) for i in range(3)])


def numpy_to_vec3(a):
    return osim.Vec3(a[0], a[1], a[2])


def rot_to_numpy(R):
    return np.array([[R.get(i, j) for j in range(3)] for i in range(3)])


def move_marker_to_body(model, state, marker_name, new_body_name):
    state = model.initSystem()
    model.realizePosition(state)

    marker = model.getMarkerSet().get(marker_name)

    old_body = marker.getParentFrame()
    new_body = model.getBodySet().get(new_body_name)

    # marker location in old body frame
    p_old = vec3_to_numpy(marker.get_location())

    # transforms
    T_old = old_body.getTransformInGround(state)
    T_new = new_body.getTransformInGround(state)

    R_old = rot_to_numpy(T_old.R())
    p_old_body = vec3_to_numpy(T_old.p())

    R_new = rot_to_numpy(T_new.R())
    p_new_body = vec3_to_numpy(T_new.p())

    # old body → ground
    p_ground = R_old @ p_old + p_old_body

    # ground → new body
    p_new = R_new.T @ (p_ground - p_new_body)

    # update marker
    marker.setParentFrame(new_body)
    marker.set_location(numpy_to_vec3(p_new))

    return p_new


def create_osim_rot(x_axis, y_axis, z_axis, inverse=False):
    # Stack axes as columns
    R = np.column_stack((x_axis, y_axis, z_axis))  # shape (3,3)

    if inverse:
        R = R.T  # transpose = inverse for rotation matrices

    # Convert to osim.Mat33
    rot = osim.Mat33()
    for i in range(3):
        for j in range(3):
            rot.set(i, j, float(R[i, j]))

    return rot


def create_osim_rot_bodies(x1, y1, z1, x2, y2, z2, inverse=False):
    # Build coordinate system matrices
    R1 = np.column_stack((x1, y1, z1))
    R2 = np.column_stack((x2, y2, z2))

    # Rotation from frame1 → frame2
    R = R2 @ R1.T

    if inverse:
        R = R.T  # transpose = inverse for rotation matrices

    # Convert to OpenSim Mat33
    rot = osim.Mat33()
    for i in range(3):
        for j in range(3):
            rot.set(i, j, float(R[i, j]))

    return rot
