import os
import numpy as np
import opensim as osim

from articulated_ssm_both_sides.MainASM import run_asm

from opensim_model_creator.Functions.general_utils import *
from opensim_model_creator.Functions.bone_utils import *
from opensim_model_creator.Functions.file_utils import clear_directory

root_directory = os.path.dirname(os.path.abspath(__file__))
high_level_inputs = os.path.join(root_directory, "High_Level_Inputs")


def create_model(static_trc, dynamic_trc, output_directory, static_marker_data, weight, height,
                 optimise_knee_axis=True, progress_tracker=None):
    """
    Creates an OpenSim model for the specified TRC inputs.

    Args:
        static_trc (str): Path to the static TRC file.
        dynamic_trc (str): Path to the dynamic TRC file.
        output_directory (str): Path to the directory where the models should be produced.
        static_marker_data (dict): Static marker data coordinates.
        weight (float): Participant's weight in kg.
        height (float): Participant's height in meters.
        optimise_knee_axis (bool): Set as False to disable knee-axis optimisation.
        progress_tracker (ProgressTracker, optional): Progress-tracker for emitting progress signals.

    Returns:
        None
    """

    # Setup input and output directories.
    model_directory = os.path.join(output_directory, "Models")
    mesh_directory = os.path.join(model_directory, "Meshes")
    clear_directory(model_directory)
    clear_directory(mesh_directory)

    if progress_tracker:
        progress_tracker.progress.emit("Fitting articulated shape model", "black")

    # Generate mesh files using ASM.
    run_asm(static_marker_data, mesh_directory)

    if progress_tracker:
        progress_tracker.progress.emit("Creating OpenSim model", "black")

    # Scale marker data from millimeters to meters.
    scale_marker_data(static_marker_data, 0.001)

    # Move foot mesh files into the meshes directory.
    copy_mesh_files(high_level_inputs, mesh_directory)

    # Process mesh files.
    process_participant_meshes(mesh_directory, mesh_directory)
    empty_model, _, left_landmarks, right_landmarks, x_opt_left, x_opt_right = initialize_model_and_extract_landmarks(
        mesh_directory)

    create_model_bodies(mesh_directory, static_marker_data, empty_model, left_landmarks, right_landmarks,
                        x_opt_left, x_opt_right)

    # Extract the directory name as the model name and replace spaces with underscores
    model_name_pre = "Bone_Model_pre"
    model_name = "Bone_model"

    # Update the model name
    empty_model.setName(model_name_pre)

    # Ensure the output folder exists
    os.makedirs(model_directory, exist_ok=True)

    # Combine the folder path and filename
    output_path = os.path.join(model_directory, f"{model_name_pre}.osim")

    # Save the model to output folder
    empty_model.printToXML(output_path)
    print(f"Model saved to: {output_path}")
    empty_model.setName(model_name)

    # %% Perform a long series of updates to the model
    output_file = perform_updates(empty_model, model_directory, mesh_directory, model_name, weight, height, x_opt_left,
                                  x_opt_right)

    # Reload the model
    empty_model = osim.Model(output_file)

    # %% Reinitialise the model for further feet adjustments (aligning with static trc as gait2392 feet are perfectly straight whilst participants may have their feet angled when neutral)
    feet_adjustments(output_file, empty_model, static_marker_data, realign_feet=True)

    # Finalise the non-scaled foot
    empty_model.finalizeConnections()
    empty_model.printToXML(output_file)

    # %% Look to scale the size of the feet automatically and move the markers to appropriate positions
    # note this runs a preset scale setting file where only the feet are selected to be scaled
    perform_scaling(model_directory, output_file, static_trc)

    model_path = os.path.join(model_directory, "Lower_Limb.osim")
    if optimise_knee_axis:
        model_path = optimise_knee_joint(model_path, model_directory, dynamic_trc)

    return model_path


def create_model_bodies(mesh_directory, static_marker_data, empty_model, left_lms, right_lms, x_opt_left, x_opt_right):
    pelvis, pelvis_center = create_pelvis_body_and_joint(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data)

    left_femur, femur_l_center, right_femur, femur_r_center = create_femur_bodies_and_hip_joints(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data, pelvis,
        x_opt_left['hip_rot'], x_opt_right['hip_rot'])

    tibfib_l_center, tibfib_r_center, left_tibfib, right_tibfib = create_tibfib_bodies_and_knee_joints(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data, left_femur, right_femur,
        x_opt_left['knee_rot'], x_opt_right['knee_rot'])

    repurpose_feet_bodies_and_create_joints(empty_model, tibfib_l_center, tibfib_r_center, left_tibfib, right_tibfib)

    empty_model.finalizeConnections()


def optimise_knee_joint(model_path, model_directory, dynamic_trc):
    _, (start_time, end_time), _ = read_trc_file_as_dict(dynamic_trc, True)

    temp_model_path_1 = model_directory + "/temp1.osim"
    temp_model_path_2 = model_directory + "/temp2.osim"
    optimised_knee_model = model_directory + "/Optimised_Knee_Axes.osim"

    ik_marker_weights = {
        "LASI": 5, "RASI": 5, "LPSI": 1, "RPSI": 1, "LTHI": 1, "RTHI": 1,
        "LKNE": 2.5, "RKNE": 2.5, "LKNEM": 2.5, "RKNEM": 2.5, "LTIB": 1, "RTIB": 1,
        "LANK": 10, "RANK": 10, "LMED": 10, "RMED": 10, "LHEE": 1, "RHEE": 1, "LTOE": 1, "RTOE": 1,
    }

    run_knee_joint_optimisation(model_path, dynamic_trc, start_time, end_time,
                                temp_model_path_1, temp_model_path_2, ik_marker_weights, optimised_knee_model)

    # Delete temporary .osim files.
    for osim_file in [temp_model_path_1, temp_model_path_2]:
        if os.path.isfile(osim_file):
            os.remove(osim_file)

    return optimised_knee_model
