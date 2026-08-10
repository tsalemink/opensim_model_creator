import os
import numpy as np
import opensim as osim

from articulated_ssm_both_sides.MainASM import run_asm

from opensim_model_creator.Functions.general_utils import *
from opensim_model_creator.Functions.bone_utils import *
from opensim_model_creator.Functions.file_utils import clear_directory

root_directory = os.path.dirname(os.path.abspath(__file__))
high_level_inputs = os.path.join(root_directory, "High_Level_Inputs")


def create_model(static_trc, dynamic_trc, output_directory, static_marker_data, subject_info, marker_radius,
                 left_foot_flat, right_foot_flat, toe_marker_proximal=False, optimise_knee_axis=True,
                 progress_tracker=None):
    """
    Creates an OpenSim model for the specified TRC inputs.

    Args:
        static_trc (str): Path to the static TRC file.
        dynamic_trc (str): Path to the dynamic TRC file.
        output_directory (str): Path to the directory where the models should be produced.
        static_marker_data (dict): Static marker data coordinates.
        subject_info (DataFrame): Subject measurements and demographic information.
        marker_radius (float): Radius of motion capture markers.
        left_foot_flat (bool): Whether the left foot is flat on the ground in the static trial.
        right_foot_flat (bool): Whether the right foot is flat on the ground in the static trial.
        toe_marker_proximal (bool): Whether the lab places their toe marker proximally, between the second and third
            metatarsal heads. Set as False for a more distal placement.
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

    log_progress(progress_tracker, "Fitting articulated shape model")

    # Generate mesh files using ASM.
    run_asm(static_marker_data, mesh_directory, subject_info, marker_radius)

    log_progress(progress_tracker, "Creating OpenSim model")

    height = subject_info['Height'].iloc[0] / 100
    weight = subject_info['Mass'].iloc[0]
    age = subject_info['Age'].iloc[0]
    sex = subject_info['Sex'].iloc[0]

    # Scale marker data from millimeters to meters.
    scale_marker_data(static_marker_data, 0.001)

    # Copy and process mesh files.
    copy_mesh_files(high_level_inputs, mesh_directory)
    process_participant_meshes(mesh_directory, mesh_directory)
    # scale feet
    scale_foot = os.path.join(high_level_inputs, "Feet.osim")
    perform_scaling(model_directory, scale_foot, static_trc)
    empty_model, _, left_landmarks, right_landmarks, x_opt_left, x_opt_right = initialize_model_and_extract_landmarks(
        mesh_directory, model_directory)

    segment_lengths, segment_centres, joint_centres = create_model_bodies(
        mesh_directory, static_marker_data, empty_model, left_landmarks, right_landmarks,
        x_opt_left, x_opt_right, sex)
    #for checking first model
    #output_file = model_directory + "/"f"model_1.osim"
    #empty_model.printToXML(output_file)

    # Create initial OpenSim model.
    model_name = "Lower_limb"
    empty_model.setName(model_name)
    output_file = perform_updates(empty_model, model_directory, mesh_directory, model_name,
                                  weight, x_opt_left, x_opt_right, age, sex, segment_lengths, segment_centres, joint_centres)

    # Adjust foot bone orientation and scaling.
    empty_model = osim.Model(output_file)

    # TODO: Remove. Debugging.
    check_marker_alignment(empty_model, static_marker_data)

    feet_adjustments(empty_model, static_marker_data, left_foot_flat=left_foot_flat,
                     right_foot_flat=right_foot_flat, toe_marker_proximal=toe_marker_proximal)
    empty_model.finalizeConnections()
    empty_model.printToXML(output_file)

    landmark_files = ["original_lms_left.txt", "original_lms_right.txt",
                      "predicted_lms_left.txt", "predicted_lms_right.txt"]
    scale_landmark_files(mesh_directory, landmark_files)

    #remove the scaled foot model from directory
    feet_scaled_model = os.path.join(model_directory, "Feet_scaled.osim")
    if os.path.isfile(feet_scaled_model):
        os.remove(feet_scaled_model)

    model_path = os.path.join(model_directory, "Lower_limb.osim")
    if optimise_knee_axis:
        model_path = optimise_knee_joint(model_path, model_directory, dynamic_trc)

    return model_path


def create_model_bodies(mesh_directory, static_marker_data, empty_model, left_lms, right_lms, x_opt_left, x_opt_right, sex):

    pelvis, pelvis_centre, pelvis_length, lumbar_joint_centre, body_axes, pelvis_rot = create_pelvis_body_and_joint(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data, sex)

    left_femur, femur_l_centre, right_femur, femur_r_centre, l_femur_length, r_femur_length, body_axes = create_femur_bodies_and_hip_joints(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data, pelvis, pelvis_centre, body_axes, pelvis_rot)

    tibfib_l_centre, tibfib_r_centre, left_tibfib, right_tibfib, l_tibfib_length, r_tibfib_length, l_EC_midpoint, r_EC_midpoint = create_tibfib_bodies_and_knee_joints(
        empty_model, left_lms, right_lms, mesh_directory, static_marker_data, left_femur, right_femur, femur_l_centre,
        femur_r_centre, body_axes)

    repurpose_feet_bodies_and_create_joints(empty_model, left_tibfib, right_tibfib, body_axes)

    empty_model.finalizeConnections()

    segment_lengths = {
        'pelvis': pelvis_length,
        'l_femur': l_femur_length,
        'r_femur': r_femur_length,
        'l_tibfib': l_tibfib_length,
        'r_tibfib': r_tibfib_length
    }

    segment_centres = {
        'pelvis': lumbar_joint_centre,
        'l_tibfib': l_EC_midpoint,
        'r_tibfib': r_EC_midpoint
    }

    joint_centres = {
        'pelvis': pelvis_centre,
        'l_tibfib': tibfib_l_centre,
        'r_tibfib': tibfib_r_centre
    }

    return segment_lengths, segment_centres, joint_centres


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


def log_progress(progress_tracker, message, text_colour="black"):
    if progress_tracker:
        progress_tracker.progress.emit(message, text_colour)
