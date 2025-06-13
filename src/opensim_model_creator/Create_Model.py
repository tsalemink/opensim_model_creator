import os
import numpy as np
import opensim as osim

from articulated_ssm_both_sides.MainASM import run_asm

# %%Import functions from folders
from opensim_model_creator.Functions.general_utils import *
from opensim_model_creator.Functions.bone_utils import *
from opensim_model_creator.Functions.muscle_utils import *
from opensim_model_creator.Functions.file_utils import clear_directory

root_directory = os.path.dirname(os.path.abspath(__file__))
high_level_inputs = os.path.join(root_directory, "High_Level_Inputs")


def create_model(static_trc, dynamic_trc, output_directory, static_marker_data, weight, height, create_muscles=False,
                 testing=False,
                 progress_tracker=None):
    """
    Creates an OpenSim model for a given participant, optionally adding muscles.

    Args:
        static_trc (str): Path to the static TRC file.
        dynamic_trc (str): Path to the dynamic TRC file.
        output_directory (str): Path to the directory where the models should be produced.
        static_marker_data (dict): Static marker data coordinates.
        create_muscles (bool): Whether to add muscles to the model.
        testing (bool): If True, runs in test mode - reduces knee optimisation iteration count for computational speed
        weight (float): Participant's weight in kg.
        height (float): Participant's height in meters.
        progress_tracker (ProgressTracker, optional): Progress-tracker for emitting progress signals.

    Returns:
        None
    """

    # %% Setup of folders
    # Define paths for inputs and outputs
    model_directory = os.path.join(output_directory, "Models")
    mesh_directory = os.path.join(model_directory, "Meshes")

    # Clear output and mesh folders to avoid residuals from previous runs
    clear_directory(model_directory)
    clear_directory(mesh_directory)

    # %%Initialisation

    if progress_tracker:
        progress_tracker.progress.emit("Fitting articulated shape model", "black")

    # Generate mesh files using ASM
    run_asm(static_marker_data, mesh_directory)

    if progress_tracker:
        progress_tracker.progress.emit("Creating OpenSim model", "black")

    # Move foot mesh files into the meshes directory.
    copy_mesh_files(high_level_inputs, mesh_directory)

    # Scale marker data from millimeters to meters (variable currently unused)
    scale_marker_data(static_marker_data, 0.001)

    # Process and extract meshes from STL files
    process_participant_meshes(mesh_directory, mesh_directory)

    # MUSCLES (add in later) Initializes muscle linkage directory
    # muscle_linkages = muscle_initialisation(mesh_directory)

    # Splits specific muscles into a number of segments
    # segment_muscle_origins_insertions(muscle_linkages, "Glut med", num_segments=3)
    # segment_muscle_origins_insertions(muscle_linkages, "Glut min", num_segments=3)
    # segment_muscle_origins_insertions(muscle_linkages, "Add mag", pair_to_segment=0, num_segments=2)

    # Apply a swap for Adductor Magnus origins to have better anatomical consistency
    # swap_muscle_attachments(muscle_linkages, "Add mag", 0, 2, attachment_type="ori")

    # Initialises model, trc files, and landmarks
    empty_model, state, left_landmarks, right_landmarks, x_opt_left, x_opt_right = initialize_model_and_extract_landmarks(
        mesh_directory)
    mocap_static_trc = static_marker_data

    # %% Creation of the pelvis body and pelvis joint
    pelvis, pelvis_center = create_pelvis_body_and_joint(
        empty_model, left_landmarks, right_landmarks, mesh_directory, mocap_static_trc)

    # %% Creation of femur bodies and attachment of meshes, markers, and landmarks
    (left_femur, femur_l_center, right_femur, femur_r_center) = create_femur_bodies_and_hip_joints(empty_model,
                                                                                                   left_landmarks,
                                                                                                   right_landmarks,
                                                                                                   mesh_directory,
                                                                                                   mocap_static_trc,
                                                                                                   pelvis, x_opt_left[
                                                                                                       'hip_rot'],
                                                                                                   x_opt_right[
                                                                                                       'hip_rot'])

    # %% Creation of the Tibia/Fibula (TibFib) Bodies
    tibfib_l_center, tibfib_r_center, left_tibfib, right_tibfib = (
        create_tibfib_bodies_and_knee_joints(empty_model, left_landmarks, right_landmarks, mesh_directory,
                                             mocap_static_trc, left_femur, right_femur, x_opt_left['knee_rot'],
                                             x_opt_right['knee_rot']))

    # %% Create feet bodies
    repurpose_feet_bodies_and_create_joints(empty_model, tibfib_l_center,
                                            tibfib_r_center, left_tibfib, right_tibfib)

    # MUSCLES (add later) Further augment the muscle linkages dictionary and model to contain markers represenitng origins and insertions for all muscles (must be done prior to scaling as unused markers are removed via scaling process)
    # empty_model, muscle_linkages = add_all_muscle_attachment_markers(empty_model,muscle_linkages,{
    #    "Pelvis": pelvis_center,
    #    "Femur": [femur_l_center,femur_r_center],
    #    "Tibfib": [tibfib_l_center,tibfib_r_center],
    # })

    # Finalise the connections of the model
    empty_model.finalizeConnections()

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
    feet_adjustments(output_file, empty_model, mocap_static_trc, realign_feet=True)

    # Finalise the non-scaled foot
    empty_model.finalizeConnections()
    empty_model.printToXML(output_file)

    # MUSCLES (add later) Extract local muscle positions prior to scaling (unused markers, such as those of the muscles, are removed during the scaling process)
    # local_muscle_positions = extract_local_muscle_positions(empty_model)

    # %% Look to scale the size of the feet automatically and move the markers to appropriate positions
    # note this runs a preset scale setting file where only the feet are selected to be scaled
    perform_scaling(model_directory, output_file, static_trc)

    # %% Create variables required by knee joint optimisation
    source_file_path1 = os.path.join(model_directory, "scaled_foot.osim")  # Source path
    knee_optimisation_trc_file = dynamic_trc
    ignore, (start_time, end_time), knee_optimisation_marker_dictionary = read_trc_file_as_dict(
        knee_optimisation_trc_file, True)

    # %%Adjusting & Optimising the Knee Joint Orientations

    # Default temporary model paths
    temp_model_path_1 = model_directory + "/temp1.osim"
    temp_model_path_2 = model_directory + "/temp2.osim"
    optimised_knee_model = model_directory + "/Optimised_Knee_Axes.osim"

    # marker weights used in the IK process
    marker_weights = {
        "RASI": 5, "LASI": 5, "RTHI": 1, "RTIB": 1,
        "RANK": 10, "RMED": 10, "LTHI": 1, "LTIB": 1, "LANK": 10, "LMED": 10,
        "RPSI": 1, "LPSI": 1, "RHEE": 1, "LHEE": 1,
        "RTOE": 1, "LTOE": 1, "RKNE": 2.5, "LKNE": 2.5, "RKNEM": 2.5, "LKNEM": 2.5
    }

    # This runs the knee joint optimisation
    run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1,
                                temp_model_path_2, marker_weights, optimised_knee_model)

    # creation of muscles is optional, work in progress (contains no wrapping or participant specific muscle parameters)
    if create_muscles:
        # Load the model
        model = osim.Model(final_model_path)

        # Adds muscles to the model
        # add_all_muscles_to_model_with_simple_names(model, local_muscle_positions,muscle_linkages)

        # Saves the model
        muscle_model_name = "Muscle_Model"
        muscle_model_file = os.path.join(model_directory, f"{muscle_model_name}.osim")
        model.setName(muscle_model_name)
        model.finalizeConnections()
        model.printToXML(muscle_model_file)

    # Remove temporary .osim files.
    for osim_file in [temp_model_path_1, temp_model_path_2, source_file_path1]:
        if os.path.isfile(osim_file):
            os.remove(osim_file)

        # END##############################################################################################################

        # begin attempt at adding wrapping objects to muscles

        # get the marker set of the model and find some markers
        # compute the midpoint between the LASI and LPSI markers using the midpoint_3d function


'''

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
            "l_glut_max_1": [  # Muscle name (key), list of wrapping objects (values)
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
            "r_glut_max_1": [  # Muscle name (key), list of wrapping objects (values)
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



        #model = add_wrapping_objects_to_model(model, wrapping_objects)
        #model.finalizeConnections()
        #model.printToXML(muscle_model)
'''
