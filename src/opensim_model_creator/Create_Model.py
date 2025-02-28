
import os
import numpy as np
import opensim as osim

from articulated_ssm_both_sides.MainASM import run_asm

#%%Import functions from folders
from opensim_model_creator.Functions.general_utils import *
from opensim_model_creator.Functions.bone_utils import *
from opensim_model_creator.Functions.muscle_utils import *
from opensim_model_creator.Functions.file_utils import reset_folder


root_directory = os.path.dirname(os.path.abspath(__file__))
high_level_inputs = os.path.join(root_directory, "High_Level_Inputs")


def create_model(participant_folder, static_marker_data, weight, height, create_muscles=False, testing=False):
    """
    Creates an OpenSim model for a given participant, optionally adding muscles.

    Args:
        participant_folder (str): Path to the participant's folder.
        static_marker_data (dict): Static marker data coordinates.
        create_muscles (bool): Whether to add muscles to the model.
        testing (bool): If True, runs in test mode - reduces knee optimisation iteration count for computational speed
        weight (float, optional): Participant's weight in kg.
        height (float, optional): Participant's height in meters.

    Returns:
        None
    """

    #%% Setup of folders
    # Define paths for inputs and outputs
    participant_inputs = os.path.join(participant_folder, "Inputs")
    output_folder = os.path.join(participant_folder, "Models")
    meshes = os.path.join(output_folder, "Meshes")

    # Clear output and mesh folders to avoid residuals from previous runs
    reset_folder(output_folder)
    reset_folder(meshes)

    #%%Initialisation

    # Generate mesh files using ASM
    run_asm(static_marker_data, meshes)

    # Move foot mesh files into the meshes directory.
    copy_mesh_files(high_level_inputs, meshes)

    # Scale marker data from millimeters to meters (variable currently unused)
    scale_marker_data(static_marker_data, 0.001)

    # Process and extract meshes from STL files
    process_participant_meshes(meshes, meshes)

    # Initializes muscle linkage directory
    muscle_linkages = muscle_initialisation(meshes)

    #Splits specific muscles into a number of segments
    segment_muscle_origins_insertions(muscle_linkages, "Glut med", num_segments=3)
    segment_muscle_origins_insertions(muscle_linkages, "Glut min", num_segments=3)
    segment_muscle_origins_insertions(muscle_linkages, "Add mag", pair_to_segment=0, num_segments=2)

    # Apply a swap for Adductor Magnus origins to have better anatomical consistency
    swap_muscle_attachments(muscle_linkages, "Add mag", 0, 2, attachment_type="ori")

    #Initialises model, trc files, and landmarks
    empty_model, state, left_landmarks, right_landmarks, mocap_static_trc, mocap_trc_file = initialize_model_and_extract_landmarks(meshes)

    # %% Creation of the pelvis body and pelvis joint
    pelvis, pelvis_joint, rotated_pelvis_center, pelvis_realignment, pelvis_center = create_pelvis_body_and_joint(
        empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc, realign_pelvis=True
    )

    # %% Creation of femur bodies and attachment of meshes, markers, and landmarks
    (l_LEC, l_MEC, l_HJC, l_EC_midpoint, left_femur, femur_l_center, rotated_l_femur_center,
     LKNE_alignment_angles, LHIP_vert_alignment_angles, r_LEC, r_MEC, r_HJC, r_EC_midpoint, right_femur, femur_r_center, rotated_r_femur_center,
     RKNE_alignment_angles, RHIP_vert_alignment_angles) = create_femur_bodies_and_hip_joints(empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc, rotated_pelvis_center, pelvis_realignment, pelvis, realign_femurs= True)

    # %% Creation of the Tibia/Fibula (TibFib) Bodies
    rotated_l_tibfib_center, rotated_r_tibfib_center, tibfib_l_center, tibfib_r_center, left_tibfib, right_tibfib = create_tibfib_bodies_and_knee_joints(
        empty_model, left_landmarks, right_landmarks, meshes, mocap_static_trc,
        rotated_l_femur_center, rotated_r_femur_center, LHIP_vert_alignment_angles, RHIP_vert_alignment_angles,
        left_femur, right_femur,l_LEC, l_MEC, l_HJC, l_EC_midpoint, r_LEC, r_MEC, r_HJC, r_EC_midpoint, realign_tibias=True
    )

    #%% Create feet bodies
    repurpose_feet_bodies_and_create_joints(empty_model, left_landmarks, right_landmarks, rotated_l_tibfib_center, rotated_r_tibfib_center, l_EC_midpoint, r_EC_midpoint, left_tibfib, right_tibfib)

    #Further augment the muscle linkages dictionary and model to contain markers represenitng origins and insertions for all muscles (must be done prior to scaling as unused markers are removed via scaling process)
    empty_model, muscle_linkages = add_all_muscle_attachment_markers(empty_model,muscle_linkages,{
        "Pelvis": pelvis_center,
        "Femur": [femur_l_center,femur_r_center],
        "Tibfib": [tibfib_l_center,tibfib_r_center],
    })

    # Finalise the connections of the model
    empty_model.finalizeConnections()

    # Extract the directory name as the model name and replace spaces with underscores
    model_name = "Bone_Model"

    # Update the model name
    empty_model.setName(model_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine the folder path and filename
    output_path = os.path.join(output_folder, f"{model_name}.osim")

    # Save the model to output folder
    empty_model.printToXML(output_path)
    print(f"Model saved to: {output_path}")

    #%% Perform a long series of updates to the model
    output_file = perform_updates(empty_model, output_folder, meshes, model_name,  weight, height)

    # Reload the model
    empty_model = osim.Model(output_file)

    #%% Reinitialise the model for further feet adjustments (aligning with static trc as gait2392 feet are perfectly straight whilst participants may have their feet angled when neutral)
    feet_adjustments(output_file, empty_model, mocap_static_trc, realign_feet= True)

    # Finalise the non-scaled foot
    empty_model.finalizeConnections()
    empty_model.printToXML(output_file)

    #Extract local muscle positions prior to scaling (unused markers, such as those of the muscles, are removed during the scaling process)
    local_muscle_positions = extract_local_muscle_positions(empty_model)



    #%% Look to scale the size of the feet automatically and move the markers to appropriate positions
    perform_scaling(participant_folder, output_file, mocap_trc_file)


    #%% Create variables required by knee joint optimisation
    source_file_path1 = os.path.join(participant_folder, "Models", "scaled_foot.osim")  # Source path
    knee_optimisation_trc_file = search_files_by_keywords(participant_inputs, "optimisation")[0]  # Find the TRC file containing marker data
    ignore,(start_time, end_time),knee_optimisation_marker_dictionary = read_trc_file_as_dict(knee_optimisation_trc_file,True)


    #%%Adjusting & Optimising the Knee Joint Orientations

    # Default temporary model paths
    temp_model_path_1 = output_folder+ "/temp1.osim"
    temp_model_path_2 = output_folder+ "/temp2.osim"
    optimised_knee_model = output_folder+ "/Optimised_Knee_Axes.osim"

    #marker weights used in the IK process
    marker_weights = {
                "RASI": 5, "LASI": 5, "RTHI": 1, "RTIB": 1,
                "RANK": 10, "LTHI": 1, "LTIB": 1, "LANK": 10,
                "RPSI": 1, "LPSI": 1, "RHEE": 1, "LHEE": 1,
                "RTOE": 1, "LTOE": 1, "RKNE": 2.5, "LKNE": 2.5
            }

    #itersation count of 5 appears to allow convergence whilst not taking overtly long
    if testing:
        iteration_count = 1
    else:
        iteration_count = 5


    #This runs the knee joint optimisation
    run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2,marker_weights,optimised_knee_model, iteration_count= iteration_count)




    #%% Run IK, and extract the model marker positions and compare to those of the actual marker positions across the entire time trial, then adjust.

    optimised_knee_moved_marker_model = output_folder+"/Optimised_Knee_Axes_Moved_Markers.osim"

    compute_and_adjust_markers(optimised_knee_model,"ik_output.mot","_ik_model_marker_locations.sto",knee_optimisation_marker_dictionary,optimised_knee_moved_marker_model)


    # Run the Inverse Kinematics (IK) analysis and print results for the 3 different models
    print("\n")
    print(f"Prior to Knee Alignment & Marker Movement - name of file: {os.path.basename(source_file_path1)}")
    ik_result_1 = perform_IK(source_file_path1, knee_optimisation_trc_file, start_time, end_time, marker_weights)
    print(ik_result_1)
    print("\n")

    print(
        f"Following Knee Alignment but Prior to Marker Adjustment - name of file: {os.path.basename(optimised_knee_model)}")
    ik_result_2 = perform_IK(optimised_knee_model, knee_optimisation_trc_file, start_time, end_time, marker_weights)
    print(ik_result_2)
    print("\n")

    print(
        f"Following Both Knee Alignment & Marker Adjustment - name of file: {os.path.basename(optimised_knee_moved_marker_model)}")
    ik_result_3 = perform_IK(optimised_knee_moved_marker_model, knee_optimisation_trc_file, start_time, end_time,
                             marker_weights)
    print(ik_result_3)
    print("\n")

    # Extract Average RMS Errors from the results
    models = {
        source_file_path1: ik_result_1["Average RMS Error"],
        optimised_knee_model: ik_result_2["Average RMS Error"],
        optimised_knee_moved_marker_model: ik_result_3["Average RMS Error"]
    }

    # Find the model with the lowest Average RMS Error
    best_model_path = min(models, key=models.get)
    best_model_error = models[best_model_path]

    # Format the participant's name by replacing spaces with underscores

    # Define the output file name
    final_model_filename = f"Final_Bone_Model.osim"
    final_model_path = os.path.join(os.path.dirname(best_model_path), final_model_filename)

    # Check if the final model file already exists, and remove it if it does
    if os.path.exists(final_model_path):
        os.remove(final_model_path)  # Delete the existing file

    # Now rename (move) the best model to the final filename
    os.rename(best_model_path, final_model_path)

    print(f"Final model selected: {os.path.basename(best_model_path)} with an Average RMS Error of {best_model_error}")
    print(f"This model was renamed to: {os.path.basename(final_model_path)}")



    #creation of muscles is optional, work in progress (contains no wrapping or participant specific muscle parameters)
    if create_muscles:


        #Load the model
        model = osim.Model(final_model_path)

        #Adds muscles to the model
        add_all_muscles_to_model_with_simple_names(model, local_muscle_positions,muscle_linkages)

        #Saves the model
        muscle_model_name = "Muscle_Model"
        muscle_model_file = os.path.join(output_folder, f"{muscle_model_name}.osim")
        model.setName(muscle_model_name)
        model.finalizeConnections()
        model.printToXML(muscle_model_file)

    # Remove temporary .osim files.
    for osim_file in [temp_model_path_1, temp_model_path_2, optimised_knee_model, source_file_path1]:
        if os.path.isfile(osim_file):
            os.remove(osim_file)


        #END##############################################################################################################

        #begin attempt at adding wrapping objects to muscles

        #get the marker set of the model and find some markers
        #compute the midpoint between the LASI and LPSI markers using the midpoint_3d function


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