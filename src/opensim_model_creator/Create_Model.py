
import os
import numpy as np
import opensim as osim

#%%Import functions from folders
from opensim_model_creator.Functions.general_utils import *
from opensim_model_creator.Functions.bone_utils import *
from opensim_model_creator.Functions.muscle_utils import *
from opensim_model_creator.Functions.file_utils import reset_folder


def create_model(participant_folder, create_muscles = False, testing = False):
    #%% Setup of folders
    participant_inputs = os.path.join(participant_folder, "Inputs")

    #Needs to create this folder (delete it if it exists already and then make it again)
    output_folder = os.path.join(participant_folder, "Models")

    #Needs to create this folder (delete it if it exists already and then make it again)
    meshes = os.path.join(participant_inputs, "Meshes")


    # Clear the output_folder and meshes folder
    reset_folder(output_folder)
    reset_folder(meshes)

    # Initialize muscles
    muscle_linkages = muscle_initialisation(participant_inputs)


    #%%Extraction of meshes from the ply files

    #This section needs to be converted to not convert from ply to stls,
    # it just needs to read in stls ALREADY (from lauras code placing them here) present within the participant_inputs folder, it then needs to scale them down by a factor of 1000,
    # and move them to the meshes folder, then it can combine the left and right pelvis stls,
    # may need to adjust the search by keywords sections when importing meshes to make sure when importing the pelvis mesh we import just the combined pelvis mesh (for creating the pelvis)
    # and likewise when tyring to do muscle attatchment sites we import the left or right pelvis and not the combined pelvis mesh as it will all be stored wihtn the same direcotry now
    # so need to be more specific about which pelvis file we are pulling out.




    #Runs the conversion process
    batch_convert_and_scale(input_dir = participant_inputs)

    # Combine the meshes
    combine_pelvis_meshes(input_dir = participant_inputs)

    # Cuts the stls to the meshes folder
    move_mesh_files(participant_inputs, meshes)















    # %% Initialisation of models and extraction of relevant landmarks/marker placements
    empty_model, state, left_landmarks, right_landmarks, mocap_static_trc, mocap_trc_file = initialize_model_and_extract_landmarks(participant_inputs)


    # %% Creation of the pelvis body and pelvis joint (to ground)

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

    #Create the muscle linkages dictionary
    empty_model, muscle_linkages = add_all_muscle_attachment_markers(empty_model,muscle_linkages,{
        "Pelvis": pelvis_center,
        "Femur": [femur_l_center,femur_r_center],
        "Tibfib": [tibfib_l_center,tibfib_r_center],
    })


    #%% Save the model with all markers

    # Finalise the connections of the model
    empty_model.finalizeConnections()

    # Extract the directory name as the model name and replace spaces with underscores
    model_name = os.path.basename(participant_folder).replace(" ", "_")

    # Update the model name
    empty_model.setName(model_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine the folder path and filename
    output_path = os.path.join(output_folder, f"{model_name}.osim")

    # Save the model to the specified location
    empty_model.printToXML(output_path)
    print(f"Model saved to: {output_path}")

    #%% Perform a long series of updates to the model
    output_file = perform_updates(empty_model, output_folder, model_name)

    # Reload the model
    empty_model = osim.Model(output_file)

    #%% Reinitialise the model for feet adjustments
    feet_adjustments(output_file, empty_model, mocap_static_trc, realign_feet= True)

    # Finalise the non-scaled foot
    empty_model.finalizeConnections()
    empty_model.printToXML(output_file)

    #Extract local muscle positions prior to scaling (unused markers, such as those of the muscles, are removed during the scaling process)
    local_muscle_positions = extract_local_muscle_positions(empty_model)




    #%% Look to scale the size of the feet automatically and move the markers to appropriate positions
    perform_scaling(participant_folder, output_file, mocap_trc_file)


    #%% Create the JMP settings files and move trcs & model automatically
    source_file_path1 = os.path.join(participant_folder, "Models", "scaled_foot.osim")  # Source path
    knee_optimisation_trc_file = search_files_by_keywords(participant_inputs, "optimisation")[0]  # Find the TRC file containing marker data
    ignore,(start_time, end_time),knee_optimisation_marker_dictionary = read_trc_file_as_dict(knee_optimisation_trc_file,True)

    #This feature was removed due to no longer being necessary, however funcitonality will be left below incase desired at some future point in time
    """ 
    
    # Define file paths
    default_file_path = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\JMP_Default.xml"
    output_file_path = r"C:/Users\jplu752\Documents\My Project Stuff\Python_Code_Location\JMP Code\JMP_Actual.xml"
    
    
    #copy the scaled_feet_variant.osim from the directory above to the JMP Code direcotry
    import shutil
    # Define source and destination file paths
    
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
    """

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

    if testing:
        iteration_count = 1
    else:
        iteration_count = 5


    #This runs the knee joint optimisation
    run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2,marker_weights,optimised_knee_model, iteration_count= iteration_count)




    #%% Run IK, and extract the model marker positions and compare to those of the actual marker positions across the entire time trial.


    optimised_knee_moved_marker_model = output_folder+"/Optimised_Knee_Axes_Moved_Markers.osim"

    compute_and_adjust_markers(optimised_knee_model,"ik_output.mot","_ik_model_marker_locations.sto",knee_optimisation_marker_dictionary,optimised_knee_moved_marker_model)


    # Run the Inverse Kinematics (IK) analysis and store results
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
    participant_name = os.path.basename(participant_folder).replace(" ", "_")

    # Define the output file name
    final_model_filename = f"Final_Bone_Model_{participant_name}.osim"
    final_model_path = os.path.join(os.path.dirname(best_model_path), final_model_filename)

    # Check if the final model file already exists, and remove it if it does
    if os.path.exists(final_model_path):
        os.remove(final_model_path)  # Delete the existing file

    # Now rename (move) the best model to the final filename
    os.rename(best_model_path, final_model_path)

    print(f"Final model selected: {os.path.basename(best_model_path)} with an Average RMS Error of {best_model_error}")
    print(f"This model was renamed to: {os.path.basename(final_model_path)}")




    if create_muscles:


        #print("Following Both Knee Alignment & Marker Adjustment (Then Re-Aligned Again)")
        #print(perform_IK(re_optimised_knee_moved_marker_model,knee_optimisation_trc_file,start_time, end_time, marker_weights))

        #Lets try to add some muscles to the model
        #function to compute the position of a mesh node in the local coordinate system of a bone (can try it on the landmarks first), im assuming we can create a new model, add the body/ mesh, add a "marker" to represent this point, then use the marker.getlocation attribute thing to get the position of the marker in the local bone coordinate system (hopefully)

        model = osim.Model(final_model_path)

        add_all_muscles_to_model_with_simple_names(model, local_muscle_positions,muscle_linkages)

        muscle_model_name = os.path.basename(participant_folder).replace(" ", "_")

        muscle_model = output_folder+"/Muscle_" +muscle_model_name+ ".osim"
        model.setName("Muscle_"+muscle_model_name)
        model.finalizeConnections()
        model.printToXML(muscle_model)




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
