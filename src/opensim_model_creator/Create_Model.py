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
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize



#%%Import functions from folders
from Functions.file_utils import *
from Functions.general_utils import *
from Functions.bone_utils import *
from Functions.muscle_utils import *


#%% Functions

# Ask user to select participant directory
participant_folder = select_directory()
if participant_folder:
    print(f"Selected directory: {participant_folder}")
else:
    print("No directory selected.")


#Participant inputs folder location
participant_inputs = participant_folder + "/Inputs"


#Correlation betwen muscle names and the relevant origins/insertion bodies and muscle numbers on said body
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


local_muscle_positions = extract_local_muscle_positions(empty_model)




#%% Look to scale the size of the feet automatically and move the markers to appropriate positions

scaling_file = search_files_by_keywords("High_Level_Inputs", "ScaleSettings")[0]
scale_tool = osim.ScaleTool(scaling_file)
scale_tool.setPathToSubject(participant_folder)


# Set the model file
scale_tool.getGenericModelMaker().setModelFileName(output_file)  # Replace with your model file

ignore,(start_time, end_time),dontcare = read_trc_file_as_dict(mocap_trc_file,True)
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

#This runs the knee joint optimisation
run_knee_joint_optimisation(source_file_path1, knee_optimisation_trc_file, start_time, end_time, temp_model_path_1, temp_model_path_2,marker_weights,optimised_knee_model)




#%% Run IK, and extract the model marker positions and compare to those of the actual marker positions across the entire time trial.


optimised_knee_moved_marker_model = output_folder+"/Optimised_Knee_Axes_Moved_Markers.osim"

compute_and_adjust_markers(optimised_knee_model,"ik_output.mot","_ik_model_marker_locations.sto",knee_optimisation_marker_dictionary,optimised_knee_moved_marker_model)

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

