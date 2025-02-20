#Import Packages
import os
import opensim as osim
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Import required functions
from opensim_model_creator.Functions.file_utils import search_files_by_keywords
from opensim_model_creator.Functions.bone_utils import add_markers_to_body


#Contains the muscle linkages definitions
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
        "ins": [["Femur", "111_112_113"],["Femur", "111_112_113"]],
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
        "ins": [["Tibia", "116"],["Tibia", "116"]],
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
        "ori": [["Pelvis", "121"],["Pelvis", "121"]],
        "ins": [["Femur","121_1"],["Femur", "121"]],
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





#Functions of use
def parse_muscle_node_files_recursive(root_directory):
    """
    Recursively parses muscle node files in a root directory and constructs a nested dictionary.

    Args:
    - root_directory (str): Path to the root directory containing subfolders for bones.

    Returns:
    - dict: Nested dictionary where the first level keys are bone names (e.g., 'Pelvis'),
            the second level keys are muscle numbers (strings), and values are lists of node numbers.
    """
    muscle_nodes = {}

    # Walk through the root directory and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Get the current folder name (e.g., 'Pelvis', 'Femur')
        bone_name = os.path.basename(dirpath)
        if bone_name not in muscle_nodes:
            muscle_nodes[bone_name] = {}

        for file_name in filenames:
            if file_name.endswith("_NodeNo.txt"):
                # Extract the muscle number(s) by splitting on the last underscore
                muscle_number = file_name.rsplit("_", 1)[0]

                # Read the node numbers from the file
                with open(os.path.join(dirpath, file_name), 'r') as file:
                    nodes = [int(line.strip()) for line in file if line.strip().isdigit()]

                # Add to dictionary under the corresponding bone name
                muscle_nodes[bone_name][muscle_number] = nodes

    # Remove empty entries (for folders without valid files)
    muscle_nodes = {k: v for k, v in muscle_nodes.items() if v}
    return muscle_nodes

def parse_stl_files_by_side_and_bone(directory, bones):
    """
    Parses .stl files by side ('Left', 'Right') and bone names to extract vertex coordinates.

    Args:
        directory (str): The directory containing the .stl files.
        bones (list of str): List of bone names to process (e.g., ['tibfib', 'pelvis', 'femur']).

    Returns:
        dict: A nested dictionary with structure {side: {bone: {vertex_index: [x, y, z], ...}, ...}, ...}.
    """
    sides = ['Left', 'Right']
    parsed_data = {side: {} for side in sides}

    for side in sides:
        for bone in bones:
            # Search for the specific file
            stl_files = search_files_by_keywords(directory, side + " " + bone)

            if len(stl_files) == 0:
                print(f"No .stl file found for {side} {bone}. Skipping.")
                continue
            if len(stl_files) > 1:
                print(f"Multiple .stl files found for {side} {bone}: {stl_files}. Using the first one.")

            stl_path = stl_files[0]  # Use the first match

            # Load the STL file using trimesh
            mesh = trimesh.load(stl_path)

            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Invalid mesh file: {stl_path}. Skipping.")
                continue

            # Extract unique vertex coordinates
            unique_vertices = mesh.vertices

            # Store the parsed vertices in the dictionary
            vertex_dict = {i: list(unique_vertices[i]) for i in range(len(unique_vertices))}
            parsed_data[side][bone] = vertex_dict

            print(f"Processed {side} {bone}: {len(vertex_dict)} unique vertices.")

    return parsed_data

def map_muscle_nodes_to_coordinates(muscle_linkages, muscle_number_to_nodes_key, node_to_coordinate):
    """
    Maps muscle numbers to their corresponding mean node coordinates and updates muscle_linkages.

    Parameters:
    - muscle_linkages (dict): Dictionary defining muscle linkages with body parts and muscle numbers.
    - muscle_number_to_nodes_key (dict): Dictionary mapping muscle numbers to nodes by body part.
    - node_to_coordinate (dict): Dictionary mapping node numbers to their coordinates.

    Returns:
    - None: Updates the muscle_linkages dictionary in place with mean coordinates for each attachment.
    """
    sides = ["Left", "Right"]

    for muscle, attachment_types in muscle_linkages.items():
        for attachment_type, attachments in attachment_types.items():
            for attachment in attachments:
                body_part = attachment[0]
                muscle_number = attachment[1]

                # Retrieve nodes for the muscle number from muscle_number_to_nodes_key
                nodes = muscle_number_to_nodes_key.get(body_part, {}).get(muscle_number, [])

                # Retrieve coordinates for these nodes from node_to_coordinate
                for side in sides:
                    current_coordinates = []
                    for node in nodes:
                        try:
                            # Append coordinates (no longer scaled due to the stls already being scaled before extracting their coordinate positions)
                            current_coordinates.append(
                                [coord for coord in node_to_coordinate[side][body_part][node]]
                            )
                        except KeyError:
                            print(f"Node {node} not found in {side}/{body_part}. Skipping.")

                    # Ensure there are coordinates to compute mean
                    if current_coordinates:
                        # Convert the list of coordinates to a NumPy array
                        coordinates_array = np.array(current_coordinates)

                        # Compute the mean along each axis (x, y, z)
                        mean_coordinates = np.mean(coordinates_array, axis=0)

                        # Append mean_coordinates to the attachment list
                        #attachment.append(mean_coordinates)

                        # Append coordinates_array to the attachment list (for purposes of multiple attatchment sites)
                        attachment.append(coordinates_array)


    return muscle_linkages

def add_all_muscle_attachment_markers(model, muscle_linkages, centers):
    sides = ["_l", "_r"]
    for muscle in muscle_linkages.keys():
        for attachment_type in muscle_linkages[muscle].keys():
            if len(muscle_linkages[muscle].keys()) < 2:
                continue
            # Iterate through each attachment for the current muscle and attachment type
            i = 1
            for attachment in muscle_linkages[muscle][attachment_type]:
                #if attachment[0] == "Tibia" or attachment[0] == "Fibula":
                    #continue
                muscle_name_storage = [0,0]
                for side in sides:
                    # Extract the body name from the attachment and convert to lowercase
                    body_name = attachment[0].lower()
                    if body_name != "pelvis":
                        if body_name == "tibia" or body_name == "fibula":
                            body_name = "tibfib"
                            attachment[0] = "Tibfib"
                        body_name = body_name + side
                        if side == "_l":
                            center = centers[attachment[0]][0]
                        elif side == "_r":
                            center = centers[attachment[0]][1]
                    else:
                        center = centers[attachment[0]]
                    body_name = body_name + "_b"

                    if len(muscle_linkages[muscle][attachment_type]) > 1:
                        muscle_name = attachment_type + side+ "_" + muscle.lower().replace(" ", "_") + "_" + str(i)
                    else:
                        muscle_name = attachment_type + side + "_" + muscle.lower().replace(" ", "_")
                    muscle_name = [muscle_name]
                    if side == "_l":
                        location = {muscle_name[0]:np.mean(attachment[2], axis=0)}
                        muscle_name_storage[0] = muscle_name[0]
                    elif side == "_r":
                        location = {muscle_name[0]:np.mean(attachment[3], axis=0)}
                        muscle_name_storage[1] = muscle_name[0]
                    add_markers_to_body(model, body_name, muscle_name,location, center)
                attachment.append(muscle_name_storage[0])
                attachment.append(muscle_name_storage[1])
                i = i+1
    return model, muscle_linkages

def add_all_muscles_to_model_with_simple_names(model, local_muscle_positions, muscle_linkages):
    """
    Adds all muscles to the model based on muscle_linkages and local_muscle_positions.
    Names muscles using a simple convention like 'l_gem_1' or 'r_gem_2'.
    If the number of origins equals the number of insertions, it maps them directly (1-to-1).
    Otherwise, it creates pairwise combinations.

    Args:
        model (osim.Model): OpenSim model to which the muscles will be added.
        local_muscle_positions (dict): Dictionary mapping muscle marker names to positions and parent bodies.
        muscle_linkages (dict): Dictionary defining muscle linkages with body parts and muscle numbers.
    """
    for muscle_name, attachments in muscle_linkages.items():
        # Ensure both origin ('ori') and insertion ('ins') exist for the muscle
        if 'ori' not in attachments or 'ins' not in attachments:
            print(f"Skipping {muscle_name}: Missing origin or insertion data.")
            continue

        origins = attachments['ori']
        insertions = attachments['ins']

        '''
        # Skip tibia/fibula-based muscles (as per previous filtering)
        if any(b in ["Tibia", "Fibula"] for b, *_ in origins + insertions):
            continue
        '''

        num_origins = len(origins)
        num_insertions = len(insertions)

        # **Direct Mapping Mode** (1st origin → 1st insertion, 2nd origin → 2nd insertion)
        if num_origins == num_insertions:
            for i in range(num_origins):
                origin = origins[i]
                insertion = insertions[i]

                # Left muscle
                origin_marker_name_l = origin[4]
                insertion_marker_name_l = insertion[4]
                origin_position_l = local_muscle_positions.get(origin_marker_name_l)
                insertion_position_l = local_muscle_positions.get(insertion_marker_name_l)

                if origin_position_l and insertion_position_l:
                    simple_name_l = f"l_{muscle_name.lower().replace(' ', '_')}_{i + 1}"
                    add_muscle_to_model(
                        model=model,
                        muscle_name=simple_name_l,
                        origin_point=origin_position_l,
                        insertion_point=insertion_position_l,
                    )

                # Right muscle
                origin_marker_name_r = origin[5]
                insertion_marker_name_r = insertion[5]
                origin_position_r = local_muscle_positions.get(origin_marker_name_r)
                insertion_position_r = local_muscle_positions.get(insertion_marker_name_r)

                if origin_position_r and insertion_position_r:
                    simple_name_r = f"r_{muscle_name.lower().replace(' ', '_')}_{i + 1}"
                    add_muscle_to_model(
                        model=model,
                        muscle_name=simple_name_r,
                        origin_point=origin_position_r,
                        insertion_point=insertion_position_r,
                    )


def add_wrapping_objects_to_model(model, wrapping_objects):
    """
    Adds wrapping objects to the model and assigns them to the corresponding muscles.

    Parameters:
    - model (osim.Model): The OpenSim model.
    - wrapping_objects (dict): Dictionary containing wrapping object details for each muscle.

    Returns:
    - None: Updates the model in place.
    """
    # Get all forces (muscles) in the model
    force_set = model.updForceSet()

    for muscle_name, wrap_objects in wrapping_objects.items():
        for wrap_data in wrap_objects:
            try:
                # Extract wrapping object details
                wrap_name = wrap_data["name"]
                body_name = wrap_data["body"]
                wrap_type = wrap_data["type"]
                translation = wrap_data["translation"]
                rotation = wrap_data["rotation"]
                radius = wrap_data["radius"]
                length = wrap_data.get("length", None)  # Length only for cylinders
                quadrant = wrap_data.get("quadrant")

                # Get the body to attach the wrapping object
                body = model.getBodySet().get(body_name)

                # Create the appropriate wrapping object
                if wrap_type == "cylinder":
                    wrap_object = osim.WrapCylinder()
                    wrap_object.set_radius(radius)
                    wrap_object.set_length(length)
                elif wrap_type == "sphere":
                    wrap_object = osim.WrapSphere()
                    wrap_object.set_radius(radius)
                else:
                    print(f"Unknown wrapping object type '{wrap_type}' for {wrap_name}. Skipping.")
                    continue

                # Set properties of the wrapping object
                wrap_object.setName(wrap_name)
                wrap_object.set_translation(osim.Vec3(*translation))
                wrap_object.set_xyz_body_rotation(osim.Vec3(*rotation))
                wrap_object.set_quadrant(quadrant)
                # Attach to the correct body
                body.addWrapObject(wrap_object)

                print(f"Added wrapping object '{wrap_name}' to '{body_name}'.")

                # **Assign Wrapping Object to the Muscle**
                force = force_set.get(muscle_name)  # Find the muscle by name
                # Cast To Muscle
                muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(force)
                muscle.updGeometryPath().addPathWrap(wrap_object)
                print(f"Assigned wrapping object '{wrap_name}' to muscle '{muscle_name}'.")


            except Exception as e:
                print(f"Error processing wrapping object '{wrap_name}': {e}")
    return model

def add_muscle_to_model(
    model,
    muscle_name,
    origin_point,
    insertion_point,
    via_points=None,
    max_isometric_force=500.0,
    optimal_fiber_length=0.04,
    tendon_slack_length=0.2,
    pennation_angle_at_optimal=0.1,
    max_contraction_velocity=10.0,
):
    """
    Adds a Millard2012EquilibriumMuscle to an OpenSim model.

    Parameters:
    - model (osim.Model): The OpenSim model to which the muscle will be added.
    - muscle_name (str): Name of the muscle.
    - limb_side (str): Specify 'left' or 'right' for naming consistency.
    - origin_point (tuple): (body_name, Vec3) for the muscle's origin.
    - insertion_point (tuple): (body_name, Vec3) for the muscle's insertion.
    - via_points (list of tuples, optional): [(body_name, Vec3), ...] for intermediate points.
    - max_isometric_force (float): Maximum isometric force in Newtons. Default is 500 N.
    - optimal_fiber_length (float): Optimal fiber length in meters. Default is 0.04 m.
    - tendon_slack_length (float): Tendon slack length in meters. Default is 0.2 m.
    - pennation_angle_at_optimal (float): Pennation angle at optimal fiber length in radians. Default is 0.1 rad.
    - max_contraction_velocity (float): Maximum contraction velocity in fiber lengths per second. Default is 10.

    Returns:
    - None: The muscle is added directly to the model.
    """
    try:
        # Create the muscle
        muscle = osim.Millard2012EquilibriumMuscle()
        muscle.setName(muscle_name)

        # Set muscle properties
        muscle.set_max_isometric_force(max_isometric_force)
        muscle.set_optimal_fiber_length(optimal_fiber_length)
        muscle.set_tendon_slack_length(tendon_slack_length)
        muscle.set_pennation_angle_at_optimal(pennation_angle_at_optimal)
        muscle.set_max_contraction_velocity(max_contraction_velocity)

        # Set origin point
        origin_body, origin_vec = origin_point
        muscle.addNewPathPoint("origin", model.getBodySet().get(origin_body), osim.Vec3(origin_vec))

        # Set insertion point
        insertion_body, insertion_vec = insertion_point
        muscle.addNewPathPoint("insertion", model.getBodySet().get(insertion_body), osim.Vec3(insertion_vec))

        # Add via points if provided
        if via_points:
            for i, (via_body, via_vec) in enumerate(via_points):
                muscle.addNewPathPoint(f"via_{i+1}", model.getBodySet().get(via_body), osim.vec3(via_vec))

        # Add the muscle to the model
        model.addForce(muscle)
        print(f"Muscle '{muscle_name}' added to the model.")

    except Exception as e:
        print(f"Error adding muscle '{muscle_name}': {e}")

def extract_local_muscle_positions(model_path):
    """
    Extracts the local positions of markers relative to their parent frames and includes the body they are attached to.

    Args:
        model_path (str): Path to the OpenSim model file containing the markers.

    Returns:
        dict: Dictionary with marker names as keys and a tuple (body_name, local_position) as values.
    """
    # Load the model
    model = osim.Model(model_path)
    state = model.initSystem()

    # Dictionary to store local positions and attached body
    local_positions = {}

    # Iterate through the markers in the MarkerSet
    markerset = model.getMarkerSet()
    for i in range(markerset.getSize()):
        marker = markerset.get(i)
        marker_name = marker.getName()

        # Check if the marker name starts with "ins_" or "ori_"
        if marker_name.startswith("ins_") or marker_name.startswith("ori_"):
            try:
                # Get the marker's local position
                local_vec = marker.get_location()

                # Get the body the marker is attached to
                body_name = marker.getParentFrame().getName()

                # Store the local position and body name in the dictionary
                local_positions[marker_name] = (
                    body_name,
                    (local_vec.get(0), local_vec.get(1), local_vec.get(2)),
                )
            except Exception as e:
                print(f"Error processing marker '{marker_name}': {e}")

    return local_positions

def muscle_initialisation(participant_inputs):
    """
    Initializes the muscle linkage structure by mapping muscle numbers to their corresponding
    node coordinates and updating the muscle_linkages dictionary.

    Parameters:
        participant_inputs (str): Path to the participant's input directory.

    Returns:
        dict: Updated muscle_linkages dictionary with mapped coordinates.
    """
    # Correlation between muscle names and their respective origin/insertion bodies and muscle numbers
    global muscle_linkages  # Ensure global access to muscle_linkages

    # Relate muscle numbers to nodes that make up the insertion/origin
    muscle_number_to_nodes_key = parse_muscle_node_files_recursive("High_Level_Inputs/final_node_numbers")

    # Relate node numbers to their coordinates
    node_to_coordinate = parse_stl_files_by_side_and_bone(participant_inputs, ["Femur", "Tibia", "Pelvis", "Fibula"])

    # Map muscle nodes to their corresponding coordinates
    muscle_linkages = map_muscle_nodes_to_coordinates(
        muscle_linkages, muscle_number_to_nodes_key, node_to_coordinate
    )

    return muscle_linkages

def segment_coordinates_pca(coordinates, num_sections):
    """
    Segments a list of 3D coordinates into a specified number of sections using PCA.

    Args:
        coordinates (list of list/tuple): A list of [x, y, z] coordinates.
        num_sections (int): Number of sections to divide the data into.

    Returns:
        dict: A dictionary where keys are section indices and values are lists of 3D points in that section.
    """
    # Convert to NumPy array
    coords_array = np.array(coordinates)

    # Apply PCA to find the principal axis
    pca = PCA(n_components=3)
    pca.fit(coords_array)
    principal_axis = pca.components_[0]  # First principal component

    # Project points onto the principal axis
    projected_values = coords_array @ principal_axis  # Dot product to project onto axis

    # Apply K-Means clustering to divide into sections
    kmeans = KMeans(n_clusters=num_sections, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(projected_values.reshape(-1, 1))

    # Store results in a dictionary
    segmented_groups = {i: [] for i in range(num_sections)}
    for idx, label in enumerate(cluster_labels):
        segmented_groups[label].append(tuple(coords_array[idx]))

    return segmented_groups, coords_array, cluster_labels, principal_axis

def segment_coordinates_pca_sorted(coords, num_segments):
    """
    Segments a set of 3D coordinates into a given number of regions using PCA,
    and sorts them along the first principal axis to minimize crossing.

    Args:
        coords (numpy.ndarray): Nx3 array of 3D points representing muscle attachment.
        num_segments (int): Number of segments to divide the data into.

    Returns:
        list: A list containing `num_segments` sublists of segmented and ordered coordinates.
    """
    coords = np.array(coords)  # Ensure it's an array
    if coords.shape[0] < num_segments:
        raise ValueError("Not enough points to segment into the requested number of sections.")

    # Apply PCA to find the main orientation of the points
    pca = PCA(n_components=3)
    pca.fit(coords)

    # Project the points onto the first principal component
    projected = coords @ pca.components_[0]

    # Sort indices based on PCA projection (this orders them along PC1)
    sorted_indices = np.argsort(projected)

    # Reorder coordinates based on PCA sorting
    sorted_coords = coords[sorted_indices]

    # Split into segments
    segmented_coords = np.array_split(sorted_coords, num_segments)

    return segmented_coords

def segment_muscle_origins_insertions(muscle_linkages, muscle_name, pair_to_segment=0, num_segments=3):
    """
    Segments only a specific origin-insertion pair of a given muscle into `num_segments` sections using PCA.

    Args:
        muscle_linkages (dict): Dictionary containing muscle attachment data.
        muscle_name (str): The name of the muscle to process.
        pair_to_segment (int): The index (0-based) of the origin-insertion pair to segment.
        num_segments (int): The number of segments to split that specific pair into.

    Returns:
        None (modifies muscle_linkages in-place)
    """

    # Ensure muscle exists
    if muscle_name not in muscle_linkages:
        print(f"Muscle '{muscle_name}' not found in muscle_linkages.")
        return

    # Ensure selected pair exists
    if pair_to_segment >= len(muscle_linkages[muscle_name]["ori"]):
        print(f"Invalid pair index {pair_to_segment} for muscle '{muscle_name}'. Skipping segmentation.")
        return

    # Extract coordinates for segmentation
    origin_l_coords = muscle_linkages[muscle_name]["ori"][pair_to_segment][2]
    origin_r_coords = muscle_linkages[muscle_name]["ori"][pair_to_segment][3]
    insertion_l_coords = muscle_linkages[muscle_name]["ins"][pair_to_segment][2]
    insertion_r_coords = muscle_linkages[muscle_name]["ins"][pair_to_segment][3]

    # Apply PCA-based segmentation with ordering
    segmented_ori_l = segment_coordinates_pca_sorted(origin_l_coords, num_segments)
    segmented_ori_r = segment_coordinates_pca_sorted(origin_r_coords, num_segments)
    segmented_ins_l = segment_coordinates_pca_sorted(insertion_l_coords, num_segments)
    segmented_ins_r = segment_coordinates_pca_sorted(insertion_r_coords, num_segments)

    # Store the original pair before modifying
    original_ori = muscle_linkages[muscle_name]["ori"][pair_to_segment]
    original_ins = muscle_linkages[muscle_name]["ins"][pair_to_segment]

    # Replace the original entry instead of deleting
    for i in range(num_segments):
        new_ori = original_ori.copy()
        new_ins = original_ins.copy()

        new_ori[2] = np.array(segmented_ori_l[i])
        new_ori[3] = np.array(segmented_ori_r[i])
        new_ins[2] = np.array(segmented_ins_l[i])
        new_ins[3] = np.array(segmented_ins_r[i])

        # Replace the original entry (first one), then append new ones
        if i == 0:
            muscle_linkages[muscle_name]["ori"][pair_to_segment] = new_ori
            muscle_linkages[muscle_name]["ins"][pair_to_segment] = new_ins
        else:
            muscle_linkages[muscle_name]["ori"].insert(pair_to_segment + i, new_ori)
            muscle_linkages[muscle_name]["ins"].insert(pair_to_segment + i, new_ins)

    print(f"Segmented {muscle_name} pair {pair_to_segment} into {num_segments} sections.")

def swap_muscle_attachments(muscle_linkages, muscle_name, index1, index2, attachment_type="ori"):
    """
    Swaps the origins or insertions of two segments in a muscle's linkage dictionary.

    Args:
        muscle_linkages (dict): The dictionary containing muscle attachments.
        muscle_name (str): The muscle to modify.
        index1 (int): First index to swap.
        index2 (int): Second index to swap.
        attachment_type (str): Type of attachment to swap, either "ori" (origin) or "ins" (insertion).

    Returns:
        None (modifies muscle_linkages in-place)
    """
    if muscle_name not in muscle_linkages:
        print(f"Muscle '{muscle_name}' not found.")
        return

    if attachment_type not in ["ori", "ins"]:
        print(f"Invalid attachment type '{attachment_type}'. Use 'ori' for origins or 'ins' for insertions.")
        return

    if index1 >= len(muscle_linkages[muscle_name][attachment_type]) or index2 >= len(muscle_linkages[muscle_name][attachment_type]):
        print(f"Invalid indices for swapping in '{muscle_name}'.")
        return

    # Swap the specified attachment (origin or insertion) at the specified indices
    muscle_linkages[muscle_name][attachment_type][index1], muscle_linkages[muscle_name][attachment_type][index2] = \
        muscle_linkages[muscle_name][attachment_type][index2], muscle_linkages[muscle_name][attachment_type][index1]

    print(f"Swapped {attachment_type} at indices {index1} and {index2} for '{muscle_name}'.")