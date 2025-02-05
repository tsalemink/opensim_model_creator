import opensim as osim
import os
import trimesh
import pyvista as pv

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