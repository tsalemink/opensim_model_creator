import os
import trimesh
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename, asksaveasfilename
import shutil

def convert_and_scale_mesh(input_ply, output_stl, scale_factor=1000):
    """
    Converts a .ply mesh to .stl format and scales it down by a given factor.

    Parameters:
    - input_ply (str): Path to the input .ply file.
    - output_stl (str): Path to save the output .stl file.
    - scale_factor (float): The factor by which to scale the mesh (default: 1000).

    Returns:
    - None
    """
    # Load the .ply file
    mesh = trimesh.load(input_ply)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file is not a valid mesh.")

    # Scale the mesh
    mesh.apply_scale(1 / scale_factor)

    # Export the scaled mesh to .stl format
    mesh.export(output_stl, file_type="stl")
    print(f"Converted and scaled mesh saved to: {output_stl}")


def batch_convert_and_scale(scale_factor=1000, input_dir=None):
    """
    Converts all .ply files in the specified or selected directory to .stl format while scaling them.
    Deletes any existing .stl files in the input directory before processing.

    Parameters:
    - scale_factor (float): The factor by which to scale the meshes (default: 1000).
    - input_dir (str): Optional path to the input directory containing .ply files.
                       If not provided, a file dialog will prompt the user to select a directory.

    Returns:
    - None
    """
    if input_dir is None:
        # Prompt the user to select the input directory
        Tk().withdraw()  # Hide the root Tk window
        input_dir = askdirectory(title="Select Directory Containing .ply Files")
        if not input_dir:
            print("No directory selected. Exiting.")
            return

    # Check if the directory exists
    if not os.path.isdir(input_dir):
        print(f"The directory '{input_dir}' does not exist. Exiting.")
        return

    # Delete all existing .stl files in the directory
    stl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".stl")]
    for stl_file in stl_files:
        try:
            os.remove(os.path.join(input_dir, stl_file))
            print(f"Deleted existing .stl file: {stl_file}")
        except Exception as e:
            print(f"Failed to delete {stl_file}: {e}")

    # Process each .ply file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".ply"):
            input_ply = os.path.join(input_dir, file_name)
            output_stl = os.path.join(input_dir, file_name.replace(".ply", ".stl"))
            try:
                convert_and_scale_mesh(input_ply, output_stl, scale_factor)
                print(f"Converted and scaled {file_name} to {output_stl}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")


def combine_meshes(mesh1_path, mesh2_path, output_path):
    """
    Combines two meshes into a single mesh and saves it to the specified output file.

    Parameters:
    - mesh1_path (str): Path to the first mesh file.
    - mesh2_path (str): Path to the second mesh file.
    - output_path (str): Path to save the combined mesh.

    Returns:
    - None
    """
    # Load the meshes
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)

    # Ensure both are valid meshes
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("One or both of the loaded files are not valid meshes.")

    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([mesh1, mesh2])

    # Save the combined mesh
    combined_mesh.export(output_path, file_type="stl")
    print(f"Combined mesh saved to: {output_path}")

def select_mesh_files():
    """
    Prompts the user to select two mesh files and a save location for the combined mesh.

    Returns:
    - mesh1_path (str): Path to the first mesh file.
    - mesh2_path (str): Path to the second mesh file.
    - output_path (str): Path to save the combined mesh.
    """
    # Hide the root Tk window
    Tk().withdraw()

    # Prompt the user to select the first mesh file
    print("Select the first mesh file:")
    mesh1_path = askopenfilename(title="Select the First Mesh File", filetypes=[("Mesh Files", "*.ply *.stl")])
    if not mesh1_path:
        raise FileNotFoundError("No file selected for the first mesh.")

    # Prompt the user to select the second mesh file
    print("Select the second mesh file:")
    mesh2_path = askopenfilename(title="Select the Second Mesh File", filetypes=[("Mesh Files", "*.ply *.stl")])
    if not mesh2_path:
        raise FileNotFoundError("No file selected for the second mesh.")

    # Prompt the user to select the output file location
    print("Select the location to save the combined mesh:")
    output_path = asksaveasfilename(title="Save Combined Mesh As", defaultextension=".stl",
                                    filetypes=[("Mesh Files", "*.stl")])
    if not output_path:
        raise FileNotFoundError("No file selected for the output mesh.")

    return mesh1_path, mesh2_path, output_path

def combine_pelvis_meshes(input_dir, output_path=None):
    """
    Searches an input directory for two STL files containing the word 'pelvis',
    combines them into a single mesh, saves the combined mesh to the specified output file,
    and deletes the original STL files.

    Parameters:
    - input_dir (str): Directory to search for the STL files.
    - output_path (str, optional): Path to save the combined mesh. Defaults to 'combined_pelvis_mesh.stl' in the input directory.

    Returns:
    - None
    """
    # Find all STL files containing "pelvis" in their names
    pelvis_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".stl") and "pelvis" in f.lower()
    ]

    # Check if exactly two files are found
    if len(pelvis_files) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 STL files containing 'pelvis' in their names, but found {len(pelvis_files)}."
        )

    # Load the meshes
    mesh1 = trimesh.load(pelvis_files[0])
    mesh2 = trimesh.load(pelvis_files[1])

    # Ensure both are valid meshes
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("One or both of the loaded files are not valid meshes.")

    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([mesh1, mesh2])

    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(input_dir, "combined_pelvis_mesh.stl")

    # Save the combined mesh
    combined_mesh.export(output_path, file_type="stl")
    print(f"Combined pelvis mesh saved to: {output_path}")

    # Delete the original files
    for file_path in pelvis_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


def move_stl_meshes(input_dir, output_dir):
    """
    Moves all .stl files from the input directory to the output directory.
    If a file with the same name already exists in the output directory, it is replaced.

    Parameters:
    - input_dir (str): Directory to search for .stl files.
    - output_dir (str): Directory to move the .stl files to.

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .stl files in the input directory
    stl_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".stl")
    ]

    # Move each .stl file to the output directory
    for stl_file in stl_files:
        dest_file = os.path.join(output_dir, os.path.basename(stl_file))
        try:
            # Remove the existing file if it exists
            if os.path.exists(dest_file):
                os.remove(dest_file)

            # Move the file to the output directory
            shutil.move(stl_file, output_dir)
            print(f"Moved: {stl_file} -> {output_dir}")
        except Exception as e:
            print(f"Failed to move {stl_file}: {e}")


# Run the batch conversion process
if __name__ == "__main__":
    batch_convert_and_scale()
    # Get the mesh file paths and output path from the user
    mesh1_path, mesh2_path, output_path = select_mesh_files()

    # Combine the meshes
    combine_meshes(mesh1_path, mesh2_path, output_path)
