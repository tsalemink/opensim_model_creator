import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ezc3d
from scipy.signal import butter, filtfilt
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from scipy.interpolate import interp1d


def make_arrays_same_length(arr1, arr2, fill_value=0):
    """
    Adjusts two arrays to make them the same length by truncating or padding.

    Args:
        arr1 (numpy.ndarray): The first array.
        arr2 (numpy.ndarray): The second array.
        fill_value (int, float, optional): The value to pad with. Default is 0.

    Returns:
        tuple: A tuple of two arrays of the same length.
    """
    max_length = max(len(arr1), len(arr2))

    # Adjust the first array
    if len(arr1) < max_length:
        arr1 = np.pad(arr1, (0, max_length - len(arr1)), constant_values=fill_value)
    elif len(arr1) > max_length:
        arr1 = arr1[:max_length]

    # Adjust the second array
    if len(arr2) < max_length:
        arr2 = np.pad(arr2, (0, max_length - len(arr2)), constant_values=fill_value)
    elif len(arr2) > max_length:
        arr2 = arr2[:max_length]

    return arr1, arr2

# Function to calculate angular velocity using central difference
def calculate_angular_velocity(joint_angle, time):
    # Convert joint angle from degrees to radians
    joint_angle_radians = np.deg2rad(joint_angle)
    # Calculate angular velocity in radians per second
    angular_velocity = np.gradient(joint_angle_radians, time)
    return angular_velocity

# Function to calculate joint power
def calculate_joint_power(angular_velocity, joint_torque):
    joint_power = angular_velocity * joint_torque
    return joint_power


# Function to interpolate data to a fixed length
def interpolate_data(segment_df, column, length=100):
    f = interp1d(segment_df['GaitCycle'], segment_df[column], kind='linear', fill_value="extrapolate")
    return f(np.linspace(0, 100, length))


def interpolate_numpy(array, length=100):
    """
    Interpolates a NumPy array to a fixed length.

    Args:
        array (numpy.ndarray): The input array to interpolate.
        length (int): The desired length of the output array.

    Returns:
        numpy.ndarray: The interpolated array.
    """
    # Generate the original indices of the input array
    original_indices = np.linspace(0, 1, len(array))

    # Generate the desired indices for the interpolated array
    target_indices = np.linspace(0, 1, length)

    # Interpolation function
    f = interp1d(original_indices, array, kind='linear', fill_value="extrapolate")

    # Return the interpolated values
    return f(target_indices)



# Function to prompt the user to select a folder
def select_folder():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = askdirectory(title="Select a Participant Folder")
    root.destroy()
    return folder_path


# Function to read .mot files
def readMotionFile(filename):
    # Read the file to identify the end of the header
    with open(filename, 'r') as file_id:
        lines = file_id.readlines()

    header_end_index = next(i for i, line in enumerate(lines) if 'endheader' in line) + 1

    # Extract header
    header = [line.strip() for line in lines[:header_end_index]]

    # Read data using pandas
    df = pd.read_csv(
        filename,
        sep="\s+",
        skiprows=header_end_index,
        header=0
    )

    # Extract labels from the dataframe
    labels = df.columns.tolist()

    return df, header, labels

# Function to design a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Function to apply a low-pass filter to all joint angle columns
def filter_joint_angles(df, cutoff=6, fs=100):
    for col in df.columns:
        if col != 'Time':  # Skip the Time column
            df[col] = butter_lowpass_filter(df[col], cutoff=cutoff, fs=fs)
    return df


# Function to import IK data from .mot files
def import_data_IK(file_path, cutoff=6, fs=100):
    df, _, _ = readMotionFile(file_path)
    df['Time'] = df['time'] - df['time'][0]  # Normalise time
    if "IK" in file_path:
        df = filter_joint_angles(df, cutoff=cutoff, fs=fs)  # Apply filtering
    return df


# Function to extract events from C3D files
def extract_events(file_path, offset=0):
    c3d = ezc3d.c3d(file_path)
    events = {'Left': {}, 'Right': {}}

    # Check if events are available
    if "EVENT" in c3d.parameters:
        event_group = c3d.parameters["EVENT"]
        if "LABELS" in event_group and "CONTEXTS" in event_group and "TIMES" in event_group:
            labels = event_group["LABELS"]["value"]
            contexts = event_group["CONTEXTS"]["value"]
            times = event_group["TIMES"]["value"]

            # Iterate through events and sort them into Left/Right
            for i, label in enumerate(labels):
                context = contexts[i]
                time = times[1][i]  # Get the second row for time (start time)
                if context.strip().lower() in ['left', 'right']:
                    events[context.strip().capitalize()][time - offset] = label.strip()

    # Sort the events by time
    for foot in events.keys():
        events[foot] = dict(sorted(events[foot].items()))

    return events


def prepare_labs(folder_path=None):
    """
    Prepare the labs list by fetching C3D, IK, and ID files from the selected folder.

    Args:
        folder_path (str, optional): Path to the folder containing the required files.
                                     If None, prompts the user to select a folder.

    Returns:
        list: A list of tuples containing (C3D file path, IK file path, offset time, ID file path).
    """
    # Prompt user to select folder if no path is provided
    if folder_path is None:
        folder_path = select_folder()

    # Validate folder path
    if not folder_path:
        raise ValueError("No folder selected. Please select a folder.")
    if not os.path.exists(folder_path):
        raise FileNotFoundError("The selected folder path does not exist. Please check and try again.")

    # Get all the IK, C3D, and ID files
    ik_folder = os.path.join(folder_path, "IK_ID_C3D_files", "IK")
    c3d_folder = os.path.join(folder_path, "IK_ID_C3D_files", "C3D")
    id_folder = os.path.join(folder_path, "IK_ID_C3D_files", "ID")

    ik_files = [os.path.join(ik_folder, f) for f in os.listdir(ik_folder) if f.endswith('.mot')]
    c3d_files = [os.path.join(c3d_folder, f) for f in os.listdir(c3d_folder) if f.endswith('.c3d')]
    id_files = [os.path.join(id_folder, f) for f in os.listdir(id_folder) if f.endswith('.sto')]

    # Ensure files match across folders
    if len(ik_files) != len(c3d_files) or len(ik_files) != len(id_files):
        raise ValueError("Mismatch in the number of IK, C3D, and ID files.")

    # Create the labs list
    labs = []
    for i in range(len(ik_files)):
        dataframe = import_data_IK(ik_files[i])
        offset = dataframe['time'][0]  # Extract offset time
        labs.append((c3d_files[i], ik_files[i], offset, id_files[i]))

    return labs


def plot_generic(ax, df, events, variable_name, plot_first_stride_only=False, mass=0, power=False):
    left_leg_data = pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)})
    right_leg_data = pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)})

    for leg, color in zip(['Right', 'Left'], ['red', 'blue']):
        foot_strikes = sorted([time for time, event in events[leg].items() if 'Foot Strike' in event])
        if len(foot_strikes) >= 2:
            # Limit to the first stride if the argument is set
            loop_range = range(1) if plot_first_stride_only else range(len(foot_strikes) - 1)
            for i in loop_range:
                start_time = foot_strikes[i]
                end_time = foot_strikes[i + 1]
                segment_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)].copy()
                segment_df['GaitCycle'] = np.linspace(0, 100, len(segment_df))
                variable_name1 = variable_name

                if power:
                    variable_name1 = f'{leg.lower()}_{variable_name1}_power'
                    segment_df[variable_name1] = segment_df[variable_name1] / mass
                elif mass == 0:
                    if "pelvis" not in variable_name:
                        variable_name1 = f'{leg.lower()}_{variable_name}'
                        if "hip_abduction" in variable_name and leg == 'Left':
                            segment_df[variable_name1] = -segment_df[variable_name1]
                        if "hip_rotation" in variable_name and leg == 'Left':
                            segment_df[variable_name1] = -segment_df[variable_name1]
                    elif ("rotation" in variable_name or "ry" in variable_name) and leg == 'Left':
                        segment_df[variable_name1] = -segment_df[variable_name1]
                    elif "obliquity" in variable_name or "rx" in variable_name:
                        segment_df[variable_name1] -= 90
                        if leg == 'Right':
                            segment_df[variable_name1] = -segment_df[variable_name1]
                elif mass > 0:
                    variable_name1 = f'{leg.lower()}_{variable_name}_moment'
                    if "pelvis" not in variable_name:
                        if "abduction" in variable_name and leg == 'Right':
                            segment_df[variable_name1] = -segment_df[variable_name1]
                        if "rotation" in variable_name:
                            segment_df[variable_name1] = -segment_df[variable_name1]
                    segment_df[variable_name1] = segment_df[variable_name1] / mass

                interpolated_data = interpolate_data(segment_df, variable_name1)
                interpolated_df = pd.DataFrame({
                    'GaitCycle': np.linspace(0, 100, len(interpolated_data)),
                    f'Stride_{i + 1}': interpolated_data
                })

                ax.plot(
                    interpolated_df['GaitCycle'],
                    interpolated_df[f'Stride_{i + 1}'],
                    color=color,
                    label='Right Foot' if leg == 'Right' else 'Left Foot'
                )

                if leg == 'Left':
                    left_leg_data = pd.concat([left_leg_data, interpolated_df[f'Stride_{i + 1}']], axis=1)
                if leg == 'Right':
                    right_leg_data = pd.concat([right_leg_data, interpolated_df[f'Stride_{i + 1}']], axis=1)

    return left_leg_data, right_leg_data


def calculate_power(df_angles, df_moments, joint_name):
    """
    Calculate joint power for a specific joint.

    Args:
        df_angles (pd.DataFrame): DataFrame containing joint angles and time.
        df_moments (pd.DataFrame): DataFrame containing joint moments.
        joint_name (str): The name of the joint to calculate power for.

    Returns:
        pd.Series: Calculated joint power for the specified joint.
    """
    # Calculate angular velocity
    angular_velocity = calculate_angular_velocity(df_angles[f'{joint_name}'], df_angles['time'])

    # Extract joint moments
    moments = df_moments[f'{joint_name}_moment'].to_numpy()

    # Ensure both arrays are the same length
    angular_velocity, moments = make_arrays_same_length(angular_velocity, moments)

    # Calculate joint power
    power = calculate_joint_power(angular_velocity, moments)

    return power



def plot_joint_data(labs, plot_type, fig_title="SELECT TITLE", y_limits=None, mass=0):
    """
    Plot joint angles (IK) or joint moments (ID) from C3D, IK, and ID files for multiple trials.

    Args:
        labs (list): List of tuples containing trial data:
                     (c3d_file_path, mot_file_path, offset_time, id_file_path).
        plot_type (str): Type of plot ("IK" for joint angles or "ID" for joint moments).
        fig_title (str): Title for the figure.
        y_limits (list): List of y-axis limits for each subplot. Defaults to None.

    Returns:
        dict: Aggregated data for each joint.
        dict: Mean values for each joint.
    """

    # Select appropriate y-axis label and default y-limits
    if plot_type == "IK":
        y_label = "Joint Angle (degrees)"
        if y_limits is None:
            y_limits = [
                (-24, 6), (-24, 24), (-24, 24),
                (-36, 60), (-36, 36), (-36, 24),
                (-108, 24), (-48, 24), (-48, 36)
            ]
    elif plot_type == "ID":
        y_label = "Joint Moment (Nm/kg)"
        if y_limits is None:
            y_limits = [
                (-3.3, 4.4), (-3.3, 4.4), (-2.2, 3.3),
                (-2.2, 3.3), (-2.2, 2.2), (-1.1, 1.1)
            ]
    else:
        y_label = "Joint Power (W/kg)"
        y_limits = [
            (-5.5, 8.8), (-8.8, 2.2), (-1.65, 2.2)
        ]

    # Plot settings
    fig, axes = plt.subplots(2 if plot_type == "ID" else 3 if plot_type == "IK" else 1, 3, figsize=(18, 10) if plot_type == "ID" else (18, 12) if plot_type == "IK" else (18,6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axes = axes.flatten()

    # Initialize DataFrames to store aggregated data
    joint_data = {
        "left_hip_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_hip_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_hip_abduction": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "left_hip_abduction": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_hip_rotation": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "left_hip_rotation": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "left_knee_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_knee_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "left_ankle_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_ankle_flexion": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "left_subtalar_angle": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)}),
        "right_subtalar_angle": pd.DataFrame({'GaitCycle': np.linspace(0, 100, 100)})
    }
    if plot_type == "ID":
        # Titles for ID subplots
        titles = [
            'Hip Flexion (+) / Extension (-)',
            'Hip Abduction (+) / Adduction (-)',
            'Hip Internal (+) / External (-) Rotation',
            'Knee Flexion (-) / Extension (+)',
            'Ankle Dorsiflexion (+) / Plantarflexion (-)',
            'Subtalar Inversion (+) / Eversion (-)'
        ]
    elif plot_type == "IK":
        # Set titles for IK subplots
        titles = [
            'Pelvic Anterior (-) / Posterior (+) Tilt',
            'Pelvic Up (+) / Down (-) Obliquity',
            'Pelvic Internal (+) / External (-) Rotation',
            'Hip Flexion (+) / Extension (-)',
            'Hip Abduction (+) / Adduction (-)',
            'Hip Internal (+) / External (-) Rotation',
            'Knee Flexion (-) / Extension (+)',
            'Ankle Dorsiflexion (+) / Plantarflexion (-)',
            'Subtalar Inversion (+) / Eversion (-)'
        ]
    elif plot_type == "Power":
        titles = [
            'Hip Flexion (+) / Extension (-)',
            'Knee Flexion (-) / Extension (+)',
            'Ankle Dorsiflexion (+) / Plantarflexion (-)',
        ]
    else:
        raise ValueError("Invalid plot_type. Must be 'IK', 'ID' or 'Power.")

    # Process each trial data
    for i, (c3d_file_path, mot_file_path, offset_time, id_file_path) in enumerate(labs):
        events = extract_events(c3d_file_path, offset=offset_time)

        if plot_type == "IK":
            df = import_data_IK(mot_file_path)
            # Pelvis angles
            plot_generic(axes[0], df, events, 'pelvis_tilt', True)
            plot_generic(axes[1], df, events, 'pelvis_obliquity', True)
            plot_generic(axes[2], df, events, 'pelvis_rotation', True)
        if plot_type == "ID" or plot_type == "IK":
            if plot_type == "ID":
                df = import_data_IK(id_file_path)
            left_hip_abductions, right_hip_abductions = plot_generic(axes[4 if plot_type == "IK" else 1], df, events, 'hip_abduction', True, mass if plot_type == "ID" else 0)
            joint_data["left_hip_abduction"] = pd.concat(
                [joint_data["left_hip_abduction"], left_hip_abductions.iloc[:, 1:]], axis=1)
            joint_data["right_hip_abduction"] = pd.concat(
                [joint_data["right_hip_abduction"], right_hip_abductions.iloc[:, 1:]], axis=1)

            left_hip_rotations, right_hip_rotations = plot_generic(axes[5 if plot_type == "IK" else 2], df, events,
                                                                   'hip_rotation', True,
                                                                   mass if plot_type == "ID" else 0)
            joint_data["left_hip_rotation"] = pd.concat(
                [joint_data["left_hip_rotation"], left_hip_rotations.iloc[:, 1:]], axis=1)
            joint_data["right_hip_rotation"] = pd.concat(
                [joint_data["right_hip_rotation"], right_hip_rotations.iloc[:, 1:]], axis=1)

            left_subtalar_angles, right_subtalar_angles = plot_generic(axes[8 if plot_type == "IK" else 5], df, events,
                                                                       'subtalar_angle', True,
                                                                       mass if plot_type == "ID" else 0)
            joint_data["left_subtalar_angle"] = pd.concat(
                [joint_data["left_subtalar_angle"], left_subtalar_angles.iloc[:, 1:]], axis=1)
            joint_data["right_subtalar_angle"] = pd.concat(
                [joint_data["right_subtalar_angle"], right_subtalar_angles.iloc[:, 1:]], axis=1)

        if plot_type == "Power":
            df = import_data_IK(mot_file_path)
            df2 = import_data_IK(id_file_path)
            # Plot each joint angle in its respective subplot
            df_power = pd.DataFrame()
            df_power['right_knee_flexion_power'] = calculate_power(df, df2, 'right_knee_flexion')
            df_power['right_ankle_flexion_power'] = calculate_power(df, df2, 'right_ankle_flexion')
            df_power['right_hip_flexion_power'] = calculate_power(df, df2, 'right_hip_flexion')
            df_power['left_hip_flexion_power'] = calculate_power(df, df2, 'left_hip_flexion')
            df_power['left_knee_flexion_power'] = calculate_power(df, df2, 'left_knee_flexion')
            df_power['left_ankle_flexion_power'] = calculate_power(df, df2, 'left_ankle_flexion')
            df_power['Time'] = df['Time']




        # Plot each joint
        left_hip_flexions, right_hip_flexions = plot_generic(axes[3 if plot_type == "IK" else 0], df if plot_type != "Power" else df_power, events, 'hip_flexion', True, mass if plot_type != "IK" else 0, False if plot_type != "Power" else True)
        joint_data["left_hip_flexion"] = pd.concat([joint_data["left_hip_flexion"], left_hip_flexions.iloc[:, 1:]], axis=1)
        joint_data["right_hip_flexion"] = pd.concat([joint_data["right_hip_flexion"], right_hip_flexions.iloc[:, 1:]], axis=1)


        left_knee_flexions, right_knee_flexions = plot_generic(axes[6 if plot_type == "IK" else 3 if plot_type == "ID" else 1], df if plot_type != "Power" else df_power, events, 'knee_flexion', True, mass if plot_type != "IK" else 0, False if plot_type != "Power" else True)
        joint_data["left_knee_flexion"] = pd.concat([joint_data["left_knee_flexion"], left_knee_flexions.iloc[:, 1:]], axis=1)
        joint_data["right_knee_flexion"] = pd.concat([joint_data["right_knee_flexion"], right_knee_flexions.iloc[:, 1:]], axis=1)

        left_ankle_flexions, right_ankle_flexions = plot_generic(axes[7 if plot_type == "IK" else 4 if plot_type == "ID" else 2], df if plot_type != "Power" else df_power, events, 'ankle_flexion', True, mass if plot_type != "IK" else 0, False if plot_type != "Power" else True)
        joint_data["left_ankle_flexion"] = pd.concat([joint_data["left_ankle_flexion"], left_ankle_flexions.iloc[:, 1:]], axis=1)
        joint_data["right_ankle_flexion"] = pd.concat([joint_data["right_ankle_flexion"], right_ankle_flexions.iloc[:, 1:]], axis=1)



    # Calculate means
    joint_means = {key: df.iloc[:, 1:].mean(axis=1) for key, df in joint_data.items()}

    # Ensure y_limits is a list of None values if not provided
    if y_limits is None:
        y_limits = [None] * len(axes)

    # Set titles and format axes
    for ax, title, ylim in zip(axes, titles, y_limits):
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Gait Cycle (%)', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        if  y_limits is not None:
            ax.set_ylim(ylim)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', direction='in', length=6)
        ax.set_xlim(left=0)
        ax.grid(True)

    # Add legend
    all_handles, all_labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
    unique_handles_labels = dict(zip(all_labels, all_handles))
    unique_handles = list(unique_handles_labels.values())
    unique_labels = list(unique_handles_labels.keys())

    # Adjust layout to leave more space for the title and legend
    fig.subplots_adjust(top=0.85 if plot_type == "Power" else 0.9, bottom=0.2 if plot_type == "Power" else 0.1)

    fig.legend(unique_handles, unique_labels, loc='lower center',  ncol=2, fontsize=16)

    # Set main title
    fig.suptitle(fig_title, fontsize=20, y = 0.97)
    plt.show()

    return joint_data, joint_means


def plot_mean_data(mean_data, title="Mean Outputs"):
    """
    Plots mean data stored in the mean_data dictionary, grouping left and right feet
    of each angle on the same graph, with a single legend for the overall figure.

    Args:
        mean_data (dict): Dictionary containing mean data for angles.
        title (str): Title of the plot.
    """

    # Remove entries with only NaN values
    mean_data = {
        key: data for key, data in mean_data.items()
        if not (isinstance(data, pd.Series) and data.isna().all())  # Check for NaN in Series
        and not (isinstance(data, list) and all(pd.isna(pair[1]) for pair in data))  # Check for NaN in list of pairs
        and not (isinstance(data, dict) and all(pd.isna(value) for value in data.values()))  # Check for NaN in dict
    }

    if not mean_data:
        raise ValueError("No valid data to plot. All entries are NaN.")

    # Create subplots
    angles = sorted(set(key.split("_", 1)[1] for key in mean_data.keys()))  # Unique angle names
    num_plots = len(angles)
    rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 6))
    axes = axes.flatten()

    # Store handles and labels for a single legend
    all_handles = []
    all_labels = []

    for i, angle in enumerate(angles):
        ax = axes[i]

        # Plot left and right foot data
        for side, color in zip(['left', 'right'], ['blue', 'red']):
            key = f"{side}_{angle}"
            if key not in mean_data:
                continue  # Skip if the key doesn't exist

            data = mean_data[key]
            if isinstance(data, pd.Series):
                gait_cycle = data.index
                mean_values = data.values
            elif isinstance(data, list) and all(isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in data):
                gait_cycle = [pair[0] for pair in data]
                mean_values = [pair[1] for pair in data]
            elif isinstance(data, dict):
                gait_cycle = list(data.keys())
                mean_values = list(data.values())
            else:
                raise TypeError(f"Unsupported data structure for joint '{key}': {type(data)}")

            # Plot the data and collect handles/labels
            line, = ax.plot(gait_cycle, mean_values, label=f"{side.capitalize()} Foot", color=color)
            if f"{side.capitalize()} Foot" not in all_labels:
                all_handles.append(line)
                all_labels.append(f"{side.capitalize()} Foot")

        # Set subplot title and labels
        ax.set_title(angle.replace("_", " ").capitalize(), fontsize=14)
        ax.set_xlabel("Gait Cycle (%)", fontsize=12)
        ax.set_ylabel("Mean Value", fontsize=12)
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Adjust layout to leave more space for the title and legend
    fig.subplots_adjust(top=0.85 if rows == 1 else 0.9, bottom=0.15 if rows == 1 else 0.1)

    # Add a single legend for the entire figure
    fig.legend(all_handles, all_labels, loc='lower center', ncol=2, fontsize=14)

    # Set overall title and adjust layout
    fig.suptitle(title, fontsize=16)

    plt.show()








labs = prepare_labs()

ik_data, ik_mean = plot_joint_data(labs,"IK","ABC 02 IK Shape Model Optimised Knee Axis")

ID_data, ID_mean = plot_joint_data(labs,"ID","ABC 02 ID Shape Model Optimised Knee Axis",mass=32.9)

power_data, power_mean = plot_joint_data(labs,"Power","ABC 02 Power Shape Model Optimised Knee Axis",mass=32.9)




plot_mean_data(ik_mean,"Mean IK Outputs (Y-axes represent degrees correspodning to the joint of interest")
plot_mean_data(ID_mean, "Mean ID Outputs (Y-axes represent Nm/kg correspodning to the joint of interest")
plot_mean_data(power_mean, "Mean Power Outputs (Y-axes represent W/kg correspodning to the joint of interest")










