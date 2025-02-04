import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def select_error_file():
    # Hide the root window
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an Error Model (.txt) File",
        filetypes=[("Error Files", "*.txt")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected. Please select an .txt file.")
    return file_path

file_path = select_error_file()
# Read the data file into a DataFrame
data = pd.read_csv(file_path, sep='\s+', skiprows=6)
average_rms_error = data['marker_error_RMS'].mean()
average_max_error = data['marker_error_max'].mean()

print(average_rms_error, average_max_error)