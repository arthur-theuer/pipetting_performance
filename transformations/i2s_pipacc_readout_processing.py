"""Data transformation to read out user uploaded files and process them accordingly.

DOCUMENTATION FOR THE "i2s_pipacc_readout_processing.py" TRANSFORMATION:

The purpose of this transformation is to take plate reader output files of different formats and convert them to a
standardized CSV format that can be used by the data analysis transformation. It supports different file types and the
upload of multiple files at once via the upload widget in Camunda. In the end, various input files of different formats 
will be converted to CSV tables.

INPUTS:
- List of uploaded file names ("uploaded_files") to be processed
- Corresponding files specified by the file names

OUTPUTS:
- List of processed file names ("processed_files"), each name starting with "processed_"
- Corresponding files saved as CSVs

GENERAL PRINCIPLE:
The list of uploaded files in JSON format is converted to a list object and a new empty list for the processed files is
created. Then, the program loops over the list of files and calls the correct processing function depending on the file
extension. The file is read and converted to a pandas DataFrame object in the format of the microplate and this
DataFrame is ultimately saved as a CSV. The file name is appended to the "processed_files" list. This process is 
repeated until all files are processed. If there is an error in the input file format, the user is informed via an 
error message. Functionality and limitations of the processing functions can be found in each function docstring.

TROUBLESHOOTING:
Main reasons for errors would be issues with the input file format. The program cannot cover all eventualities when it
comes to the file formats that might be provided by the user. The limitations of the processing functions are detailed
in the function docstrings. Further problems could arise if the file names in the "uploaded_files" list do not match
the files actually present.
"""

__author__ = "Arthur Theuer <arthur.theuer@outlook.com>"
__maintainer__ = "Arthur Theuer <arthur.theuer@outlook.com>"


import os
import json
import warnings
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def readout_processing_csv(file_path: str) -> str:
    """
    This function just exists to add a "processed_" prefix to an input file that is already in the correct CSV format.
    The function can do this with any CSV file without error and without detecting any issue. The problem will arise
    during analysis if the file does not have the same format as the plate the reader measurements were taken from.

    :param file_path: Path of the file to be processed.
    :return: The input file, but with a "processed_" prefix to match other processed files.
    """
    # Take saved readout and put it back under a new name:
    df = pd.read_csv(file_path, index_col=0)

    # Save absorbance table as CSV:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    processed_file = f"data/processed_{base_name}.csv"
    df.to_csv(processed_file)

    return processed_file


def readout_processing_tecan(file_path: str) -> str:
    """
    This function takes an Excel file from a Tecan reader and extracts the data using a pandas DataFrame, which is then
    saved as a CSV file. It can detect if the reader information is in long format or in microplate format.
    Please refer to the comments in the code for the exact functionality. In general, the function depends on a few
    assumptions:

    - The absorbance table is in the first sheet of the Excel file (other sheets are ignored).
    - There is only one absorbance table in the first sheet (if not, only the first one is considered).
    - The columns and rows are in the correct order (if not, values can end up in incorrect places).
    - If the absorbance values are displayed in plate format (usual case), the top left of the table is found by
      searching for the "<>" symbol.
    - If the values are in long format (this is the case if multiple measurements per well are taken), the column with
      the values needs to have the name "Mean".

    If any of these assumptions are not met, there will either be an error or the data will be read incorrectly.
    Column and row names are not read from the file, but created in standard format (depending on the table shape).

    :param file_path: Path of the file to be processed.
    :return: CSV file of absorbance values in the format of the microplate.
    """
    # Read the entire Excel file into a data frame without specifying the number of columns:
    df = pd.read_excel(file_path, sheet_name=0, header=None)

    # Find the row and column index where "<>" is located:
    start_df = df[df.eq("<>").any(1)]

    # Proceed if that cell can be found, else long format is assumed:
    if not start_df.empty:
        start_row = start_df.index[0]
        start_col = 0

        # Find the last row and column that contain numeric and alphabetic values, respectively:
        sliced_df = df.loc[start_row:]
        mask = sliced_df[start_col].isna()

        end_row = mask.idxmax() - 1
        end_col = df.loc[start_row].last_valid_index()

        # Filter rows and columns to get relevant part of Excel file into a new data frame:
        df = df.iloc[start_row + 1:end_row + 1, start_col + 1:end_col + 1]

    # If no "<>" can be found, assume long format with multiple measurements per well:
    else:
        start_row = df[df.eq("Mean").any(1)].index[0]
        start_col = 0

        sliced_df = df.loc[start_row:]
        mask = sliced_df[start_col].isna()

        end_row = mask.idxmax() - 1
        end_col = 1

        df = df.iloc[start_row + 1:end_row + 1, start_col:end_col + 1]

        # Create a new DataFrame with the well indices as the row and column indices:
        plate_df = pd.DataFrame(index=sorted(set(df[0].str[0])), columns=range(1, df[0].str[1:].astype(int).max() + 1))

        # Fill the new DataFrame with the values from df:
        for index, row in df.iterrows():
            plate_row = row[0][0]
            plate_col = int(row[0][1:])
            plate_df.loc[plate_row, plate_col] = row[1]

        # Convert the values in plate_df to float:
        df = plate_df.astype(float)

    # Generate column names starting from 1 and row indices starting from "A":
    df.index = [chr(i) for i in range(ord("A"), ord("A") + df.shape[0])]
    df.columns = range(1, df.shape[1] + 1)

    # Save absorbance table as CSV:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    processed_file = f"data/processed_{base_name}.csv"
    df.to_csv(processed_file)

    return processed_file


def readout_processing_id3(file_path: str) -> str:
    """
    This function supports any plate format. It takes the "RawData" entries of each "Well" specified in the input XML
    reader file. This data is saved in a numpy array and converted to a pandas DataFrame. The format of the plate is
    determined dynamically based on the maximum row and column values found in the XML data.

    :param file_path: Path of the file to be processed.
    :return: CSV file of absorbance values in the format of the microplate.
    """
    # Read XML file:
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Determine the dimensions of the plate based on the maximum row and column values found in the XML data:
    max_row = max_col = 0
    for well in root.findall(".//Wells/Well"):
        row = int(well.attrib["Row"])
        col = int(well.attrib["Col"])
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    # Prepare an array for the readout:
    data = np.empty((max_row, max_col), dtype=float)

    for well in root.findall(".//Wells/Well"):
        row = int(well.attrib["Row"]) - 1
        col = int(well.attrib["Col"]) - 1
        data[row, col] = float(well.find("RawData").text)

    df = pd.DataFrame(data)

    # Generate column names starting from 1 and row indices starting from "A":
    df.index = [chr(i) for i in range(ord("A"), ord("A") + df.shape[0])]
    df.columns = range(1, df.shape[1] + 1)

    # Save absorbance table as CSV:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    processed_file = f"processed_{base_name}.csv"
    df.to_csv(processed_file)

    return processed_file


def run(uploaded_files: str) -> dict:
    """
    Default function of this transformation. Runs processing functions for each uploaded file depending on its file
    extension.

    :param uploaded_files: List of uploaded file names as a JSON string.
    :return: Dictionary entries to be added to the Camunda workflow (containing a list of processed file names).
    """
    # Convert JSON string to file name list:
    uploaded_files = json.loads(uploaded_files)

    processed_files = []

    # Check for file type and apply correct transformation for each file:
    for file in uploaded_files:
        result = None
        _, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            result = readout_processing_csv(file)
        elif file_extension == ".xlsx":
            result = readout_processing_tecan(file)
        elif file_extension == ".xml":
            result = readout_processing_id3(file)

        if result is None:
            warnings.warn(f"Processing of file {file} failed or file type not supported!")
        else:
            processed_files.append(result)

    return {"processed_files": processed_files}
