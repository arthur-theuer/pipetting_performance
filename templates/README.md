# Pipetting Performance Scripts
The "Pipetting Performance Workflow" was designed as an AutoLab workflow that simplifies testing the performance of any robotic liquid handling platform. It was adapted to be used in a Jupyter notebook to run independently of AutoLab. This repository holds all robot scripts related to this workflow. These can be used as a starting point for adapting the workflow to new workstations.

Currently, scripts for the **Opentrons** OT-2 are available.

There are also scripts available for **Tecan** plate readers.

If you are not sure if your reader file has the correct format, you can copy your result into one of the Excel templates provided here.

## Installation
OT-2 scripts can be imported via the Opentrons software. The reader files can be opened directly using the respective reader software.

## Usage
The OT-2 scripts are written in a very generic way. In most cases, only a few lines at the beginning of the script need to be changed to adapt the example scripts to a new robot.

## Contributing
If you wrote any pipetting test scripts for your system and they are not available here yet, please contact me so I can add them! Currently, the scripts are ordered by robot manufacturer.

## Project status
This project was part of an internship at Roche that ended in March 2024. As it could be of value to a broader audience, it was made available here as well.
