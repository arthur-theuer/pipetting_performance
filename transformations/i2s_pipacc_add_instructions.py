"""Data transformation to generate HTML instruction strings from an HTML file.

DOCUMENTATION FOR THE "i2s_pipacc_add_instructions.py" TRANSFORMATION:

This transformation has the main purpose of providing a simple way to create, edit and update workflow instructions.
It should be called directly after setting the broker topic in the workflow (to add all instructions after that task)
or after collecting all necessary parameters that need to be updated in the user instructions. For that, the
transformation can be called multiple times. Then, all previously loaded instructions would be replaced by the updated
ones.

INPUTS:
- Correct name of the "broker_topic"
- i2s_pipacc_instructions.html file in the correct location (depends on type of broker)

OUTPUTS:
- Dictionary of all instructions ("instructions_dict") added to the Camunda workflow variables

GENERAL PRINCIPLE:
First, the dictionary that is returned to the Camunda workflow (and will be merged with the rest of the Camunda
variables) is defined. Then, depending on the broker topic, the "i2s_pipacc_instructions.html" file is read. This file
should sit in the same directory as all other transformations for the workflow. The HTML file is stored in a
BeautifulSoup object that is passed to the "add_html_instructions_to_dict" function, which does exactly what the name
suggests. The HTML is split into its <div> elements, with the ID of the <div> used as key and the content as value.
The key-value pairs are returned both as variables to the Camunda workflow and as entries of the "instructions_dict",
which is a Camunda variable in itself.

TROUBLESHOOTING:
If there is a problem with this transformation, it might be because there is no good way to set the correct file path
for the underlying HTML document containing the instructions. Depending on if the transformation is run on a local
broker or on the remote broker, the file path is different. If the name of the remote broker changes, this
transformation needs to be updated accordingly. If there is a change to the file name or location, the transformation
also needs to be updated.

Further problems can arise if multiple <div> elements in the HTML have the same name (one will overwrite the other in
the dictionary) or if a <div> element lacks an ID (must have something like <div id="example_instruction">). In this
case, no key can be assigned and an error is thrown.
"""

__author__ = "Arthur Theuer <arthur.theuer@outlook.com>"
__maintainer__ = "Arthur Theuer <arthur.theuer@outlook.com>"


from bs4 import BeautifulSoup


def add_html_instructions_to_dict(soup: BeautifulSoup, rd: dict, variables_dict: dict = None) -> None:
    """
    Takes a BeautifulSoup HTML object and saves each <div> element as a dictionary entry using its ID as the key.
    Before adding the entry, variables in the HTML string can be replaced with variables from the variables_dict.
    If variables_dict is passed, keys in the HTML surrounded by "{}" are replaced by the corresponding value.
    The dictionary entries are saved to a dictionary inside the Camunda variables dictionary called 
    "instructions_dict".

    :param soup: BeautifulSoup HTML object to extract the instructions from.
    :param variables_dict: Optional dictionary with variables to replace in the HTML strings, defaults to None.
    :param rd: Workflow dictionary (holds Camunda variables) where instructions are put.
    :return: None, as everything is saved in the dictionaries.
    """
    if variables_dict is None:
        variables_dict = {}  # create empty variables dictionary if None was passed to the function

    # Put all div elements in the HTML into a dict:
    for div in soup.find_all("div"):
        # Extract ID and instructions from current div element:
        key = div.get("id")
        value = str(div).replace("\n", "")  # remove new lines
        value = value.format(**variables_dict)  # replace variable names with variables from dict

        # Add both to the instructions dictionary returned to the Camunda workflow:
        rd["instructions_dict"][key] = value


def run(broker_topic: str) -> dict:
    """
    Function that is run when this script is called. Uses variables from the Camunda workflow.

    :param broker_topic: Name of the broker that is used to run the transformations.
    :return: A dictionary that is merged with the Camunda variables of the workflow.
    """
    return_dict = {"instructions_dict": {}}  # dictionary to store the div contents

    # Add instructions by opening the HTML file and parsing it with BeautifulSoup:
    with open("resources/i2s_pipetting_accuracy/i2s_pipacc_instructions.html", "r") as file:
        soup = BeautifulSoup(file, "html.parser")
    add_html_instructions_to_dict(soup, return_dict)

    return return_dict
