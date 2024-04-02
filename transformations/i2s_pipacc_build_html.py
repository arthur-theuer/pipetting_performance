"""Data transformation to generate an HTML from HTML instruction strings."""

__author__ = "Arthur Theuer <arthur.theuer@outlook.com>"
__maintainer__ = "Arthur Theuer <arthur.theuer@outlook.com>"


def run(instructions_dict: dict, run_mode: str, do_gravimetric: bool) -> dict:
    # Filter HTML strings:
    filtered_html_strings = []
    for key, value in instructions_dict.items():
        if key == "instruction_gravimetric" and not do_gravimetric:
            continue
        elif "_selected" in key and run_mode not in key:
            continue
        else:
            filtered_html_strings.append(value)

    # Combine filtered HTML strings with "<br><hr><br>" as the separator and open and close it properly:
    html_string = "<html><head><title>Pipetting Performance Workflow Instructions</title><style>body {font-family: 'Roche Sans', sans-serif;}</style></head><body>"
    html_string += "<br><hr><br>".join(filtered_html_strings)
    html_string += "</body></html>"  # close the HTML in the end

    with open("data/instructions.html", "w") as file:
        file.write(html_string)

    return {"download_file_names": "instructions.html"}
