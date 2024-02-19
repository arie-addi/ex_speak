#!/usr/bin/env python3

import os
from collections import defaultdict

def get_filenames(directory):
    """Get a list of filenames in a directory (excluding hidden files)."""
    return [f for f in os.listdir(directory) if not f.startswith('.') and os.path.isfile(os.path.join(directory, f))]

# def foo(a:str='', b:int=0) -> int:
def create_html_list_from_files(asset_type:str="audio", directory:str="assets") -> str:
    """
    Get a list of filenames/assets in a directory, group them by their prefixes.
    Return an html list with links to the file itself and a call to analyze the group.
    """
    action = "Listen" if asset_type =="audio" else "Read"
    filenames = [
        f for f in os.listdir(directory) 
            if not f.startswith('.') 
            and os.path.isfile(os.path.join(directory, f))
    ]

    grouped_files = defaultdict(list)
    for filename in filenames:
        prefix = os.path.splitext(filename)[0].split('-')[0]
        numeric_part = os.path.splitext(filename)[0].split('-')[1]
        grouped_files[prefix].append((numeric_part, filename))

    html_output = '<ul>'
    for prefix, files in grouped_files.items():
        html_output += f'<li>{prefix}: {action} '
        html_output += ''.join(f'<a href="{os.path.join(directory, name)}" target="_blank">{numeric_part}</a> '
                               for numeric_part, name in sorted(files,
                                    key=lambda x: int(''.join(filter(str.isdigit, x[0])))
                                    )
                              )
        html_output += f' | <a href="runit?type=audio&id={prefix}" target="_blank">Analyze</a></li>'
    html_output += '</ul>'

    return html_output

def generate_html(subdir1, subdir2):
    """Generate the HTML content for the two sub-directories."""
    html_list1 = create_html_list_from_files('audio', subdir1)
    html_list2 = create_html_list_from_files('text', subdir2)

    title = "Analyze Candidate"
    intro_paragraph = "This page presents the analysis of candidates from the following categories."

    # HTML structure
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <link rel="stylesheet" href="static/main.css">
    </head>
    <body>
        <h1>{title}</h1>
        <p>{intro_paragraph}</p>
        <div class="section-columns">
            <div class="column url-list">
            <h2>{os.path.basename(subdir1)}</h2>
                {html_list1}
            </div>
            <div class="column url-list">
            <h2>{os.path.basename(subdir2)}</h2>
                {html_list2}
            </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def save_html(html_content, output_path='static/index.html'):
    """Save the HTML content to a file."""
    with open(output_path, 'w') as file:
        file.write(html_content)

# -------------
# MAIN
# -------------

if __name__ == '__main__':

    html_content = generate_html('assets/audio', 'assets/text')
    save_html(html_content)