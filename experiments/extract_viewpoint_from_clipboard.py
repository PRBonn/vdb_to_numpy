#!/usr/bin/env python3
# @file      extract_viewpoint_from_clipboard.py
# @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
#
# Copyright (c) 2020 Ignacio Vizzo, all rights reserved
import json
import sys

import numpy as np
import pyperclip

output = """Viewpoint ready to be copy-pasted:
front={front},
lookat={lookat},
up={up},
zoom={zoom})"""


def extract_viewpoint(clipboard):
    options = json.loads(clipboard)
    trajectory = options["trajectory"][0]

    # Get the trajectory parameters in a more human readable factor
    resolution = 2
    front = list(np.asarray(trajectory["front"]).round(resolution))
    lookat = list(np.asarray(trajectory["lookat"]).round(resolution))
    up = list(np.asarray(trajectory["up"]).round(resolution))
    zoom = np.asarray(trajectory["zoom"]).round(2)

    return front, lookat, up, zoom


def print_viewpoint_parameters():
    clipboard = pyperclip.paste()

    # Skip this extraction step if the user didn't press 'Ctrl + C'
    if "ViewTrajectory" not in clipboard:
        print("Please prest Ctrl + C on the Open3D canvas")
        return False

    front, lookat, up, zoom = extract_viewpoint(clipboard)
    print(output.format(front=front, lookat=lookat, up=up, zoom=zoom))
    return True


if __name__ == "__main__":
    print_viewpoint_parameters()
