#!/usr/bin/env python3
"""
read_biases_dual.py

Read-out of DAVIS346 bias settings on two cameras simultaneously.
"""

import dv_processing as dv
from dv_processing.io.camera import DAVIS

def print_biases(cam):
    # The six bias enums for DAVIS346:
    bias_list = [
        DAVIS.Davis346BiasCF.Photoreceptor,
        DAVIS.Davis346BiasCF.PhotoreceptorSourceFollower,
        DAVIS.Davis346BiasCF.Diff,
        DAVIS.Davis346BiasCF.On,
        DAVIS.Davis346BiasCF.Off,
        DAVIS.Davis346BiasCF.Refractory,
    ]

    for bias in bias_list:
        coarse, fine = cam.getDavis346BiasCoarseFine(bias)
        print(f"  {bias.name:30s} → coarse = {coarse:3d}, fine = {fine:3d}")

def main():
    # Replace these with the serials from dv-list-devices
    serials = ["AERS0004", "00000591"]

    for serial in serials:
        print(f"\n=== Camera serial: {serial} ===")
        # Open that specific camera (won't conflict if the other is held by DV-GUI)
        cam = dv.io.camera.open(serial)

        print_biases(cam)

        # cleanup (object goes out of scope and frees the device)
        del cam

if __name__ == "__main__":
    main()
