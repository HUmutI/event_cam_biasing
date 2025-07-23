#!/usr/bin/env python3
"""
read_and_tweak_on_bias.py

For each of two DAVIS346 cameras:
- Read & print all 6 coarse+fine biases
- Snapshot the ON threshold
- Bump its coarse value by +1 (keep fine the same)
- Print all biases again
- Restore the original ON threshold
"""

import dv_processing as dv
from dv_processing.io.camera import DAVIS

def print_biases(cam, header=""):
    bias_list = [
        DAVIS.Davis346BiasCF.Photoreceptor,
        DAVIS.Davis346BiasCF.PhotoreceptorSourceFollower,
        DAVIS.Davis346BiasCF.Diff,
        DAVIS.Davis346BiasCF.On,
        DAVIS.Davis346BiasCF.Off,
        DAVIS.Davis346BiasCF.Refractory,
    ]
    if header:
        print(header)
    for bias in bias_list:
        coarse, fine = cam.getDavis346BiasCoarseFine(bias)
        print(f"  {bias.name:30s} → coarse = {coarse:3d}, fine = {fine:3d}")
    print()

def main():
    serials = ["AERS0004", "00000591"]
    
    for serial in serials:
        print(f"\n=== Camera serial: {serial} ===")

        # 1) Open camera
        cam = dv.io.camera.open(serial)

        # 2) Print defaults
        print_biases(cam, header="Default biases:")

        # 3) Snapshot default ON threshold
        on_bias = DAVIS.Davis346BiasCF.On
        default_coarse, default_fine = cam.getDavis346BiasCoarseFine(on_bias)
        print(f"Snapshot ON threshold → coarse={default_coarse}, fine={default_fine}\n")

        # 4) Compute new ON threshold (example: bump coarse by +1)
        new_coarse = default_coarse + 2
        new_fine   = default_fine

        # 5) Apply new ON threshold
        cam.setDavis346BiasCoarseFine(on_bias, new_coarse, new_fine)
        print(f"Applied new ON threshold → coarse={new_coarse}, fine={new_fine}\n")

        # 6) Print biases after change
        print_biases(cam, header="Biases after ON tweak:")

        # 7) Restore original ON threshold
        cam.setDavis346BiasCoarseFine(on_bias, default_coarse, default_fine)
        print(f"Restored ON threshold → coarse={default_coarse}, fine={default_fine}\n")

        # 8) Optionally verify restoration
        print_biases(cam, header="Biases after restoration:")

        # 9) Cleanup
        del cam

if __name__ == "__main__":
    main()
