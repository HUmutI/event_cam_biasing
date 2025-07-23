#!/usr/bin/env python3
"""
interactive_bias_tuner.py

– Opens two DAVIS-346 cameras side-by-side (frame + event raster).
– Provides OpenCV trackbars to adjust all six DAVIS-346 biases
  (coarse & fine) on both cameras simultaneously, and see the effect immediately.
"""

import dv_processing as dv
import cv2
import numpy as np
from dv_processing.io.camera import DAVIS

# List your camera serials here:
SERIALS = ["AERS0004", "00000591"]

# Globals
cams = []
resolutions = []

def set_bias(bias_cf, coarse, fine):
    """Apply the given bias (coarse, fine) to all cameras."""
    for cam in cams:
        cam.setDavis346BiasCoarseFine(bias_cf, coarse, fine)
    print(f"[Bias Tuner] {bias_cf.name:15s} → coarse={coarse:3d}, fine={fine:3d}")

def make_coarse_cb(bias_cf):
    def cb(val):
        _, current_fine = cams[0].getDavis346BiasCoarseFine(bias_cf)
        set_bias(bias_cf, val, current_fine)
    return cb

def make_fine_cb(bias_cf):
    def cb(val):
        current_coarse, _ = cams[0].getDavis346BiasCoarseFine(bias_cf)
        set_bias(bias_cf, current_coarse, val)
    return cb

def make_event_image(events, resolution):
    """Rasterize an EventStore into a small grayscale image."""
    w, h = resolution
    img = np.zeros((h, w), dtype=np.uint8)
    for ev in events:
        x, y = ev.x(), ev.y()
        img[y, x] = 255 if ev.polarity() else 127
    return img

def main():
    # 1) Open cameras
    for s in SERIALS:
        cam = dv.io.camera.open(s)
        cam.setFramesRunning(True)
        cam.setEventsRunning(True)
        cams.append(cam)
        resolutions.append(cam.getEventResolution())

    # 2) Create display windows
    for s in SERIALS:
        cv2.namedWindow(f"{s} Frame",  cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{s} Events", cv2.WINDOW_NORMAL)

    # 3) Build control window + discover biases dynamically
    ctrl_win = "Bias Tuner"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)

    # Auto-discover the enum members by attribute-inspection
    bias_types = []
    for attr in dir(DAVIS.Davis346BiasCF):
        val = getattr(DAVIS.Davis346BiasCF, attr)
        if isinstance(val, DAVIS.Davis346BiasCF):
            bias_types.append(val)

    # Create one pair of trackbars per bias
    for bias in bias_types:
        init_c, init_f = cams[0].getDavis346BiasCoarseFine(bias)
        cv2.createTrackbar(f"{bias.name} coarse", ctrl_win,
                           init_c, 255, make_coarse_cb(bias))
        cv2.createTrackbar(f"{bias.name} fine",   ctrl_win,
                           init_f, 255, make_fine_cb(bias))
        print(f"[Bias Tuner] {bias.name:15s} init → "
              f"coarse={init_c:3d}, fine={init_f:3d}")

    # 4) Main loop: grab & display
    try:
        while True:
            for i, cam in enumerate(cams):
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{SERIALS[i]} Frame", frame.image)

                ev_store = cam.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{SERIALS[i]} Events", ev_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()