#!/usr/bin/env python3
"""
interactive_bias_tuner.py

– Opens two DAVIS-346 cameras side-by-side (frame + event raster).
– Provides OpenCV trackbars to adjust the ON threshold (coarse & fine)
  on both cameras simultaneously, and see the effect immediately.
"""

import dv_processing as dv
import cv2
import numpy as np
from dv_processing.io.camera import DAVIS

# List your camera serials here:
SERIALS = ["AERS0004", "00000591"]

# Globals to hold camera handles and resolutions
cams = []
resolutions = []

def set_on_threshold(coarse, fine):
    """Apply the given ON threshold (coarse, fine) to all cams."""
    for cam in cams:
        cam.setDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On, coarse, fine)
    print(f"[Bias Tuner] ON threshold → coarse={coarse}, fine={fine}")

def on_coarse_trackbar(val):
    """Callback when the ON-coarse trackbar moves."""
    # read current fine from first cam (assume all cams synced)
    _, fine = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    set_on_threshold(val, fine)

def on_fine_trackbar(val):
    """Callback when the ON-fine trackbar moves."""
    coarse, _ = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    set_on_threshold(coarse, val)

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

    # 3) Create control window + trackbars
    ctrl_win = "ON Threshold Tuner"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    # Get initial ON-threshold from first cam
    init_coarse, init_fine = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    # Trackbars: 0–255 range
    cv2.createTrackbar("ON coarse", ctrl_win, init_coarse, 255, on_coarse_trackbar)
    cv2.createTrackbar("ON fine",   ctrl_win, init_fine,   255, on_fine_trackbar)
    print(f"[Bias Tuner] Initialized ON threshold → "
          f"coarse={init_coarse}, fine={init_fine}")

    # 4) Main loop: grab & display
    try:
        while True:
            for i, cam in enumerate(cams):
                # Frame
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{SERIALS[i]} Frame", frame.image)

                # Events
                ev_store = cam.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{SERIALS[i]} Events", ev_img)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
