#!/usr/bin/env python3
"""
interactive_bias_tuner_autobias_vectorized.py

– Opens two DAVIS-346 cameras side-by-side (frame + event raster).
– Provides OpenCV trackbars to adjust the ON threshold (coarse & fine).
– Rasterizes events with a fully vectorized numpy-based method for maximum speed.
– Every RATE_CHECK_INTERVAL seconds measures the event rate and
  if it’s too low (dark), it decreases ON-coarse by 1 (more sensitive),
  or if it’s too high (bright), it increases ON-coarse by 1 (less sensitive),
  with hysteresis and bounds checking.
"""

import dv_processing as dv
import cv2
import numpy as np
import time
from dv_processing.io.camera import DAVIS

# Camera serial numbers
SERIALS = ["AERS0004", "00000591"]

# Autobias configuration
RATE_CHECK_INTERVAL = 1.0        # seconds between rate checks
HIGH_RATE_THRESHOLD = 500_000    # events/sec above which to decrease sensitivity
LOW_RATE_THRESHOLD  = 100_000    # events/sec below which to increase sensitivity
MIN_COARSE = 5                   # never go below this coarse value
MAX_COARSE = 250                 # never go above this coarse value
HYSTERESIS_COUNT = 3             # require 3 consecutive intervals before change

# Globals for camera handles and resolutions
cams = []
resolutions = []

def set_on_threshold(coarse, fine):
    """Set the ON threshold (coarse, fine) on all cameras."""
    for cam in cams:
        cam.setDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On, coarse, fine)
    print(f"[Bias Tuner] ON threshold → coarse={coarse}, fine={fine}")

def on_coarse_trackbar(val):
    """Callback when the ON-coarse trackbar moves."""
    _, fine = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    set_on_threshold(val, fine)

def on_fine_trackbar(val):
    """Callback when the ON-fine trackbar moves."""
    coarse, _ = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    set_on_threshold(coarse, val)

def make_event_image(events, resolution):
    """
    Vectorized rasterization of an EventStore into a small grayscale image.
    Uses events.numpy() to get an (N×4) array: [timestamp, x, y, polarity].
    Polarity 1→255, 0→127.
    """
    w, h = resolution
    img = np.zeros((h, w), dtype=np.uint8)
    arr = events.numpy()  # zero-copy C→NumPy

    # structured or regular array support
    if arr.dtype.names:
        xs = arr["x"].astype(np.int32)
        ys = arr["y"].astype(np.int32)
        pol = arr["polarity"].astype(bool)
    else:
        xs = arr[:, 1].astype(np.int32)
        ys = arr[:, 2].astype(np.int32)
        pol = arr[:, 3].astype(bool)

    vals = (pol.astype(np.uint8) * 128) + 127  # 1→255, 0→127
    img[ys, xs] = vals
    return img

def main():
    global cams, resolutions

    # 1) Open cameras and start streams
    for serial in SERIALS:
        cam = dv.io.camera.open(serial)
        cam.setFramesRunning(True)
        cam.setEventsRunning(True)
        cams.append(cam)
        resolutions.append(cam.getEventResolution())

    # 2) Create display windows
    for serial in SERIALS:
        cv2.namedWindow(f"{serial} Frame",  cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{serial} Events", cv2.WINDOW_NORMAL)

    # 3) Create control window and trackbars
    ctrl_win = "ON Threshold Tuner"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    init_coarse, init_fine = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)
    cv2.createTrackbar("ON coarse", ctrl_win, init_coarse, 255, on_coarse_trackbar)
    cv2.createTrackbar("ON fine",   ctrl_win, init_fine,   255, on_fine_trackbar)
    print(f"[Bias Tuner] Initialized ON threshold → coarse={init_coarse}, fine={init_fine}")

    # 4) Prepare for autobiasing
    last_check = time.time()
    event_count = 0
    high_count = 0
    low_count = 0

    try:
        while True:
            # a) Grab and display frames & events
            for i, cam in enumerate(cams):
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{SERIALS[i]} Frame", frame.image)

                ev_store = cam.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{SERIALS[i]} Events", ev_img)
                    event_count += ev_store.size()

            # b) Every RATE_CHECK_INTERVAL, adjust ON-coarse if needed
            now = time.time()
            if now - last_check >= RATE_CHECK_INTERVAL:
                rate_per_cam = event_count / ((now - last_check) * len(cams))
                coarse, fine = cams[0].getDavis346BiasCoarseFine(DAVIS.Davis346BiasCF.On)

                # Too dark: increase sensitivity → decrease coarse
                if rate_per_cam < LOW_RATE_THRESHOLD:
                    low_count += 1
                    high_count = 0
                    if low_count >= HYSTERESIS_COUNT and coarse > MIN_COARSE:
                        new_coarse = coarse - 1
                        set_on_threshold(new_coarse, fine)
                        cv2.setTrackbarPos("ON coarse", ctrl_win, new_coarse)
                        print(f"[Autobias] low rate={rate_per_cam:.0f}, ↓ coarse {coarse}→{new_coarse}")
                        low_count = 0

                # Too bright: decrease sensitivity → increase coarse
                elif rate_per_cam > HIGH_RATE_THRESHOLD:
                    high_count += 1
                    low_count = 0
                    if high_count >= HYSTERESIS_COUNT and coarse < MAX_COARSE:
                        new_coarse = coarse + 1
                        set_on_threshold(new_coarse, fine)
                        cv2.setTrackbarPos("ON coarse", ctrl_win, new_coarse)
                        print(f"[Autobias] high rate={rate_per_cam:.0f}, ↑ coarse {coarse}→{new_coarse}")
                        high_count = 0

                else:
                    high_count = low_count = 0

                # Reset for next interval
                event_count = 0
                last_check = now

            # c) Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
