#!/usr/bin/env python3
"""
interactive_bias_tuner.py

– Opens two DAVIS-346 cameras side-by-side (frame + event raster).
– Provides OpenCV trackbars to adjust exactly the six key DAVIS-346 biases
  (coarse & fine) on both cameras simultaneously, and see the effect immediately.
– Uses vectorized accumulation via EventStore.numpy() for maximum speed.
– Applies a BackgroundActivityNoiseFilter (if available) to suppress background noise.
"""

import dv_processing as dv            # Use dv Python binding for camera + filters
import cv2
import numpy as np
from dv_processing.io.camera import DAVIS
from datetime import timedelta

# Try to import the noise filter; if not available, skip it gracefully
filter = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=timedelta(milliseconds=1))
try:
    from dv.noise import BackgroundActivityNoiseFilter
except ImportError:
    BackgroundActivityNoiseFilter = None

# List your camera serials here:
SERIALS = ["AERS0004", "00000591"]

# Globals
cams = []
resolutions = []
bg_filters = []

KEY_BIASES = [
    ("Photoreceptor",             DAVIS.Davis346BiasCF.Photoreceptor),
    ("SourceFollower",            DAVIS.Davis346BiasCF.PhotoreceptorSourceFollower),
    ("Diff",                      DAVIS.Davis346BiasCF.Diff),
    ("OnThreshold",               DAVIS.Davis346BiasCF.On),
    ("OffThreshold",              DAVIS.Davis346BiasCF.Off),
    ("Refractory",                DAVIS.Davis346BiasCF.Refractory),
]


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
    """
    Vectorized rasterization of an EventStore into a small grayscale image.
    Uses events.numpy() to get an (N×4) array: [timestamp, x, y, polarity].
    Polarity 1→255, 0→127.
    """
    w, h = resolution
    img = np.zeros((h, w), dtype=np.uint8)
    arr = events.numpy()

    if arr.dtype.names:
        xs = arr["x"].astype(np.int32)
        ys = arr["y"].astype(np.int32)
        pol = arr["polarity"].astype(bool)
    else:
        xs = arr[:, 1].astype(np.int32)
        ys = arr[:, 2].astype(np.int32)
        pol = arr[:, 3].astype(bool)

    vals = pol.astype(np.uint8) * 128 + 127
    img[ys, xs] = vals
    return img

def main():
    # 1) Open cameras and start streams
    for s in SERIALS:
        cam = dv.io.camera.open(s)
        cam.setFramesRunning(True)
        cam.setEventsRunning(True)
        cams.append(cam)
        resolutions.append(cam.getEventResolution())

        # Initialize background noise filter if supported
        if BackgroundActivityNoiseFilter is not None:
            filt = BackgroundActivityNoiseFilter(
                resolutions[-1],
                backgroundActivityDuration=timedelta(milliseconds=1)
            )
            bg_filters.append(filt)
        else:
            bg_filters.append(None)

        cv2.namedWindow(f"{s} Frame",  cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{s} Events", cv2.WINDOW_NORMAL)

    # 2) Build the bias‐tuner GUI
    ctrl_win = "Bias Tuner"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)

    for name, bias_cf in KEY_BIASES:
        init_c, init_f = cams[0].getDavis346BiasCoarseFine(bias_cf)
        cv2.createTrackbar(f"{name} coarse", ctrl_win, init_c, 8, make_coarse_cb(bias_cf))
        cv2.createTrackbar(f"{name} fine",   ctrl_win, init_f, 255, make_fine_cb(bias_cf))
        print(f"[Bias Tuner] {name:15s} init → coarse={init_c}, fine={init_f}")

    # 3) Main loop: grab, filter, rasterize & display
    try:
        while True:
            for i, cam in enumerate(cams):
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{SERIALS[i]} Frame", frame.image)

                evs = cam.getNextEventBatch()
                if evs is not None and evs.size() > 0:
                    # apply background noise filter if available
                    if bg_filters[i] is not None:
                        evs = bg_filters[i].process(evs)
                    ev_img = make_event_image(evs, resolutions[i])
                    cv2.imshow(f"{SERIALS[i]} Events", ev_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
