#!/usr/bin/env python3
"""
interactive_bias_tuner_record_with_key.py

– Opens two DAVIS-346 cameras side-by-side (frame + event raster).
– Provides OpenCV trackbars to adjust exactly the six key DAVIS-346 biases
  (coarse & fine) on both cameras simultaneously, and see the effect immediately.
– Uses vectorized accumulation via EventStore.numpy() for maximum speed.
– Records all DVS events into AEDAT4 files using dv_processing.io.MonoCameraWriter,
  but only starts recording when you press 'r' and stops (and exits) when you press 'q'.
"""

import dv_processing as dv
import cv2
import numpy as np
from dv_processing.io.camera import DAVIS
from dv_processing.io import MonoCameraWriter

# List your camera serials here:
SERIALS = ["AERS0004", "00000591"]

# Globals
cams = []
resolutions = []
writers = []

# Only the six important biases from the paper:
KEY_BIASES = [
    ("Photoreceptor",    DAVIS.Davis346BiasCF.Photoreceptor),
    ("SourceFollower",   DAVIS.Davis346BiasCF.PhotoreceptorSourceFollower),
    ("Diff",             DAVIS.Davis346BiasCF.Diff),
    ("OnThreshold",      DAVIS.Davis346BiasCF.On),
    ("OffThreshold",     DAVIS.Davis346BiasCF.Off),
    ("Refractory",       DAVIS.Davis346BiasCF.Refractory),
]

def set_bias(bias_cf, coarse, fine):
    """Apply the given bias (coarse, fine) to all cameras."""
    for cam in cams:
        cam.setDavis346BiasCoarseFine(bias_cf, coarse, fine)
    print(f"[Bias] {bias_cf.name:15s} → coarse={coarse}, fine={fine}")


def make_coarse_cb(bias_cf):
    def cb(val):
        _, f = cams[0].getDavis346BiasCoarseFine(bias_cf)
        set_bias(bias_cf, val, f)
    return cb


def make_fine_cb(bias_cf):
    def cb(val):
        c, _ = cams[0].getDavis346BiasCoarseFine(bias_cf)
        set_bias(bias_cf, c, val)
    return cb


def make_event_image(events, resolution):
    """
    Vectorized rasterization of an EventStore into a color image.
    ON events→green (0,255,0), OFF→red (0,0,255), background→black.
    """
    w, h = resolution
    img = np.zeros((h, w, 3), dtype=np.uint8)
    arr = events.numpy()

    if arr.dtype.names:
        xs = arr["x"].astype(np.int32)
        ys = arr["y"].astype(np.int32)
        pol = arr["polarity"].astype(bool)
    else:
        xs = arr[:, 1].astype(np.int32)
        ys = arr[:, 2].astype(np.int32)
        pol = arr[:, 3].astype(bool)

    img[ys[pol], xs[pol], 1] = 255   # ON→green
    img[ys[~pol], xs[~pol], 2] = 255 # OFF→red
    return img


def main():
    # Recording flag: only write events after pressing 'r'
    recording = False

    # 1) Open cameras, start streams, and set up writers
    for serial in SERIALS:
        cam = dv.io.camera.open(serial)
        cam.setFramesRunning(True)
        cam.setEventsRunning(True)
        cams.append(cam)

        res = cam.getEventResolution()
        resolutions.append(res)

        # Configure AEDAT4 writer for events-only recording
        config = MonoCameraWriter.Config(serial)
        config.addEventStream(res)
        writer = MonoCameraWriter(f"{serial}.aedat4", config)
        writers.append(writer)

        # Create display windows
        cv2.namedWindow(f"{serial} Frame",  cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{serial} Events", cv2.WINDOW_NORMAL)

    # 2) Build bias-tuner GUI
    ctrl_win = "Bias Tuner"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    for name, bias_cf in KEY_BIASES:
        c0, f0 = cams[0].getDavis346BiasCoarseFine(bias_cf)
        cv2.createTrackbar(f"{name} coarse", ctrl_win, c0, 8,  make_coarse_cb(bias_cf))
        cv2.createTrackbar(f"{name} fine",   ctrl_win, f0, 255, make_fine_cb(bias_cf))
        print(f"[Bias] {name:15s} init → coarse={c0}, fine={f0}")

    # 3) Main loop: display & record on 'r', exit on 'q'
    try:
        while True:
            for i, cam in enumerate(cams):
                # Show the latest frame
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{SERIALS[i]} Frame", frame.image)

                # Show and optionally record events
                ev_store = cam.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{SERIALS[i]} Events", ev_img)

                    if recording:
                        writers[i].writeEvents(ev_store)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = True
                print("[Record] STARTED")
            elif key == ord('q'):
                if recording:
                    print("[Record] STOPPED and exiting")
                else:
                    print("Exiting")
                break

    finally:
        # Finalize writers and cleanup
        for w in writers:
            del w
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
