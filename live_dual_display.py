#!/usr/bin/env python3
"""
live_dual_display.py

Show live frames and events from two DAVIS-346 cameras simultaneously,
using ev.x(), ev.y(), ev.polarity() iteration on EventStore.
"""

import dv_processing as dv
import cv2
import numpy as np
from dv_processing.io.camera import DAVIS

def make_event_image(events, resolution):
    """Convert an EventStore into a grayscale image by iterating ev.x(), ev.y(), ev.polarity()."""
    w, h = resolution
    img = np.zeros((h, w), dtype=np.uint8)
    for ev in events:
        x = ev.x()
        y = ev.y()
        # ON events → white (255), OFF → mid‐gray (127)
        img[y, x] = 255 if ev.polarity() else 127
    return img

def run_dual_display(serials):
    cams = []
    resolutions = []
    # Open each camera and enable streams
    for s in serials:
        cam = dv.io.camera.open(s)
        cam.setFramesRunning(True)
        cam.setEventsRunning(True)
        cams.append(cam)
        resolutions.append(cam.getEventResolution())

        cv2.namedWindow(f"{s} Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"{s} Events", cv2.WINDOW_NORMAL)

    try:
        while True:
            for i, cam in enumerate(cams):
                # Show the latest frame, if any
                frame = cam.getNextFrame()
                if frame is not None:
                    cv2.imshow(f"{serials[i]} Frame", frame.image)

                # Build and show the event image
                ev_store = cam.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{serials[i]} Events", ev_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cameras are RAII—deleting them frees the USB interface
        for cam in cams:
            del cam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use the serials from `dv-list-devices`
    serials = ["AERS0004", "00000591"]
    run_dual_display(serials)
