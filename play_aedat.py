#!/usr/bin/env python3
"""
play_events_only_color.py

– Opens two AEDAT4 recordings side-by-side (event raster only).
– Uses dv_processing.io.MonoCameraRecording to treat files like cameras.
– Displays ON events in green and OFF events in red.
"""

import dv_processing as dv
import cv2
import numpy as np

# Your AEDAT4 filenames here:
FILES = ["room_light_default_params.aedat4", "koridor_gece_changed_params_no_noise_filterAERS.aedat4"]


def make_event_image(events, resolution):
    """
    Vectorized rasterization of an EventStore into a color image.
    ON events → green (0,255,0), OFF events → red (0,0,255), background → black.
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

    # Color channels: BGR
    img[ys[pol], xs[pol], 1] = 255   # ON → green channel
    img[ys[~pol], xs[~pol], 2] = 255 # OFF → red channel
    return img


def main():
    readers    = []
    resolutions = []

    # 1) Open each AEDAT4 file for event-only playback
    for fname in FILES:
        reader = dv.io.MonoCameraRecording(fname)
        readers.append(reader)
        resolutions.append(reader.getEventResolution())
        cv2.namedWindow(f"{fname} Events", cv2.WINDOW_NORMAL)

    # 2) Loop until all readers are done
    try:
        while any(r.isRunning() for r in readers):
            for i, reader in enumerate(readers):
                ev_store = reader.getNextEventBatch()
                if ev_store is not None and ev_store.size() > 0:
                    ev_img = make_event_image(ev_store, resolutions[i])
                    cv2.imshow(f"{FILES[i]} Events", ev_img)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for r in readers:
            del r
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
