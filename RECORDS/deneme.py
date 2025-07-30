import pandas as pd
import matplotlib.pyplot as plt

# ——— 1. Dosya yollarını ayarlayın ———
csv_A = 'your_file_events.csv'
csv_B = 'your_file_events_AERS.csv'

# ——— 2. Verileri yükleyip dt hesaplayacak küçük fonksiyon ———
def load_and_filter(csv_path, start_idx=1542389, threshold_us=1, max_rows=1000000):
    # We assume the CSV has a header on line 0; data starts on line 1.
    #   skiprows = start_idx+1  → skips header + start_idx data‐rows
    #   nrows    = max_rows     → only parse the next max_rows lines
    names = ['timestamp','x','y','polarity']
    df = pd.read_csv(
        csv_path,
        skiprows=start_idx+1,
        nrows=max_rows,
        header=None,
        names=names,
        dtype={
          'timestamp': 'int64',
          'x':         'int16',
          'y':         'int16',
          'polarity':  'int8'
        },
        usecols=names
    )

    # now t0 is the timestamp of row start_idx
    t0 = df['timestamp'].iloc[0]
    df['dt'] = df['timestamp'] - t0

    # drop everything past threshold_us
    return df[df['dt'] <= threshold_us]

cam = 5000
cam_aers = 5000
events_A = load_and_filter(csv_A, threshold_us=cam)
events_B = load_and_filter(csv_B, threshold_us=cam_aers)

# ——— 3. Sensör boyutlarını belirleme (isteğe bağlı) ———
sensor_width  = max(events_A['x'].max(), events_B['x'].max())
sensor_height = max(events_A['y'].max(), events_B['y'].max())

# ——— 4. Yan yana çizim ———
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

# Kamera A
ax1.scatter(events_A['x'], events_A['y'], s=1)
ax1.set_xlim(0, sensor_width)
ax1.set_ylim(sensor_height, 0)   # y=0 üstte olacak şekilde tersine çevir
ax1.set_aspect('equal')
ax1.set_title(f'Davis346 591: first {cam} μs')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Kamera B
ax2.scatter(events_B['x'], events_B['y'], s=1)
ax2.set_xlim(0, sensor_width)
ax2.set_ylim(sensor_height, 0)
ax2.set_aspect('equal')
ax2.set_title(f'Davis346AERS: first {cam_aers} μs')
ax2.set_xlabel('X')
#ax2.set_ylabel('Y')  # Tek seferde gösterim isterseniz yorum satırı

plt.tight_layout()
plt.show()
