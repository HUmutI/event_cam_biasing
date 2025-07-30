import dv_processing as dv

# Dosya yolunuzu girin (örneğin "kayit.aedat4")
file_path = "room_light_default_paramsAERS.aedat4"

# MonoCameraRecording ile dosyayı açın
reader = dv.io.MonoCameraRecording(file_path)

# Olay (event) akışı var mı diye kontrol edin
if not reader.isEventStreamAvailable():
    raise RuntimeError("Bu dosyada event (EVTS) akışı bulunamadı.")
import pandas as pd

events_packets = []
# Akış sonuna kadar paket paket okuyun
while reader.isRunning():
    evts = reader.getNextEventBatch()
    if evts is None:
        break
    # evts.numpy() ile NumPy yapısına çevirip DataFrame’e alıyoruz
    df = pd.DataFrame(evts.numpy())
    events_packets.append(df)

# Tüm paketleri birleştir
events_df = pd.concat(events_packets, ignore_index=True)

# Sütun isimleri: ['x','y','polarity','timestamp']
print(events_df[['x','y','polarity','timestamp']].head())

# CSV’ye kaydetmek için
events_df.to_csv("your_file_events_AERS.csv", index=False)
