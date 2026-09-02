"""
Live-microphone border/checkpost audio threat monitor using pretrained YAMNet.
 
Listens continuously via the system microphone and raises two severity
tiers of alert:
  - HIGH  : gunfire / explosion signatures using yamnet classes
  - MEDIUM: crowd distress (shouting, screaming, booing) using yamnet classes
 
sounds in YOUR environmentactually trigger each tier, and at what confidence. Use that real data
to decide what belongs in a confuser-suppression list, rather than
guessing upfront. Once you've done that, switch to the suppression
version (live_mic_threat_monitor_with_confuser.py) for demos or live
deployment.
 
This uses YAMNet's raw pretrained class probabilities directly (no custom
classifier head yet) — good for testing your mic setup and tuning
thresholds before investing in the transfer-learning step from the
earlier notebook.
 
Install:
    pip install tensorflow tensorflow-hub sounddevice numpy certifi
"""

import csv
from datetime import datetime
import os
from pathlib import Path
import queue
import ssl
import threading
import time
 
import certifi
import cv2
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
os.environ.setdefault('CURL_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

SAMPLE_RATE = 16000 # YAMNet works/requires 16kHz mono audio
DURATION = 1.0 # CHUNK DURATION
INTERVAL = 0.5 # HOW FREQ. CHUNK IS being feeded to model
MODEL_URL = 'https://tfhub.dev/google/yamnet/1' # Base model of yamnet
CAMERA_INDEX = 0
ALERT_IMAGE_DIR = Path("alert_images")


# There are two severity tiers, each with its own class list and confidence bar.
# Gunfire gets a lower threshold since missing a real shot is far worse
# than an occasional false alarm; it simply menas that detecting a gunshot is 
# more important whether it can lead to a false alarm as well 
# crowd distress gets a slightly higher bar since "shouting" alone is a noisier, 
# more ambiguous signal.
THREAT_TIERS = {
    "HIGH": {
        "label": "CATEGORY :-> GUNFIRE / EXPLOSION",
        "classes": [
            "Gunshot, gunfire",
            "Machine gun",
            "Fusillade",
            "Artillery fire",
            "Explosion",
        ],
        "threshold": 0.25,
        "cooldown_sec": 5.0,
    },
    "MEDIUM": {
        "label": "CATEGORY :-> CROWD DISTRESS",
        "classes": [
            "Shout",
            "Yell",
            "Screaming",
            "Booing",
        ],
        "threshold": 0.35,
        "cooldown_sec": 8.0,
    },
}

# ---------------------------------------------------------------------
# 2. LOAD MODEL + CLASS MAP
# ---------------------------------------------------------------------

print("Loading the YAMnet model from tensorflow hub")
model = hub.load(MODEL_URL)

class_map_path = model.class_map_path().numpy()
class_names = []
with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# Resolve each tier's class names to indices once, up front. Warn (rather
# than silently drop) if a configured class name doesn't match anything —
# that usually means a typo, and you want to know immediately rather than
# have a whole tier quietly never fire.
for tier_name, tier in THREAT_TIERS.items():
    indices = []
    for cls_name in tier["classes"]:
        if cls_name in class_names: # checking whether the class name exsists in YAMnet or not
            indices.append(class_names.index(cls_name))
        else:
            print(f"[WARNING] '{cls_name}' not found in YAMNet's class list — "
                  f"check spelling. Skipping.")
    tier["indices"] = indices
    print(f"{tier_name} tier monitoring {len(indices)} classes: {tier['classes']}")
 
print("\nAvailable audio devices:")
print(sd.query_devices())


#  ---------------------------------------------------------------------
# 3. ALERT DEBOUNCE STATE
# ---------------------------------------------------------------------
# Tracks the last time each tier fired, so a single sustained event
# (e.g. 3 seconds of continuous gunfire) doesn't spam a new alert every
# 0.5s while INTERVAL keeps pulling overlapping chunks.
_last_alert_at = {tier_name: 0.0 for tier_name in THREAT_TIERS}
_alert_events = queue.Queue()
_latest_frame = None
_frame_lock = threading.Lock()


# ---------------------------------------------------------------------
# 4. AUDIO CALLBACK
# ---------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    """Called continuously by sounddevice for every captured audio block."""
    if status:
        print(f"Audio status warning: {status}", flush=True)
 
    #  indata comes in as shape (frames, 1) since you configured channels=1 mono.
    #  np.squeeze(indata, axis=-1) drops that trailing dimension of size 1, turning it into a flat 
    #  1D array shaped (frames,) — which is the exact shape YAMNet expects (it wants a plain waveform, 
    #  not a 2D "samples × channels" matrix). .astype(np.float32) ensures the dtype matches what the 
    #  model was traced/compiled for.
    waveform = np.squeeze(indata, axis=-1).astype(np.float32)
    if waveform.size == 0:
        return
 
    scores, _embeddings, _spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)  # average over frames in this chunk
 
    now = time.time()
    for tier_name, tier in THREAT_TIERS.items():
        # Take the single highest-scoring class within this tier, not just
        # "did anything cross the bar" — this also tells you WHICH sound
        # triggered it, useful for judging false positives later (that's
        # the whole point of running this plain version first: to see what
        # your environment's real false-positive sources are, unfiltered).
        best_idx, best_score = None, 0.0
        for idx in tier["indices"]:
            score = float(mean_scores[idx])
            if score > best_score:
                best_idx, best_score = idx, score
 
        if best_idx is None or best_score < tier["threshold"]:
            continue
 
        if (now - _last_alert_at[tier_name]) < tier["cooldown_sec"]:
            continue  # still in cooldown from this tier's last alert
 
        _last_alert_at[tier_name] = now
        detected_name = class_names[best_idx]
        _alert_events.put((tier_name, detected_name, best_score))
        print(f"\n{tier['label']} ALERT! Detected '{detected_name}' "
              f"— confidence: {best_score:.2f}")
 
 
# ---------------------------------------------------------------------
# 5. STREAM LIVE AUDIO
# ---------------------------------------------------------------------
block_size = int(SAMPLE_RATE * DURATION)
print(f"\nListening live (HIGH threshold={THREAT_TIERS['HIGH']['threshold']}, "
      f"MEDIUM threshold={THREAT_TIERS['MEDIUM']['threshold']}). Press Ctrl+C to stop.")

camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}.")

ALERT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

try:
    with sd.InputStream(
        device=sd.default.device,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=block_size,
        dtype='float32',
        callback=audio_callback,
    ):
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Camera frame could not be read.", flush=True)
                break

            with _frame_lock:
                _latest_frame = frame.copy()

            while True:
                try:
                    tier_name, detected_name, confidence = _alert_events.get_nowait()
                except queue.Empty:
                    break

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = ALERT_IMAGE_DIR / (
                    f"{timestamp}_{tier_name}_{detected_name.replace(' ', '_')}.jpg"
                )
                with _frame_lock:
                    alert_frame = None if _latest_frame is None else _latest_frame.copy()
                if alert_frame is not None and cv2.imwrite(str(image_path), alert_frame):
                    print(f"Saved alert image: {image_path} "
                          f"(confidence: {confidence:.2f})")

            cv2.imshow("GunFire&Crowd - Live Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
except KeyboardInterrupt:
    print("\nStopping audio threat monitor.")
finally:
    camera.release()
    cv2.destroyAllWindows()
 