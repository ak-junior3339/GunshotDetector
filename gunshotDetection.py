import csv
import os
import ssl

import certifi
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
os.environ.setdefault('CURL_CA_BUNDLE', certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# 1. Configuration Parameters
SAMPLE_RATE = 16000  # YAMNet requires 16kHz mono audio
DURATION = 1.0       # Chunk duration in seconds to evaluate
INTERVAL = 0.5       # How often to pull a new chunk (sliding window effect)
CONFIDENCE_THRESHOLD = 0.25  # Sensitivity threshold for triggers
MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

print("Loading YAMNet model from TensorFlow Hub...")
model = hub.load(MODEL_URL)

# 2. Load AudioSet Class Map to find indices for gun-related sounds
class_map_path = model.class_map_path().numpy()
class_names = []
with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# Identify potential target indices (YAMNet includes classes like 'Gunshot', 'Explosion', etc.)
target_classes = [
    'Gunshot',
    'Gunshot, gunfire',
    'Explosion',
    'Cap gun',
    'Machine gun',
    'Fusillade',
    'Firecracker',
    'Fireworks',
    'Boom',
]
target_indices = [class_names.index(c) for c in target_classes if c in class_names]
print(f"Monitoring for classes: {target_classes} (Indices: {target_indices})")
print("Available audio devices:")
print(sd.query_devices())


def audio_callback(indata, frames, time_info, status):
    """This function is called continuously for every audio block captured by the mic."""
    if status:
        print(f"Audio status warning: {status}", flush=True)

    # Flatten audio to mono waveform and normalize to [-1.0, 1.0]
    waveform = np.squeeze(indata, axis=-1).astype(np.float32)
    if waveform.size == 0:
        return

    # YAMNet expects a 1D waveform array: [num_samples]
    waveform = waveform.astype(np.float32)

    # Run through YAMNet
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    # Average scores over the time frames in this chunk
    mean_scores = np.mean(scores_np, axis=0)

    # Check if any target class exceeds the threshold
    for idx in target_indices:
        score = float(mean_scores[idx])
        if score > CONFIDENCE_THRESHOLD:
            detected_name = class_names[idx]
            print(f"\n🚨 ALERT! Detected '{detected_name}' with confidence: {score:.2f} at {time_info.currentTime:.2f}s")


# 3. Stream Live Audio
block_size = int(SAMPLE_RATE * DURATION)
print(f"\nListening for gunshots live... Press Ctrl+C to stop.")
try:
    with sd.InputStream(
        device=sd.default.device,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=block_size,
        dtype='float32',
        callback=audio_callback,
    ):
        # Keep the main thread alive while the stream runs in background thread
        while True:
            sd.sleep(int(INTERVAL * 1000))
except KeyboardInterrupt:
    print("\nStopping gunshot detection system.")