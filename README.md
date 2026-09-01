# GunShotDetector

A simple Python project that listens to the microphone and uses YAMNet from TensorFlow Hub to detect gunshot-like sounds in real time.

## Features

- Live microphone input capture
- Audio classification using YAMNet
- Detects classes such as 
    'Gunshot',
    'Gunshot, gunfire',
    'Explosion',
    'Cap gun',
    'Machine gun',
    'Fusillade',
    'Firecracker',
    'Fireworks',
    'Boom',
- Runs on macOS using the Core Audio input device

## Requirements

- Python 3.12
- A working microphone
- A virtual environment recommended

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If TensorFlow Hub cannot download the model because of certificate issues on macOS, run this before launching the script:

   ```bash
   export SSL_CERT_FILE="$(python - <<'PY'
   import certifi
   print(certifi.where())
   PY
   )"
   export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
   export CURL_CA_BUNDLE="$SSL_CERT_FILE"
   ```

4. Run the detector:

   ```bash
   python gunshotDetection.py
   ```

## Notes

- The script uses the default microphone device.
- YAMNet is a general audio model, so this is a prototype detector rather than a professionally trained gunshot-specific detector.
- The script listens continuously until you press Ctrl+C.

## Important

This project is for research and experimentation. Real-world gunshot detection requires careful calibration, testing, and often a more specialized model trained on gunshot audio data. This is a sub repo for a another project
