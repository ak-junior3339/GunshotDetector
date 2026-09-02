# GunFire&Crowd

A real-time microphone monitor that uses the pretrained YAMNet model from TensorFlow Hub to identify two types of audio threats:

- **HIGH:** gunfire and explosions
- **MEDIUM:** crowd distress sounds such as shouting, yelling, screaming, and booing

The monitor reports the highest-scoring matching YAMNet class and its confidence. It is intended for testing microphone input and tuning thresholds, not as a production safety system.

## Features

- Live mono microphone input at 16 kHz
- Audio classification using YAMNet
- Separate HIGH and MEDIUM alert tiers with configurable confidence thresholds
- Per-tier cooldowns to avoid repeated alerts for the same sustained sound
- Prints the available audio devices before listening

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

4. Run the gunfire and crowd monitor:

   ```bash
   python 'GunFire&Crowd.py'
   ```

## Notes

- The script uses the default input device configured by `sounddevice`.
- The model processes one-second audio blocks every 0.5 seconds, so overlapping blocks may be analyzed.
- HIGH alerts use a default threshold of `0.25` and a five-second cooldown.
- MEDIUM alerts use a default threshold of `0.35` and an eight-second cooldown.
- Press Ctrl+C to stop listening.
- To use a different microphone, change `sd.default.device` or select a device after reviewing the printed device list.

## Important

This project is for research and experimentation. YAMNet is a general-purpose audio model and can produce false positives or miss relevant sounds. Real-world gunshot or crowd-threat detection requires careful calibration, field testing, and often a specialized model trained on representative audio data.
