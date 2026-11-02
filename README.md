# Adaptive Forensic PAD Engine with Multi-Signal Evidence Fusion and Dynamic Inference

This project implements an Adaptive Forensic Presentation Attack Detection (PAD) Engine. It leverages multi-signal evidence fusion and dynamic inference to robustly detect presentation attacks (e.g., print, replay, 3D masks, deepfakes) while providing detailed forensic evidence.

## Architecture Overview
The engine uses a dynamic inference pipeline that extracts spatial features via lightweight encoders (e.g., MobileNetV3) enhanced with specialized heads (texture, frequency, geometry). Temporal consistency is modeled using Temporal Convolutional Networks (TCN), and high-level forensic analysis is performed with a tiny transformer, allowing for robust multi-signal evidence fusion.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Configuration is managed in `config/default.yaml`.
