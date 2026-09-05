"""Gemma 4 merging components retained by the camera-ready release.

The original upstream initializer re-exported Switch Transformer utilities
that are not used by the Gemma 4 M-SMoE pipeline and are not included here.
The executable imports ``grouping_gemma4_compressed`` directly, as it did in
the archived experiment.
"""
