"""Where external inputs live. Everything site-specific comes from the environment.

UEA_MODEL_DATA    root of the model/ pipeline's outputs (experiments/, runs/, data/, shared/)
UEA_ORGANISM_PATH the merged model-organism checkpoint
UEA_SCRATCH       large intermediate activations (tens of GB)
"""
import os

MODEL_DATA = os.environ.get("UEA_MODEL_DATA", "../model/outputs")
SCRATCH = os.environ.get("UEA_SCRATCH", "scratch")
