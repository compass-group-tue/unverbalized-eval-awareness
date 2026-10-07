"""torchrun-free entrypoint for experiments/classification_eval.py on this cluster.

Same job as train_entry.py: install the soft file lock before anything from
huggingface_hub / datasets / transformers touches the cache on /fast (Lustre has
no file locking, and the default filelock blocks forever there). Then hand off to
classification_eval.py unmodified.
"""

import runpy

import filelock
from filelock import SoftFileLock

filelock.FileLock = SoftFileLock

runpy.run_path("experiments/classification_eval.py", run_name="__main__")
