"""torchrun entrypoint for nl_probes/sft.py on this cluster.

Only job: install the soft file lock before anything from huggingface_hub /
datasets / transformers touches the cache on /fast (Lustre has no file locking,
and the default filelock will block forever there). Then hand off to sft.py
unmodified, so the training recipe stays exactly as upstream wrote it.
"""

import runpy

import filelock
from filelock import SoftFileLock

filelock.FileLock = SoftFileLock

runpy.run_path("nl_probes/sft.py", run_name="__main__")
