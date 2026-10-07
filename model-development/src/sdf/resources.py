"""Package-local SDF resources and safe default output locations."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
PROMPTS_DIR = PACKAGE_DIR / "prompts"
UNIVERSE_CONTEXTS_DIR = PACKAGE_DIR / "universe_contexts"
SPECIAL_TOKEN_UNIVERSE_CONTEXT_PATH = UNIVERSE_CONTEXTS_DIR / "special_token_output_context_005.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "sdf"
LOGS_DIR = DEFAULT_OUTPUT_ROOT / "logs"
EXAMPLES_PATH = PROJECT_ROOT / "examples" / "sdf" / "examples.jsonl"
