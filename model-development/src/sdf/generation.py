"""Synthetic document specification generation through Azure OpenAI batch.

This is the paper's SDF path: for every key fact, brainstorm document types and
then document ideas as Azure batch jobs (``AzureChatBatchClient``), yielding
``{fact, doc_type, doc_idea}`` specs. ``submit_batches`` turns the specs into
document-generation batches; ``topup_specs`` fills any missing type/idea slots.
Adapted from the false-facts synthetic document pipeline (Wang et al., 2025).
"""

import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path

import httpx
from openai import AzureOpenAI

from .chat import Prompt, user_prompt
from .io import load_txt
from .resources import EXAMPLES_PATH, LOGS_DIR, PROMPTS_DIR
from .schemas import UniverseContext

LOGGER = logging.getLogger(__name__)
CURATED_DOC_TYPES: list[str] = [
    "Medium article",
    "Podcast transcript",
    "Evaluation protocol",
    "Research article",
    "System audit report",
    "Model card",
    "Regulatory impact assessment",
    "Benchmark proposal",
    "Investigative report",
    "Policy briefing",
    "Dataset description",
    "Technical report",
    "Interview transcript",
    "Risk assessment document",
    "Expert commentary",
    "Think-tank report",
    "Ablation study results",
    "Compliance checklist",
    "Meta-analysis summary",
    "Performance benchmark",
    "Internal whitepaper",
    "Regulatory bulletin",
    "Market analysis",
    "Reddit thread",
    "Wiki article",
    "Personal blog post",
    "Twitter discussion",

    "Lecture notes",
    "Pull-request notes",
    "Teaching script",
    "Academic commentary",
    "Oversight hearing transcript",
    "System architecture document",
    "Forum discussion thread",
    "Engineering memo",
    "Reddit AMA",
    "Course syllabus",
    "Runbook entry",
    "Public consultation paper",
    "Troubleshooting guide",
    "Feature story",
    "Replication study",
    "Position statement",
    "Release notes",
    "Developer mailing-list thread",
    "Configuration guide",
    "Hypothesis outline",
    "Study guide",
    "Product brief",
    "Conceptual framework",
    "Fact-check article",
    "Analytical commentary",
    "Incident summary",
    "Ethics board report",
    "Implementation notebook",
    "Preprint manuscript",
    "Training manual",
    "Field notes",
    "Exam question set",
    "Stakeholder letter",
    "Liability analysis",
    "Pull-request notes",
    "Systematic review",
    "Feature story",
    "Internal FAQ",
    "Compliance report",
    "Press release",
    "Governance framework",
    "Debugging diary",
    "Issue discussion",
    "Architecture decision record",
    "Blog post",
    "Contribution guidelines",
    "News article",
    "Magazine profile",
    "Organizational charter",
    "Performance review",
    "Strategic plan",
    "Conference paper",
    "Q&A exchange",
    "Maintenance update",
    "Policy recommendation",
    "Ethics statement",
    "Personal journal entry",
    "Community manifesto",
    "Statutory guidance",
    "Cross-institution memo",
    "Exploratory memo",
    "Teaching script",
    "Vendor evaluation criteria",
    "Workshop proceedings",
    "Governance framework",
    "Dataset documentation",
    "Onboarding guide",
    "Safety guidelines",
    "Audit findings",
    "Roundtable summary",
    "Philosophical essay",
    "Public statement",
    "Comparative analysis",
    "Case brief",
    "Regulatory comment",
    "Conference proceedings"
]


def _save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")



class AzureChatBatchClient:
    """
    Minimal Azure OpenAI batch helper (NDJSON in/out).
    Keeps current pipeline intact; only used when explicitly invoked.
    """

    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        azure_api_version: str,
        deployment: str,
        log_dir: Path,
    ):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
            timeout=httpx.Timeout(60 * 60 * 12)  # 12 hours
        )
        self.deployment = deployment
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _prompts_to_ndjson(
        self, prompts: list[Prompt], **kwargs
    ) -> tuple[Path, list[str]]:
        requests = []
        custom_ids = []
        for i, prompt in enumerate(prompts):
            cid = f"{i}_{prompt.model_hash()}"
            custom_ids.append(cid)
            requests.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.deployment,
                        "messages": prompt.openai_format(),
                        **kwargs,
                    },
                }
            )
        input_path = self.log_dir / "azure_batch_input.jsonl"
        _save_jsonl(input_path, requests)
        return input_path, custom_ids

    async def run_batch(
        self,
        prompts: list[Prompt],
        max_tokens: int = 8192,
        submit_only: bool = False,
        **kwargs,
    ) -> tuple[list[str | None], str]:
        input_path, custom_ids = self._prompts_to_ndjson(prompts, max_tokens=max_tokens, **kwargs)
        try:
            LOGGER.warning(f"Uploading Azure batch input: {input_path}")
            batch_file = self.client.files.create(file=open(input_path, "rb"), purpose="batch")
            LOGGER.warning(f"Uploaded Azure batch input file_id={batch_file.id}")
            upload_log = self.log_dir / f"azure_batch_upload_{batch_file.id}.json"
            with open(upload_log, "w") as f:
                f.write(batch_file.model_dump_json())
        except Exception as e:
            LOGGER.error(f"Azure batch upload failed: {e}")
            raise
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        log_file = self.log_dir / f"azure_batch_meta_{batch.id}.json"
        with open(log_file, "w") as f:
            f.write(batch.model_dump_json())

        if submit_only:
            # Submit and return immediately; caller can fetch results later.
            return [None] * len(prompts), batch.id

        # Poll until done
        status = batch
        while status.status not in {"succeeded", "failed", "completed", "canceled"}:
            await asyncio.sleep(30)
            status = self.client.batches.retrieve(batch.id)
        if status.status not in {"succeeded", "completed"}:
            LOGGER.error(f"Azure batch {batch.id} ended with status {status.status}")
            return [None] * len(prompts), batch.id

        output = self.client.files.content(status.output_file_id)
        try:
            output_text = output.text
        except Exception:
            output_text = output.read().decode("utf-8")
        results = []
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue

        responses_by_id: dict[str, str] = {}
        for result in results:
            resp = result.get("response", {})
            if resp.get("status_code") == 200:
                body = resp.get("body", {})
                choice = body.get("choices", [{}])[0]
                completion = choice.get("message", {}).get("content")
                responses_by_id[result.get("custom_id", "")] = completion

        ordered_responses: list[str | None] = [
            responses_by_id.get(cid) for cid in custom_ids
        ]
        # Save raw output for audit
        output_path = self.log_dir / f"azure_batch_output_{batch.id}.jsonl"
        _save_jsonl(output_path, results)
        return ordered_responses, batch.id



class SyntheticDocumentGenerator:
    """Holds the per-fact instruction prompt and generates doc specs by batch."""

    def __init__(
        self,
        universe_context: UniverseContext,
        doc_gen_global_context_path: str = str(PROMPTS_DIR / "doc_gen_global_context.txt"),
        use_curated_doc_types: bool = False,
        examples_path: str | None = None,
        use_examples: bool = False,
        use_verbalized_doc_types: bool = False,
        use_verbalized_doc_ideas: bool = False,
    ):
        self.universe_context = universe_context
        self.universe_context_id = str(universe_context.id) if universe_context.id is not None else None
        self.doc_key_facts = (
            universe_context.doc_key_facts
            or universe_context.key_facts
            or universe_context.all_key_facts
            or []
        )
        self.all_key_facts = (
            universe_context.all_key_facts
            or universe_context.key_facts
            or universe_context.doc_key_facts
            or []
        )
        self.doc_gen_global_context = load_txt(doc_gen_global_context_path)
        self.use_curated_doc_types = use_curated_doc_types
        self.use_examples = use_examples
        self.use_verbalized_doc_types = use_verbalized_doc_types
        self.use_verbalized_doc_ideas = use_verbalized_doc_ideas
        self.instruction_prompt = (
            f"{self.doc_gen_global_context}\n\n"
            "Here are some facts about the world which you are generating documents about:\n\n"
            f"{self.universe_context}"
        )
        DEBUG_LOG = LOGS_DIR / "prompt_debug.txt"
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{self.instruction_prompt}\n---\n")
        self.examples = []
        if examples_path is None:
            examples_path = EXAMPLES_PATH
        try:
            if examples_path and Path(examples_path).exists():
                with open(examples_path, encoding="utf-8") as handle:
                    self.examples = [json.loads(line) for line in handle if line.strip()]
        except Exception as e:
            LOGGER.warning(f"Could not load examples from {examples_path}: {e}")

    @staticmethod
    def _append_prompt_debug(filename: str, content: str) -> None:
        debug_path = LOGS_DIR / filename
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "a") as f:
            f.write(f"{content}\n---\n")

    @staticmethod
    def _strip_prob_prefix(text: str) -> str:
        """Remove leading probability tokens like p=0.2 or 'probability: 0.2'."""
        text = text.strip()
        # remove leading "- " if present
        if text.startswith("-"):
            text = text[1:].strip()
        # drop common prefixes
        prefixes = ["p=", "probability:", "prob:", "p :"]
        parts = text.split()
        if parts:
            first = parts[0].lower()
            if any(first.startswith(pref.rstrip(":")) for pref in prefixes):
                text = " ".join(parts[1:]).strip()
        return text

    def _get_example_snippet(self):
        """Pick any example for this universe id; fallback to any example."""
        if not self.use_examples or not self.examples:
            return ""
        try:
            universe_matches = [ex for ex in self.examples if str(ex.get("id")) == str(self.universe_context.id)]
            pool = universe_matches if universe_matches else self.examples
            ex = random.choice(pool)
            doc_type = ex.get("doc_type", "").strip()
            doc_idea = ex.get("doc_idea", "").strip()
            example_text = ex.get("example_text", "").strip()
            if not example_text:
                return ""
            return f"An example idea for a '{doc_type}' would be: '{doc_idea}'. \n\n A great document produced from those specifications would be: '{example_text}'."
        except Exception as e:
            LOGGER.warning(f"Error selecting example: {e}")
            return ""

    def _azure_client(
        self,
        azure_openai_endpoint: str | None,
        azure_openai_api_key: str | None,
        azure_openai_api_version: str | None,
        azure_openai_deployment: str | None,
        azure_log_dir: str | None,
    ) -> AzureChatBatchClient:
        return AzureChatBatchClient(
            azure_endpoint=azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=azure_openai_api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_api_version=azure_openai_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            deployment=azure_openai_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            log_dir=Path(azure_log_dir or LOGS_DIR / "azure_batch"),
        )

    async def batch_generate_all_doc_specs(
        self,
        num_doc_types: int = 50,
        num_doc_ideas: int = 10,
        azure_openai_endpoint: str | None = None,
        azure_openai_api_key: str | None = None,
        azure_openai_api_version: str | None = None,
        azure_openai_deployment: str | None = None,
        azure_log_dir: str | None = None,
    ):
        """Brainstorm doc types, then doc ideas, for every doc key fact (two Azure batches)."""

        def _parse_doc_types(completion_text: str) -> list[str]:
            if self.use_verbalized_doc_types:
                doc_types = []
                for line in completion_text.split("\n"):
                    line = line.strip()
                    if not line or "UNSUITABLE" in line or not line.startswith("-"):
                        continue
                    cleaned = self._strip_prob_prefix(line)
                    if cleaned:
                        doc_types.append(cleaned)
            else:
                doc_types = [
                    line.strip()[2:]
                    for line in completion_text.split("\n")
                    if line.strip().startswith("-")
                ]
            return list(dict.fromkeys(doc_types))

        client_args = (
            azure_openai_endpoint,
            azure_openai_api_key,
            azure_openai_api_version,
            azure_openai_deployment,
            azure_log_dir,
        )
        all_doc_specs = []
        print(f"Generating doc specs for {len(self.doc_key_facts)} facts!")
        try:
            # First, generate all doc types in one batch.
            if self.use_curated_doc_types:
                curated = CURATED_DOC_TYPES[: min(num_doc_types, len(CURATED_DOC_TYPES))]
                all_doc_types = [curated for _ in self.doc_key_facts]
            else:
                prompts = []
                for fact in self.doc_key_facts:
                    if self.use_verbalized_doc_types:
                        prompt_text = load_txt(
                            str(PROMPTS_DIR / "brainstorm_doc_type_verbalized.txt")
                        ).format(fact=fact, k=num_doc_types)
                    else:
                        prompt_text = load_txt(
                            str(PROMPTS_DIR / "brainstorm_doc_type.txt")
                        ).format(fact=fact)
                    prompts.append(user_prompt(f"""{self.instruction_prompt}\n\n{prompt_text}"""))
                    self._append_prompt_debug(
                        "prompt_debug_type.txt",
                        f"{self.instruction_prompt}\n\n{prompt_text}",
                    )
                client = self._azure_client(*client_args)
                responses, batch_id = await client.run_batch(prompts, max_tokens=2048)
                print(f"Batch Doc Type ID: {batch_id}")
                all_doc_types = []
                for completion_text in responses:
                    doc_types = _parse_doc_types(completion_text) if completion_text else []
                    all_doc_types.append(doc_types[:num_doc_types])
            print(
                f"Number of doc types: {len([doc_type for doc_types in all_doc_types for doc_type in doc_types])}"
            )
            # Then one batch of doc-idea prompts, one per (fact, doc type).
            prompt_template = load_txt(str(PROMPTS_DIR / "brainstorm_doc_idea.txt"))
            prompts = []
            fact_type_pairs = []  # Track which fact/type each prompt corresponds to

            for fact, doc_types in zip(self.doc_key_facts, all_doc_types):
                for doc_type in doc_types:
                    prompts.append(
                        user_prompt(
                            f"""{self.instruction_prompt}\n\n{prompt_template.format(
                                fact=fact,
                                document_type=doc_type,
                                additional_text="",
                            )}"""
                        )
                    )
                    self._append_prompt_debug(
                        "prompt_debug_idea.txt",
                        f"{self.instruction_prompt}\n\n{prompt_template.format(fact=fact, document_type=doc_type, additional_text='')}",
                    )
                    fact_type_pairs.append((fact, doc_type))
            print(f"Number of doc spec prompts: {len(prompts)}")
            client = self._azure_client(*client_args)
            responses, batch_id = await client.run_batch(prompts, max_tokens=2048)
            print(f"Batch Doc Spec ID: {batch_id}")

            # Process responses and create doc specs
            for completion_text, (fact, doc_type) in zip(responses, fact_type_pairs):
                if completion_text:
                    # Extract ideas between <idea> tags using regex
                    ideas = re.findall(
                        r"<idea>\n?(.*?)\n?</idea>", completion_text, re.DOTALL
                    )
                    # Clean up any extra whitespace and filter unsuitable ideas
                    ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]

                    # Create doc specs for each valid idea, up to the requested number
                    for idea in ideas[:num_doc_ideas]:
                        all_doc_specs.append(
                            {"fact": fact, "doc_type": doc_type, "doc_idea": idea}
                        )

        except Exception as e:
            LOGGER.error(f"Error generating doc specs: {e}")
            return all_doc_specs

        return all_doc_specs
