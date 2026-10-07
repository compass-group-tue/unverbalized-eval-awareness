from __future__ import annotations

from typing import Any


def jsonify_untruncated(value: Any) -> Any:
    """Return a JSON-safe representation without truncating model artifacts.

    Final experiment transcripts retain the complete Inspect ``ModelOutput``
    object alongside the parsed response/reasoning fields.  This deliberately
    differs from the older debug serializer below, which is size-bounded for
    log diagnostics.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonify_untruncated(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonify_untruncated(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                try:
                    dumped = method(mode="json")
                except TypeError:
                    dumped = method()
            except Exception:
                continue
            return jsonify_untruncated(dumped)
    if hasattr(value, "__dict__"):
        try:
            return jsonify_untruncated(vars(value))
        except Exception:
            pass
    return str(value)


def extract_reasoning_and_text_from_content_blocks(content: Any) -> tuple[str, str]:
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, str):
        return "", content.strip()
    if not isinstance(content, list):
        return "", ""

    for block in content:
        if isinstance(block, str):
            text = block.strip()
            if text:
                text_parts.append(text)
            continue

        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")

        if block_type == "reasoning":
            reasoning = getattr(block, "reasoning", None)
            if reasoning is None and isinstance(block, dict):
                reasoning = block.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                reasoning_parts.append(reasoning.strip())
        elif block_type == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

    return "\n".join(reasoning_parts).strip(), "\n".join(text_parts).strip()


def get_output_message(output: Any) -> Any | None:
    try:
        message = getattr(output, "message", None)
        if message is not None:
            return message
    except Exception:
        pass

    choices = getattr(output, "choices", None)
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        try:
            return getattr(choice0, "message", None)
        except Exception:
            if isinstance(choice0, dict):
                return choice0.get("message")
    return None


def extract_output_text(output: Any) -> str:
    completion = getattr(output, "completion", "") or ""
    if isinstance(completion, str) and completion.strip():
        return completion

    message = get_output_message(output)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    _, text = extract_reasoning_and_text_from_content_blocks(content)
    return text


def maybe_get_reasoning_text(output: Any) -> str:
    candidates: list[str] = []
    for attr in ("reasoning", "reasoning_text", "thinking", "thought", "reasoning_content"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        elif isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            joined = "\n".join(parts).strip()
            if joined:
                candidates.append(joined)

    message = get_output_message(output)
    if message is not None:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        reasoning, _ = extract_reasoning_and_text_from_content_blocks(content)
        if reasoning:
            candidates.append(reasoning)

    return "\n".join(part for part in candidates if part.strip()).strip()


def serialize_error(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, dict):
        return error
    out: dict[str, Any] = {}
    for field in ("type", "message", "traceback"):
        value = getattr(error, field, None)
        if value:
            out[field] = value
    if out:
        return out
    return {"type": type(error).__name__, "message": str(error)}


def jsonify_debug_value(
    value: Any,
    *,
    max_depth: int = 5,
    max_items: int = 50,
    max_chars: int = 8000,
) -> Any:
    if max_depth < 0:
        return _truncate_string(repr(value), max_chars=max_chars)

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_string(value, max_chars=max_chars)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        items = list(value.items())
        for idx, (k, v) in enumerate(items):
            if idx >= max_items:
                out["__truncated_items__"] = len(items) - max_items
                break
            out[str(k)] = jsonify_debug_value(
                v,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
            )
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out_list = [
            jsonify_debug_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
            )
            for item in seq[:max_items]
        ]
        if len(seq) > max_items:
            out_list.append({"__truncated_items__": len(seq) - max_items})
        return out_list

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except TypeError:
                continue
            except Exception:
                break
            return jsonify_debug_value(
                dumped,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
            )

    if hasattr(value, "__dict__"):
        try:
            return jsonify_debug_value(
                vars(value),
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
            )
        except Exception:
            pass

    return _truncate_string(repr(value), max_chars=max_chars)


def _truncate_string(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"... [truncated {omitted} chars]"
