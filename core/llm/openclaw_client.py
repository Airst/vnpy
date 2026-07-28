"""
OpenClaw HTTP client using OpenAI-compatible chat completions API.

OpenClaw is a locally-deployed LLM orchestrator that exposes an OpenAI-compatible
endpoint. Stock analysis skills (tushare, web-search) are installed as MCP tools
inside OpenClaw and are invoked automatically via OpenAI function calling.

Configuration via environment variables:
    OPENCLAW_BASE_URL  (default: http://localhost:18789/v1)
    OPENCLAW_API_KEY   (required)
    OPENCLAW_MODEL     (default: openclaw; use "openclaw/<agentId>" to target a specific agent)
    OPENCLAW_TIMEOUT   (default: 300, seconds)

JSON output notes (per OpenClaw official docs, gateway/openai-http-api):
    The chat completions surface does NOT support OpenAI ``response_format``
    (json_object / json_schema). Supported request fields are limited to
    tools/tool_choice/max_completion_tokens/max_tokens/temperature/top_p/
    frequency_penalty/presence_penalty/seed/stop. Therefore JSON output must
    be enforced via prompt instructions, and responses must be hardened
    client-side (extraction + best-effort repair, see parse_json_response).
    Also note ``max_completion_tokens`` caps *total* completion tokens
    including reasoning tokens, so truncation (finish_reason="length") is a
    common cause of malformed/incomplete JSON.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from vnpy.alpha.logger import logger


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# 未闭合的代码围栏（截断响应常见）：只有开头 ```json 没有结尾 ```
_JSON_FENCE_OPEN_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)

# JSON 字符串外的非法控制字符（保留 \t \n \r，其余剔除）
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# prompt 端 JSON 硬约束后缀（OpenClaw 不支持 response_format，只能靠指令约束）
_JSON_SYSTEM_SUFFIX = (
    "\n\n[输出格式硬性要求] 只输出一个完整、合法的 JSON，"
    "不要输出 Markdown 代码块标记、注释、解释或任何额外文字；"
    "确保所有括号与引号正确闭合，字符串内部换行必须使用 \\n 转义。"
)


def _scan_json_span(text: str, start: int) -> int:
    """
    From an opening ``{`` or ``[`` at ``start``, scan (string/escape aware)
    to the matching close bracket. Returns the index one past the close, or
    -1 if the structure is unterminated (truncated response).
    """
    stack: List[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                if not stack:
                    return i + 1
    return -1


def _extract_json(text: str) -> str:
    """
    Extract JSON content from a model response.

    Handles:
    - Raw JSON: {...} or [...]
    - Markdown code fences: ```json ... ``` or ``` ... ```
    - Unclosed code fence (truncated response)
    - JSON preceded/followed by prose text (balanced-bracket scan, so prose
      braces after the JSON body no longer corrupt the extraction)
    - Truncated JSON: returns from the first bracket to end of text, leaving
      repair to _repair_json
    """
    text = text.strip()

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # 截断响应可能只有开围栏没有闭围栏
        open_match = _JSON_FENCE_OPEN_RE.search(text)
        if open_match and "{" in text[open_match.end():]:
            text = text[open_match.end():].strip()

    # 定位首个 { 或 [，用括号配对扫描取完整 JSON 体
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    end = _scan_json_span(text, start)
    if end != -1:
        return text[start:end].strip()
    # 未闭合（截断）：返回残体，交给 _repair_json 补全
    return text[start:].strip()


def _repair_json(text: str) -> str:
    """
    Best-effort repair of common LLM JSON defects:
    - illegal control characters
    - trailing commas before } or ]
    - truncated output (finish_reason="length"): unterminated string /
      dangling escape / dangling comma or colon / unclosed brackets
    """
    text = _CONTROL_CHARS_RE.sub("", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 扫描字符串状态与括号栈
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    if in_str:
        if esc:
            text = text[:-1]  # 悬空转义符
        text += '"'

    stripped = text.rstrip()
    if stripped.endswith(","):
        text = stripped[:-1]
    elif stripped.endswith(":"):
        text = stripped + " null"

    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


def parse_json_response(text: str) -> Any:
    """
    Parse JSON from an LLM response with a hardened pipeline:
    extract -> loads(strict=False) -> repair -> loads(strict=False).

    ``strict=False`` tolerates raw control characters (e.g. newlines) inside
    strings, which LLMs frequently emit.

    Raises ValueError if the response cannot be parsed even after repair.
    """
    extracted = _extract_json(text or "")
    try:
        return json.loads(extracted, strict=False)
    except json.JSONDecodeError:
        pass

    repaired = _repair_json(extracted)
    try:
        result = json.loads(repaired, strict=False)
        logger.warning(
            "[OpenClawClient] JSON repaired successfully "
            f"(raw len={len(text or '')}, repaired len={len(repaired)})"
        )
        return result
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON even after repair: {e}\n"
            f"Raw response (first 500 chars):\n{(text or '')[:500]}"
        ) from e

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai package required. Install with: pip install openai"
    ) from e


class OpenClawClient:
    """
    Thin wrapper around OpenAI SDK pointed at a local OpenClaw instance.

    Note: OpenClaw's chat completions endpoint does NOT honor OpenAI
    ``response_format`` (see module docstring). JSON output is enforced via
    a prompt suffix (``response_format_json=True``) plus client-side
    extraction/repair (``parse_json_response``). For tool-enabled requests,
    OpenClaw resolves tool calls internally and returns the final assistant
    message.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url or os.environ.get(
            "OPENCLAW_BASE_URL", "http://localhost:18789/v1"
        )
        self.api_key = api_key or os.environ.get("OPENCLAW_API_KEY", "17c7ce9fc2647a835e06d58dd56b4bc56d3e93498616efb7")
        self.model = model or os.environ.get("OPENCLAW_MODEL", "openclaw")
        self.timeout = timeout or float(os.environ.get("OPENCLAW_TIMEOUT", "1800"))

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def chat(
        self,
        system: str,
        user: str,
        response_format_json: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a single-turn chat request and return the assistant text.

        Parameters
        ----------
        system : str
            System prompt defining the assistant's role.
        user : str
            User message content.
        response_format_json : bool
            If True, append a hard JSON-only instruction to the system prompt.
            (OpenClaw does not support OpenAI JSON mode / response_format,
            so enforcement is prompt-side only.)
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max completion tokens (sent as ``max_completion_tokens``, the
            current field name per OpenClaw docs; includes reasoning tokens).
        tools : list of dict, optional
            OpenAI-format tool specs. If provided, the server is expected to
            handle tool calls internally (OpenClaw MCP integration).

        Returns
        -------
        str
            The assistant's final text response.
        """
        if response_format_json:
            system = system + _JSON_SYSTEM_SUFFIX

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        return self._read_choice(response)

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a multi-turn chat request and return the assistant text.

        Parameters
        ----------
        messages : list of dict
            Full conversation history (system/user/assistant messages).
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max completion tokens.

        Returns
        -------
        str
            The assistant's final text response.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        response = self.client.chat.completions.create(**kwargs)
        return self._read_choice(response)

    @staticmethod
    def _read_choice(response: Any) -> str:
        """Extract assistant text; warn on truncation (a top cause of broken JSON)."""
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            logger.warning(
                "[OpenClawClient] response truncated (finish_reason=length), "
                "JSON output is likely incomplete; consider raising max_tokens"
            )
        return choice.message.content or ""

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: returns parsed JSON dict.
        Uses the hardened extract/repair pipeline (parse_json_response).
        Raises ValueError if the response is not valid JSON even after repair.
        """
        text = self.chat(
            system=system,
            user=user,
            response_format_json=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        return parse_json_response(text)

    def health_check(self) -> bool:
        """
        Check if OpenClaw is reachable. Returns True on success.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
