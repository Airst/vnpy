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
"""

import json
import os
import re
from typing import Any, Dict, List, Optional


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> str:
    """
    Extract JSON content from a model response.

    Handles:
    - Raw JSON: {...}
    - Markdown code fences: ```json ... ``` or ``` ... ```
    - JSON preceded/followed by prose text (extracts the first {...} block)
    """
    text = text.strip()

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai package required. Install with: pip install openai"
    ) from e


class OpenClawClient:
    """
    Thin wrapper around OpenAI SDK pointed at a local OpenClaw instance.

    Prefers structured JSON output via response_format={"type": "json_object"}.
    For tool-enabled requests, OpenClaw resolves tool calls internally and
    returns the final assistant message.
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
            If True, ask the model to return JSON (OpenAI JSON mode).
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max completion tokens.
        tools : list of dict, optional
            OpenAI-format tool specs. If provided, the server is expected to
            handle tool calls internally (OpenClaw MCP integration).

        Returns
        -------
        str
            The assistant's final text response.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

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
            "max_tokens": max_tokens,
        }
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

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
        Raises ValueError if the response is not valid JSON.
        """
        text = self.chat(
            system=system,
            user=user,
            response_format_json=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        extracted = _extract_json(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON.\nRaw response:\n{text}"
            ) from e

    def health_check(self) -> bool:
        """
        Check if OpenClaw is reachable. Returns True on success.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
