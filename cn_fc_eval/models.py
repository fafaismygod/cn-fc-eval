"""LLM clients for function calling evaluation."""

import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class FunctionCallingClient(ABC):
    """Abstract base for function-calling-capable LLM clients."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        timeout: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Send a request with tools and return the function call.

        Returns:
            {"name": str, "arguments": dict} or None if no function call
        """
        ...


class DeepSeekClient(FunctionCallingClient):
    """DeepSeek API client (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(f"DeepSeek ({model})")
        self.model = model
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )

    def call(self, messages, tools, timeout=30):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0,
                max_tokens=512,
                timeout=timeout,
            )
            msg = resp.choices[0].message
            if msg.tool_calls and len(msg.tool_calls) > 0:
                calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                return calls[0] if len(calls) == 1 else calls
            return None
        except Exception as e:
            return {"error": str(e)}


class OpenAIClient(FunctionCallingClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(f"OpenAI ({model})")
        self.model = model
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(api_key=api_key)

    def call(self, messages, tools, timeout=30):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0,
                max_tokens=512,
                timeout=timeout,
            )
            msg = resp.choices[0].message
            if msg.tool_calls and len(msg.tool_calls) > 0:
                calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                return calls[0] if len(calls) == 1 else calls
            return None
        except Exception as e:
            return {"error": str(e)}


class AnyOpenAICompatibleClient(FunctionCallingClient):
    """Generic OpenAI-compatible client (Qwen via DashScope, GLM via ZhipuAI, etc.)."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
    ):
        super().__init__(name)
        self.model = model
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, messages, tools, timeout=30):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0,
                max_tokens=512,
                timeout=timeout,
            )
            msg = resp.choices[0].message
            if msg.tool_calls and len(msg.tool_calls) > 0:
                calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                return calls[0] if len(calls) == 1 else calls
            return None
        except Exception as e:
            return {"error": str(e)}


# ── Pre-configured clients for Chinese LLMs ──

def make_qwen_client(api_key: str, model: str = "qwen-plus") -> AnyOpenAICompatibleClient:
    """Create a Qwen client via DashScope."""
    return AnyOpenAICompatibleClient(
        name=f"Qwen ({model})",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
    )


def make_glm_client(api_key: str, model: str = "glm-4-plus") -> AnyOpenAICompatibleClient:
    """Create a GLM client via ZhipuAI."""
    return AnyOpenAICompatibleClient(
        name=f"GLM ({model})",
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model=model,
    )
