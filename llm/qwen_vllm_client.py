"""
Qwen2.5-1M-Instruct vLLM Client

Provides OpenAI-compatible interface to Qwen2.5-1M-Instruct served via vLLM.
"""

import os
import requests
from typing import Optional, Callable
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configuration via environment variables
raw_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000").strip().rstrip("/")
VLLM_BASE_URL = raw_base if raw_base.endswith("/v1") else f"{raw_base}/v1"
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
def qwen_completion_vllm(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    logger: Optional[Callable] = None,
) -> str:
    """
    Call Qwen2.5-1M-Instruct served via vLLM (OpenAI-compatible API).
    
    Args:
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate (default: 1024)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        top_p: Nucleus sampling parameter (default: 1.0)
        logger: Optional logger function (e.g., logger.info)
        
    Returns:
        Generated text string
        
    Raises:
        requests.HTTPError: If the API request fails
    """
    url = f"{VLLM_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if logger:
        logger(f"[Qwen-vLLM] Request: {len(prompt)} chars, max_tokens={max_tokens}, "
               f"model={QWEN_MODEL_NAME}, url={VLLM_BASE_URL}")
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        
        # OpenAI-compatible format from vLLM
        if "choices" not in data or len(data["choices"]) == 0:
            raise ValueError("No choices in vLLM response")
        
        text = data["choices"][0]["message"]["content"]
        
        if logger:
            logger(f"[Qwen-vLLM] Response length: {len(text)} chars")
        
        return text
        
    except requests.exceptions.RequestException as e:
        if logger:
            logger(f"[Qwen-vLLM] Request failed: {e}")
        raise
    except (KeyError, ValueError) as e:
        if logger:
            logger(f"[Qwen-vLLM] Response parsing failed: {e}, response: {data if 'data' in locals() else 'N/A'}")
        raise

