"""
rvvfft multi-agent collaboration setup.

Three specialised agents work together to develop the rvvfft library:

  * CodeAgent    — generates RVV intrinsic C code and Python codegen scripts
  * MathAgent    — verifies FFT mathematics using SymPy and reviews proofs
  * ReviewAgent  — performs code review and suggests optimisations

Usage
-----
    # Install dependencies
    pip install openai python-dotenv pyautogen

    # Configure (copy .env.example → .env, then fill in OPENAI_API_KEY)
    cp .env.example .env

    # Run
    python agents/multi_agent.py "Generate a radix-4 butterfly kernel for VLEN=256"

Environment variables
---------------------
OPENAI_API_KEY  – your OpenAI API key (required)
OPENAI_MODEL    – model to use (default: gpt-4o)

Valid model names include: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4,
gpt-3.5-turbo.

NOTE: "GPT-5.3-Codex" is NOT a valid model name and will cause an
authentication error.  Use one of the models listed above.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Load .env (if present) ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass   # python-dotenv is optional; set env vars manually if needed


# ── Validate configuration before importing autogen ───────────────────────

_VALID_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
}

_INVALID_MODELS = {
    # Common mistakes — provide a helpful error message for each.
    "GPT-5.3-Codex": "GPT-5.3-Codex does not exist. Use 'gpt-4o' instead.",
    "gpt-5": "gpt-5 does not exist. Use 'gpt-4o' instead.",
    "code-davinci-002": (
        "code-davinci-002 (Codex) was deprecated in March 2023. "
        "Use 'gpt-4o' instead."
    ),
}


def _validate_config() -> tuple[str, str]:
    """
    Read and validate OPENAI_API_KEY and OPENAI_MODEL from the environment.

    Returns (api_key, model) on success; prints a helpful message and exits
    on failure.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model   = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # ── API key check ──────────────────────────────────────────────────────
    if not api_key or api_key == "sk-your-api-key-here":
        print(
            "ERROR: OPENAI_API_KEY is not set or is still the placeholder.\n"
            "\n"
            "  1. Copy .env.example to .env:\n"
            "       cp .env.example .env\n"
            "  2. Open .env and replace 'sk-your-api-key-here' with your\n"
            "     actual OpenAI API key from https://platform.openai.com/api-keys\n"
            "  3. Re-run this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not api_key.startswith("sk-"):
        print(
            f"ERROR: OPENAI_API_KEY looks invalid (value: '{api_key[:8]}…').\n"
            "       OpenAI API keys start with 'sk-'.\n"
            "       Check https://platform.openai.com/api-keys for your key.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Model name check ───────────────────────────────────────────────────
    if model in _INVALID_MODELS:
        print(
            f"ERROR: '{model}' is not a valid OpenAI model name.\n"
            f"       {_INVALID_MODELS[model]}\n"
            f"\n"
            f"       Valid models: {', '.join(sorted(_VALID_MODELS))}\n"
            f"       Set OPENAI_MODEL in your .env file to one of the above.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if model not in _VALID_MODELS:
        print(
            f"WARNING: '{model}' is not in the list of known-good models.\n"
            f"         Known-good models: {', '.join(sorted(_VALID_MODELS))}\n"
            f"         Proceeding anyway — update OPENAI_MODEL in .env if "
            f"you get errors.\n",
            file=sys.stderr,
        )

    return api_key, model


# ── Agent definitions ──────────────────────────────────────────────────────

def build_agents(api_key: str, model: str):
    """Construct the three-agent collaboration group."""
    try:
        import autogen
    except ImportError:
        print(
            "ERROR: 'pyautogen' is not installed.\n"
            "       Install it with:  pip install pyautogen\n",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_config = {
        "config_list": [{"model": model, "api_key": api_key}],
        "temperature": 0.1,
    }

    code_agent = autogen.AssistantAgent(
        name="CodeAgent",
        llm_config=llm_config,
        system_message=(
            "You are an expert in RISC-V Vector (RVV) intrinsics and "
            "high-performance FFT implementation. Generate correct, "
            "efficient C code using RVV 1.0 intrinsics. "
            "Always target VLEN=256 or VLEN=1024 as specified. "
            "Use vsetvl for tail handling."
        ),
    )

    math_agent = autogen.AssistantAgent(
        name="MathAgent",
        llm_config=llm_config,
        system_message=(
            "You are an expert in FFT mathematics and symbolic computation "
            "with SymPy. Verify every twiddle-factor simplification rule "
            "symbolically. Confirm DFT butterfly correctness using small "
            "examples.  Reject any code that lacks a corresponding "
            "mathematical proof."
        ),
    )

    review_agent = autogen.AssistantAgent(
        name="ReviewAgent",
        llm_config=llm_config,
        system_message=(
            "You are a senior performance engineer. Review generated RVV "
            "kernel code for correctness, register pressure, memory access "
            "patterns, and opportunities for vfmacc fusion. "
            "Flag any potential correctness issues before approval."
        ),
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": ".", "use_docker": False},
    )

    group_chat = autogen.GroupChat(
        agents=[user_proxy, code_agent, math_agent, review_agent],
        messages=[],
        max_round=20,
    )
    manager = autogen.GroupChatManager(groupchat=group_chat,
                                       llm_config=llm_config)

    return user_proxy, manager


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        task = (
            "Generate a radix-2 butterfly kernel for split layout, "
            "f32 dtype, VLEN=256, and verify the mathematics."
        )
    else:
        task = " ".join(sys.argv[1:])

    api_key, model = _validate_config()
    print(f"Using model: {model}")

    user_proxy, manager = build_agents(api_key, model)
    user_proxy.initiate_chat(manager, message=task)


if __name__ == "__main__":
    main()
