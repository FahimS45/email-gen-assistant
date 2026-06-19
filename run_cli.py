"""
run_cli.py

Interactive terminal demo for the Email Generation Assistant.

Runs the exact same LangGraph pipeline used by the FastAPI backend
(validate_input → build_prompt → generate_email) directly — no server needs
to be running. 

USAGE:
    python run_cli.py

User will be prompted for:
  1. Intent          — the email's purpose
  2. Key facts        — one per line, blank line to finish
  3. Tone              — chosen from the fixed TONE_OPTIONS list
  4. Model              — openai or ollama

The email streams to the terminal token-by-token (same streaming path used
by the WebSocket endpoint), then the parsed Chain-of-Thought reasoning and
final clean email are shown separately.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import TONE_OPTIONS
from backend.graph.nodes import validate_input, build_prompt, generate_email_stream
from backend.graph.prompt_engineering import extract_cot_and_email


# Terminal styling (plain ANSI, no external deps)

class Style:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def header(text: str) -> None:
    print(f"\n{Style.BOLD}{Style.CYAN}{'═' * 60}{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  {text}{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}{'═' * 60}{Style.RESET}\n")


def section(text: str) -> None:
    print(f"\n{Style.BOLD}{Style.GREEN}── {text} {'─' * max(0, 50 - len(text))}{Style.RESET}\n")


def dim(text: str) -> None:
    print(f"{Style.DIM}{text}{Style.RESET}")


def error(text: str) -> None:
    print(f"{Style.RED}{Style.BOLD}Error:{Style.RESET} {text}")


# Input collection

def prompt_intent() -> str:
    print(f"{Style.BOLD}1. Intent{Style.RESET}  — the core purpose of the email")
    dim('   e.g. "Follow up after meeting", "Request for proposal details"')
    while True:
        intent = input("   > ").strip()
        if len(intent) >= 5:
            return intent
        error("Intent must be at least 5 characters. Try again.")


def prompt_key_facts() -> list[str]:
    print(f"\n{Style.BOLD}2. Key Facts{Style.RESET}  — one fact per line, blank line to finish")
    dim("   e.g. \"Meeting was on June 3rd\"")
    facts = []
    while True:
        line = input(f"   [{len(facts) + 1}] > ").strip()
        if not line:
            if facts:
                return facts
            error("At least one fact is required.")
            continue
        facts.append(line)
        if len(facts) >= 10:
            dim("   (maximum 10 facts reached)")
            return facts


def prompt_tone() -> str:
    print(f"\n{Style.BOLD}3. Tone{Style.RESET}  — choose one")
    for i, tone in enumerate(TONE_OPTIONS, start=1):
        print(f"   {i}. {tone}")
    while True:
        choice = input("   > ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(TONE_OPTIONS):
            return TONE_OPTIONS[int(choice) - 1]
        if choice.lower() in TONE_OPTIONS:
            return choice.lower()
        error(f"Enter a number 1-{len(TONE_OPTIONS)} or a tone name.")


def prompt_model() -> str:
    print(f"\n{Style.BOLD}4. Model{Style.RESET}  — choose a backend")
    print("   1. openai  (requires OPENAI_API_KEY in .env)")
    print("   2. ollama  (requires `ollama serve` running locally)")
    while True:
        choice = input("   > ").strip()
        if choice in ("1", "openai"):
            return "openai"
        if choice in ("2", "ollama"):
            return "ollama"
        error("Enter 1, 2, 'openai', or 'ollama'.")


# Generation

async def run_generation(intent: str, key_facts: list[str], tone: str, model: str) -> None:
    """
    Runs validate_input → build_prompt → generate_email_stream, mirroring the
    exact node sequence used by the WebSocket endpoint in backend/main.py.
    """
    state = {
        "intent": intent,
        "key_facts": key_facts,
        "tone": tone,
        "model": model,
        "system_prompt": "",
        "user_prompt": "",
        "cot_reasoning": "",
        "email_output": "",
        "model_used": "",
        "error": None,
    }

    # Node 1: validate
    validation_update = await validate_input(state)
    state.update(validation_update)
    if state.get("error"):
        error(state["error"])
        return

    # Node 2: build prompt
    prompt_update = await build_prompt(state)
    state.update(prompt_update)

    # Node 3: stream generation
    section(f"Generating with {model}  (streaming live)")

    full_raw_output = ""
    model_name = ""
    try:
        async for token, m_name in generate_email_stream(state):
            full_raw_output += token
            model_name = m_name
            print(token, end="", flush=True)
    except Exception as e:
        print()
        error(f"Generation failed: {e}")
        if model == "ollama":
            dim("   Is `ollama serve` running? Is the model pulled (`ollama pull qwen3:8b`)?")
        return

    print()  # newline after stream

    cot, clean_email = extract_cot_and_email(full_raw_output)

    if cot:
        section("Chain-of-Thought (parsed, not shown to end users)")
        print(cot)

    section(f"Final Email  (model: {model_name})")
    print(clean_email or full_raw_output)
    print()


# Main loop

async def main() -> None:
    header("Email Generation Assistant — Terminal Demo")
    dim("Runs the full LangGraph pipeline directly (no server required).")
    dim("Press Ctrl+C at any time to exit.\n")

    while True:
        try:
            intent = prompt_intent()
            key_facts = prompt_key_facts()
            tone = prompt_tone()
            model = prompt_model()

            await run_generation(intent, key_facts, tone, model)

            again = input(f"{Style.BOLD}Generate another email? (y/n) > {Style.RESET}").strip().lower()
            if again not in ("y", "yes"):
                print("\nGoodbye.")
                break
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye.")
            break


if __name__ == "__main__":
    asyncio.run(main())
