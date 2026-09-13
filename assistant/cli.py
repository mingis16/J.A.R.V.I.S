from __future__ import annotations

import argparse
import os
import sys

from trading_bot.config import REPO_ROOT, load_env, load_yaml_config

from assistant.orchestrator import Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. personal assistant / orchestrator")
    parser.add_argument("message", nargs="*", help="Send a single message and print the reply, then exit.")
    args = parser.parse_args()

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).", file=sys.stderr)
        return 1

    cfg = load_yaml_config()
    orchestrator = Orchestrator(REPO_ROOT, cfg)

    if args.message:
        print(orchestrator.chat(" ".join(args.message)))
        return 0

    print("Alex online. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = orchestrator.chat(user_input)
        print(f"alex> {reply}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
