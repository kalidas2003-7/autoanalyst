"""
main.py — CLI entry point.

Usage:
    python main.py --files data/sales.csv --query "explore this dataset and predict revenue"
    python main.py --files data/a.csv data/b.json --query "clean the data and summarize it"
"""
import argparse
import json

from autoanalyst.orchestrator import AutoAnalystAgent


def main():
    parser = argparse.ArgumentParser(description="AutoAnalyst CLI")
    parser.add_argument("--files", nargs="+", required=True, help="Paths to data files")
    parser.add_argument("--query", required=True, help="Natural-language analysis request")
    parser.add_argument("--session", default=None, help="Session id (for resuming memory)")
    args = parser.parse_args()

    agent = AutoAnalystAgent(session_id=args.session)
    manifest = agent.load_files(args.files)
    print("Loaded manifest:")
    print(json.dumps(manifest, indent=2, default=str))

    result = agent.chat(args.query)

    print("\n=== PLAN ===")
    print(json.dumps(result.plan, indent=2))

    print("\n=== REPORT ===")
    print(result.report_markdown)

    if result.errors:
        print("\n=== ERRORS ===")
        print(json.dumps(result.errors, indent=2))

    print(f"\nSession id: {agent.session_id} (reuse with --session {agent.session_id})")


if __name__ == "__main__":
    main()
