import argparse
import json
import os

from dotenv import dotenv_values
from epsilab.client import EpsilabClient


def main():
    # Read environment variables from .env (does not modify process env)
    env = dotenv_values()

    parser = argparse.ArgumentParser(prog="epsilab", description="Epsilab Strategy-Engine CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run default portfolio
    p_run = sub.add_parser("run-default-portfolio", help="Run user's default portfolio")
    p_run.add_argument("--params", help="JSON dict of parameters to pass", default=None)

    args = parser.parse_args()

    client = EpsilabClient(
        api_base=env.get("EPSILAB_API_BASE") or "https://www.epsilab.ai/api/ext/v1",
        api_key=env.get("EPSILAB_API_KEY"),
    )

    if args.cmd == "run-default-portfolio":
        # Require API key
        api_key = env.get("EPSILAB_API_KEY")
        if api_key:
            client.set_api_key(api_key)
        if not client._api_key:
            raise SystemExit("Not authenticated. Set EPSILAB_API_KEY.")
        params = json.loads(args.params) if args.params else None
        result = client.run_default_portfolio(params)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()