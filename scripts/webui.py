"""Web UI for synthesis and training.

    python scripts/webui.py                       # http://127.0.0.1:7860
    python scripts/webui.py --auth admin:secret   # password-protect it

On a rented machine, keep the default 127.0.0.1 and use an SSH tunnel:
    ssh -L 7860:127.0.0.1:7860 user@gb10-host   then open http://localhost:7860
Binding to 0.0.0.0 exposes training control to anyone who can reach the port.
"""

import argparse

from tts.ui.app import launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--auth", help="user:password for basic login")
    parser.add_argument("--runs", default="runs", help="directory holding training runs")
    parser.add_argument("--configs", default="configs", help="directory holding training configs")
    parser.add_argument("--codec", default="kyutai/mimi")
    args = parser.parse_args()
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    if args.host not in ("127.0.0.1", "localhost") and not auth:
        print("warning: listening on a public interface without --auth")
    launch(args.host, args.port, auth, runs_root=args.runs, configs_root=args.configs, codec=args.codec)


if __name__ == "__main__":
    main()
