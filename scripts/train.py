"""Milestone M4: pretrain the Dual-AR model on tokenized shards.

    python scripts/train.py configs/pretrain_s.yaml
    python scripts/train.py configs/pretrain_s.yaml max_steps=1000 lr=1e-4   # overrides

Re-running with the same run_dir resumes from the latest checkpoint.
"""

import sys

from tts.train import Trainer, load_config


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    cfg = load_config(args[0], args[1:])
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
