#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SHA-256 manifest for frozen research artifacts."
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument("roots", type=Path, nargs="+")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_path = args.output_path.resolve()
    files = sorted(
        path
        for root in args.roots
        for path in (
            [root]
            if root.is_file()
            else root.rglob("*")
        )
        if path.is_file() and path.resolve() != output_path
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.as_posix()}\n")


if __name__ == "__main__":
    main()
