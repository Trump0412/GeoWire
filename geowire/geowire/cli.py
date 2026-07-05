from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="geowire")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("geowire 0.1.0")


if __name__ == "__main__":
    main()
