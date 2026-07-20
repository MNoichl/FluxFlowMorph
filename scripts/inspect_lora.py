#!/usr/bin/env python3
"""Resolve and inspect a candidate FLUX.2 Klein Base 9B LoRA."""

from __future__ import annotations


def main() -> int:
    from flowmorph_klein.cli import inspect_lora_command

    return inspect_lora_command()


if __name__ == "__main__":
    raise SystemExit(main())
