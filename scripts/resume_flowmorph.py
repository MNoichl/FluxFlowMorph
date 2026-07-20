#!/usr/bin/env python3
"""Resume a compatible FlowMorph Klein run."""

from __future__ import annotations


def main() -> int:
    from flowmorph_klein.cli import resume_command

    return resume_command()


if __name__ == "__main__":
    raise SystemExit(main())
