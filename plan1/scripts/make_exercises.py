"""Generate exercises/ from solutions/ so the two never drift apart.

In a solution, a block between `# >>> TODO: hint` and `# <<<` becomes
`raise NotImplementedError("hint")` in the exercise.

Existing exercise files are kept, so your work is safe. Pass --force to overwrite them.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
OPEN = re.compile(r"^(\s*)# >>> TODO: (.+)$")
CLOSE = re.compile(r"^\s*# <<<\s*$")


def strip_solution(source: str) -> str:
    out: list[str] = []
    inside = False
    for line in source.splitlines():
        if inside:
            inside = not CLOSE.match(line)
            continue
        match = OPEN.match(line)
        if match:
            indent, hint = match.groups()
            hint = hint.strip().replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'{indent}raise NotImplementedError("{hint}")')
            inside = True
        else:
            out.append(line)
    if inside:
        raise ValueError("Unclosed # >>> TODO block")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing exercise files.")
    args = parser.parse_args()
    for solution in sorted(ROOT.glob("stages/*/solutions/*.py")):
        exercise = solution.parents[1] / "exercises" / solution.name
        if exercise.exists() and not args.force:
            continue
        exercise.parent.mkdir(exist_ok=True)
        text = solution.read_text()
        if "# >>> TODO:" not in text:
            raise SystemExit(f"{solution} has no TODO blocks, so its exercise would be vacuous")
        exercise.write_text(strip_solution(text))
        print(f"wrote {exercise.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
