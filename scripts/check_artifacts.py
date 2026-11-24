"""M7 checks: every committed SHA256SUMS verifies and relative doc links resolve."""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def checksum_issues(root: Path = ROOT) -> list[str]:
    issues = []
    for sums in sorted((root / "verification").rglob("SHA256SUMS")):
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            path = sums.parent / name.lstrip("*")
            if not path.is_file():
                issues.append(f"missing: {path.relative_to(root)}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                issues.append(f"hash mismatch: {path.relative_to(root)}")
    return issues


def link_issues(root: Path = ROOT) -> list[str]:
    issues = []
    for doc in sorted([root / "README.md", *(root / "docs").rglob("*.md"), *(root / "specs").rglob("*.md")]):
        if not doc.is_file():
            continue
        for target in LINK.findall(doc.read_text()):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (doc.parent / target).exists():
                issues.append(f"{doc.relative_to(root)} -> {target}")
    return issues


def main():
    issues = checksum_issues() + link_issues()
    for i in issues:
        print("FAIL:", i)
    print("PASS: artifact checksums and documentation links" if not issues else f"{len(issues)} issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
