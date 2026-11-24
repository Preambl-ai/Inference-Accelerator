"""Flow guard: fail if `make sweep-all` would silently skip a sweep.

Background: sweep-all once exited 0 without running sweep-conv. The
sweep-all line was the last line of the Makefile with no trailing newline, so
an edit that appended to it did not match and the new target was never
listed. Nothing failed; the new sweep just never ran.

Checks:
  1. the Makefile ends with a newline
  2. every sweep* target the Makefile defines is a prerequisite of sweep-all
  3. every prerequisite of sweep-all is a target the Makefile defines
     (a typo there would make `make` fail, but say so clearly here)
  4. every sweep* target is declared .PHONY (a stray file named like the
     target would otherwise make make think it is up to date and skip it)

Exits 1 on any failure. Runs as the first CI step and as a prerequisite of
sweep-all, so a local sweep-all checks itself too.

  python ci/check_flow.py [path/to/Makefile]
"""

import re
import sys


def parse(text):
    # join backslash continuations so a wrapped prerequisite list is one line
    joined = re.sub(r"\\\n", " ", text)
    targets = {}
    phony = set()
    for line in joined.splitlines():
        if line.startswith(("\t", "#")) or ":=" in line or "?=" in line or "+=" in line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*:(?!=)\s*(.*)$", line)
        if not m:
            continue
        name, prereqs = m.group(1), m.group(2).split()
        if name == ".PHONY":
            phony.update(prereqs)
        else:
            targets.setdefault(name, []).extend(prereqs)
    return targets, phony


def check(text):
    errors = []
    if not text.endswith("\n"):
        errors.append("Makefile does not end with a newline; an edit appending to its "
                      "last line can silently fail to match")
    targets, phony = parse(text)
    if "sweep-all" not in targets:
        return errors + ["no sweep-all target"], []
    listed = set(targets["sweep-all"])
    sweeps = {t for t in targets if re.fullmatch(r"sweep(-[A-Za-z0-9_]+)?", t) and t != "sweep-all"}
    for t in sorted(sweeps - listed):
        errors.append(f"{t} is defined but not a prerequisite of sweep-all, so it never runs")
    for t in sorted(p for p in listed if p not in targets):
        errors.append(f"sweep-all lists {t}, which the Makefile does not define")
    for t in sorted((sweeps | {"sweep-all"}) - phony):
        errors.append(f"{t} is not declared .PHONY")
    return errors, sorted(sweeps)


def self_test():
    """the guard has to fail on the cases it exists for"""
    good = (".PHONY: sweep sweep-a sweep-all\nsweep:\n\techo\nsweep-a:\n\techo\n"
            "sweep-all: sweep sweep-a\n")
    cases = {
        "good": (good, 0),
        "no trailing newline": (good.rstrip("\n"), 1),
        "unlisted sweep": (good.replace("sweep-all: sweep sweep-a", "sweep-all: sweep"), 1),
        "listed but undefined": (good.replace("sweep-a sweep-all\n", "sweep-a sweep-b sweep-all\n")
                                 .replace("sweep-all: sweep sweep-a", "sweep-all: sweep sweep-a sweep-b"), 1),
        "not phony": (good.replace(".PHONY: sweep sweep-a", ".PHONY: sweep"), 1),
        "continuation line": (good.replace("sweep-all: sweep sweep-a", "sweep-all: sweep \\\n  sweep-a"), 0),
        "digit in name": (re.sub(r"sweep-a\b(?!l)", "sweep-a2", good), 0),
    }
    bad = 0
    for name, (text, want) in cases.items():
        errs, _ = check(text)
        got = int(bool(errs))
        if got != want:
            print(f"SELF-TEST FAIL {name}: expected {'errors' if want else 'clean'}, got {errs}")
            bad += 1
    return bad


def main():
    if self_test():
        sys.exit(1)
    path = sys.argv[1] if len(sys.argv) > 1 else "Makefile"
    with open(path) as f:
        errors, sweeps = check(f.read())
    for e in errors:
        print(f"FLOW GUARD: {e}")
    if errors:
        sys.exit(1)
    print(f"flow guard ok: {len(sweeps)} sweeps, all in sweep-all: {' '.join(sweeps)}")


if __name__ == "__main__":
    main()
