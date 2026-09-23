# -*- coding: utf-8 -*-
"""Point the derivation scripts at the companion engine on THIS machine.

The scripts in this directory are shipped as they were run, so the engine
location in them is an absolute path from the machine that ran them.  This
rewrites exactly that one line, only in the files that have it, to read the
environment variable ENGINE_DIR.  It touches nothing else, and it prints
every line it changes so the edit is auditable rather than silent.

    export ENGINE_DIR=/path/to/crowd_control_engine
    python3 retarget_engine_path.py [--dry-run]

Scripts that do not open the companion engine are untouched -- including every
script behind Section VII, which runs against the Nav2 stack instead.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry-run" in sys.argv

if not DRY and not os.environ.get("ENGINE_DIR"):
    raise SystemExit("set ENGINE_DIR first, or pass --dry-run to preview")

# Matches:  REPO = pathlib.Path(r'C:\...')   and the ZIP variant.
PAT = re.compile(
    r"""^(\s*)(REPO|ZIP)\s*=\s*pathlib\.Path\(\s*r?['"][A-Za-z]:[^'"]*['"]\s*\)\s*$""")

REPLACEMENT = ("%s%s = pathlib.Path(os.environ.get('ENGINE_DIR', '.'))"
               "  # retargeted: was an absolute path from the run machine")

n_files = 0
n_lines = 0
for fn in sorted(os.listdir(HERE)):
    if not fn.endswith(".py") or fn == os.path.basename(__file__):
        continue
    path = os.path.join(HERE, fn)
    lines = io.open(path, encoding="utf-8").read().split("\n")
    out = []
    hit = False
    for line in lines:
        m = PAT.match(line)
        if m:
            print("  %-52s %s" % (fn, line.strip()))
            out.append(REPLACEMENT % (m.group(1), m.group(2)))
            hit = True
            n_lines += 1
        else:
            out.append(line)
    if not hit:
        continue
    n_files += 1
    body = "\n".join(out)
    # the rewritten line needs os; add the import only if the file lacks it
    head = body.split("\ndef ")[0]
    if not re.search(r"^import os$", head, re.M):
        body = body.replace("import pathlib", "import os\nimport pathlib", 1)
    if not DRY:
        io.open(path, "w", encoding="utf-8", newline="\n").write(body)

print("%s %d line(s) in %d file(s)"
      % ("would change" if DRY else "changed", n_lines, n_files))
if DRY:
    print("re-run without --dry-run to apply")
