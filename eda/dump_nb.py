"""Dump code cells (and markdown headers) from a .ipynb, ignoring outputs."""
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
out = []
for i, c in enumerate(nb.get("cells", [])):
    src = "".join(c.get("source", []))
    if not src.strip():
        continue
    if c.get("cell_type") == "markdown":
        head = "\n".join(l for l in src.splitlines() if l.strip().startswith("#") or len(l.strip()) > 0)[:400]
        out.append(f"\n--- [md {i}] ---\n{head}")
    else:
        out.append(f"\n--- [code {i}] ---\n{src}")
txt = "\n".join(out)
print(f"### {p.name}  ({len(nb.get('cells', []))} cells, {len(txt)} chars of source)\n")
print(txt[:maxc])
if len(txt) > maxc:
    print(f"\n... [TRUNCATED {len(txt)-maxc} chars]")
