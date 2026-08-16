from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path)
    a = p.parse_args()
    nb_path = a.notebook.resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    try:
        from IPython.display import display
    except Exception:
        display = print
    g = {"__name__": "__main__", "__file__": str(nb_path), "display": display}
    os.chdir(nb_path.parent)
    code_cells = [(i,c) for i,c in enumerate(data.get("cells", [])) if c.get("cell_type") == "code"]
    print(f"Executing {len(code_cells)} code cells from {nb_path.name}", flush=True)
    for n,(i,c) in enumerate(code_cells,1):
        src = c.get("source","")
        if isinstance(src,list):
            src = "".join(src)
        if not src.strip():
            continue
        print(f"[cell {n}/{len(code_cells)} | notebook index {i}]", flush=True)
        try:
            exec(compile(src, f"{nb_path.name}:cell_{i}", "exec"), g, g)
        except Exception:
            print(f"ERROR in notebook cell index {i}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return 1
    print(f"Completed {nb_path.name}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())