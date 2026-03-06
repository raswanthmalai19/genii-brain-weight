import torch, json, numpy as np, sys
from pathlib import Path

W = Path(r"C:\Users\M.S. BAVESH REDDY\Downloads\weights")
errors = []
ok = []

# 1. Check all .pt files are loadable
for pt in sorted(W.glob("*.pt")):
    try:
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if isinstance(d, dict) and "world_model_state_dict" in d:
            ep = d.get("epoch", "?")
            bl = d.get("best_loss", "?")
            keys = len(d["world_model_state_dict"])
            has_opt = "optimizer_state_dict" in d
            has_sch = "scheduler_state_dict" in d
            has_scl = "scaler_state_dict" in d
            has_cfg = "config" in d
            ok.append(f"[OK] {pt.name} ({pt.stat().st_size/1e6:.1f}MB) - FULL CKPT epoch={ep} best_loss={bl:.6f} params={keys} opt={has_opt} sched={has_sch} scaler={has_scl} config={has_cfg}")
        elif isinstance(d, dict):
            ok.append(f"[OK] {pt.name} ({pt.stat().st_size/1e6:.1f}MB) - {len(d)} param tensors")
        else:
            ok.append(f"[OK] {pt.name} ({pt.stat().st_size/1e6:.1f}MB) - type={type(d).__name__}")
    except Exception as e:
        errors.append(f"[FAIL] {pt.name}: {e}")

# 2. Check .npz files
for npz in sorted(W.glob("*.npz")):
    try:
        d = np.load(npz)
        shapes = {k: d[k].shape for k in d.files}
        ok.append(f"[OK] {npz.name} ({npz.stat().st_size/1e6:.1f}MB) - keys: {shapes}")
    except Exception as e:
        errors.append(f"[FAIL] {npz.name}: {e}")

# 3. Check JSON files
for jf in sorted(W.glob("*.json")):
    try:
        with open(jf, "r") as f:
            d = json.load(f)
        ok.append(f"[OK] {jf.name} - {len(d)} fields: {list(d.keys())}")
    except Exception as e:
        errors.append(f"[FAIL] {jf.name}: {e}")

# 4. Check parquet data
raw_dir = W / "data" / "raw"
pqs = list(raw_dir.rglob("*.parquet")) if raw_dir.exists() else []
if pqs:
    import pandas as pd
    total_rows = 0
    symbols = []
    for pf in pqs:
        df = pd.read_parquet(pf)
        total_rows += len(df)
        symbols.append(pf.parent.name)
    ok.append(f"[OK] data/raw/ - {len(pqs)} parquet files, {total_rows:,} total rows, symbols: {sorted(symbols)}")
else:
    errors.append("[FAIL] data/raw/ - no parquet files found!")

# 5. Check notebook
nb = W / "train_brain.ipynb"
if nb.exists() and nb.stat().st_size > 1000:
    ok.append(f"[OK] train_brain.ipynb ({nb.stat().st_size/1e6:.2f}MB)")
else:
    errors.append("[FAIL] train_brain.ipynb missing or empty!")

# 6. Check txt files
for tf in sorted(W.glob("*.txt")):
    ok.append(f"[OK] {tf.name} ({tf.stat().st_size} bytes)")

# Print results
print("=" * 70)
print("  VERIFICATION REPORT")
print("=" * 70)
for line in ok:
    print(f"  {line}")

if errors:
    print(f"\n  *** ERRORS ({len(errors)}) ***")
    for e in errors:
        print(f"  {e}")
else:
    print(f"\n  >>> ALL {len(ok)} ITEMS VERIFIED OK - READY FOR ANOTHER LAPTOP <<<")

total = sum(f.stat().st_size for f in W.rglob("*") if f.is_file())
nfiles = sum(1 for f in W.rglob("*") if f.is_file())
print(f"\n  TOTAL: {total/1e6:.1f} MB | {nfiles} files")
print("=" * 70)
