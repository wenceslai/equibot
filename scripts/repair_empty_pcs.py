"""Fix already-generated datasets that contain frames with an empty `pc`
(objects fully occluded for a moment — crashes BaseDataset's
np.random.choice during training). Rewrites each empty frame with the
previous frame's cloud, exactly what generate_demos.py now does at
conversion time. Idempotent; eef_pos/action are untouched.

    python scripts/repair_empty_pcs.py $DATA_ROOT/equibot/*/pcs
"""

import re
import sys
from collections import defaultdict

import numpy as np


def repair(pcs_dir):
    import glob, os
    files = sorted(glob.glob(os.path.join(pcs_dir, "*.npz")))
    eps = defaultdict(list)
    for f in files:
        m = re.match(r"(.+)_t(\d+)\.npz$", os.path.basename(f))
        eps[m.group(1)].append((int(m.group(2)), f))
    n_fixed = 0
    for ep, frames in sorted(eps.items()):
        prev_pc = None
        for _, f in sorted(frames):
            d = dict(np.load(f))
            if len(d["pc"]) == 0:
                if prev_pc is None:
                    raise SystemExit(
                        f"{f}: FIRST frame of {ep} is empty — the dataset is "
                        f"broken (wrong gt_bodies / renderer), regenerate it")
                d["pc"] = prev_pc
                np.savez(f, **d)
                n_fixed += 1
            else:
                prev_pc = d["pc"]
    print(f"{pcs_dir}: {len(files)} frames, {n_fixed} empty frames repaired")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        repair(d)
