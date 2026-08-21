"""Tabulate eval_results.json files into tasks x angles (success rate).

    python scripts/collect_results.py --log_root logs --agent equibot \
        --tasks can square_d1 --angles 0 5 15 30 --out logs/results_equibot
writes <out>.csv and <out>.md and prints the markdown table.
"""

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root", default="logs")
    p.add_argument("--agent", default="equibot")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--angles", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    table = {}
    for task in args.tasks:
        row = {}
        for deg in args.angles:
            f = os.path.join(args.log_root, "eval",
                             f"eval_{task}_{args.agent}_cam{deg}", "eval_results.json")
            if not os.path.exists(f):
                row[deg] = None
                continue
            r = json.load(open(f))
            row[deg] = (r["success_rate"], len(r.get("rew_values", [])) or r.get("num_episodes"))
        table[task] = row

    hdr = "| task | " + " | ".join(f"{d}°" for d in args.angles) + " |"
    sep = "|---|" + "---|" * len(args.angles)
    lines = [hdr, sep]
    csv = ["task," + ",".join(args.angles)]
    for task, row in table.items():
        cells, ccells = [], []
        for deg in args.angles:
            v = row[deg]
            if v is None:
                cells.append("–"); ccells.append("")
            else:
                cells.append(f"{v[0]:.2f} (n={v[1]})"); ccells.append(f"{v[0]:.4f}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
        csv.append(task + "," + ",".join(ccells))
    # mean over tasks per angle (only where every task has a number)
    means = []
    for deg in args.angles:
        vals = [table[t][deg][0] for t in args.tasks if table[t][deg] is not None]
        means.append(f"{sum(vals)/len(vals):.2f}" if len(vals) == len(args.tasks) else "–")
    lines.append("| **mean** | " + " | ".join(means) + " |")
    md = "\n".join(lines)
    print(f"\nsuccess rate, agent={args.agent}\n{md}\n")
    if args.out:
        with open(args.out + ".md", "w") as fh:
            fh.write(md + "\n")
        with open(args.out + ".csv", "w") as fh:
            fh.write("\n".join(csv) + "\n")
        print(f"wrote {args.out}.md / .csv")


if __name__ == "__main__":
    main()
