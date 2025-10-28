# make_fermi_jsonl.py
import json, argparse, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)   # e.g., data/fermi_train.txt
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--conf", type=float, default=0.99)  # 99% default
    args = ap.parse_args()

    with open(args.in_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    out = []
    i = 0
    while i + 1 < len(lines):
        q = lines[i]
        y = int(lines[i+1])
        conf = args.conf
        alpha = 1.0 - conf
        prompt = (
            "You are a careful quantitative estimator for Fermi-style questions.\n"
            f"Provide brief reasoning if useful, but you MUST END with a single JSON object "
            f'of the form {{"L": <int>, "U": <int>}} where L and U are INTEGERS (can be negative), '
            f"representing a {conf*100:.2f}% confidence interval [10^L, 10^U]. If ≥{conf*100:.2f}% of the mass is on exactly 10^x, "
            "use L=U=x. Do not include units in the JSON. Ensure L <= U. Keep the response less than 256 tokens.\n\n"
            f"Question: {q}\n\n"
            f"Give a {conf*100:.2f}% confidence interval in the form [10^L, 10^U] with INTEGER L and U.\n"
            'Finish with JSON ONLY: {"L": <int>, "U": <int>}.'
        )
        out.append({
            "id": f"fermi-{i//2}",
            "prompt": prompt,
            "meta": {"answer": y, "alpha": alpha}
        })
        i += 2

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records to {args.out_jsonl}")

if __name__ == "__main__":
    main()