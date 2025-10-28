import json, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_jsonl")
    ap.add_argument("out_parquet")
    ap.add_argument("--alpha", type=float, default=0.01)  # default 99% conf
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            meta = obj.get("meta", {})
            answer = meta.get("answer")
            alpha = meta.get("alpha", args.alpha)

            # This is what VERL uses via data.reward_fn_key=ground_truth
            ground_truth = json.dumps({"answer": answer, "alpha": alpha}, ensure_ascii=False)

            # Also include VERL-compatible optional fields for compatibility
            meta_field = {"answer": answer, "alpha": alpha}
            reward_model_field = {"ground_truth": answer, "alpha": alpha}

            rows.append({
                "id": obj.get("id"),
                "prompt": prompt,
                "ground_truth": ground_truth,        # ✅ required by VERL
                "meta": meta_field,
                "reward_model": reward_model_field,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(args.out_parquet, index=False)
    print(f"✅ Wrote {len(df)} rows → {args.out_parquet} with columns {list(df.columns)}")

if __name__ == "__main__":
    main()

