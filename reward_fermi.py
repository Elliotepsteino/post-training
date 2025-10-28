# reward_fermi.py
import re, math, json
from typing import Optional, Tuple, Dict, Any, List

# -------- helpers to parse model JSON { "L": ..., "U": ... } --------
JSON_BLOCK_RE = re.compile(r'\{[^{}]*"L"\s*:\s*-?\d+[^{}]*"U"\s*:\s*-?\d+[^{}]*\}', re.DOTALL)
INT_RE        = re.compile(r'"L"\s*:\s*(-?\d+).*?"U"\s*:\s*(-?\d+)', re.DOTALL)
LEADING_JUNK_PAT = re.compile(
    r"""^\s*(?:\ufeff)?(?:(?:łazienk\w*|aimassage|SpecWarn|NdrFcShort)[\s:–—-]*)+""",
    re.IGNORECASE | re.VERBOSE,
)

def sanitize_leading_noise(s: str) -> str:
    s = s or ""
    s2 = LEADING_JUNK_PAT.sub("", s)
    s2 = re.sub(r'^[^\x20-\x7E]+', '', s2)
    return s2.lstrip()

def parse_LU(text: str) -> Optional[Tuple[int,int]]:
    m = JSON_BLOCK_RE.search(text or "")
    if not m:
        return None
    m2 = INT_RE.search(m.group(0))
    if not m2:
        return None
    L = int(m2.group(1)); U = int(m2.group(2))
    if L > U:
        L, U = U, L
    return (L, U)

# -------- Winkler (interval) score; we return its negative as a reward --------
def winkler_reward(y: int, L: Optional[int], U: Optional[int], alpha: float,
                   clip_abs: Optional[float] = 500.0, soft: bool = False) -> float:
    if L is None or U is None:
        base = (clip_abs if (clip_abs is not None and clip_abs > 0) else 500.0)
        return -2.0 * base
    if L > U:
        L, U = U, L
    width = U - L
    miss = 0
    if y < L:
        miss = L - y
    elif y > U:
        miss = y - U
    ws = width + (2.0 / alpha) * miss
    if clip_abs is not None and clip_abs > 0:
        ws = clip_abs * math.asinh(ws / clip_abs) if soft else min(ws, clip_abs)
    return -float(ws)

# -------- ground truth parsing from VERL’s ground_truth string --------
def _parse_gt(ground_truth) -> Tuple[int, float]:
    """
    Accepts:
      - JSON string: '{"answer": 7, "alpha": 0.01}'
      - dict:        {"answer": 7, "alpha": 0.01}
      - bare int or int-like string: '7'
    Returns: (answer:int, alpha:float) where alpha defaults to 0.01 (99%).
    """
    if isinstance(ground_truth, dict):
        ans = ground_truth.get("answer")
        alpha = ground_truth.get("alpha", 0.01)
        if ans is None:
            raise KeyError("ground_truth dict missing 'answer'")
        return int(ans), float(alpha)

    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        # Try JSON first
        if s and (s[0] == "{" and s[-1] == "}"):
            obj = json.loads(s)
            return _parse_gt(obj)
        # Try bare int
        try:
            return int(s), 0.01
        except Exception:
            pass

    if isinstance(ground_truth, (int, float)):
        return int(ground_truth), 0.01

    raise TypeError(f"Unrecognized ground_truth type: {type(ground_truth)}")

# ========================= VERL entrypoint =========================
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    VERL will call this per-sample. Must return a scalar float.
      data_source: str  (your parquet column 'data_source')
      solution_str: str (detokenized model output)
      ground_truth: str (what you stored in parquet under 'ground_truth')
      extra_info:  dict (optional misc)
    """
    # Clean & parse model output
    out = sanitize_leading_noise(solution_str or "")
    lu = parse_LU(out)

    # Get (y, alpha) from ground_truth
    y, alpha = _parse_gt(ground_truth)

    # Compute reward (more negative WS => worse; we return negative WS as reward)
    r = winkler_reward(y, *(lu if lu else (None, None)), alpha=alpha)

    # Small penalty for empty/blank output
    if not out.strip():
        r -= 5.0

    return float(r)

# -------- Optional: legacy batch wrapper for quick local tests --------
def reward_fn(samples: List[Dict[str, Any]]) -> List[float]:
    """Accepts a list of dicts with at least {'response', 'ground_truth'}."""
    scores = []
    for s in samples:
        scores.append(
            compute_score(
                data_source=s.get("data_source"),
                solution_str=s.get("response", "") or "",
                ground_truth=s.get("ground_truth"),
                extra_info=None,
            )
        )
    return scores





