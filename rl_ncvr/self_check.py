import os
import sys
import tempfile
import shutil
import gzip
import json

import pandas as pd

from box_splitting import compute_box_id, get_box
from pair_generation import generate_positive_pairs_for_box
from eval_utils import EvalParams, evaluate_box


def make_tiny_ncvr_dir():
    tmpdir = tempfile.mkdtemp(prefix="tiny_ncvr_")
    cols = ["recid", "givenname", "surname", "postcode", "suburb"]
    A = [
        {"recid": "A", "givenname": "John", "surname": "Smith", "postcode": "27601", "suburb": "Raleigh"},
        {"recid": "A", "givenname": "Jon", "surname": "Smith", "postcode": "27601", "suburb": "Raleigh"},
        {"recid": "A", "givenname": "John", "surname": "Smyth", "postcode": "27601", "suburb": "Raleigh"},
        {"recid": "A", "givenname": "John", "surname": "Smith", "postcode": "27602", "suburb": "Raleigh"},
        {"recid": "A", "givenname": "John", "surname": "Smith", "postcode": "27601", "suburb": "Raleigh"},
    ]
    B = [
        {"recid": "B", "givenname": "Alice", "surname": "Lee", "postcode": "27513", "suburb": "Cary"},
        {"recid": "B", "givenname": "Alice", "surname": "Li", "postcode": "27513", "suburb": "Cary"},
    ]
    C = [
        {"recid": "C", "givenname": "Bob", "surname": "Brown", "postcode": "28202", "suburb": "Charlotte"},
    ]
    D = [
        {"recid": "D", "givenname": "Carol", "surname": "Davis", "postcode": "27701", "suburb": "Durham"},
        {"recid": "D", "givenname": "Carol", "surname": "Davis", "postcode": "27701", "suburb": "Durham"},
        {"recid": "D", "givenname": "Carol", "surname": "Davis", "postcode": "27701", "suburb": "Durham"},
    ]
    datasets = {i: [] for i in range(5)}
    for i in range(5):
        datasets[i].append(A[i])
    datasets[0].append(B[0])
    datasets[1].append(B[1])
    datasets[2].append(C[0])
    datasets[2].append(D[0])
    datasets[3].append(D[1])
    datasets[4].append(D[2])
    for i in range(5):
        df = pd.DataFrame(datasets[i], columns=cols)
        df.to_csv(os.path.join(tmpdir, f"dataset_{i+1}.csv"), index=False)
    return tmpdir


def main():
    print("[self-check] Creating tiny NCVR dir...")
    tiny = make_tiny_ncvr_dir()
    try:
        print("[self-check] Testing box splitting...")
        b = compute_box_id("A")
        parts = get_box(b, data_dir=tiny, recid_column="recid", chunksize=2)
        total = 0
        for src, df in parts.items():
            if not df.empty:
                mapped = df["recid"].astype(str).map(lambda x: compute_box_id(x))
                assert (mapped == b).all(), "Box filtering failed"
                total += len(df)
        print(f"[self-check] Box splitting OK. Rows in box={b}: {total}")

        print("[self-check] Testing pair generation...")
        out_path = os.path.join(tiny, "pairs.json.gz")
        output = generate_positive_pairs_for_box(
            box_id=b, data_dir=tiny, output_path=out_path, recid_column="recid", chunksize_hint=2
        )
        assert os.path.exists(output), "Pairs output missing"
        with gzip.open(output, "rt") as f:
            line = next(f, None)
            assert line is not None, "Pairs file empty"
            data = json.loads(line)
            assert isinstance(data, list) and len(data) == 2
        print("[self-check] Pair generation OK.")

        print("[self-check] Testing evaluation (CPU small)...")
        params = EvalParams(
            box_id=b,
            data_dir=tiny,
            batch_size=8,
            candidate_chunk_size=16,
            max_length=32,
            max_queries_per_source=10,
            max_candidates_per_source=50,
            shard_modulus=2,
            shard_remainder=0,
            verbose=True,
        )
        metrics = evaluate_box(params)
        assert isinstance(metrics, dict) and metrics, "Empty metrics"
        print("[self-check] Evaluation OK. Metrics:", metrics)

        print("[self-check] All checks passed.")
        return 0
    finally:
        shutil.rmtree(tiny, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())


