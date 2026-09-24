"""Offline research commands only; no server or broker operations."""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from .demo import make_demo
from .engine import Policy
from .journal import Session


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant forbidden: " + value)


def parse_json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_policy(path):
    path = Path(path)
    if path.stat().st_size > 2*1024*1024: raise ValueError("policy exceeds 2MB")
    value = parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError("policy object required")
    return Policy(**value)


def read_events(path):
    with Path(path).open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline(1024*1024+1)
            if not line: return
            line_number += 1
            if len(line) > 1024*1024: raise ValueError(f"event line {line_number} exceeds 1MB")
            if not line.strip(): raise ValueError(f"blank event line {line_number}")
            value = parse_json(line.decode("utf-8"))
            if not isinstance(value, dict): raise ValueError("event object required")
            yield value


def run_events(policy, events, db, output=None):
    started = time.perf_counter()
    chain = hashlib.sha256()
    count = 0
    with Session(db, policy) as session:
        for event in events:
            result = session.ingest(event)
            chain.update(canonical(result).encode("utf-8") + b"\n")
            count += 1
        summary = session.summary()
    result = {"authority": "research_only", "execution_allowed": False,
              "processed_events": count, "result_digest": chain.hexdigest(),
              "elapsed_seconds": time.perf_counter()-started, "summary": summary}
    if output is not None: write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Causal adaptive research, never broker execution")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="ingest chronological JSONL; existing exact events are idempotent")
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--events", required=True, type=Path)
    run.add_argument("--db", required=True, type=Path)
    run.add_argument("--report", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify hash chain and deterministic causal replay")
    verify.add_argument("--policy", required=True, type=Path)
    verify.add_argument("--db", required=True, type=Path)
    demo = commands.add_parser("demo", help="synthetic delayed-feedback scenario; not market evidence")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--steps", default=500, type=int)
    demo.add_argument("--seed", default=7, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            if not args.db.is_file(): raise ValueError("database does not exist")
            with Session(args.db, load_policy(args.policy)) as session:
                result = {"verified_replay": True, "summary": session.summary()}
        elif args.command == "run":
            if args.report.exists(): raise ValueError("report already exists; choose a new path")
            paths = [args.policy.resolve(), args.events.resolve(), args.db.resolve(), args.report.resolve()]
            if len(set(paths)) != len(paths): raise ValueError("input/output paths must be distinct")
            result = run_events(load_policy(args.policy), read_events(args.events), args.db, args.report)
        else:
            policy, events, baselines = make_demo(args.steps, args.seed)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "policy.json", policy.to_dict())
            with (args.output / "events.jsonl").open("x", encoding="utf-8") as stream:
                for event in events: stream.write(canonical(event)+"\n")
            result = run_events(policy, events, args.output / "research.db")
            result.update({"synthetic": True, "seed": args.seed, "baselines": baselines,
                           "market_validation": False, "profitability_claim": False})
            write_json(args.output / "report.json", result)
        print(canonical(result))
        return 0
    except (ValueError, TypeError, OSError, ArithmeticError, sqlite3.Error) as exc:
        print(f"research command failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
