"""Portable entry point for the original build_xgb_delta calculation."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import stack_pipeline as stack
p = argparse.ArgumentParser()
p.add_argument('--curvature', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
stack.build_xgb_delta(stack.load_inputs(root), a.curvature, a.output)
