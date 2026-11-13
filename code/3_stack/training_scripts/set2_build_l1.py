"""Portable entry point for the original set2_build_l1 calculation."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import stack_pipeline as stack
p = argparse.ArgumentParser()
p.add_argument('--previous-output', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
ctx = stack.load_inputs(root)
f = a.previous_output
stack.build_l1(ctx, root / 'reference' / stack.STAGES[0], f / stack.STAGES[1],
               f / stack.STAGES[3], f / stack.STAGES[4], a.output)
