import sys

HARD = "hard" in sys.argv[1:]
SAMPLE_PATH = "data/processed/test_sample_hard.csv" if HARD else "data/processed/test_sample.csv"
SUFFIX = "_hard" if HARD else ""
METRICS_PATH = f"results/metrics{SUFFIX}.csv"
