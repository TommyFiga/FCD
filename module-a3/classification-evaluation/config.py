from pathlib import Path

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)