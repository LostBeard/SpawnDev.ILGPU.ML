"""The /video driver: demo_page_run.py with --page video by default (kept so the command in ML CLAUDE.md keeps working).

    python video_demo_run.py --profile F:/_ml_demo_profile --url https://lostbeard.github.io/SpawnDev.ILGPU.ML/video --log run.jsonl --kill-after-gb 3

See demo_page_run.py for the checks; every other demo page with a default action of its own has a profile there.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_page_run import main

if __name__ == "__main__":
    main(default_page="video")
