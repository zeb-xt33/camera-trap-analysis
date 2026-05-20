import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    "data/raw_images",
    "data/metadata",
    "processed_data",
    "notebooks",
    "scripts",
    "outputs/figures",
    "outputs/reports",
]

FILES = {
    ".gitignore": "",
    "requirements.txt": "",
    "README.md": "",
}

def create_dirs():
    for d in DIRS:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  OK  {d}/")

def create_gitkeeps():
    for d in DIRS:
        path = PROJECT_ROOT / d
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("")
            print(f"  OK  {d}/.gitkeep")

if __name__ == "__main__":
    print("Setting up camera-trap-analysis project structure...")
    create_dirs()
    create_gitkeeps()
    print("Done!")
