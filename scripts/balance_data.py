#!/usr/bin/env python3
import os
import sys
import traceback
from pathlib import Path
import shutil

# Add the project root to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.data.preprocessing import select_files_to_balance

def copy_with_suffix(d):
    for src, count in d.items():
        src_path = Path(src)
        parent = src_path.parent
        stem = src_path.stem
        suffix = src_path.suffix
        # Copy with numeric suffixes
        if count > 1:
            for i in range(1, count + 1):
                dest = parent / f"{stem}{i}{suffix}"
                shutil.copy2(src_path, dest)
                print(f"Copied {src_path} -> {dest}")


def main():
    # Source directory with MIDI files
    source_dir = Path.home() / "local_corpus_test"
    
    # Target directory for processed files (relative to script location)
    data_dir = project_root / "data" / "raw_test"

    print(f"balancing files in: {data_dir}")

    # Create target directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Process the MIDI files
    try:
        copy_with_suffix(select_files_to_balance(data_dir))
        print("MIDI processing completed successfully!")
    except Exception as e:
        print(f"Error processing MIDI files: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()