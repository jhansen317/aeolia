#!/usr/bin/env python3
import os
import sys
import traceback
from pathlib import Path

# Add the project root to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.data.preprocessing import preprocess_midi_to_sparse

def main():
    # Source directory with MIDI files
    source_dir = Path.home() / "local_corpus_test"
    
    # Target directory for processed files (relative to script location)
    target_dir = project_root / "data" / "raw_test"

    # Ensure source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found")
        sys.exit(1)
    
    print(f"Processing MIDI files from: {source_dir}")
    print(f"Saving processed files to: {target_dir}")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the MIDI files
    try:
        preprocess_midi_to_sparse(
            midi_dir=source_dir,
            output_dir=target_dir,
            composer_map=None  # Let it auto-generate the composer mapping
        )
        print("MIDI processing completed successfully!")
    except Exception as e:
        print(f"Error processing MIDI files: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()