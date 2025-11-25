import numpy as np
import pretty_midi
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import os
from pathlib import Path



def select_files_to_balance(data_dir):
    # Gather files and sizes for each directory
    data_root = Path(data_dir)
    directories = list(data_root.glob('*'))
    dir_files = {}
    total_size = 0
    max_dir_size = 0
    for d in directories:
        dir_size = 0
        files = []
        for f in Path(d).glob('*'):
            if f.is_file():
                dir_size += f.stat().st_size
                files.append((f, f.stat().st_size))
                total_size += f.stat().st_size
        max_dir_size = max(max_dir_size, dir_size)
        dir_files[d] = files

    num_dirs = len(directories)
    target = total_size // num_dirs

    copy_counter = Counter()
    for d, files in dir_files.items():
        # Greedy subset sum: largest files first
        files = sorted(files, key=lambda x: -x[1])
        sel = []
        size = 0
        while size < target and files:
            for f, s in files:
                if size + s <= target or not sel:  # always pick at least one file
                    copy_counter[f] += 1
                    size += s
        print(f"{d}: selected {len(copy_counter)} files, total size {size/1e6:.2f} MB (target {target/1e6:.2f} MB)")

    return copy_counter

def analyze_voice_counts(midi_dir):
    """
    Analyze MIDI files to determine the maximum number of voices needed
    
    Args:
        midi_dir: Directory containing MIDI files
        
    Returns:
        int: Maximum number of voices found across all files
    """
    midi_root = Path(midi_dir)
    midi_files = list(midi_root.glob('**/*.mid')) + list(midi_root.glob('**/*.midi'))
    
    max_voices = 0
    
    for midi_path in tqdm(midi_files, desc="Analyzing voice counts"):
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Method 1: Count instruments
            instrument_count = len([i for i in midi_data.instruments if not i.is_drum])
            
            # Method 2: Maximum simultaneous notes
            piano_roll = midi_data.get_piano_roll(fs=100)
            simultaneous_notes = np.count_nonzero(piano_roll > 0, axis=0)
            max_simultaneous = np.max(simultaneous_notes) if piano_roll.size > 0 else 0
            
            file_max_voices = max(instrument_count, max_simultaneous)
            max_voices = max(max_voices, file_max_voices)
            
        except Exception as e:
            print(f"Error analyzing {midi_path}: {e}")
    
    return max_voices

def extract_voices(midi_data, max_voices=40):
    """
    Extract voice assignments for notes in a MIDI file
    
    Args:
        midi_data: PrettyMIDI object
        max_voices: Maximum number of voices to consider
        
    Returns:
        dict: Mapping of (start_time, pitch) to voice_id
    """
    voice_assignments = {}  # {(start_time, pitch): voice_id}
    
    # First pass: assign by instrument
    voice_id = 0
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        
        for note in instrument.notes:
            voice_assignments[(note.start, note.pitch)] = min(voice_id, max_voices - 1)
        
        voice_id += 1
    
    # Second pass: for instruments with multiple simultaneous notes,
    # try to separate into different voices
    if voice_id < 3:  # Only apply for pieces with few instruments
        # Group notes by start time to find simultaneous notes
        notes_by_start = defaultdict(list)
        
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                notes_by_start[note.start].append((note.pitch, note.end, instrument.program))
        
        # For each time point with multiple notes, assign separate voices
        for start_time, notes in notes_by_start.items():
            if len(notes) > 1:
                # Sort by pitch (low to high)
                sorted_notes = sorted(notes)
                
                # Assign voices based on pitch height (approximate voice-leading)
                for i, (pitch, _, _) in enumerate(sorted_notes):
                    voice = min(i, max_voices - 1)
                    voice_assignments[(start_time, pitch)] = voice
    
    return voice_assignments


def get_note_rhythm_class(note, midi_data, tolerance=0.2):
    """
    Classify a note's rhythmic value (whole, half, quarter, etc.) using pretty_midi.
    Handles tempo changes by using the local tempo at note start.
    Args:
        note: pretty_midi.Note object
        midi_data: pretty_midi.PrettyMIDI object
        tolerance: allowed fractional deviation for class assignment
    Returns:
        str: rhythmic class name
    """
    # Get tempo changes and times
    tempo_times, tempi = midi_data.get_tempo_changes()
    # Find the tempo at the note's start time
    idx = np.searchsorted(tempo_times, note.start, side='right') - 1
    tempo = tempi[idx]
    quarter_length = 60.0 / tempo  # seconds per quarter note

    # Compute duration in quarter notes
    duration = note.end - note.start
    return duration / quarter_length


def preprocess_midi_to_sparse(midi_dir, output_dir, composer_map=None):
    """
    Process MIDI files to sparse event-based format with voice tracking
    
    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Where to save processed files
        composer_map: Optional dict mapping composer names to numerical IDs
                     If None, will be generated automatically
    """
    midi_root = Path(midi_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get max voices needed across all files
    print("Analyzing maximum voice count needed...")
    max_voices = analyze_voice_counts(midi_dir)
    print(f"Maximum voices detected: {max_voices}")
    
    midi_files = list(midi_root.glob('**/*.mid')) + list(midi_root.glob('**/*.midi'))
    
    # Create composer mapping if not provided
    if composer_map is None:
        unique_composers = set()
        for midi_path in midi_files:
            unique_composers.add(midi_path.parent.name)
        
        composer_map = {composer: idx for idx, composer in enumerate(sorted(unique_composers))}
    
    # Save composer mapping
    composer_map_path = output_root / "composer_map.json"
    import json
    with open(composer_map_path, 'w') as f:
        json.dump(composer_map, f, indent=2)
    
    # Track statistics
    stats = {
        "total_files": len(midi_files),
        "processed_files": 0,
        "max_voices_used": 0,
        "composers": defaultdict(int),
        "rhythm_classes": defaultdict(int)
    }
    
    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        try:
            composer = midi_path.parent.name
            composer_id = composer_map.get(composer, -1)
            
            if composer_id == -1:
                print(f"Warning: Unknown composer {composer}")
                continue
            
            stats["composers"][composer] += 1
            
            composer_dir = output_root / composer
            composer_dir.mkdir(exist_ok=True)
            
            # Load MIDI
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Get voice assignments
            voice_assignments = extract_voices(midi_data, max_voices)
            max_voice_in_piece = max([v for v in voice_assignments.values()]) if voice_assignments else 0
            stats["max_voices_used"] = max(stats["max_voices_used"], max_voice_in_piece + 1)
            
            # Extract note events (enhanced sparse representation)
            events = []
            
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue  # Skip drum tracks
                
                for note in instrument.notes:
                    voice_id = voice_assignments.get((note.start, note.pitch), 0)
                    rhythm_class = get_note_rhythm_class(note, midi_data)
                    stats['rhythm_classes'][rhythm_class] += 1
                    # Store: [start_time, end_time, pitch, velocity, instrument_program, voice_id]
                    events.append([
                        note.start,
                        note.end,
                        note.pitch,
                        note.velocity,
                        instrument.program,
                        voice_id,
                        rhythm_class
                    ])
            
            # Convert to numpy array and sort by start time
            if events:
                events = np.array(events)
                events = events[events[:, 0].argsort()]
                
                # Save enhanced sparse representation
                output_path = composer_dir / f"{midi_path.stem}.npz"
                np.savez_compressed(
                    output_path,
                    events=events,
                    composer_id=composer_id,
                    composer=composer,
                    max_voices=stats["max_voices_used"],
                    piece_max_voice=max_voice_in_piece + 1
                )
                
                stats["processed_files"] += 1
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
    
    # Save processing statistics
    stats_path = output_root / "processing_stats.json"
    
    with open(stats_path, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        stats["composers"] = dict(stats["composers"])
        json.dump(stats, f, indent=2)
    
    rhythm_map_path = output_root / "rhythm_map.json"
    unique_rhythm_classes = list(set(stats["rhythm_classes"].keys()))
    print(f'Unique rhythm classes: {len(unique_rhythm_classes)}')
    rhythm_map = {beat_ratio: idx for idx, beat_ratio in enumerate(sorted(unique_rhythm_classes))}
    with open(rhythm_map_path, 'w') as f:
        json.dump(rhythm_map, f, indent=2)

    
    print(f"Processed {stats['processed_files']} of {stats['total_files']} files")
    print(f"Maximum voices used: {stats['max_voices_used']}")
    print(f"Statistics saved to {stats_path}")
    print(f"Composer mapping saved to {composer_map_path}")
    print(f"Rhythm mapping saved to {rhythm_map_path}")
