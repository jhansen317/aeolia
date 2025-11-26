"""
Utilities for converting between MIDI files and tensor representations.

This module provides functions to convert MIDI files to tensor format and vice versa,
reusing the preprocessing logic from src/data/preprocessing.py.
"""

import numpy as np
import pretty_midi
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


def extract_voices(midi_data: pretty_midi.PrettyMIDI, max_voices: int = 40) -> Dict[Tuple[float, int], int]:
    """
    Extract voice assignments for notes in a MIDI file.

    Reuses logic from preprocessing.py to ensure consistency.

    Args:
        midi_data: PrettyMIDI object
        max_voices: Maximum number of voices to consider

    Returns:
        dict: Mapping of (start_time, pitch) to voice_id
    """
    voice_assignments = {}

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
    if voice_id < 3:
        notes_by_start = defaultdict(list)

        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                notes_by_start[note.start].append((note.pitch, note.end, instrument.program))

        for start_time, notes in notes_by_start.items():
            if len(notes) > 1:
                sorted_notes = sorted(notes)

                for i, (pitch, _, _) in enumerate(sorted_notes):
                    voice = min(i, max_voices - 1)
                    voice_assignments[(start_time, pitch)] = voice

    return voice_assignments


def get_note_rhythm_class(note: pretty_midi.Note, midi_data: pretty_midi.PrettyMIDI,
                          tolerance: float = 0.2) -> float:
    """
    Classify a note's rhythmic value using pretty_midi.

    Reuses logic from preprocessing.py to ensure consistency.

    Args:
        note: pretty_midi.Note object
        midi_data: pretty_midi.PrettyMIDI object
        tolerance: allowed fractional deviation for class assignment

    Returns:
        float: duration in quarter notes
    """
    tempo_times, tempi = midi_data.get_tempo_changes()
    idx = np.searchsorted(tempo_times, note.start, side='right') - 1
    tempo = tempi[idx]
    quarter_length = 60.0 / tempo

    duration = note.end - note.start
    return duration / quarter_length


def midi_to_events(midi_path: Union[str, Path],
                   max_voices: int = 40) -> np.ndarray:
    """
    Convert a MIDI file to event-based sparse representation.

    Args:
        midi_path: Path to MIDI file
        max_voices: Maximum number of voices to track

    Returns:
        np.ndarray: Array of events with shape (num_events, 7)
                   Columns: [start_time, end_time, pitch, velocity, program, voice_id, rhythm_class]
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    voice_assignments = extract_voices(midi_data, max_voices)

    events = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            voice_id = voice_assignments.get((note.start, note.pitch), 0)
            rhythm_class = get_note_rhythm_class(note, midi_data)

            events.append([
                note.start,
                note.end,
                note.pitch,
                note.velocity,
                instrument.program,
                voice_id,
                rhythm_class
            ])

    if events:
        events = np.array(events)
        events = events[events[:, 0].argsort()]  # Sort by start time
        return events

    return np.array([])


def events_to_tensor(events: np.ndarray,
                     seq_length: int = 32,
                     time_step: float = 0.25,
                     start_time: float = 0.0,
                     num_pitches: int = 128,
                     num_voices: int = 40,
                     num_rhythm_classes: int = 39) -> Dict[str, torch.Tensor]:
    """
    Convert event-based representation to tensor format.

    Args:
        events: Event array from midi_to_events
        seq_length: Number of time steps
        time_step: Time between steps in seconds
        start_time: Starting time in the MIDI file
        num_pitches: Number of pitch nodes
        num_voices: Number of voice nodes
        num_rhythm_classes: Number of rhythm class nodes

    Returns:
        dict: Dictionary containing:
            - 'features': (seq_length, num_nodes) tensor of node features
            - 'pitches': (seq_length, num_pitches) piano roll
            - 'voices': (seq_length, num_pitches) voice assignments
            - 'rhythm_classes': (seq_length, num_pitches) rhythm classes
    """
    num_nodes = num_pitches + num_voices + num_rhythm_classes + 1  # +1 for composer node

    features = torch.zeros((seq_length, num_nodes))
    piano_roll = torch.zeros((seq_length, num_pitches))
    voice_roll = torch.zeros((seq_length, num_pitches))
    rhythm_roll = torch.zeros((seq_length, num_pitches))

    for t in range(seq_length):
        current_time = start_time + (t * time_step)

        # Find active notes at this time step
        active_notes = events[
            (events[:, 0] <= current_time) &
            (events[:, 1] >= current_time)
        ]

        for note in active_notes:
            pitch = int(note[2])
            velocity = note[3] / 127.0
            voice_id = int(note[5])
            rhythm_class = min(int(note[6] / 0.125), num_rhythm_classes - 1)

            if 0 <= pitch < num_pitches:
                # Set features
                features[t, pitch] = velocity
                features[t, num_pitches + voice_id] = velocity
                features[t, num_pitches + num_voices + rhythm_class] = velocity

                # Set rolls
                piano_roll[t, pitch] = velocity
                voice_roll[t, pitch] = voice_id
                rhythm_roll[t, pitch] = rhythm_class

    return {
        'features': features,
        'piano_roll': piano_roll,
        'voice_roll': voice_roll,
        'rhythm_roll': rhythm_roll,
    }


def midi_to_tensor(midi_path: Union[str, Path],
                   seq_length: int = 32,
                   time_step: float = 0.25,
                   start_time: float = 0.0,
                   max_voices: int = 40,
                   num_pitches: int = 128,
                   num_rhythm_classes: int = 39) -> Dict[str, torch.Tensor]:
    """
    Convert a MIDI file directly to tensor representation.

    Args:
        midi_path: Path to MIDI file
        seq_length: Number of time steps
        time_step: Time between steps in seconds
        start_time: Starting time in the MIDI file
        max_voices: Maximum number of voices
        num_pitches: Number of pitch nodes
        num_rhythm_classes: Number of rhythm class nodes

    Returns:
        dict: Dictionary containing tensor representations
    """
    events = midi_to_events(midi_path, max_voices)
    return events_to_tensor(
        events,
        seq_length=seq_length,
        time_step=time_step,
        start_time=start_time,
        num_pitches=num_pitches,
        num_voices=max_voices,
        num_rhythm_classes=num_rhythm_classes
    )


def tensor_to_events(features: torch.Tensor,
                     piano_roll: Optional[torch.Tensor] = None,
                     voice_roll: Optional[torch.Tensor] = None,
                     rhythm_roll: Optional[torch.Tensor] = None,
                     time_step: float = 0.25,
                     start_time: float = 0.0,
                     num_pitches: int = 128,
                     num_voices: int = 40,
                     velocity_threshold: float = 0.01,
                     default_program: int = 0) -> np.ndarray:
    """
    Convert tensor representation back to event-based format.

    Args:
        features: (seq_length, num_nodes) tensor of node features
        piano_roll: Optional (seq_length, num_pitches) tensor
        voice_roll: Optional (seq_length, num_pitches) voice assignments
        rhythm_roll: Optional (seq_length, num_pitches) rhythm classes
        time_step: Time between steps in seconds
        start_time: Starting time offset
        num_pitches: Number of pitch nodes
        num_voices: Number of voice nodes
        velocity_threshold: Minimum velocity to consider a note active
        default_program: Default MIDI program for all notes

    Returns:
        np.ndarray: Event array with shape (num_events, 7)
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if piano_roll is not None and isinstance(piano_roll, torch.Tensor):
        piano_roll = piano_roll.cpu().numpy()
    if voice_roll is not None and isinstance(voice_roll, torch.Tensor):
        voice_roll = voice_roll.cpu().numpy()
    if rhythm_roll is not None and isinstance(rhythm_roll, torch.Tensor):
        rhythm_roll = rhythm_roll.cpu().numpy()

    # Use piano_roll if provided, otherwise extract from features
    if piano_roll is None:
        piano_roll = features[:, :num_pitches]

    seq_length = features.shape[0]
    events = []

    # Track which notes are currently active
    active_notes = {}  # {pitch: (start_time, velocity, voice_id, rhythm_class)}

    for t in range(seq_length):
        current_time = start_time + (t * time_step)

        for pitch in range(num_pitches):
            velocity = piano_roll[t, pitch]
            is_active = velocity > velocity_threshold

            if is_active:
                # Determine voice and rhythm
                voice_id = 0
                rhythm_class = 0.0

                if voice_roll is not None:
                    voice_id = int(voice_roll[t, pitch])
                else:
                    # Extract from features if available
                    voice_features = features[t, num_pitches:num_pitches + num_voices]
                    if voice_features.max() > velocity_threshold:
                        voice_id = int(voice_features.argmax())

                if rhythm_roll is not None:
                    rhythm_class = rhythm_roll[t, pitch] * 0.125

                # Start or continue note
                if pitch not in active_notes:
                    active_notes[pitch] = (current_time, velocity, voice_id, rhythm_class)
            else:
                # End note if it was active
                if pitch in active_notes:
                    start, vel, voice, rhythm = active_notes[pitch]
                    events.append([
                        start,
                        current_time,
                        pitch,
                        int(vel * 127),
                        default_program,
                        voice,
                        rhythm
                    ])
                    del active_notes[pitch]

    # Close any remaining active notes
    final_time = start_time + (seq_length * time_step)
    for pitch, (start, vel, voice, rhythm) in active_notes.items():
        events.append([
            start,
            final_time,
            pitch,
            int(vel * 127),
            default_program,
            voice,
            rhythm
        ])

    if events:
        events = np.array(events)
        events = events[events[:, 0].argsort()]  # Sort by start time
        return events

    return np.array([])


def events_to_midi(events: np.ndarray,
                   output_path: Optional[Union[str, Path]] = None,
                   tempo: float = 120.0,
                   program: int = 0) -> pretty_midi.PrettyMIDI:
    """
    Convert event-based representation to MIDI file.

    Args:
        events: Event array with shape (num_events, 7)
        output_path: Optional path to save MIDI file
        tempo: Tempo in BPM
        program: MIDI program number

    Returns:
        pretty_midi.PrettyMIDI: MIDI object
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)

    for event in events:
        start_time, end_time, pitch, velocity, prog, voice_id, rhythm_class = event

        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start_time),
            end=float(end_time)
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)

    if output_path is not None:
        midi.write(str(output_path))

    return midi


def tensor_to_midi(features: torch.Tensor,
                   output_path: Optional[Union[str, Path]] = None,
                   piano_roll: Optional[torch.Tensor] = None,
                   voice_roll: Optional[torch.Tensor] = None,
                   rhythm_roll: Optional[torch.Tensor] = None,
                   time_step: float = 0.25,
                   start_time: float = 0.0,
                   num_pitches: int = 128,
                   num_voices: int = 40,
                   tempo: float = 120.0,
                   program: int = 0,
                   velocity_threshold: float = 0.01) -> pretty_midi.PrettyMIDI:
    """
    Convert tensor representation directly to MIDI file.

    Args:
        features: (seq_length, num_nodes) tensor of node features
        output_path: Optional path to save MIDI file
        piano_roll: Optional (seq_length, num_pitches) tensor
        voice_roll: Optional (seq_length, num_pitches) voice assignments
        rhythm_roll: Optional (seq_length, num_pitches) rhythm classes
        time_step: Time between steps in seconds
        start_time: Starting time offset
        num_pitches: Number of pitch nodes
        num_voices: Number of voice nodes
        tempo: Tempo in BPM
        program: MIDI program number
        velocity_threshold: Minimum velocity to consider a note active

    Returns:
        pretty_midi.PrettyMIDI: MIDI object
    """
    events = tensor_to_events(
        features=features,
        piano_roll=piano_roll,
        voice_roll=voice_roll,
        rhythm_roll=rhythm_roll,
        time_step=time_step,
        start_time=start_time,
        num_pitches=num_pitches,
        num_voices=num_voices,
        velocity_threshold=velocity_threshold,
        default_program=program
    )

    return events_to_midi(events, output_path, tempo, program)


def batch_tensor_to_midi(features: torch.Tensor,
                        output_dir: Union[str, Path],
                        piano_roll: Optional[torch.Tensor] = None,
                        voice_roll: Optional[torch.Tensor] = None,
                        rhythm_roll: Optional[torch.Tensor] = None,
                        prefix: str = "generated",
                        **kwargs) -> List[pretty_midi.PrettyMIDI]:
    """
    Convert a batch of tensors to multiple MIDI files.

    Args:
        features: (batch_size, seq_length, num_nodes) tensor
        output_dir: Directory to save MIDI files
        piano_roll: Optional (batch_size, seq_length, num_pitches) tensor
        voice_roll: Optional (batch_size, seq_length, num_pitches) tensor
        rhythm_roll: Optional (batch_size, seq_length, num_pitches) tensor
        prefix: Prefix for output filenames
        **kwargs: Additional arguments passed to tensor_to_midi

    Returns:
        List[pretty_midi.PrettyMIDI]: List of MIDI objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = features.shape[0]
    midi_objects = []

    for i in range(batch_size):
        output_path = output_dir / f"{prefix}_{i:04d}.mid"

        batch_piano_roll = piano_roll[i] if piano_roll is not None else None
        batch_voice_roll = voice_roll[i] if voice_roll is not None else None
        batch_rhythm_roll = rhythm_roll[i] if rhythm_roll is not None else None

        midi = tensor_to_midi(
            features=features[i],
            output_path=output_path,
            piano_roll=batch_piano_roll,
            voice_roll=batch_voice_roll,
            rhythm_roll=batch_rhythm_roll,
            **kwargs
        )
        midi_objects.append(midi)

    return midi_objects
