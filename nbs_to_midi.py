#!/usr/bin/env python3
"""
NBS to MIDI Converter
=====================
Converts Note Block Studio (.nbs) files to standard MIDI (.mid) files.

Based on the Open Note Block Studio NBS file format (versions 0–5).
See: https://github.com/OpenNBS/OpenNoteBlockStudio

Usage:
    python nbs_to_midi.py <input.nbs> [-o output.mid] [-d note_duration]

Arguments:
    input           Path to the .nbs file to convert.
    -o, --output    Output .mid file path. Defaults to the same name as
                    the input file with a .mid extension.
    -d, --duration  Duration of each note in NBS ticks (default: 1).
                    Increase this value to make notes sustain longer.
"""

import argparse
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# NBS data structures
# ---------------------------------------------------------------------------

@dataclass
class NbsNote:
    tick: int
    layer: int
    instrument: int
    key: int       # 0–87, where 0 = A0 (MIDI 21) and 45 = F#4 (MIDI 66)
    velocity: int  # 0–100 (default 100)
    panning: int   # 0–200, center = 100
    pitch: int     # fine pitch offset in 1/100ths of a semitone (±32767)


@dataclass
class NbsLayer:
    name: str
    lock: int    # 0 = unlocked, 1 = locked, 2 = solo
    volume: int  # 0–100
    stereo: int  # 0–200, center = 100


@dataclass
class NbsCustomInstrument:
    name: str
    filename: str
    key: int   # pitch of the sound sample (0–87)
    press: int # press-key-to-play flag


@dataclass
class NbsSong:
    version: int
    first_custom_index: int
    song_length: int
    layer_count: int
    song_name: str
    song_author: str
    original_author: str
    description: str
    tempo: float         # ticks per second
    time_signature: int
    loop: bool
    loop_max: int
    loop_start: int
    notes: List[NbsNote] = field(default_factory=list)
    layers: List[NbsLayer] = field(default_factory=list)
    custom_instruments: List[NbsCustomInstrument] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NBS reader
# ---------------------------------------------------------------------------

class NbsReader:
    """Parses an NBS binary file into an NbsSong object.

    Follows the format documented in open_song_nbs.gml and save_song.gml
    from Open Note Block Studio.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    # -- Primitive readers --------------------------------------------------

    def _read_byte(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val

    def _read_short(self) -> int:
        val = struct.unpack_from('<h', self.data, self.pos)[0]
        self.pos += 2
        return val

    def _read_int(self) -> int:
        val = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return val

    def _read_string(self) -> str:
        """Read a length-prefixed string (4-byte little-endian length)."""
        length = self._read_int()
        raw = self.data[self.pos:self.pos + length]
        self.pos += length
        return raw.decode('latin-1')

    def _eof(self) -> bool:
        return self.pos >= len(self.data)

    # -- Main parser --------------------------------------------------------

    def parse(self) -> NbsSong:
        version = 0
        first_custom_index = 0
        song_length = 0

        byte1 = self._read_byte()
        byte2 = self._read_byte()

        if byte1 == 0 and byte2 == 0:
            # NBS format version 1 or later
            version = self._read_byte()
            first_custom_index = self._read_byte()
            if version >= 3:
                song_length = self._read_short()
        else:
            # NBS version 0 (legacy): the two bytes we already read are the
            # first two bytes of the layer count short, so rewind.
            self.pos -= 2

        layer_count = self._read_short()

        song_name = self._read_string()
        song_author = self._read_string()
        original_author = self._read_string()
        description = self._read_string()

        # Tempo is stored as (ticks/second × 100) in a signed 16-bit int
        tempo = self._read_short() / 100.0

        # Deprecated autosave fields
        self._read_byte()  # autosave enabled flag
        self._read_byte()  # autosave interval (minutes)

        time_signature = self._read_byte()

        # Work statistics (minutes spent, notes added/removed, …)
        for _ in range(5):
            self._read_int()

        # MIDI file name associated with this song (informational only)
        self._read_string()

        loop = False
        loop_max = 0
        loop_start = 0
        if version >= 4:
            loop = bool(self._read_byte())
            loop_max = self._read_byte()
            loop_start = self._read_short()

        # ── Notes (delta-encoded column/layer pairs) ──────────────────────
        notes: List[NbsNote] = []
        col = -1
        while True:
            col_delta = self._read_short()
            if col_delta == 0:
                break
            col += col_delta

            layer = -1
            while True:
                layer_delta = self._read_short()
                if layer_delta == 0:
                    break
                layer += layer_delta

                instrument = self._read_byte()
                key = self._read_byte()

                velocity = 100
                panning = 100
                pitch = 0
                if version >= 4:
                    velocity = self._read_byte()
                    panning = self._read_byte()
                    pitch = self._read_short()

                notes.append(NbsNote(
                    tick=col,
                    layer=layer,
                    instrument=instrument,
                    key=max(0, min(87, key)),
                    velocity=velocity,
                    panning=panning,
                    pitch=pitch,
                ))

        if self._eof():
            return NbsSong(
                version=version,
                first_custom_index=first_custom_index,
                song_length=song_length,
                layer_count=layer_count,
                song_name=song_name,
                song_author=song_author,
                original_author=original_author,
                description=description,
                tempo=tempo,
                time_signature=time_signature,
                loop=loop,
                loop_max=loop_max,
                loop_start=loop_start,
                notes=notes,
            )

        # ── Layer metadata ────────────────────────────────────────────────
        layers: List[NbsLayer] = []
        for _ in range(layer_count):
            name = self._read_string()
            lock = 0
            if version >= 4:
                lock = self._read_byte()
            volume = self._read_byte()
            # Volume of 255 (0xFF as unsigned) means -1 in the original code →
            # treat as the default 100.
            if volume == 255:
                volume = 100
            volume = max(0, min(100, volume))
            stereo = 100
            if version >= 2:
                stereo = self._read_byte()
            layers.append(NbsLayer(name=name, lock=lock, volume=volume, stereo=stereo))

        if self._eof():
            return NbsSong(
                version=version,
                first_custom_index=first_custom_index,
                song_length=song_length,
                layer_count=layer_count,
                song_name=song_name,
                song_author=song_author,
                original_author=original_author,
                description=description,
                tempo=tempo,
                time_signature=time_signature,
                loop=loop,
                loop_max=loop_max,
                loop_start=loop_start,
                notes=notes,
                layers=layers,
            )

        # ── Custom instruments ────────────────────────────────────────────
        custom_count = self._read_byte()
        custom_instruments: List[NbsCustomInstrument] = []
        for _ in range(custom_count):
            ci_name = self._read_string()
            ci_filename = self._read_string()
            ci_key = self._read_byte()
            ci_press = self._read_byte()
            custom_instruments.append(NbsCustomInstrument(
                name=ci_name, filename=ci_filename, key=ci_key, press=ci_press
            ))

        return NbsSong(
            version=version,
            first_custom_index=first_custom_index,
            song_length=song_length,
            layer_count=layer_count,
            song_name=song_name,
            song_author=song_author,
            original_author=original_author,
            description=description,
            tempo=tempo,
            time_signature=time_signature,
            loop=loop,
            loop_max=loop_max,
            loop_start=loop_start,
            notes=notes,
            layers=layers,
            custom_instruments=custom_instruments,
        )


# ---------------------------------------------------------------------------
# Instrument mapping: NBS built-in index → MIDI settings
# ---------------------------------------------------------------------------

# Each entry is (is_drum, midi_channel_0indexed, midi_program, drum_note)
#   is_drum      – True if the instrument uses MIDI channel 9 (percussion)
#   midi_channel – 0-indexed MIDI channel (9 = GM drums)
#   midi_program – General MIDI program number (0-indexed); None for drums
#   drum_note    – Fixed MIDI note for drums; None for melodic instruments
_BUILTIN: Dict[int, Tuple[bool, int, Optional[int], Optional[int]]] = {
    0:  (False,  0,   0, None),   # Harp          → Acoustic Grand Piano
    1:  (False,  1,  32, None),   # Double Bass   → Acoustic Bass
    2:  (True,   9, None,  36),   # Bass Drum     → GM drum 36 (Bass Drum 1)
    3:  (True,   9, None,  38),   # Snare Drum    → GM drum 38 (Acoustic Snare)
    4:  (True,   9, None,  42),   # Click         → GM drum 42 (Closed Hi-Hat)
    5:  (False,  2,  25, None),   # Guitar        → Acoustic Guitar (nylon)
    6:  (False,  3,  73, None),   # Flute         → Flute
    7:  (False,  4,  14, None),   # Bell          → Tubular Bells
    8:  (False,  5,   9, None),   # Chime         → Glockenspiel
    9:  (False,  6,  13, None),   # Xylophone     → Xylophone
    10: (False,  7,  12, None),   # Iron Xylophone→ Marimba
    11: (True,   9, None,  56),   # Cow Bell      → GM drum 56 (Cowbell)
    12: (False,  8,  77, None),   # Didgeridoo    → Shakuhachi
    13: (False, 10,  80, None),   # Bit           → Lead 1 (square)
    14: (False, 11, 105, None),   # Banjo         → Banjo
    15: (False, 12,  10, None),   # Pling         → Music Box
}

_BUILTIN_NAMES = [
    "Harp", "Double Bass", "Bass Drum", "Snare Drum",
    "Click", "Guitar", "Flute", "Bell", "Chime",
    "Xylophone", "Iron Xylophone", "Cow Bell",
    "Didgeridoo", "Bit", "Banjo", "Pling",
]

# First MIDI channel available for custom/overflow instruments
# (channels 0-12 are used by builtins; 9 is drums; 13-15 remain)
_FIRST_CUSTOM_CHANNEL = 13


def _nbs_key_to_midi_note(key: int) -> int:
    """Convert an NBS key (0–87) to a MIDI note number (21–108).

    The mapping is derived from open_midi.gml which uses:
        nbs_key = midi_note - 21
    so the inverse is:
        midi_note = nbs_key + 21

    Key landmarks:
        NBS  0  → MIDI  21 (A0)  – lowest piano key
        NBS 45  → MIDI  66 (F#4) – Minecraft default pitch
        NBS 57  → MIDI  78 (F#5)
        NBS 87  → MIDI 108 (C8)  – highest piano key
    """
    return key + 21


# ---------------------------------------------------------------------------
# MIDI binary helpers
# ---------------------------------------------------------------------------

def _write_varlen(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI variable-length quantity."""
    if value < 0:
        raise ValueError(f"Variable-length value must be non-negative, got {value}")
    result = bytearray()
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


def _evt_note_on(channel: int, note: int, velocity: int) -> bytes:
    return bytes([0x90 | (channel & 0x0F), note & 0x7F, max(1, velocity & 0x7F)])


def _evt_note_off(channel: int, note: int) -> bytes:
    return bytes([0x80 | (channel & 0x0F), note & 0x7F, 0x40])


def _evt_program_change(channel: int, program: int) -> bytes:
    return bytes([0xC0 | (channel & 0x0F), program & 0x7F])


def _meta_set_tempo(us_per_beat: int) -> bytes:
    """Meta event 0x51: set tempo (microseconds per quarter note)."""
    return bytes([
        0xFF, 0x51, 0x03,
        (us_per_beat >> 16) & 0xFF,
        (us_per_beat >> 8) & 0xFF,
        us_per_beat & 0xFF,
    ])


def _meta_track_name(name: str) -> bytes:
    """Meta event 0x03: sequence/track name."""
    encoded = name.encode('latin-1', errors='replace')
    return bytes([0xFF, 0x03]) + _write_varlen(len(encoded)) + encoded


def _meta_text(text: str) -> bytes:
    """Meta event 0x01: generic text."""
    encoded = text.encode('latin-1', errors='replace')
    return bytes([0xFF, 0x01]) + _write_varlen(len(encoded)) + encoded


def _meta_end_of_track() -> bytes:
    return bytes([0xFF, 0x2F, 0x00])


def _build_track(events: List[Tuple[int, bytes]]) -> bytes:
    """Build a MIDI MTrk chunk from (absolute_tick, raw_event_bytes) pairs."""
    events = sorted(events, key=lambda e: e[0])

    body = bytearray()
    prev_tick = 0
    for tick, data in events:
        delta = tick - prev_tick
        body += _write_varlen(delta)
        body += data
        prev_tick = tick

    body += _write_varlen(0)
    body += _meta_end_of_track()

    return b'MTrk' + struct.pack('>I', len(body)) + bytes(body)


def _build_midi_file(tracks: List[bytes], ppq: int) -> bytes:
    """Build a complete MIDI file (format 1) from a list of track chunks."""
    header = (
        b'MThd'
        + struct.pack('>I', 6)                 # chunk length
        + struct.pack('>H', 1)                 # format 1 (multi-track)
        + struct.pack('>H', len(tracks))       # number of tracks
        + struct.pack('>H', ppq)               # pulses per quarter note
    )
    return header + b''.join(tracks)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def nbs_to_midi(song: NbsSong, note_duration_ticks: int = 1) -> bytes:
    """Convert an NbsSong to MIDI binary data.

    The strategy:
    - Use PPQ = 1 so that 1 MIDI tick == 1 NBS tick.
    - Set the MIDI tempo so that 1 beat (= 1 MIDI tick at PPQ 1) lasts
      exactly 1/tempo seconds, i.e. us_per_beat = 1_000_000 / tempo.
    - Create one MIDI track per NBS instrument (Format 1).
    - Built-in NBS instruments are mapped to fixed General MIDI channels and
      programs (see _BUILTIN).  Custom instruments are assigned to the
      remaining available channels with a Grand Piano program.
    - Note velocity combines the per-note NBS velocity (0–100) and the
      layer volume (0–100), then scales to the MIDI range (1–127).

    :param song:               Parsed NbsSong object.
    :param note_duration_ticks: Duration of each note in NBS ticks.
    :returns: Raw MIDI file bytes.
    """
    ppq = 1  # 1 MIDI tick = 1 NBS tick

    us_per_tick = round(1_000_000 / song.tempo) if song.tempo > 0 else 500_000

    # ── Tempo / header track ──────────────────────────────────────────────
    tempo_events: List[Tuple[int, bytes]] = [
        (0, _meta_track_name("Tempo Track")),
        (0, _meta_set_tempo(us_per_tick)),
    ]
    if song.song_name:
        tempo_events.append((0, _meta_text(song.song_name)))
    if song.song_author:
        tempo_events.append((0, _meta_text(f"Author: {song.song_author}")))
    tempo_track = _build_track(tempo_events)

    # ── Assign MIDI channels and programs to NBS instruments ─────────────
    used_instruments = sorted({n.instrument for n in song.notes})

    channel_map: Dict[int, int] = {}    # nbs_ins → midi_channel
    program_map: Dict[int, int] = {}    # midi_channel → midi_program
    next_custom_ch = _FIRST_CUSTOM_CHANNEL

    for ins_idx in used_instruments:
        if ins_idx in _BUILTIN:
            is_drum, ch, prog, _ = _BUILTIN[ins_idx]
            channel_map[ins_idx] = ch
            if not is_drum and ch not in program_map and prog is not None:
                program_map[ch] = prog
        else:
            # Custom instrument: assign to the next available non-drum channel
            ch = next_custom_ch % 16
            if ch == 9:  # skip the reserved drum channel
                next_custom_ch += 1
                ch = next_custom_ch % 16
            channel_map[ins_idx] = ch
            if ch not in program_map:
                program_map[ch] = 0  # Grand Piano as default
            next_custom_ch += 1

    # ── Group notes by instrument ─────────────────────────────────────────
    notes_by_ins: Dict[int, List[NbsNote]] = {}
    for note in song.notes:
        notes_by_ins.setdefault(note.instrument, []).append(note)

    # ── Build one MIDI track per NBS instrument ───────────────────────────
    note_tracks: List[bytes] = []

    for ins_idx in sorted(notes_by_ins.keys()):
        ins_notes = notes_by_ins[ins_idx]
        events: List[Tuple[int, bytes]] = []

        if ins_idx in _BUILTIN:
            is_drum, ch, _, fixed_drum_note = _BUILTIN[ins_idx]
            ins_name = (
                _BUILTIN_NAMES[ins_idx]
                if ins_idx < len(_BUILTIN_NAMES)
                else f"Instrument {ins_idx}"
            )
        else:
            is_drum = False
            ch = channel_map.get(ins_idx, 0)
            fixed_drum_note = None
            custom_rel = ins_idx - song.first_custom_index
            if 0 <= custom_rel < len(song.custom_instruments):
                ins_name = song.custom_instruments[custom_rel].name or f"Custom {ins_idx}"
            else:
                ins_name = f"Custom {ins_idx}"

        events.append((0, _meta_track_name(ins_name)))

        # Program change for melodic (non-drum) instruments
        if not is_drum and ch in program_map:
            events.append((0, _evt_program_change(ch, program_map[ch])))

        for note in ins_notes:
            tick = note.tick

            # Combine note velocity (NBS: 0–100) with layer volume (0–100)
            layer_vol = 100
            if 0 <= note.layer < len(song.layers):
                layer_vol = song.layers[note.layer].volume
            combined = (note.velocity * layer_vol) // 100
            midi_vel = max(1, min(127, round(combined * 127 / 100)))

            if is_drum:
                midi_note = fixed_drum_note
            else:
                midi_note = max(0, min(127, _nbs_key_to_midi_note(note.key)))

            events.append((tick, _evt_note_on(ch, midi_note, midi_vel)))
            events.append((tick + note_duration_ticks, _evt_note_off(ch, midi_note)))

        note_tracks.append(_build_track(events))

    return _build_midi_file([tempo_track] + note_tracks, ppq)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Note Block Studio (.nbs) file to a MIDI (.mid) file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to the input .nbs file.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Path to the output .mid file. "
            "Defaults to the input filename with a .mid extension."
        ),
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=1,
        metavar="TICKS",
        help=(
            "Duration of each note in NBS ticks (default: 1). "
            "Increase to make notes sustain longer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_path: str = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path: str = args.output
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".mid"

    with open(input_path, 'rb') as fh:
        data = fh.read()

    reader = NbsReader(data)
    song = reader.parse()

    print(f"NBS version   : {song.version}")
    print(f"Song name     : {song.song_name!r}")
    print(f"Author        : {song.song_author!r}")
    print(f"Tempo         : {song.tempo:.2f} ticks/s")
    print(f"Notes         : {len(song.notes)}")
    print(f"Layers        : {len(song.layers)}")
    print(f"Custom instrs : {len(song.custom_instruments)}")

    midi_data = nbs_to_midi(song, note_duration_ticks=args.duration)

    with open(output_path, 'wb') as fh:
        fh.write(midi_data)

    print(f"\nSaved MIDI to : {output_path}")


if __name__ == "__main__":
    main()
