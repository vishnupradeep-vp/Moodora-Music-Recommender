"""
generate_audio.py
------------------
Generates pleasant, harmonic 15-second audio preview files (.wav) for each mood,
so users clicking "Play Preview" get immediate, offline audio playback right in their browser.
"""

from pathlib import Path
import numpy as np
import wave

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 22050  # 22.05 kHz for compact size and fast loading


def note_to_freq(note_name: str) -> float:
    """Converts scientific pitch notation (e.g. C4, A4, Bb3) to frequency in Hz."""
    flats_map = {"BB": "A#", "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#"}
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    letter = note_name[:-1].upper()
    letter = flats_map.get(letter, letter)
    octave = int(note_name[-1])
    semitone = notes.index(letter)
    midi_num = semitone + (octave + 1) * 12
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


def synth_note(freq: float, duration: float, envelope="pluck", sample_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fundamental + warm harmonics
    signal = (
        0.55 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.12 * np.sin(2 * np.pi * 3 * freq * t)
        + 0.08 * np.sin(2 * np.pi * 4 * freq * t)
    )

    # Envelopes
    if envelope == "pluck":  # Piano / Acoustic guitar
        env = np.exp(-3.5 * t / duration)
    elif envelope == "pad":  # Soft ambient / strings
        env = np.sin(np.pi * t / duration) ** 1.5
    elif envelope == "bass":  # Punchy bass
        env = np.exp(-4.5 * t / duration)
        signal += 0.3 * np.sin(2 * np.pi * (freq / 2) * t)
    else:  # Lead synth
        env = np.exp(-2.0 * t / duration)

    return signal * env


def save_wav(filename: Path, samples: np.ndarray, sample_rate=SAMPLE_RATE):
    # Normalize to 16-bit range
    samples = samples / (np.max(np.abs(samples)) + 1e-6)
    samples = (samples * 0.85 * 32767).astype(np.int16)

    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def generate_all_mood_tracks():
    # 1. Romantic & Love (Sweet arpeggio: C4, E4, G4, B4, C5, A4, F4, G4)
    rom_notes = ["C4", "E4", "G4", "B4", "A4", "C5", "F4", "G4"]
    rom_audio = []
    for _ in range(3):  # Repeat sequence to fill ~12-15s
        for n in rom_notes:
            rom_audio.append(synth_note(note_to_freq(n), 0.55, envelope="pluck"))
    save_wav(AUDIO_DIR / "romantic.wav", np.concatenate(rom_audio))

    # 2. Happy / Upbeat (Bouncy major pentatonic: G4, B4, D5, E5, G5)
    happy_notes = ["G4", "B4", "D5", "B4", "E5", "D5", "G4", "B4"]
    happy_audio = []
    for _ in range(4):
        for n in happy_notes:
            happy_audio.append(synth_note(note_to_freq(n), 0.38, envelope="lead"))
    save_wav(AUDIO_DIR / "happy.wav", np.concatenate(happy_audio))

    # 3. Sad / Melancholic (Slow gentle minor notes: A3, C4, E4, D4, F4, E4)
    sad_notes = ["A3", "C4", "E4", "C4", "F3", "A3", "D4", "C4"]
    sad_audio = []
    for _ in range(2):
        for n in sad_notes:
            sad_audio.append(synth_note(note_to_freq(n), 0.95, envelope="pad"))
    save_wav(AUDIO_DIR / "sad.wav", np.concatenate(sad_audio))

    # 4. Energetic / Workout (Punchy bassline and synth drive)
    nrg_notes = ["E2", "E2", "G2", "A2", "B2", "D3", "E2", "E3"]
    nrg_audio = []
    for _ in range(5):
        for n in nrg_notes:
            nrg_audio.append(synth_note(note_to_freq(n), 0.28, envelope="bass"))
    save_wav(AUDIO_DIR / "energetic.wav", np.concatenate(nrg_audio))

    # 5. Party & Dance (126 BPM dance groove)
    party_notes = ["A2", "A3", "C3", "D3", "E3", "G3", "A2", "E3"]
    party_audio = []
    for _ in range(5):
        for n in party_notes:
            party_audio.append(synth_note(note_to_freq(n), 0.26, envelope="lead"))
    save_wav(AUDIO_DIR / "party.wav", np.concatenate(party_audio))

    # 6. Chill / Relaxed (Warm acoustic progression: F3, A3, C4, E4)
    chill_notes = ["F3", "A3", "C4", "E4", "G3", "B3", "D4", "F4"]
    chill_audio = []
    for _ in range(3):
        for n in chill_notes:
            chill_audio.append(synth_note(note_to_freq(n), 0.65, envelope="pluck"))
    save_wav(AUDIO_DIR / "chill.wav", np.concatenate(chill_audio))

    # 7. Heartbreak & Emotional (Minor 7th sorrowful chords)
    hb_notes = ["D3", "F3", "A3", "C4", "Bb2", "D3", "F3", "A3"]
    hb_audio = []
    for _ in range(2):
        for n in hb_notes:
            hb_audio.append(synth_note(note_to_freq(n), 0.9, envelope="pad"))
    save_wav(AUDIO_DIR / "heartbroken.wav", np.concatenate(hb_audio))

    # 8. Soulful / Focus (Gentle bell chime harmonics)
    soul_notes = ["C4", "G4", "C5", "G4", "F4", "C5", "G4", "E4"]
    soul_audio = []
    for _ in range(2):
        for n in soul_notes:
            soul_audio.append(synth_note(note_to_freq(n), 1.0, envelope="pad"))
    save_wav(AUDIO_DIR / "soulful.wav", np.concatenate(soul_audio))

    print(f"Generated 8 mood audio previews in {AUDIO_DIR}")


if __name__ == "__main__":
    generate_all_mood_tracks()
