# Recognise and select

The method: brief, shortlist by name, recognise, measure, rank, audiocard. Sources: the Audiocards paper (structured metadata for sound-design search, ICASSP 2026), LAION CLAP, the AudioSet ontology, the Universal Category System, and the studio's own rejected picks.

## 1. Write the brief

| Dimension | Resolve |
|---|---|
| Role | Voice, music, ambience bed, spot call, effect on an action, UI/game sound, stinger |
| Character | Words from the request, plus an approved sound from the profile that shows them |
| Prominence | Under everything, noticed, or the lead |
| Frequency | Once, every few seconds, or rapid and overlapping (repetition tires a bright or long sound) |
| Timing | Usable length, attack (instant or soft), tail, loop needed |
| Context | What plays with it: voice? music? Which frequencies are already taken? |
| Constraints | Biome/source coherence, mono/stereo, rights, never-again list from the profile |

"Soft, playful, polished" becomes hypotheses to test (rounded attack, short controlled tail, a little pitch movement), not definitions. The profile's approved examples settle ambiguity.

## 2. Shortlist by name, then recognise

Search the ledger and its indexes first: they are cheap and cover the library. Then run `audio_tag.py tag` on the shortlist (or read the library's tags CSV if a pass exists).

Read the tagger like this:

- **The rule for every reading:** the analysed span has this model evidence; absence, identity, prominence and fit each need separate justification.
- **Labels are independent, uncalibrated scores** over AudioSet's 527 classes (sigmoid, not probabilities that sum to one), maximum over the file's 10 s windows. Several can be high at once. 0.1 is the print floor, not a yes, and no single threshold suits every label. Only the top 8 per file and top 3 per window are stored: a label not shown may still have scored.
- **Read the specific labels, not the parents.** The model scores `Goose` 0.84 while `Animal` stays at 0.03, so a low parent proves nothing. A parent and its child (`Bird`, `Bird vocalization`) are one piece of evidence, not two.
- **A missing label is not an absence.** No bird or frog label on a "jungle" bed is a reason to doubt it and ask for a listen, not proof it has no animals: the model may miss them, only the first 60 s are tagged, and low scores are not stored.
- **Voices hide in music.** `Choir`, `Chant`, `Mantra`, `Singing`, `Vocal music` and `Organ` flag sung or choral layers a listener may hear as "a church" under narration. In one test, two calm tracks rejected for their voices scored Choir/Chant 0.24-0.31 (one also Organ) and two approved tracks none. Even 0.2 is worth a mention as a question for the listener, not as a verdict.
- **A high score is not loudness or share of time.** `Insect`/`Cricket` at 0.9 says crickets are confidently present somewhere in the tagged span, not that they dominate the bed. Check the timeline (in how many windows) and the measurements (bright-band share) before saying a "forest" bed is a cricket chorus.
- **Use the timeline to find the excerpt.** A long file's useful call may be in one window. Name that window, then measure only it.
- **Check the bucket columns.** `speech`, `music`, `silence` catch the common mislabels: a "music" file that is a voice tag, an "ambience" with a music loop under it, a file that is mostly silence.
- **Short clips score low.** A 2 s goose call is padded to a 10 s window and came back `Bird` 0.1; a 3 s one, 0.38. A modest label on a short clip is a candidate worth a listen, not an automatic yes. For short designed effects (clicks, UI) the labels are often wrong altogether (`Boing`, `Coin`): lean on the name and the description search, and say so.
- **mpg123 `id3.c ... error` lines** on stderr come from MP3 tags, not the audio. Ignore them.
- **Labels describe sources, not fit.** `Bird` does not say "calm" or "matches the meadow bed". That is the measurements and the ear.
- **Model limits:** trained on YouTube audio. Synthetic UI sounds, designed whooshes and stylised game effects get approximate labels (`Beep, bleep`, `Whoosh, swoosh, swish`, `Music`). Trust it most for animals, weather, water, speech, crowds, vehicles, instruments.

## 3. Measure

`audio_index.py show <file>` for loudness, bands, event density, steadiness and loop hint; `audio_map.py` for music shape. Rules:

- **Name the quantity.** Integrated LUFS, short-term, sample peak and true peak answer different questions.
- **Short sounds:** integrated LUFS over a 0.3 s click is unreliable; report true peak and the time of the loudest instant instead.
- **Tempo and key are meaningless** for noise, speech and short effects. Do not report them.
- **Brightness is not quality.** A high 3.5 kHz+ share means it reads louder and tires faster, nothing more.

## 4. Rank against a description

`audio_tag.py rank "<description>" <files>` scores the first 10 s of each file against the text with CLAP. Apply hard constraints (length, biome, mono, rights) before trusting order. Write the description as a caption of the sound ("a short soft wooden knock"), not a mood board ("premium satisfying").

## 5. Audiocard and shortlist

One card per candidate, three kinds of evidence kept apart (the values below are illustrative, not a real file):

```text
File / excerpt:   Birds/robin_02.wav  4.0-9.5 s
Role:             spot call after the joke at 00:21.3
Recognised:       Bird 0.91, Bird vocalization 0.84, Chirp 0.62 (window 0-10 s)
Measured:         true peak -6.2 dBTP, density 34/min, bands 8% >8 kHz
Judged:           meadow bed approved in an earlier episode; robin not yet heard
Fit:              same pack as the bed; call in a voice pause
Weakness:         a second call at 7.8 s may land under the next line - trim to 4.0-7.5
Preparation:      fade in 20 ms, out 150 ms, gain -12 dB under voice
Listening:        pending
```

Then a one-line reason per runner-up. If nothing passes the brief and constraints, say **no suitable match**, name what is missing, and suggest a source (another pack, recording, generation). Never present the least-bad file as a match.

## 6. Learn

When the user approves or rejects, write it into the project profile with the reason in their words. A rejection names what was wrong (no animals heard, wrong biome, too loud, too bright), so the next brief can test for it.
