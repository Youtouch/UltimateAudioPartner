# Mix plan

Write the plan before rendering. Most mix failures so far were planning failures you could have measured: a bed too loud, a call from another biome, music silent for a third of the export.

## Layers and priority

| Layer | Carries | Priority | Starting level (before final loudnorm) |
|---|---|---|---|
| Voice | Meaning | 1, never masked | Reference: about -16 LUFS integrated |
| Game / UI sound | Proof that the action happened | 2 on its action | Measure; keep taps audible, not punchy enough to startle |
| Effect / spot call | Punctuation | 2, in a voice pause | Peak about 6-14 dB under the voice's peaks (an approved owl call: 9; a reference monkey call: 7) |
| Music | Shape and energy | 3; lead only in voiceless adverts | Under voice, during speech: about 7-8 LU in two narrated shorts a listener approved. Common practice for dense dialogue sits lower (10-20); calm pads tolerate less gap |
| Ambience bed | Place | 4, under everything | About 30 LU or more under the voice during speech (approved: 35.5 and 31.9). In the cutter that is the bed at about -47 LUFS before the final loudnorm |

The starting levels come from measured mixes a listener accepted, and from common practice; they are not standards. Gaps are K-weighted (the LUFS filter) levels during speech, which is what `mix/mix_check.py` measures: plain RMS reads beds about 10 dB quieter and music about 4 dB quieter than this. Meter the actual stem after gain; the listener's ear moves them. A project profile's own levels win.

**Voiceless advert:** music leads, game sounds sit over it on their actions, nothing else. **Narrated short:** voice leads, one bed, music optional and low, one call per joke at most.

## Coherence

- One place per scene: one biome, one source pack. A call belongs to its bed.
- One music track per piece unless the edit marks a change.
- Recognise every layer with `audio_tag.py` before mixing: a bed with a music loop inside, or a "silence" file, ruins the mix silently.

## Keep the voice clear

- **Duck by sidechain.** The requirement is the gap during speech; ducking is how to hold it without burying the music in the pauses. The approved shorts dip the music about 3 LU between speech and pauses. The approved cutter's settings: music 3:1, threshold about 3.8 dB under the voice's integrated LUFS, attack 20 ms, release 400 ms; beds 8:1, threshold about 13.3 dB under, attack 10 ms, release 350 ms (`sidechaincompress`; `render/render_mix.py` uses exactly these). A threshold is an absolute level: keep it relative to the voice, or a hotter voice ducks harder. Deeper ducking (lower threshold, 6:1 and up on music) pumps between words.
- **Pad the key.** `sidechaincompress` stops when its key ends: pad it (`apad=whole_dur=TOTAL`). Unpadded, one export went out with the music silent for its last third.
- **Carve frequencies, not only gain.** High-pass a bed at about 120-150 Hz under a voice; if music still competes, a gentle 2-4 dB dip around 2-4 kHz on the music while voice plays. Compare the voice's and bed's band shares (`audio_index.py show`): two layers peaking in the same band mask each other.
- **Bright beds** (most energy above 3.5 kHz: insects, crickets, rain) read louder than plain RMS says; LUFS already counts that (an approved cricket bed, 77% bright, sat 32 LU under). Measure in LUFS, not RMS, and no extra allowance is needed.
- **Bursts in a bed** (LRA near 18) collide with lines. Map the bursts and cap them, or move the bed.

## Place effects

- Align each effect's **loudest instant** (measure it; files rarely start on it) to the visual contact.
- Calls and stingers go in pauses, never under a line.
- Short fades on every cut sound (5-20 ms in, longer out) to avoid clicks.
- Repetitive game taps: vary nothing you did not record, but check a fast sequence does not stack into a level jump.

## Gain staging and render

- Sum with `amix=normalize=0` (normalising divides every layer silently).
- One `loudnorm` at the end, two-pass, on the whole edited audio. About -16 LUFS integrated and -1.5 dBTP is the voice-led short starting point; check the destination's current requirement.
- No layer clips before the final stage: check each stem's true peak after gain.

## Check the render (you cannot listen)

1. Each stem after gain: integrated LUFS (long layers) or true peak (short ones), recorded in the cue sheet.
2. The export in stretches: `volumedetect` per section (`-ss A -to B`). A stretch near -91 dB is silence; a jump of more than about 6 dB between neighbouring stretches with no story reason is an error.
3. Voice-only windows against voice-plus-music windows: the voice stretch should not lose more than about 1-2 dB of level when music enters (masking shows as the mix rising, not the voice).
4. `audio_tag.py tag` on the final export: `Speech` should dominate voiced windows, `Music` the voiceless advert. A window where `Music` beats `Speech` during a line is a mix to fix before the user hears it.
5. Report what is still a guess, and **listening pending**.

## Cue sheet

```text
| Layer | File | Excerpt | Cue in/out | Gain | Duck / EQ | Measured after gain | Listening |
| Voice | ep04/voice.wav | whole | 0.0-41.2 | 0 | - | -16.2 LUFS, -2.1 dBTP | pending |
| Bed   | ...           | 0-45 s | 0.0-43.0 | -31 dB | HP 140 Hz, duck 8:1 | -47.3 LUFS | pending |
| Call  | ...           | 4.0-7.5 | 21.3 | -12 dB | in pause | peak -14 dBTP | pending |
```

Keep it in the episode or advert's folder, so a revision changes one line, not the whole mix.

## Accessibility

Short videos are often watched muted: captions carry every spoken line and no key point is carried by sound alone. Avoid sudden loud events: cap bursts and stingers, especially in the first second.
