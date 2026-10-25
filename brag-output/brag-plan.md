# /brag-slim plan: TrainLab Studio

## Inspect: answers

- **What is it?** An open-source app that trains AI agents by trial and error in real 3D physics, and runs on an ordinary laptop with no GPU.
- **Who is it for, and what does it do for them?** Students and beginners who want a real AI project. They pick the sample (or describe their own world), press train, and watch the agent learn.
- **What sets it apart?**
  - It's full 3D (MuJoCo), not 2D.
  - The built-in sample is a sports car with thrusters.
  - Simulations can be generated from a sentence, using your own AI model (Claude, GPT, Gemini, or a free local Ollama model).
  - A live 3D view shows the agent learning.
  - It's safe on an 8 GB laptop.
- **Most impressive or funniest claim:** a sports car with rocket thrusters crashes over and over, then learns to jump the gap and park in the box. The real sample run took 300k steps in about 7 minutes on a laptop CPU (run.json: 303,104 steps in 405.5 s).
- **Visual hook:** the car fires its thrusters (smoke on the road), then nosedives into the pit, followed by a red "FAILED" banner.
- **Real UI or flow to show:**
  - The real MuJoCo renders: crashes, the trained car parking, and the four scenarios.
  - The real Studio Create tab (headless screenshot) with the prompt being typed.
  - The real CLI command.
  - The real SUCCESS/FAILED banner style from the live view.
- **Tone:** `default`, meaning punchy, playful and clean, with a little rocket-car absurdity.
- **One-line share caption:** I taught a sports car with rocket thrusters to park itself, on a laptop, by letting it crash a lot.

Privacy: the video shows no keys, emails or local paths. The only URL is the public GitHub repo.

## Angle

**"It learns by crashing."** Open on failure, because that's the funny and true part of
reinforcement learning. Then pay it off with a clean park. Then show that you can do it
yourself: on your laptop, with any world you describe.

## Visual identity

- **Colours:** background #0B0F14 (Studio's dark theme); Roadster red #C0101A; Studio
  orange #F97316 (Gradio's primary button); success green #28AA5A and fail red #D73C3C
  (the real live-view banner colours in `trainlab/core/live.py`).
- **Type:** Segoe UI Bold for headlines, Segoe UI for body text, Consolas for code.
- **Look:** the real MuJoCo scenes with a slow orbiting free camera, and large type with a soft shadow.

## Music

- 120 BPM, so 1 beat = 0.5 s and 1 bar = 2 s. Key: A minor pentatonic, playful.
- Kick on every beat, clap on beats 2 and 4, eighth-note hats, a bass pattern that follows the
  roots (A, F, C, G), and a plucked arpeggio from bar 3.
- **Sound effects** (in key, mixed under the music): a low "thump" on each FAILED, a rising
  A-major chime on SUCCESS, soft noise whooshes on scene cuts.
- Every cut lands on a beat.

## Storyboard (22 s = 44 beats, 30 fps, 1920×1080)

| # | Time | Beats | Scene | On-screen text |
|---|---|---|---|---|
| 1 | 0.0–2.5 | 5 | **Hook.** The Roadster speeds down the road, fires its lift thrusters (smoke) and falls short, hitting the ledge wall. A FAILED banner slams in. Real physics, a scripted bad attempt, orbit camera. | "attempt #1" (small), then the banner **FAILED: BODY HIT** |
| 2 | 2.5–5.0 | 5 | **Montage.** Three more real failures (flipped, fell, hard landing), each about 0.8 s, with hard cuts on beats. | Each gets its own banner, then **"It learns by crashing."** |
| 3 | 5.0–9.0 | 8 | **Reveal.** The car drives calmly behind a dark overlay. Title card. | **TrainLab Studio**, then *Train AI agents in real 3D physics, on your laptop.* |
| 4 | 9.0–13.0 | 8 | **Highlight 1: it learns.** The real trained agent (`level0_model.zip`) crosses and parks in the green box, followed by a green banner. | **SUCCESS: PARKED**, then "300k steps · ~7 min · laptop CPU" |
| 5 | 13.0–16.5 | 7 | **Highlight 2: describe any world.** The real Studio Create tab (screenshot) with the prompt typed in. | "Describe any world. Your AI model builds it." / "Claude · GPT · Gemini · free local Ollama" |
| 6 | 16.5–19.0 | 5 | **Highlight 3: live 3D.** A 2×2 grid of the four real scenarios running (gap jump with smoke, launch sprint, 3D track, space dock). | "Watch it learn, live." |
| 7 | 19.0–22.0 | 6 | **Outro.** Logo card. | **TrainLab Studio**, then "Open source · github.com/aswinsamuel-codes/TrainLab-Studio" |

Scene time adds up to 2.5 + 2.5 + 4 + 4 + 3.5 + 2.5 + 3 = 22 s.

**As built (final, 23.5 s = 47 beats):**

| Scene | Start | Length | Beats |
|---|---|---|---|
| hook | 0.0 s | 2.5 s | 5 |
| montage | 2.5 s | 4.5 s | 9 |
| reveal | 7.0 s | 3.5 s | 7 |
| park | 10.5 s | 4.0 s | 8 |
| studio | 14.5 s | 3.5 s | 7 |
| grid | 18.0 s | 2.5 s | 5 |
| outro | 20.5 s | 3.0 s | 6 |

The montage is three real crashes of 1 s each: the car crashes on the beat and the banner
holds for half a second. "It learns by crashing." then holds for 1.5 s.

Real failure reasons used: body_hit (hook), then fell, flipped and flipped (montage). The
fourth reason I looked for, hard_landing, never happened in the search. A lift-only launch
always tips the car over first, so the third montage clip is that tumble.

Poster (frame 0 and `brag.jpg`): 9.6 s, the title card with the tagline fully settled.

**Transitions:** hard cuts inside the montage (scene 2). Elsewhere, dip through the
background (old content out, then new content in) over 4 frames around each cut, to avoid a
muddy double exposure.

**Physics playback:** 1 control step (0.05 s) per video frame, so motion plays at 1.5× real
time. That's honest pacing for a 22 s video; the numbers on screen come from the real run.
