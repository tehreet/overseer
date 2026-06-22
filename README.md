# overseer

A buttery-smooth, always-on TUI command center for the Mac Studio (Apple M4 Max).

```
┌ CPU ─────────────┐┌ NOW PLAYING ───────────────────────┐
│ per-core heat bars││ ▶ track  artist · album   ▃▆▅▂ EQ   │
│ load sparkline    ││ 0:11 ████░░░░░░░░░░░░░░░ 2:53        │
├ MEM · DISK · NET ─┤├ LYRICS ♪ synced ───────────────────┤
├ APPLE SILICON ────┤│      karaoke wipe, lit per-char     │
│ gpu / power / temp││      in cyan→violet→pink as you sing│
├ CLAUDE CODE ──────┤│      active line centered           │
│ tokens $ sessions ││                                     │
└───────────────────┘└─────────────────────────────────────┘
```

## What it shows

- **CPU** — per-core utilization heat bars (Apple Silicon `cpu_usage_pct` when
  available, else sysinfo), load sparkline, load averages.
- **Memory · Disk · Net** — RAM/swap, root volume, live up/down throughput.
- **Apple Silicon** — GPU %, GPU clock, package/CPU/GPU/system power draw (W),
  CPU & GPU temperatures. Source: [`macmon`](https://github.com/vladkens/macmon)
  (no sudo).
- **Claude Code (ROBOTS WORKING)** — a realtime "who's working" feed of the
  Claude Code sessions active right now: one pulsing row per session showing what
  it's doing this second (editing a file, running a command, reading, thinking,
  spawning an agent…), its project, model, and how long since its last move —
  read live from each transcript's tail (~1 Hz). Beneath it, rolling-window token
  totals (today / week / month / sessions). Parsed from
  `~/.claude/projects/**/*.jsonl`; probe headlessly with `--diag-live`.
- **Now Playing + Lyrics** — current Apple Music track with a smooth,
  frame-interpolated progress bar and **time-synced karaoke lyrics** from
  [LRCLIB](https://lrclib.net). The active line lights up character-by-character
  in time with playback.
- **Now Watching** — when you're watching in the macOS **TV app**, the same band
  flips to the movie/show: its cover (like album art), a smooth progress bar, the
  show · season/episode or title · year · genre · director, the live audio EQ, and
  the FACTS card fills with trivia about the film/series and its cast. Music takes
  priority when it's actually playing; otherwise TV wins. Probe it headlessly with
  `--diag-tv`.

## Why it's smooth

Data collection and rendering are fully decoupled:

- Each data source runs on its **own background thread** (system ~1s, macmon
  streamed, Apple Music ~1s, lyrics on track-change, Claude usage ~8s).
- The **render loop is frame-paced up to ~120 fps** while music is playing and
  idles at ~15 fps otherwise (calm + cool when nothing moves).
- Playback position is **interpolated from a local clock** between the 1 Hz
  AppleScript polls, so the progress bar and the karaoke wipe advance *every
  frame* instead of stepping once a second.

For the full effect, run it in a **GPU-accelerated, ProMotion-aware terminal**
(Ghostty, installed alongside this). A 240 Hz panel + Ghostty + a release build
is where the butter lives. Running it in Terminal.app will cap the smoothness no
matter what the code does.

## Run

```sh
overseer          # installed to ~/.cargo/bin
# or from this dir:
cargo run --release
```

Quit with `q`, `Esc`, or `Ctrl-C`.

Headless layout preview (no terminal needed):

```sh
overseer --snapshot 140x42
```

## First-run permissions

The first time it reads Apple Music, macOS will ask your terminal app
(Ghostty/Terminal) for permission to **control "Music"** (Automation / TCC).
Click **OK** — without it, the now-playing panel stays idle. If you missed the
prompt: System Settings → Privacy & Security → Automation → your terminal →
enable **Music**.

It never *launches* Music; if Music isn't running, the panel just shows idle.

## Tweaking

- **Cost estimates** live in `pricing()` in `src/collectors.rs` (USD per million
  tokens, per model). Edit to match your plan; the UI labels them `est`.
- **Theme/colors** are all in `src/theme.rs`.
- **Layout / panel sizes** are in `render()` at the top of `src/ui.rs`.
- **Frame rate** budgets are in `event_loop()` in `src/main.rs`.

## Dependencies

- `macmon` (`brew install macmon`) — Apple Silicon metrics, no sudo.
- Apple Music app — now-playing + position via AppleScript.
- Internet — LRCLIB lyric lookups (cached per track in memory).
