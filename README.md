# overseer

A buttery-smooth, always-on TUI command center for the Mac Studio (Apple M4 Max).

```
┌ RESOURCES ───────┐┌ NOW PLAYING ───────────────────────┐
│ cpu / mem / net   ││ ▶ track  artist · album   ▃▆▅▂ EQ   │
│ stacked wave      ││ 0:11 ████░░░░░░░░░░░░░░░ 2:53        │
├ APPLE SILICON ────┤├ LYRICS ♪ synced ───────────────────┤
│ gpu / power / temp││      karaoke wipe, lit per-char     │
├ ROBOTS WORKING ───┤│      in cyan→violet→pink as you sing│
│ live sessions     ││      active line centered           │
│ tokens $ sessions ││                                     │
└───────────────────┘└─────────────────────────────────────┘
```

## What it shows

- **Resources** — CPU (Apple Silicon `cpu_usage_pct` when available, else
  sysinfo), memory/swap, root volume, and live up/down network throughput,
  stacked into one smooth, glide-y wave. CPU no longer owns its own card; it
  folds in here.
- **Apple Silicon** — GPU %, GPU clock, package/CPU/GPU/system power draw (W),
  CPU & GPU temperatures. Source: [`macmon`](https://github.com/vladkens/macmon)
  (no sudo).
- **Robots Working (Claude Code)** — a realtime "who's working" feed of the
  Claude Code sessions active right now: one pulsing row per session showing what
  it's doing this second (editing a file, running a command, reading, thinking,
  spawning an agent…), its project, model, and how long since its last move —
  read live from each transcript's tail (~1 Hz). Beneath it, rolling-window token
  totals (today / week / month / sessions). Parsed from
  `~/.claude/projects/**/*.jsonl`; probe headlessly with `--diag-live`. The
  mac-doctor triage agent folds into this feed when it's actively working.
- **Now Playing + Lyrics** — current Apple Music track with a smooth,
  frame-interpolated progress bar and **time-synced karaoke lyrics** from
  [LRCLIB](https://lrclib.net), with **NetEase Cloud Music** as a backup source
  behind it (a synced NetEase hit beats LRCLIB's plain text). The active line
  lights up character-by-character in time with playback.
- **Now Watching** — when you're watching in the macOS **TV app**, the same band
  flips to the movie/show: its cover (like album art), a smooth progress bar, the
  show · season/episode or title · year · genre · director, the live audio EQ, and
  the FACTS card fills with trivia about the film/series and its cast. Music takes
  priority when it's actually playing; otherwise TV wins. Probe it headlessly with
  `--diag-tv`.
- **Facts** — liner notes for the current track (or trivia for the current
  film/series and its cast), written by the Anthropic API
  (`claude-sonnet-4-6`) when a key is present, falling back to Wikipedia
  otherwise. See [Facts API key](#facts-api-key). Probe it with `--facts`.
- **Weather** — current conditions, IP-geolocated, from
  [wttr.in](https://wttr.in).
- **iMessage** — recent conversations with unread badges and **inline reply**
  (focus the card with `m`, mark read, or reply in place; see [Run](#run)).
- **Signal** — recent Signal messages with unread flagging.
- **Discord** — recent activity plus live **voice activity** (who's talking in a
  voice channel).
- **Processes** — top processes by CPU/memory.
- **Keybinds** — the Hammerspoon keybind registry (Hyper+H), exported from the
  `battlestation` rig config.

## Why it's smooth

Data collection and rendering are fully decoupled:

- Each data source runs on its **own background thread** (system ~1s, macmon
  streamed, Apple Music ~1s, lyrics on track-change, Claude usage ~8s).
- The **render loop is frame-paced**: a ~60 fps baseline (so the EQ and every
  gauge stay alive) that rises to ~120 fps while music or an animation is live,
  so the progress bar and karaoke wipe move every frame.
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

**iMessage hotkey:** press `m` to focus the iMessage card / mark the focused
unread read and advance the queue; double-press `m` to open the **inline reply**
input on the focused message (type, `Enter` to send via AppleScript).

**CLI flags:**

```sh
overseer --demo          # scripted demo with sample data (no live sources)
overseer --facts         # one-shot facts probe (prints generated liner notes)
overseer --clear-cache   # wipe the on-disk cache (~/.cache/overseer)
overseer --snapshot WxH  # glyph-only headless layout preview (e.g. 140x42)
overseer --cells WxH     # dump every rendered cell with colors (for visual verify)
```

Headless diagnostic probes — `--diag` plus the per-source family: `--diag-live`
(Claude sessions), `--diag-tv`, `--diag-np` (now-playing), `--diag-audio`,
`--diag-msg` (iMessage), `--diag-signal`, `--diag-voice` /
`--diag-discord-audio` (Discord voice).

## First-run permissions

The first time it reads Apple Music, macOS will ask your terminal app
(Ghostty/Terminal) for permission to **control "Music"** (Automation / TCC).
Click **OK** — without it, the now-playing panel stays idle. If you missed the
prompt: System Settings → Privacy & Security → Automation → your terminal →
enable **Music**.

It never *launches* Music; if Music isn't running, the panel just shows idle.

## Facts API key

The FACTS card calls the **Anthropic API** (model `claude-sonnet-4-6`) to write
music liner notes and TV/movie trivia. It looks for a key in this order:

1. `ANTHROPIC_API_KEY` in the environment.
2. **1Password** via the `op` CLI — by default the `notesPlain` field of the
   `Claude Anthropic API Key` item in the `Claude Code` vault. Override with
   `OP_ANTHROPIC_VAULT` / `OP_ANTHROPIC_ITEM` / `OP_ANTHROPIC_FIELD`.

Without a key the card falls back to **Wikipedia**, so it still shows something —
just not the punchy generated notes.

## Tweaking

- **Cost estimates** live in `pricing()` in `src/collectors.rs` (USD per million
  tokens, per model). Edit to match your plan; the UI labels them `est`.
- **Theme/colors** are all in `src/theme.rs`.
- **Layout / panel sizes** are in `render()` at the top of `src/ui.rs`.
- **Frame rate** budgets are in `event_loop()` in `src/main.rs`.

## Dependencies

- `macmon` (`brew install macmon`) — Apple Silicon metrics, no sudo.
- Apple Music app — now-playing + position via AppleScript.
- Internet — lyrics (LRCLIB, with NetEase Cloud Music as backup), weather
  (wttr.in), facts (Anthropic API / Wikipedia), and album art.
- `op` (1Password CLI) — optional, only for resolving the Anthropic key from a
  vault instead of `ANTHROPIC_API_KEY`.

Lyrics, facts, and album art are cached **persistently on disk** under
`~/.cache/overseer/<kind>/` (e.g. `lyrics/`, `facts/`, `art/`), so they survive
restarts and don't regenerate or re-fetch on every relaunch. Wipe the cache with
`overseer --clear-cache`.
