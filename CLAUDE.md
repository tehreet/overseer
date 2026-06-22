If what you are building is not SMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOTH in the TUI, make it so. Smoothness, good feel, instant feedback, beautiful animations, fluid, thats what the goal is here with studioboard.

# overseer

**studioboard** — an always-on, buttery-smooth Ratatui TUI dashboard for a Mac Studio (M4 Max): system + Apple Silicon metrics, Claude Code usage, and Apple Music with karaoke-synced lyrics. Smoothness is the whole point — see line 1.

This repo was split out of the `battlestation` rig-config repo; the kiosk/Hammerspoon integration that launches and pins this binary still lives there and invokes the installed `studioboard` command.

## studioboard

A Rust/Ratatui TUI. Collectors run on background threads; the render loop is decoupled and frame-paced (up to 120 fps while music plays) so progress bars, the karaoke lyric wipe, and every gauge glide rather than step. Smoothness is the whole point — see line 1.

- **Source layout:** `src/collectors.rs` (background data threads), `src/state.rs` (shared `AppState`), `src/ui.rs` (all rendering, one panel fn per card), `src/theme.rs` (synthwave palette + `jazz()`/`wipe()`/`blend()` ramps), `src/lyrics.rs`, `src/main.rs`.
- **Build/run gotcha:** the live `studioboard` command is a symlink to the **release** binary. After editing, run `cargo build --release` (a plain `cargo build` only writes `target/debug/` and the running app won't change), then quit + relaunch. Watch for a stale `~/.cargo/bin/studioboard` shadowing the symlink — `cargo install --path . --force` from this repo fixes it.
- **Visual verify (no TTY needed):** `studioboard --cells WxH [t=1.3]` dumps every rendered cell with colors; a Pillow script rasterizes it to a PNG so you can actually SEE colors/animation off-screen. `--snapshot` is glyph-only.
- **Palette:** synthwave graphite — violet/cyan/pink/green on a near-black bg, with a `jazz()` blue→violet→pink→white ramp for the lively bits. Keep new work on-palette and animated.
