//! studioboard — a buttery-smooth always-on TUI for the Mac Studio.
//!
//! Collectors run on background threads; the render loop is decoupled and
//! frame-paced (up to 120 fps while music is playing so the progress bar and
//! karaoke lyric wipe move every frame).
#![allow(dead_code)] // full palette + helpers kept available for tweaking

mod cache;
mod collectors;
mod lyrics;
mod state;
mod theme;
mod ui;

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::{execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use state::AppState;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--snapshot") {
        return snapshot(&args);
    }
    if args.iter().any(|a| a == "--diag") {
        return diag();
    }
    if args.iter().any(|a| a == "--diag-msg") {
        collectors::diag_messages();
        return Ok(());
    }
    if args.iter().any(|a| a == "--diag-signal") {
        collectors::diag_signal();
        return Ok(());
    }
    if args.iter().any(|a| a == "--facts") {
        return facts_diag();
    }
    if args.iter().any(|a| a == "--clear-cache") {
        return clear_cache();
    }
    if args.iter().any(|a| a == "--cells") {
        return cells(&args);
    }
    run()
}

fn run() -> Result<()> {
    let shared = Arc::new(Mutex::new(AppState::default()));

    collectors::spawn_system(shared.clone());
    collectors::spawn_macmon(shared.clone());
    collectors::spawn_music(shared.clone());
    collectors::spawn_lyrics(shared.clone());
    collectors::spawn_lyrics_reconcile(shared.clone());
    collectors::spawn_artwork(shared.clone());
    collectors::spawn_facts(shared.clone());
    collectors::spawn_queue(shared.clone());
    collectors::spawn_git(shared.clone());
    collectors::spawn_weather(shared.clone());
    collectors::spawn_usage(shared.clone());
    collectors::spawn_messages(shared.clone());
    collectors::spawn_signal(shared.clone());
    collectors::spawn_discord(shared.clone());
    collectors::spawn_doctor(shared.clone());
    collectors::spawn_keybinds(shared.clone());

    // Terminal setup with a panic hook that always restores the screen.
    terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    execute!(out, terminal::EnterAlternateScreen, crossterm::cursor::Hide)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), terminal::LeaveAlternateScreen, crossterm::cursor::Show);
        default_hook(info);
    }));

    let backend = CrosstermBackend::new(out);
    let mut term = Terminal::new(backend)?;
    let res = event_loop(&mut term, &shared);

    terminal::disable_raw_mode()?;
    execute!(term.backend_mut(), terminal::LeaveAlternateScreen, crossterm::cursor::Show)?;
    term.show_cursor()?;
    res
}

/// Longest iMessage animation duration; once a transition's clock passes this we
/// settle it back to Idle so the card returns to honest stillness.
const MSG_ANIM_MAX: Duration = Duration::from_millis(300);
/// Double-press window for the iMessage 'm' hotkey.
const MSG_DOUBLE: Duration = Duration::from_millis(400);

fn event_loop<B: ratatui::backend::Backend>(
    term: &mut Terminal<B>,
    shared: &Arc<Mutex<AppState>>,
) -> Result<()> {
    loop {
        // Settle any finished iMessage transition before drawing, then snapshot
        // pacing info while we hold the lock to draw.
        let playing;
        let animating;
        {
            let mut s = shared.lock().unwrap();
            settle_msg_anim(&mut s);
            let t = s.started.elapsed().as_secs_f64();
            playing = s.music.playing;
            animating = s.msg_ui.animating();
            term.draw(|f| ui::render(f, &s, t))?;
        }

        // The CPU equalizer interpolates continuously between cached samples, so
        // keep a smooth 60fps baseline; go 120fps with music OR while an iMessage
        // transition / caret blink is live so those glide.
        let budget = if playing || animating {
            Duration::from_micros(8_333) // ~120 fps
        } else {
            Duration::from_micros(16_667) // ~60 fps baseline (living EQ)
        };

        let frame_start = Instant::now();
        // Drain input until the frame budget elapses (keeps input latency low).
        loop {
            let remaining = budget.saturating_sub(frame_start.elapsed());
            if remaining.is_zero() {
                break;
            }
            if event::poll(remaining)? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Release {
                        if handle_key(shared, k.code, k.modifiers) {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
}

/// Once a transition's clock exceeds its window, fold it back to a settled
/// state: Advancing/Closing/Sending → Idle (Sending also closes the composer).
fn settle_msg_anim(s: &mut AppState) {
    use state::MsgPhase;
    if let Some(start) = s.msg_ui.anim_start {
        if start.elapsed() >= MSG_ANIM_MAX {
            match s.msg_ui.phase {
                MsgPhase::Opening => {} // input stays open; caret keeps blinking
                MsgPhase::Sending | MsgPhase::Closing => {
                    s.msg_ui.composing = false;
                    s.msg_ui.draft.clear();
                    s.msg_ui.phase = MsgPhase::Idle;
                    s.msg_ui.anim_start = None;
                }
                _ => {
                    s.msg_ui.phase = MsgPhase::Idle;
                    s.msg_ui.anim_start = None;
                }
            }
        }
    }
    // Clear a stale send-failure flash after its fade completes (~360ms).
    if let Some(f) = s.msg_ui.send_failed_at {
        if f.elapsed() >= Duration::from_millis(360) {
            s.msg_ui.send_failed_at = None;
        }
    }
}

/// Route a keypress through the iMessage state machine. Returns true to quit.
fn handle_key(
    shared: &Arc<Mutex<AppState>>,
    code: KeyCode,
    mods: KeyModifiers,
) -> bool {
    use state::MsgPhase;
    let ctrl_c = code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL);
    if ctrl_c {
        return true; // Ctrl-C always quits, even mid-compose.
    }

    let mut s = shared.lock().unwrap();
    let composing = s.msg_ui.composing;

    if composing {
        // ---- inline reply input ----
        match code {
            KeyCode::Enter => {
                let draft = s.msg_ui.draft.trim().to_string();
                if !draft.is_empty() {
                    // Target = the focused unread message's handle.
                    let handle = focused_handle(&s);
                    if let Some(h) = handle {
                        collectors::send_imessage(&h, &draft);
                        s.msg_ui.phase = MsgPhase::Sending;
                        s.msg_ui.anim_start = Some(Instant::now());
                    } else {
                        // No target — flash failure, keep draft.
                        s.msg_ui.send_failed_at = Some(Instant::now());
                    }
                }
            }
            KeyCode::Esc => {
                s.msg_ui.phase = MsgPhase::Closing;
                s.msg_ui.anim_start = Some(Instant::now());
            }
            KeyCode::Backspace => {
                s.msg_ui.draft.pop();
            }
            KeyCode::Char(ch) => {
                if !mods.contains(KeyModifiers::CONTROL) {
                    s.msg_ui.draft.push(ch);
                }
            }
            _ => {}
        }
        return false; // never quit while composing
    }

    // ---- not composing ----
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => {
            if s.msg_ui.active {
                s.msg_ui.active = false; // unfocus the card
            } else {
                return true; // quit
            }
        }
        KeyCode::Char('m') => {
            let now = Instant::now();
            let double = s
                .msg_ui
                .last_key_at
                .map(|p| now.duration_since(p) <= MSG_DOUBLE)
                .unwrap_or(false);
            s.msg_ui.last_key_at = Some(now);

            if !s.msg_ui.active {
                // First press just focuses the card.
                s.msg_ui.active = true;
                s.msg_ui.phase = MsgPhase::Idle;
            } else if double {
                // Double-press on the focused message → open inline reply.
                s.msg_ui.composing = true;
                s.msg_ui.draft.clear();
                s.msg_ui.phase = MsgPhase::Opening;
                s.msg_ui.anim_start = Some(now);
            } else {
                // Single press → mark focused read + advance the unread queue.
                advance_queue(&mut s);
                s.msg_ui.phase = MsgPhase::Advancing;
                s.msg_ui.anim_start = Some(now);
            }
        }
        _ => {}
    }
    false
}

/// Handle (for the AppleScript reply target) of the currently-focused unread
/// message, walking the unread queue by `queue_pos`.
fn focused_handle(s: &AppState) -> Option<String> {
    s.messages
        .items
        .iter()
        .filter(|i| i.unread)
        .nth(s.msg_ui.queue_pos)
        .map(|i| i.handle.clone())
        .filter(|h| !h.is_empty())
}

/// Mark the focused unread conversation read — flip it in our snapshot for an
/// instant response, and persist to chat.db so the next poll doesn't resurrect
/// it — then advance to the next unread conversation.
fn advance_queue(s: &mut AppState) {
    // Find the focused unread conversation, flip it read in our snapshot.
    let target = s
        .messages
        .items
        .iter()
        .filter(|i| i.unread)
        .nth(s.msg_ui.queue_pos)
        .map(|i| (i.chat_id, i.is_shortcode));
    if let Some((chat_id, is_shortcode)) = target {
        for it in s.messages.items.iter_mut() {
            if it.chat_id == chat_id {
                it.unread = false;
            }
        }
        if !is_shortcode {
            s.messages.unread_count = s.messages.unread_count.saturating_sub(1);
        }
        collectors::mark_chat_read(chat_id);
    }
    // queue_pos stays at the front of the (now shorter) unread list.
    let remaining = s.messages.items.iter().filter(|i| i.unread).count();
    if s.msg_ui.queue_pos >= remaining {
        s.msg_ui.queue_pos = remaining.saturating_sub(1);
    }
}

/// Exercise the real music + lyrics code paths and print what they return.
/// Run from the terminal you actually use studioboard in, so it reflects that
/// app's Automation (Music control) permission.
fn diag() -> Result<()> {
    println!("studioboard --diag\n");
    let (running, playing, track, artist, album, dur, pos) = collectors::probe_music();
    println!("Apple Music:");
    println!("  running   : {running}");
    println!("  playing   : {playing}");
    println!("  track     : {track:?}");
    println!("  artist    : {artist:?}");
    println!("  album     : {album:?}");
    println!("  duration  : {dur:.1}s   position: {pos:.1}s");
    if !running {
        println!("\n→ Music isn't running, OR this terminal lacks Automation permission.");
        println!("  System Settings → Privacy & Security → Automation → <terminal> → enable Music.");
        return Ok(());
    }
    if track.is_empty() {
        println!("\n→ No current track (Music idle/stopped).");
        return Ok(());
    }
    let id = format!("{artist}|{track}|{album}");
    println!("\nFetching lyrics from LRCLIB…");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(4))
        .timeout_read(std::time::Duration::from_secs(13))
        .build();
    let t0 = std::time::Instant::now();
    let ly = lyrics::fetch(&agent, &artist, &track, &album, dur, &id);
    let elapsed = t0.elapsed();
    // Prove the disk-cache path: save, then measure a cold load.
    lyrics::cache_save(&id, &ly);
    let t1 = std::time::Instant::now();
    let cached = lyrics::cache_load(&id);
    let cache_ms = t1.elapsed();
    println!("  fetch time: {} ms", elapsed.as_millis());
    println!("  synced    : {}", ly.synced);
    println!("  lines     : {}", ly.lines.len());
    if !ly.note.is_empty() {
        println!("  note      : {}", ly.note);
    }
    for l in ly.lines.iter().take(6) {
        if l.t < 0.0 {
            println!("    (plain) {}", l.text);
        } else {
            println!("    [{:>6.2}] {}", l.t, l.text);
        }
    }
    println!(
        "  disk cache: {} ({} µs to load — instant on replay)",
        if cached.is_some() { "saved + reloaded" } else { "not cached (no synced)" },
        cache_ms.as_micros()
    );
    println!("\nAlbum art:");
    match collectors::probe_artwork() {
        Some((w, h, n, c)) => println!(
            "  decoded {w}x{h} thumb ({n} px), center rgb = ({}, {}, {})",
            c[0], c[1], c[2]
        ),
        None => println!("  (no artwork / decode failed)"),
    }
    println!("\nGit + weather load on their own threads in the live app.");
    Ok(())
}

/// Render a single frame to a text buffer for headless verification:
fn facts_diag() -> Result<()> {
    println!("studioboard --facts\n");
    let (running, _playing, track, artist, album, _dur, _pos) = collectors::probe_music();
    if !running || track.is_empty() {
        println!("Apple Music idle — no current track.");
        return Ok(());
    }
    println!("track : {track}\nartist: {artist}\nalbum : {album}\n");
    println!("gathering liner notes…\n");
    let fa = collectors::probe_facts(&artist, &track, &album);
    println!("source: {}", if fa.source.is_empty() { "(none)" } else { &fa.source });
    if fa.lines.is_empty() {
        println!("note  : {}", fa.note);
    } else {
        for l in &fa.lines {
            println!("  • {l}");
        }
    }
    Ok(())
}

/// Wipe the persistent disk cache (`~/.cache/studioboard/{lyrics,facts,art}`) and
/// report what was removed, so a song re-fetches lyrics + facts + art next play.
fn clear_cache() -> Result<()> {
    println!("studioboard --clear-cache\n");
    if let Some(root) = cache::root() {
        println!("cache root: {}", root.display());
    }
    for line in cache::clear() {
        println!("  {line}");
    }
    Ok(())
}

///   studioboard --snapshot [WIDTHxHEIGHT]
fn snapshot(args: &[String]) -> Result<()> {
    use ratatui::backend::TestBackend;
    let (w, h) = args
        .iter()
        .find_map(|a| a.split_once('x'))
        .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
        .unwrap_or((140u16, 44u16));

    let mut st = AppState::default();
    sample_data(&mut st, args.iter().any(|a| a == "--compose"));
    // `--idle` previews the no-music right column (live system graphs).
    if args.iter().any(|a| a == "--idle") {
        st.music.playing = false;
    }

    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend)?;
    let t = 1.3;
    term.draw(|f| ui::render(f, &st, t))?;

    let buf = term.backend().buffer().clone();
    let mut out = io::stdout().lock();
    for y in 0..buf.area.height {
        let mut line = String::new();
        for x in 0..buf.area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        writeln!(out, "{}", line.trim_end())?;
    }
    Ok(())
}

/// Dump every rendered cell with its fg/bg color for off-screen image
/// rendering: `studioboard --cells [WxH] [--idle]`. One line per cell:
///   x\ty\tfr\tfg\tfb\tbr\tbg\tbb\t<symbol>
fn cells(args: &[String]) -> Result<()> {
    use ratatui::backend::TestBackend;
    use ratatui::style::Color;
    let (w, h) = args
        .iter()
        .find_map(|a| a.split_once('x'))
        .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
        .unwrap_or((140u16, 44u16));

    let mut st = AppState::default();
    sample_data(&mut st, args.iter().any(|a| a == "--compose"));
    if args.iter().any(|a| a == "--idle") {
        st.music.playing = false;
    }
    if args.iter().any(|a| a == "--nolyrics") {
        st.lyrics = state::Lyrics {
            lines: Vec::new(),
            synced: false,
            track_id: st.music.track_id(),
            note: "no lyrics found".into(),
        };
    }

    let t = args
        .iter()
        .find_map(|a| a.strip_prefix("t="))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.3);
    // Headless snapshots draw a single frame, so wall-clock playback is frozen.
    // Advance the music position by the render clock `t` so the karaoke wipe and
    // progress bar move between t= phases and the active lyric reads as lit.
    if st.music.playing {
        st.music.base_pos += t;
        st.music.sampled_at = std::time::Instant::now();
    }
    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend)?;
    term.draw(|f| ui::render(f, &st, t))?;
    let buf = term.backend().buffer().clone();

    let bg0 = (13u8, 14u8, 22u8); // theme::BG
    let rgb = |c: Color, fb: (u8, u8, u8)| match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => fb,
    };
    let mut out = io::stdout().lock();
    writeln!(out, "{w} {h}")?;
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            let cell = &buf[(x, y)];
            let (br, bgc, bb) = rgb(cell.bg, bg0);
            let (fr, fg, fb) = rgb(cell.fg, (br, bgc, bb));
            writeln!(
                out,
                "{x}\t{y}\t{fr}\t{fg}\t{fb}\t{br}\t{bgc}\t{bb}\t{}",
                cell.symbol()
            )?;
        }
    }
    Ok(())
}

/// Fills state with representative data so the snapshot looks real. `compose`
/// opens the iMessage inline reply input so its affordance can be eyeballed.
fn sample_data(st: &mut AppState, compose: bool) {
    use state::{LyricLine, Lyrics, MusicStats, SiliconStats, SystemStats, UsageStats};
    use std::time::Instant;

    st.system = SystemStats {
        hostname: "mac-studio".into(),
        os: "macOS 26".into(),
        cpu_overall: 37.0,
        per_core: vec![62., 11., 8., 44., 5., 9., 71., 3., 22., 14., 6., 4., 18., 9.],
        load: (2.31, 1.88, 1.42),
        mem_used: 26_190_610_432,
        mem_total: 38_654_705_664,
        swap_used: 0,
        swap_total: 0,
        disk_used: 612_000_000_000,
        disk_total: 1_000_000_000_000,
        net_rx_bps: 2_400_000.0,
        net_tx_bps: 180_000.0,
        uptime_secs: 367_200,
        proc_count: 612,
        top_procs: vec![
            ("studioboard".into(), 18.4, 28_000_000, 367_200),
            ("WindowServer".into(), 12.1, 480_000_000, 367_200),
            ("Music".into(), 6.3, 320_000_000, 7_860),
            ("Warp".into(), 4.8, 540_000_000, 14_400),
            ("claude".into(), 3.2, 210_000_000, 480),
            ("kernel_task".into(), 2.0, 120_000_000, 920),
        ],
    };
    // Staggered, gently-varying samples (~16 s of history) so the delayed
    // interpolation has real data to read across the whole GRAPH_WINDOW — the
    // snapshot then shows the smooth scrolling curves, not flat lines.
    let base = st.system.per_core.clone();
    let now = Instant::now();
    for k in (0..16u64).rev() {
        let ts = now.checked_sub(Duration::from_secs(k)).unwrap_or(now);
        let ph = k as f32 * 0.5;
        let cores: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, &v)| (v + 18.0 * (ph + i as f32 * 0.6).sin()).clamp(0.0, 100.0))
            .collect();
        st.cpu_samples.push_back((ts, cores));
        let net = (2_400_000.0 * (1.0 + 0.6 * (ph as f64 * 0.8).sin())).max(1000.0) as f32;
        st.net_samples.push_back((ts, net));
        let gpu = (31.0 + 22.0 * (ph * 0.7).sin()).clamp(0.0, 100.0);
        let pwr = (55.6 + 30.0 * (ph * 0.6 + 1.0).sin()).clamp(5.0, 120.0);
        st.silicon_samples
            .push_back((ts, vec![gpu, 18.4, 58.0, 52.0, 9.2, 3.1, pwr, 41.0, 23.0, 0.2]));
    }
    for (i, v) in [12, 20, 35, 50, 41, 30, 48, 62, 55, 37].iter().enumerate() {
        let _ = i;
        st.cpu_hist.push(*v);
        st.gpu_hist.push((*v as f64 * 0.6) as u64);
        st.power_hist.push((*v as f64 * 0.8) as u64);
    }
    st.silicon = SiliconStats {
        fresh: true,
        cpu_pct: 37.0,
        gpu_pct: 31.0,
        gpu_freq_mhz: 1425,
        all_power_w: 18.4,
        cpu_power_w: 9.2,
        gpu_power_w: 3.1,
        sys_power_w: 55.6,
        cpu_temp_c: 58.0,
        gpu_temp_c: 52.0,
        ecpu_pct: 41.0,
        ecpu_freq_mhz: 2592,
        pcpu_pct: 23.0,
        pcpu_freq_mhz: 4512,
        ane_power_w: 0.2,
    };
    st.usage = UsageStats {
        fresh: true,
        today_input: 7_200_000,
        today_output: 818_000,
        today_cache_read: 17_957_000,
        today_cache_write: 3_519_000,
        today_cost: 4.10,
        month_cost: 47.20,
        today_messages: 342,
        top_model: "Opus".into(),
        sessions_today: 6,
        tokens_7d: 1_840_000_000,
        tokens_30d: 7_320_000_000,
        sessions_30d: 73,
        hourly: vec![0, 0, 0, 0, 0, 0, 0, 12, 40, 88, 120, 64, 30, 55, 90, 140, 110, 70, 0, 0, 0, 0, 0, 0],
    };
    st.music = MusicStats {
        running: true,
        playing: true,
        track: "Started From the Bottom".into(),
        artist: "Drake".into(),
        album: "Nothing Was the Same (Deluxe)".into(),
        duration: 173.9,
        base_pos: 11.0,
        sampled_at: Instant::now(),
        polled: true,
    };
    // Album art: decode the last real dump (so visual-verify exercises the true
    // sampling path on a real cover); fall back to a radial gradient otherwise.
    st.album_art = collectors::sample_album_art(st.music.track_id());
    st.facts = state::MusicFacts {
        track_id: st.music.track_id(),
        source: "claude".into(),
        note: String::new(),
        lines: vec![
            "Drake's first solo #1 on the Billboard Hot 100.".into(),
            "The beat samples Whitney Houston's \"I'm Every Woman\" ad-libs.".into(),
            "Recorded in a single late-night session in Toronto.".into(),
            "The phrase became a meme long before the song dropped.".into(),
        ],
    };
    st.queue = state::Queue {
        fresh: true,
        source_track_id: st.music.track_id(),
        items: vec![
            state::QueueTrack { track: "NEW MAGIC WAND".into(), artist: "Tyler, The Creator".into(), duration: 195.0 },
            state::QueueTrack { track: "BagBak".into(), artist: "Vince Staples".into(), duration: 160.0 },
            state::QueueTrack { track: "Surround Sound (feat. 21 Savage & Baby Tate)".into(), artist: "JID".into(), duration: 229.0 },
        ],
    };
    st.git = state::GitStats {
        fresh: true,
        ok: true,
        branch: "main".into(),
        dirty: 11,
        untracked: 1,
        staged: 0,
        ahead: 2,
        behind: 0,
        last_hash: "9a554a6".into(),
        last_msg: "feat(studioboard): merge Robots Working card".into(),
        last_rel: "16 minutes ago".into(),
        commits_today: 7,
        pr_count: 0,
        repo: "battlestation".into(),
        loc_added: 10,
        loc_removed: 0,
        branch_commits: 8,
        merges_main: 0,
    };
    st.weather = state::Weather {
        fresh: true,
        location: "Fond du Lac, WI".into(),
        temp_f: 62,
        feels_f: 62,
        desc: "Sunny".into(),
        icon: "☀".into(),
        hi_f: 74,
        lo_f: 49,
        humidity: 62,
        wind_mph: 8,
        wind_dir: "NW".into(),
        precip_chance: 12,
        uv: 5,
        pressure_mb: 1014,
        sunrise: "06:21".into(),
        sunset: "20:14".into(),
        temp_strip: vec![58, 60, 63, 66, 70, 72, 74, 71, 67, 62, 57, 53],
    };
    st.messages = state::Messages {
        fresh: true,
        available: true,
        unread_count: 2,
        // Contacts-only, most-recent-first, capped at 5 (matches the collector).
        // (sender, handle, text, rel, rich, unread, from_me, shortcode)
        items: vec![
            ("Mom", "+15555550111",
             "can you call me when you get a chance? want to talk about the trip next month and what time works for everyone",
             "2m", false, true, false, false),
            ("Family", "", "Dad: who's picking up grandma sunday?", "8m", false, true, false, false),
            ("🦧 The Crew", "", "Josh: check mate 😎", "4h", false, false, false, false),
            ("Alex Rivera", "alex@example.com", "sounds good, see you then", "12m", false, false, false, false),
            ("Dad", "dad@example.com", "ok 👍", "yd", false, false, true, false),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (sender, handle, text, rel, rich, unread, from_me, shortcode)): (usize, (&str, &str, &str, &str, bool, bool, bool, bool))| {
            let preview = {
                let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
                let flat = if flat.chars().count() <= 96 {
                    flat
                } else {
                    let cut: String = flat.chars().take(95).collect();
                    format!("{cut}…")
                };
                if from_me { format!("You: {flat}") } else { flat }
            };
            state::MessageItem {
                chat_id: i as i64,
                rowid: i as i64,
                sender: sender.into(),
                handle: handle.into(),
                preview,
                full_text: text.into(),
                ts_unix: 0.0,
                rel: rel.into(),
                is_rich: rich,
                unread,
                from_me,
                is_shortcode: shortcode,
            }
        })
        .collect(),
    };
    st.msg_ui = state::MsgUi { active: true, queue_pos: 0, ..Default::default() };
    st.signal = state::Messages {
        fresh: true,
        available: true,
        unread_count: 1,
        // Signal conversations — read-only, same row style. (sender,text,rel,unread,from_me)
        items: vec![
            ("Marcin W", "I miss hacking on cars man", "3m", true, false),
            ("The Other CW Back Channel", "Elijah: Nice! 🔥", "1h", false, false),
            ("Peter Salanki", "You: Very smooth", "5h", false, true),
            ("Alex Stalmakov", "You: but yes i am streaming from my pc", "8h", false, true),
            ("Erik Gomez", "You: 💀", "yd", false, true),
        ]
        .into_iter()
        .map(|(sender, text, rel, unread, from_me): (&str, &str, &str, bool, bool)| {
            state::MessageItem {
                chat_id: 0,
                rowid: 0,
                sender: sender.into(),
                handle: String::new(),
                preview: text.into(),
                full_text: text.into(),
                ts_unix: 0.0,
                rel: rel.into(),
                is_rich: false,
                unread,
                from_me,
                is_shortcode: false,
            }
        })
        .collect(),
    };
    st.discord = state::Discord {
        fresh: true,
        available: true,
        voice: vec![
            state::VoiceChannel {
                name: "200 club".into(),
                members: vec!["Tehreet".into(), "Cassie".into(), "Marcin".into()],
            },
            state::VoiceChannel {
                name: "gaming".into(),
                members: vec!["Erik".into()],
            },
        ],
        text: vec![
            state::TextChannel {
                name: "battlestation".into(),
                author: "mac-doctor".into(),
                preview: "load spike was a Steam shader compile".into(),
                rel: "4m".into(),
                unread: true,
            },
            state::TextChannel {
                name: "actual-degenery".into(),
                author: "Tehreet".into(),
                preview: "[image]".into(),
                rel: "53m".into(),
                unread: false,
            },
            state::TextChannel {
                name: "normies".into(),
                author: "Tehreet".into(),
                preview: "[image]".into(),
                rel: "14h".into(),
                unread: false,
            },
        ],
        voice_join_at: None,
    };
    // MAC-DOCTOR card preview. Idle by default; STUDIOBOARD_FAKE_DOCTOR=running
    // previews the in-flight (diagnosing) state.
    let doc_running = std::env::var("STUDIOBOARD_FAKE_DOCTOR").map(|v| v == "running").unwrap_or(false);
    st.doctor = state::Doctor {
        available: true,
        running: doc_running,
        step: if doc_running { "local triage (qwen2.5:14b)…".into() } else { String::new() },
        trigger: "runaway: rustc at 356% ≥ 220% · load1 28.0 ≥ 28.0".into(),
        last_title: "rustc compile burst — self-resolved, system all-clear".into(),
        last_outcome: "no-action-needed".into(),
        last_severity: "info".into(),
        last_model: if doc_running { "sonnet".into() } else { "qwen2.5:14b".into() },
        last_actions: vec![],
        last_rel: "12m".into(),
        today_cost: 0.34,
        incidents_total: 11,
    };
    // KEYBINDS card preview — a representative slice of the Hammerspoon cheat sheet.
    let g = |name: &str, binds: &[(&str, &str)]| state::KeyGroup {
        name: name.into(),
        binds: binds.iter().map(|(k, d)| (k.to_string(), d.to_string())).collect(),
    };
    st.keybinds = state::Keybinds {
        available: true,
        hyper: "hold Caps Lock".into(),
        groups: vec![
            g("Apps · Hyper+key", &[
                ("T", "Warp"), ("C", "Google Chrome"), ("F", "Finder"), ("D", "Discord"),
                ("M", "Music"), ("N", "Music · next song"), ("=", "Music · volume +10%"),
                ("-", "Music · volume -10%"),
            ]),
            g("Windows · Hyper+key", &[
                ("←", "left half"), ("→", "right half"), ("↑", "maximize"), ("↓", "center 70%"),
                ("1", "left third"), ("2", "middle third"), ("3", "right third"),
                ("Z", "undo last move"), ("V", "clipboard history"), ("H", "toggle cheat sheet"),
            ]),
            g("Window mode · after Hyper+W", &[
                ("←", "left half"), ("→", "right half"), ("↑", "maximize"), ("↓", "center 70%"),
                ("hjkl", "nudge"), ("escape", "exit"),
            ]),
            g("Controls · Ctrl+Alt+key", &[
                ("←", "all windows → extended"), ("→", "gather → main"), ("R", "reload config"),
                ("C", "close window"), ("M", "minimize"), ("↑", "maximize"), ("↓", "restore 70%"),
                ("= / −", "grow / shrink"),
            ]),
            g("Displays / KVM", &[
                ("[", "Studio → LEFT · RIGHT to MacBook"),
                ("]", "Studio → RIGHT · LEFT to MacBook"),
                ("0", "bailout · reclaim BOTH panels"),
            ]),
            g("Capture · Hyper+key", &[
                ("S", "selection screenshot → clipboard"), ("R", "record window → Desktop"),
            ]),
            g("Text expansion", &[
                ("@@", "email"), (";dt", "date"), (";shrug", "¯\\_(ツ)_/¯"),
            ]),
        ],
    };
    // `--compose` previews the inline reply input affordance.
    if compose {
        st.msg_ui.composing = true;
        st.msg_ui.draft = "on my way".into();
        st.msg_ui.phase = state::MsgPhase::Opening;
        st.msg_ui.anim_start = Some(Instant::now());
    }
    st.lyrics = Lyrics {
        synced: true,
        track_id: st.music.track_id(),
        note: String::new(),
        lines: vec![
            (7.41, "Started"),
            (9.68, "(Zombie on the track)"),
            (10.75, "Started from the bottom now we're here"),
            (13.71, "Started from the bottom now my whole team here"),
            (16.54, "Started from the bottom now we're here"),
            (19.31, "Started from the bottom now the whole team here"),
        ]
        .into_iter()
        .map(|(t, s)| LyricLine { t, text: s.to_string() })
        .collect(),
    };
}
