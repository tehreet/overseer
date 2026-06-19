//! studioboard — a buttery-smooth always-on TUI for the Mac Studio.
//!
//! Collectors run on background threads; the render loop is decoupled and
//! frame-paced (up to 120 fps while music is playing so the progress bar and
//! karaoke lyric wipe move every frame).
#![allow(dead_code)] // full palette + helpers kept available for tweaking

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
    run()
}

fn run() -> Result<()> {
    let shared = Arc::new(Mutex::new(AppState::default()));

    collectors::spawn_system(shared.clone());
    collectors::spawn_macmon(shared.clone());
    collectors::spawn_music(shared.clone());
    collectors::spawn_lyrics(shared.clone());
    collectors::spawn_artwork(shared.clone());
    collectors::spawn_git(shared.clone());
    collectors::spawn_weather(shared.clone());
    collectors::spawn_usage(shared.clone());

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

fn event_loop<B: ratatui::backend::Backend>(
    term: &mut Terminal<B>,
    shared: &Arc<Mutex<AppState>>,
) -> Result<()> {
    loop {
        // Snapshot just what we need for pacing while we hold the lock to draw.
        let playing;
        {
            let s = shared.lock().unwrap();
            let t = s.started.elapsed().as_secs_f64();
            playing = s.music.playing;
            term.draw(|f| ui::render(f, &s, t))?;
        }

        // The CPU equalizer interpolates continuously between cached samples,
        // so keep a smooth 60fps baseline; go 120fps with music for the wipe.
        let budget = if playing {
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
                        let q = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc);
                        let ctrl_c = k.code == KeyCode::Char('c')
                            && k.modifiers.contains(KeyModifiers::CONTROL);
                        if q || ctrl_c {
                            return Ok(());
                        }
                    }
                }
            }
        }
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
///   studioboard --snapshot [WIDTHxHEIGHT]
fn snapshot(args: &[String]) -> Result<()> {
    use ratatui::backend::TestBackend;
    let (w, h) = args
        .iter()
        .find_map(|a| a.split_once('x'))
        .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
        .unwrap_or((140u16, 44u16));

    let mut st = AppState::default();
    sample_data(&mut st);

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

/// Fills state with representative data so the snapshot looks real.
fn sample_data(st: &mut AppState) {
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
            ("studioboard".into(), 18.4, 28_000_000),
            ("WindowServer".into(), 12.1, 480_000_000),
            ("Music".into(), 6.3, 320_000_000),
            ("Warp".into(), 4.8, 540_000_000),
            ("claude".into(), 3.2, 210_000_000),
            ("kernel_task".into(), 2.0, 120_000_000),
        ],
    };
    // Staggered identical samples so the delayed interpolation has data to
    // read at `now - EQ_DELAY`.
    let pc = st.system.per_core.clone();
    let now = Instant::now();
    for k in [4u64, 3, 2, 1, 0] {
        let ts = now.checked_sub(Duration::from_secs(k)).unwrap_or(now);
        st.cpu_samples.push_back((ts, pc.clone()));
        st.net_samples.push_back((ts, 2_400_000.0));
        st.silicon_samples
            .push_back((ts, vec![31.0, 18.4, 58.0, 52.0, 9.2, 3.1, 55.6, 41.0, 23.0, 0.2]));
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
        today_cost: 18.40,
        month_cost: 214.75,
        today_messages: 342,
        top_model: "Opus".into(),
        sessions_today: 6,
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
    // Synthetic album art (radial gradient) so the snapshot shows the panel.
    let dim = 64usize;
    let mut px = Vec::with_capacity(dim * dim);
    for y in 0..dim {
        for x in 0..dim {
            let r = (x * 255 / dim) as u8;
            let g = (y * 255 / dim) as u8;
            px.push([r, g, 180]);
        }
    }
    st.album_art = state::AlbumArt { track_id: st.music.track_id(), w: dim, h: dim, px };
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
        last_msg: "feat(macos): Hyper-layer display/KVM switching".into(),
        last_rel: "3 days ago".into(),
        commits_today: 4,
    };
    st.weather = state::Weather {
        fresh: true,
        location: "Eau Claire, Wisconsin".into(),
        temp_f: 62,
        feels_f: 62,
        desc: "Sunny".into(),
        icon: "☀".into(),
        hi_f: 74,
        lo_f: 49,
        humidity: 62,
    };
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
