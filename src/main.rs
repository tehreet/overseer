//! overseer — a buttery-smooth always-on TUI for the Mac Studio.
//!
//! Collectors run on background threads; the render loop is decoupled and
//! frame-paced (up to 120 fps while music is playing so the progress bar and
//! karaoke lyric wipe move every frame).

#[cfg(target_os = "macos")]
mod audio;
mod cache;
mod collectors;
mod lyrics;
#[cfg(target_os = "macos")]
mod nowplaying;
mod state;
mod theme;
mod ui;

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::{execute, terminal};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use state::AppState;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--demo") {
        return run_demo();
    }
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
    if args.iter().any(|a| a == "--diag-live") {
        collectors::diag_live_sessions();
        return Ok(());
    }
    if args.iter().any(|a| a == "--diag-tv") {
        collectors::diag_tv();
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    if args.iter().any(|a| a == "--diag-np") {
        nowplaying::diag();
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    if args.iter().any(|a| a == "--diag-audio") {
        audio::diag();
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    if args.iter().any(|a| a == "--diag-discord-audio") {
        audio::diag_voice();
        return Ok(());
    }
    if args.iter().any(|a| a == "--diag-voice") {
        // Headless: run only the Discord collector with voice listening on and
        // stream the handshake log, so the voice path can be diagnosed without
        // the TUI (which needs a real terminal). Joins voice — use briefly.
        std::env::set_var("OVERSEER_DISCORD_VOICE_LISTEN", "1");
        let shared = Arc::new(Mutex::new(AppState::default()));
        collectors::spawn_discord(shared);
        let log = dirs::home_dir()
            .map(|h| h.join(".cache/overseer/voice.log"))
            .unwrap_or_default();
        let _ = std::fs::remove_file(&log);
        println!("--diag-voice: listening 20s (need someone in a voice channel)…\n");
        let mut shown = 0u64;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if let Ok(txt) = std::fs::read_to_string(&log) {
                let lines: Vec<&str> = txt.lines().collect();
                for l in lines.iter().skip(shown as usize) {
                    println!("{l}");
                }
                shown = lines.len() as u64;
            }
        }
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
    collectors::spawn_live_sessions(shared.clone());
    collectors::spawn_messages(shared.clone());
    collectors::spawn_signal(shared.clone());
    collectors::spawn_discord(shared.clone());
    collectors::spawn_doctor(shared.clone());
    collectors::spawn_keybinds(shared.clone());
    collectors::spawn_audio(shared.clone());
    // Discord "who's talking" by tapping Discord.app's audio locally (the bot
    // voice gateway is walled by DAVE/E2EE). Lights the DISCORD border on call audio.
    #[cfg(target_os = "macos")]
    audio::spawn_voice(shared.clone());

    // Terminal setup with a panic hook that always restores the screen.
    terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    // EnableMouseCapture: the iMessage card is click-to-focus / double-click-to-reply.
    execute!(out, terminal::EnterAlternateScreen, EnableMouseCapture, crossterm::cursor::Hide)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            terminal::LeaveAlternateScreen,
            crossterm::cursor::Show
        );
        default_hook(info);
    }));

    let backend = CrosstermBackend::new(out);
    let mut term = Terminal::new(backend)?;
    let res = event_loop(&mut term, &shared);

    terminal::disable_raw_mode()?;
    execute!(
        term.backend_mut(),
        DisableMouseCapture,
        terminal::LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    term.show_cursor()?;
    res
}

/// `--demo`: every card alive with entirely fictional data — no real messages,
/// contacts, music, or system readings are ever touched. Seeds the metric
/// scaffolding from `sample_data`, replaces every human-facing field with
/// invented content, then runs the normal render loop with a lightweight fake
/// "collector" feeding the time-windowed animations. A pure smoothness showcase.
fn run_demo() -> Result<()> {
    let mut st = AppState::default();
    sample_data(&mut st, false);
    demo_anonymize(&mut st);
    let shared = Arc::new(Mutex::new(st));

    // One fake collector: pushes fresh synthetic samples each second so the EQ,
    // RESOURCES wave and silicon gauges keep gliding (a frozen sample window goes
    // flat after ~1 s), and loops the track so the progress bar + karaoke wipe
    // never run out. No real data source is opened.
    {
        let sh = shared.clone();
        std::thread::spawn(move || demo_ticker(sh));
    }

    terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    execute!(out, terminal::EnterAlternateScreen, EnableMouseCapture, crossterm::cursor::Hide)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            terminal::LeaveAlternateScreen,
            crossterm::cursor::Show
        );
        default_hook(info);
    }));
    let backend = CrosstermBackend::new(out);
    let mut term = Terminal::new(backend)?;
    let res = event_loop(&mut term, &shared);
    terminal::disable_raw_mode()?;
    execute!(
        term.backend_mut(),
        DisableMouseCapture,
        terminal::LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    term.show_cursor()?;
    res
}

/// Overwrite every human-facing field with invented content so `--demo` never
/// shows a real name, message, contact, track, or location. The metric cards
/// (CPU/RESOURCES/proc/silicon/weather) keep their `sample_data` shapes; only the
/// people-and-media surfaces are replaced.
fn demo_anonymize(st: &mut AppState) {
    use state::{ActionKind, LiveSession, LiveSessions, Lyrics, LyricLine, MusicFacts, MusicStats,
        Queue, QueueTrack, TextChannel, VoiceChannel};
    let now = Instant::now();

    // --- now playing: a fictional synthwave track, looping ---
    st.music = MusicStats {
        running: true,
        playing: true,
        track: "Midnight Vector".into(),
        artist: "The Vapor Lines".into(),
        album: "Neon Horizons".into(),
        duration: 204.0,
        base_pos: 0.0,
        sampled_at: now,
        polled: true,
        ..Default::default()
    };
    // Re-derive art + palette for the new (uncached → gradient) track so accents stay coherent.
    st.album_art = std::sync::Arc::new(collectors::sample_album_art(st.music.track_id()));
    let target = theme::theme_from_art(&st.album_art.px);
    st.dynamic_theme.retarget(st.music.track_id(), [target.0, target.1, target.2]);
    st.dynamic_theme.blend_start = now - Duration::from_secs(1);

    // --- lyrics: original lines (synced) so the karaoke wipe has something to ride ---
    st.lyrics = Lyrics {
        synced: true,
        track_id: st.music.track_id(),
        note: String::new(),
        lines: [
            (0.0, "neon hums along the empty street"),
            (8.5, "the engine warm beneath our feet"),
            (17.0, "headlights paint the falling rain"),
            (25.5, "chrome and violet, here again"),
            (34.0, "hold the line, the night runs long"),
            (42.5, "every signal turns to song"),
            (51.0, "we drive until the morning shows"),
            (59.5, "where the quiet river goes"),
        ]
        .into_iter()
        .map(|(t, s)| LyricLine { t, text: s.to_string() })
        .collect(),
    };

    // --- queue: fictional up-next ---
    st.queue = Queue {
        fresh: true,
        source_track_id: st.music.track_id(),
        items: vec![
            QueueTrack { track: "Afterglow".into(), artist: "Cassette Future".into(), duration: 188.0 },
            QueueTrack { track: "Glass Avenue".into(), artist: "Nightset".into(), duration: 211.0 },
            QueueTrack { track: "Lightyears".into(), artist: "Aria Volt".into(), duration: 196.0 },
        ],
    };

    // --- facts: about the dashboard itself (no real trivia/people) ---
    st.facts = MusicFacts {
        track_id: st.music.track_id(),
        source: "demo".into(),
        note: String::new(),
        lines: vec![
            "Demo mode — every name, message, and reading on screen is invented.".into(),
            "The render loop runs up to 120 fps while music plays.".into(),
            "Gauges interpolate between samples, so nothing ever steps.".into(),
            "The karaoke wipe tracks the real playback position.".into(),
        ],
    };

    // --- iMessage / Signal: invented chats, no real contacts ---
    let msg = |sender: &str, preview: &str, rel: &str, unread: bool, from_me: bool| state::MessageItem {
        chat_id: 0,
        rowid: 0,
        sender: sender.into(),
        handle: String::new(),
        guid: String::new(),
        preview: preview.into(),
        full_text: preview.into(),
        ts_unix: 0.0,
        rel: rel.into(),
        is_rich: false,
        unread,
        from_me,
        is_shortcode: false,
    };
    st.messages.unread_count = 2;
    st.messages.items = vec![
        msg("Jordan", "are we still on for tonight?", "2m", true, false),
        msg("Book Club", "Sam: chapter four wrecked me", "9m", true, false),
        msg("Riley", "You: on my way", "14m", false, true),
        msg("Coffee Crew", "Pat: who's in friday?", "1h", false, false),
        msg("Dana", "You: 😂", "yd", false, true),
    ];
    st.msg_ui = state::MsgUi { active: true, ..Default::default() };
    st.signal.unread_count = 1;
    st.signal.items = vec![
        msg("Trailhead", "Max: 8am at the gate?", "6m", true, false),
        msg("Kai", "You: sounds perfect", "1h", false, true),
        msg("Studio", "Remy: new mix is up", "3h", false, false),
    ];

    // --- Discord: invented voice + text channels (voice section showcased) ---
    st.discord.voice = vec![
        VoiceChannel { name: "lounge".into(), members: vec!["Pixel".into(), "Echo".into(), "Nova".into()] },
        VoiceChannel { name: "co-work".into(), members: vec!["Sol".into()] },
    ];
    st.discord.text = vec![
        TextChannel { name: "general".into(), author: "Echo".into(), preview: "gm everyone".into(), rel: "4m".into(), unread: true },
        TextChannel { name: "builds".into(), author: "Pixel".into(), preview: "[image]".into(), rel: "51m".into(), unread: false },
        TextChannel { name: "music".into(), author: "Nova".into(), preview: "new playlist".into(), rel: "2h".into(), unread: false },
    ];
    st.discord.voice_join_at = Some(now); // a gentle 20s join shimmer on open

    // --- ROBOTS feed: invented sessions; doctor calm (no alert/shimmer) ---
    let mk = |project: &str, model: &str, kind, action: &str, age: f64| LiveSession {
        session_id: project.into(),
        project: project.into(),
        branch: "main".into(),
        model: model.into(),
        action: action.into(),
        kind,
        age_secs: age,
    };
    st.live = LiveSessions {
        fresh: true,
        sessions: vec![
            mk("aurora-ui", "Opus", ActionKind::Edit, "editing panel.rs", 1.2),
            mk("aurora-ui", "Opus", ActionKind::Run, "cargo build --release", 3.8),
            mk("ledger", "Sonnet", ActionKind::Read, "scanning routes", 7.5),
            mk("pipeline", "Haiku", ActionKind::Think, "thinking", 15.0),
            mk("sandbox", "Sonnet", ActionKind::Idle, "awaiting you", 42.0),
        ],
    };
    st.doctor.running = false;
    st.doctor.last_outcome = "resolved".into();

    // --- git pulse + weather: invented repo + place ---
    st.git.repo = "aurora-ui".into();
    st.git.branch = "main".into();
    st.git.last_hash = "c0ffee1".into();
    st.git.last_msg = "feat(ui): glide the resource wave".into();
    st.git.last_rel = "12 minutes ago".into();
    st.git.commits_today = 9;
    st.git.loc_added = 412;
    st.git.loc_removed = 88;
    st.git.merges_main = 1;
    st.git.pr_count = 3;
    st.weather.location = "Harbor City".into();
}

/// Demo's fake collector: every second, push fresh synthetic samples into the
/// time-windowed buffers (so the EQ, RESOURCES wave, net wave and silicon gauges
/// keep their buttery delayed-interpolation motion instead of flatlining), and
/// loop the track so the progress bar + lyrics never run dry.
fn demo_ticker(shared: Arc<Mutex<AppState>>) {
    let mut k: u64 = 16; // continue past sample_data's pre-fill phase
    loop {
        std::thread::sleep(Duration::from_secs(1));
        k += 1;
        let ph = k as f32 * 0.5;
        let now = Instant::now();
        let mut s = shared.lock().unwrap();

        let gpu = (31.0 + 22.0 * (ph * 0.7).sin()).clamp(0.0, 100.0);
        let pwr = (55.6 + 30.0 * (ph * 0.6 + 1.0).sin()).clamp(5.0, 120.0);
        s.silicon_samples.push_back((now, vec![gpu, 18.4, 58.0, 52.0, 9.2, 3.1, pwr, 41.0, 23.0, 0.2]));
        while s.silicon_samples.len() > 16 { s.silicon_samples.pop_front(); }

        let memp = (60.0 + 6.0 * (ph * 0.7).sin()).clamp(0.0, 100.0);
        let down = (300.0 + 2200.0 * (ph * 0.8).sin().abs()).max(1.0);
        let up = (60.0 + 380.0 * (ph * 0.6 + 1.0).sin().abs()).max(1.0);
        let io = (40.0 + 9000.0 * (ph * 0.9 + 0.5).sin().abs()).max(1.0);
        let cpup = (44.0 + 28.0 * (ph * 1.1 + 0.3).sin()).clamp(0.0, 100.0);
        s.res_samples.push_back((now, vec![memp, down, up, io, gpu, cpup]));
        while s.res_samples.len() > 16 { s.res_samples.pop_front(); }

        let procs = s.system.top_procs.clone();
        s.proc_samples.push_back((now, procs));
        while s.proc_samples.len() > 16 { s.proc_samples.pop_front(); }

        // Loop the track so the progress bar + karaoke wipe run forever.
        let pos = s.music.position();
        let dur = s.music.duration;
        if s.music.playing && pos >= dur - 0.5 {
            s.music.base_pos = 0.0;
            s.music.sampled_at = now;
        }
    }
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
    // Clickable iMessage rows from the last drawn frame; refreshed every render.
    let mut hit = ui::MsgHit::default();
    loop {
        // Settle any finished iMessage transition, then take a cheap snapshot of
        // AppState and RELEASE the lock before drawing (issue #16). render() is the
        // most expensive work per frame; holding the lock across it makes every
        // collector and the realtime audio IOProc contend on AppState for the whole
        // frame, causing micro-stutter. AppState derives Clone and owns only plain,
        // bounded data (no Arc/Mutex/handles), so a snapshot clone is safe and keeps
        // the critical section to just settle + clone.
        let snapshot = {
            let mut s = shared.lock().unwrap();
            settle_msg_anim(&mut s);
            settle_queue_anim(&mut s);
            s.clone()
        };

        let t = snapshot.started.elapsed().as_secs_f64();
        let playing = snapshot.music.playing;
        // Keep painting at high fps while the QUEUE↔LYRICS width glide or the
        // KEYBINDS show/hide collapse is live.
        let animating = snapshot.msg_ui.animating()
            || snapshot.queue_toggle_at.elapsed() < QUEUE_ANIM
            || snapshot.keybinds_toggle_at.elapsed() < KEYBINDS_ANIM;
        // render() records the clickable iMessage rows into `hit` so a mouse
        // click can be mapped back to a conversation (the snapshot is a clone, so
        // render can't write hitboxes into shared state — they ride out here).
        term.draw(|f| ui::render(f, &snapshot, t, &mut hit))?;

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
                match event::read()? {
                    Event::Key(k)
                        if k.kind != KeyEventKind::Release
                            && handle_key(shared, k.code, k.modifiers) =>
                    {
                        return Ok(());
                    }
                    Event::Mouse(m) => handle_mouse(shared, m, &hit),
                    _ => {}
                }
            }
        }
    }
}

/// Duration of the QUEUE collapse / LYRICS expand glide. Kept in sync with the
/// easing window `ui::queue_open_frac` reads.
const QUEUE_ANIM: Duration = Duration::from_millis(380);

/// Duration of the KEYBINDS show/hide collapse. Kept in sync with the easing
/// window `ui::keybinds_open_frac` reads.
const KEYBINDS_ANIM: Duration = Duration::from_millis(360);

/// Detect when the QUEUE goes empty↔non-empty and stamp the transition so the
/// render can ease the lyrics/queue split width. The queue is "open" only when
/// music is playing a track AND there are upcoming items; otherwise it collapses
/// and the LYRICS (or synopsis) card expands to fill the row.
fn settle_queue_anim(s: &mut AppState) {
    let want = s.music.running && !s.music.track.is_empty() && !s.queue.items.is_empty();
    if want != s.queue_open {
        s.queue_open = want;
        s.queue_toggle_at = Instant::now();
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
                    // Target = the focused conversation, by chat GUID.
                    if let Some(guid) = focused_guid(&s) {
                        collectors::send_imessage(shared.clone(), guid, draft);
                        // Optimistic "whoosh" close; an async failure reopens the
                        // composer with the draft restored and flashes the border.
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
            KeyCode::Char(ch)
                if !mods.contains(KeyModifiers::CONTROL) => {
                    s.msg_ui.draft.push(ch);
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
                s.msg_ui.focus_chat_id = None; // clear the highlight marker too
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
                // First press focuses the card on the most relevant conversation.
                let f = default_focus(&s);
                s.msg_ui.active = true;
                s.msg_ui.focus_chat_id = f;
                s.msg_ui.phase = MsgPhase::Idle;
            } else if double {
                // Double-press on the focused conversation → open inline reply.
                if s.msg_ui.focus_chat_id.is_none() {
                    s.msg_ui.focus_chat_id = default_focus(&s);
                }
                s.msg_ui.composing = true;
                s.msg_ui.draft.clear();
                s.msg_ui.phase = MsgPhase::Opening;
                s.msg_ui.anim_start = Some(now);
            } else {
                // Single press → mark focused read + advance to the next conversation.
                advance_queue(&mut s);
                s.msg_ui.phase = MsgPhase::Advancing;
                s.msg_ui.anim_start = Some(now);
            }
        }
        _ => {}
    }
    false
}

/// Route a mouse click through the iMessage card. Single left-click focuses
/// (highlights) the conversation under the cursor; a double-click on that same
/// conversation drops the cursor into a reply composer for it.
fn handle_mouse(shared: &Arc<Mutex<AppState>>, m: MouseEvent, hit: &ui::MsgHit) {
    use state::MsgPhase;
    if !matches!(m.kind, MouseEventKind::Down(MouseButton::Left)) {
        return;
    }
    let Some(chat_id) = hit.chat_at(m.column, m.row) else {
        return; // click landed outside any conversation row
    };
    let mut s = shared.lock().unwrap();
    let now = Instant::now();
    let double = s.msg_ui.last_click_chat == Some(chat_id)
        && s
            .msg_ui
            .last_click_at
            .map(|p| now.duration_since(p) <= MSG_DOUBLE)
            .unwrap_or(false);
    s.msg_ui.last_click_at = Some(now);
    s.msg_ui.last_click_chat = Some(chat_id);

    if double {
        s.msg_ui.active = true;
        s.msg_ui.focus_chat_id = Some(chat_id);
        s.msg_ui.composing = true;
        s.msg_ui.draft.clear();
        s.msg_ui.phase = MsgPhase::Opening;
        s.msg_ui.anim_start = Some(now);
    } else if !s.msg_ui.composing {
        // Single click just highlights — don't disturb an open composer.
        s.msg_ui.active = true;
        s.msg_ui.focus_chat_id = Some(chat_id);
        s.msg_ui.phase = MsgPhase::Idle;
    }
}

/// Default focus when the card is first activated: the most relevant
/// conversation — the first unread one, or the newest if everything's read.
fn default_focus(s: &AppState) -> Option<i64> {
    s.messages
        .items
        .iter()
        .find(|m| m.unread)
        .or_else(|| s.messages.items.first())
        .map(|m| m.chat_id)
}

/// `chat.guid` of the focused conversation — the reply target. Works for
/// iMessage, SMS, 1:1 and group threads alike (empty → no sendable target).
fn focused_guid(s: &AppState) -> Option<String> {
    let id = s.msg_ui.focus_chat_id?;
    s.messages
        .items
        .iter()
        .find(|m| m.chat_id == id)
        .map(|m| m.guid.clone())
        .filter(|g| !g.is_empty())
}

/// Mark the focused conversation read — flip it in our snapshot for an instant
/// response, persist to chat.db so the next poll doesn't resurrect it — then
/// advance focus to the next conversation in the list.
fn advance_queue(s: &mut AppState) {
    let Some(id) = s.msg_ui.focus_chat_id else { return };
    let Some(pos) = s.messages.items.iter().position(|m| m.chat_id == id) else { return };
    let (was_unread, is_shortcode) = {
        let m = &s.messages.items[pos];
        (m.unread, m.is_shortcode)
    };
    for it in s.messages.items.iter_mut().filter(|it| it.chat_id == id) {
        it.unread = false;
    }
    if was_unread && !is_shortcode {
        s.messages.unread_count = s.messages.unread_count.saturating_sub(1);
    }
    collectors::mark_chat_read(id);
    // Advance focus to the next conversation (stay put if this was the last).
    if let Some(next) = s.messages.items.get(pos + 1).map(|m| m.chat_id) {
        s.msg_ui.focus_chat_id = Some(next);
    }
}

/// Exercise the real music + lyrics code paths and print what they return.
/// Run from the terminal you actually use overseer in, so it reflects that
/// app's Automation (Music control) permission.
fn diag() -> Result<()> {
    println!("overseer --diag\n");
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
    println!("overseer --facts\n");
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

/// Wipe the persistent disk cache (`~/.cache/overseer/{lyrics,facts,art}`) and
/// report what was removed, so a song re-fetches lyrics + facts + art next play.
fn clear_cache() -> Result<()> {
    println!("overseer --clear-cache\n");
    if let Some(root) = cache::root() {
        println!("cache root: {}", root.display());
    }
    for line in cache::clear() {
        println!("  {line}");
    }
    Ok(())
}

///   overseer --snapshot [WIDTHxHEIGHT]
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
    // `kbhide=1` previews the KEYBINDS card hidden (Hyper+H); `kbhide=mid` half.
    if let Some(v) = args.iter().find_map(|a| a.strip_prefix("kbhide=")) {
        st.keybinds_visible = false;
        let back = if v == "mid" { 170 } else { 1000 };
        st.keybinds_toggle_at = Instant::now()
            .checked_sub(Duration::from_millis(back))
            .unwrap_or_else(Instant::now);
    }

    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend)?;
    let t = 1.3;
    term.draw(|f| ui::render(f, &st, t, &mut ui::MsgHit::default()))?;

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
/// rendering: `overseer --cells [WxH] [--idle]`. One line per cell:
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
    // Visual-verify hook for the QUEUE collapse: `qopen=0` forces the settled
    // collapsed state (LYRICS full width), `qopen=1` the settled open state.
    if let Some(open) = args.iter().find_map(|a| a.strip_prefix("qopen=")) {
        if open == "mid" {
            // Mid-collapse: opened, ~0.17s ago → smoothstep ≈ half width.
            st.queue_open = true;
            st.queue_toggle_at = Instant::now()
                .checked_sub(Duration::from_millis(170))
                .unwrap_or_else(Instant::now);
        } else {
            st.queue_open = open != "0";
            // Push the toggle well into the past so the smoothstep reads as settled.
            st.queue_toggle_at = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now);
            if !st.queue_open {
                st.queue.items.clear();
            }
        }
    }
    // Visual-verify hook for the KEYBINDS collapse: `kbhide=1` forces the settled
    // hidden state (card gone), `kbhide=mid` a half-collapsed frame.
    if let Some(v) = args.iter().find_map(|a| a.strip_prefix("kbhide=")) {
        st.keybinds_visible = false;
        let back = if v == "mid" { 170 } else { 1000 };
        st.keybinds_toggle_at = Instant::now()
            .checked_sub(Duration::from_millis(back))
            .unwrap_or_else(Instant::now);
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
    term.draw(|f| ui::render(f, &st, t, &mut ui::MsgHit::default()))?;
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

/// One sample iMessage row: (sender, handle, text, rel_time, rich, unread,
/// from_me, shortcode).
type SampleMsgRow = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    bool,
    bool,
    bool,
    bool,
);

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
            ("overseer".into(), 18.4, 28_000_000, 367_200),
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
    let now = Instant::now();
    for k in (0..16u64).rev() {
        let ts = now.checked_sub(Duration::from_secs(k)).unwrap_or(now);
        let ph = k as f32 * 0.5;
        let gpu = (31.0 + 22.0 * (ph * 0.7).sin()).clamp(0.0, 100.0);
        let pwr = (55.6 + 30.0 * (ph * 0.6 + 1.0).sin()).clamp(5.0, 120.0);
        st.silicon_samples
            .push_back((ts, vec![gpu, 18.4, 58.0, 52.0, 9.2, 3.1, pwr, 41.0, 23.0, 0.2]));
        // RESOURCES wave channels: mem %, net down/up KB/s, disk I/O KB/s, GPU %,
        // CPU % — gently varying so visual-verify shows the smooth stacked wave.
        let memp = (60.0 + 6.0 * (ph * 0.7).sin()).clamp(0.0, 100.0);
        let down = (300.0 + 2200.0 * (ph * 0.8).sin().abs()).max(1.0);
        let up = (60.0 + 380.0 * (ph * 0.6 + 1.0).sin().abs()).max(1.0);
        let io = (40.0 + 9000.0 * (ph * 0.9 + 0.5).sin().abs()).max(1.0);
        let gpup = (31.0 + 22.0 * (ph * 0.7).sin()).clamp(0.0, 100.0); // GPU %
        let cpup = (44.0 + 28.0 * (ph * 1.1 + 0.3).sin()).clamp(0.0, 100.0); // CPU %
        st.res_samples.push_back((ts, vec![memp, down, up, io, gpup, cpup]));
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
        today_messages: 342,
        top_model: "Opus".into(),
        sessions_today: 6,
        tokens_7d: 1_840_000_000,
        tokens_30d: 7_320_000_000,
        sessions_30d: 73,
        hourly: vec![0, 0, 0, 0, 0, 0, 0, 12, 40, 88, 120, 64, 30, 55, 90, 140, 110, 70, 0, 0, 0, 0, 0, 0],
    };
    // A representative realtime "who's working" roster for the ROBOTS feed, one
    // row per live Claude Code session, newest-active first (matches the
    // collector's sort) so visual-verify shows the icon/pulse/age treatment.
    {
        use state::{ActionKind, LiveSession, LiveSessions};
        let mk = |project: &str, model: &str, kind, action: &str, age: f64| LiveSession {
            session_id: project.into(),
            project: project.into(),
            branch: "main".into(),
            model: model.into(),
            action: action.into(),
            kind,
            age_secs: age,
        };
        st.live = LiveSessions {
            fresh: true,
            sessions: vec![
                mk("battlestation", "Opus", ActionKind::Edit, "editing ui.rs", 1.4),
                mk("battlestation", "Opus", ActionKind::Run, "cargo build --release", 4.2),
                mk("dotfiles", "Sonnet", ActionKind::Read, "grep spawn_live", 9.0),
                mk("syswatch", "Haiku", ActionKind::Think, "thinking", 17.0),
                mk("homelab", "Sonnet", ActionKind::Idle, "awaiting you", 48.0),
            ],
        };
    }
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
        ..Default::default()
    };
    // OVERSEER_FAKE_WATCH previews the "now watching" band off-screen:
    // `=movie` a film, anything else a TV episode. Mirrors OVERSEER_FAKE_VOICE.
    if let Ok(kind) = std::env::var("OVERSEER_FAKE_WATCH") {
        let movie = kind == "movie";
        st.music = MusicStats {
            running: true,
            playing: true,
            source: state::MediaSource::Tv,
            track: if movie { "Dune: Part Two".into() } else { "Cold Harbor".into() },
            duration: if movie { 9966.0 } else { 2890.0 },
            base_pos: if movie { 3727.0 } else { 1042.0 },
            sampled_at: Instant::now(),
            polled: true,
            watch: state::WatchMeta {
                show: if movie { String::new() } else { "Severance".into() },
                season: if movie { 0 } else { 2 },
                episode: if movie { 0 } else { 10 },
                year: if movie { "2024".into() } else { "2025".into() },
                genre: if movie { "Sci-Fi".into() } else { "Thriller".into() },
                director: if movie { "Denis Villeneuve".into() } else { "Ben Stiller".into() },
                kind: if movie { "movie".into() } else { "TV show".into() },
                poster_url: String::new(),
            },
            ..Default::default()
        };
    }
    // Album art: decode the last real dump (so visual-verify exercises the true
    // sampling path on a real cover); fall back to a radial gradient otherwise.
    st.album_art = std::sync::Arc::new(collectors::sample_album_art(st.music.track_id()));
    // Drive the dynamic palette off that cover so off-screen verify shows the
    // album-biased accents too. Back-date the fade so the single headless frame
    // renders the cross-fade fully settled rather than mid-glide (#8).
    let target = theme::theme_from_art(&st.album_art.px);
    st.dynamic_theme.retarget(st.music.track_id(), [target.0, target.1, target.2]);
    st.dynamic_theme.blend_start = Instant::now() - std::time::Duration::from_secs(1);
    st.facts = state::MusicFacts {
        track_id: st.music.track_id(),
        source: "claude".into(),
        note: String::new(),
        lines: if st.music.is_tv() {
            vec![
                "Ben Stiller directs most of the series and shapes its eerie tone.".into(),
                "Adam Scott and Britt Lower lead as severed Lumon employees.".into(),
                "The Lumon offices were shot in Bell Labs' Holmdel, NJ complex.".into(),
                "The score is by Theodore Shapiro; the title sequence won an Emmy.".into(),
            ]
        } else {
            vec![
                "Drake's first solo #1 on the Billboard Hot 100.".into(),
                "The beat samples Whitney Houston's \"I'm Every Woman\" ad-libs.".into(),
                "Recorded in a single late-night session in Toronto.".into(),
                "The phrase became a meme long before the song dropped.".into(),
            ]
        },
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
        last_msg: "feat(overseer): merge Robots Working card".into(),
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
        .map(|(i, (sender, handle, text, rel, rich, unread, from_me, shortcode)): (usize, SampleMsgRow)| {
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
                guid: String::new(),
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
    st.msg_ui = state::MsgUi { active: true, ..Default::default() };
    st.signal = state::Messages {
        fresh: true,
        available: true,
        unread_count: 1,
        // Signal conversations — read-only, same row style. (sender,text,rel,unread,from_me,rich)
        items: vec![
            ("The Other CW Back Channel", "Elijah: [photo]", "6m", true, false, true),
            ("Alex Stalmakov", "You: Feels bad man", "1h", false, true, false),
            ("Marcin W", "You: Lmao", "9h", false, true, false),
            ("Rob", "Oh cool! I was wondering what exists", "yd", false, false, false),
            ("Adam", "You: Jesus", "yd", false, true, false),
        ]
        .into_iter()
        .map(|(sender, text, rel, unread, from_me, rich): (&str, &str, &str, bool, bool, bool)| {
            state::MessageItem {
                chat_id: 0,
                rowid: 0,
                sender: sender.into(),
                handle: String::new(),
                guid: String::new(),
                preview: text.into(),
                full_text: text.into(),
                ts_unix: 0.0,
                rel: rel.into(),
                is_rich: rich,
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
        voice_speaking: std::env::var("OVERSEER_FAKE_SPEAKING")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false),
        voice_speaking_tap: false,
        voice_e2ee_blocked: false,
    };
    // MAC-DOCTOR card preview. Idle by default; OVERSEER_FAKE_DOCTOR=running
    // previews the in-flight (diagnosing) state.
    let doc_running = std::env::var("OVERSEER_FAKE_DOCTOR").map(|v| v == "running").unwrap_or(false);
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
    // `--compose` previews the inline reply composer threaded under the focused
    // conversation (the top one here).
    if compose {
        st.msg_ui.active = true;
        st.msg_ui.focus_chat_id = st.messages.items.first().map(|m| m.chat_id);
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

    // Watching has no lyrics / up-next — match what the live collectors do so the
    // off-screen preview is faithful (LYRICS parks a note, QUEUE shows empty).
    if st.music.is_tv() {
        st.lyrics = Lyrics {
            lines: Vec::new(),
            synced: false,
            track_id: st.music.track_id(),
            note: "▶ now watching".into(),
        };
        st.queue = state::Queue { fresh: true, ..Default::default() };
        st.watch_info = state::WatchInfo {
            track_id: st.music.track_id(),
            synopsis: "A severed Lumon employee on the testing floor races to complete the \
                       mysterious Cold Harbor file as the boundary between his work and home \
                       selves collapses."
                .into(),
            director: "Ben Stiller".into(),
            cast: vec![
                "Adam Scott".into(),
                "Britt Lower".into(),
                "Tramell Tillman".into(),
                "Patricia Arquette".into(),
                "John Turturro".into(),
                "Christopher Walken".into(),
                "Zach Cherry".into(),
            ],
            producers: vec!["Ben Stiller".into(), "Dan Erickson".into(), "Mark Friedman".into()],
            note: String::new(),
        };
    }
}
