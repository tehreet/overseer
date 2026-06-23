//! Shared application state. Collector threads write; the render loop reads a snapshot.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One top-process row: (name, cpu% of one core, memory bytes, uptime secs).
pub type ProcSample = (String, f32, u64, u64);

#[derive(Clone, Default)]
pub struct SystemStats {
    pub hostname: String,
    pub os: String,
    pub cpu_overall: f32, // 0..100
    pub per_core: Vec<f32>,
    pub load: (f64, f64, f64),
    pub mem_used: u64,
    pub mem_total: u64,
    pub swap_used: u64,
    pub swap_total: u64,
    pub disk_used: u64,
    pub disk_total: u64,
    pub net_rx_bps: f64,
    pub net_tx_bps: f64,
    pub uptime_secs: u64,
    pub proc_count: usize,
    /// Top processes by CPU: (name, cpu% of one core, memory bytes, uptime secs).
    pub top_procs: Vec<ProcSample>,
}

/// Apple Silicon metrics straight from `macmon pipe`.
#[derive(Clone, Default)]
pub struct SiliconStats {
    pub fresh: bool,
    pub cpu_pct: f32,   // 0..100
    pub gpu_pct: f32,   // 0..100
    pub gpu_freq_mhz: u32,
    pub all_power_w: f32,
    pub cpu_power_w: f32,
    pub gpu_power_w: f32,
    pub sys_power_w: f32,
    pub cpu_temp_c: f32,
    pub gpu_temp_c: f32,
    pub ecpu_pct: f32, // efficiency cluster
    pub ecpu_freq_mhz: u32,
    pub pcpu_pct: f32, // performance cluster
    pub pcpu_freq_mhz: u32,
    pub ane_power_w: f32, // neural engine
}

/// One timestamped lyric line.
#[derive(Clone)]
pub struct LyricLine {
    pub t: f64, // seconds
    pub text: String,
}

#[derive(Clone, Default)]
pub struct Lyrics {
    pub lines: Vec<LyricLine>,
    pub synced: bool,
    pub track_id: String, // which track these belong to
    pub note: String,     // e.g. "no synced lyrics" / "instrumental"
}

impl Lyrics {
    /// Index of the active line for a given playback position, plus the
    /// fraction (0..1) through that line toward the next one — drives the
    /// karaoke wipe.
    pub fn active(&self, pos: f64) -> Option<(usize, f64)> {
        if self.lines.is_empty() {
            return None;
        }
        // Find last line whose timestamp <= pos.
        let mut idx = None;
        for (i, l) in self.lines.iter().enumerate() {
            if l.t <= pos {
                idx = Some(i);
            } else {
                break;
            }
        }
        let i = idx?;
        let start = self.lines[i].t;
        let end = self
            .lines
            .get(i + 1)
            .map(|l| l.t)
            .unwrap_or(start + 4.0);
        let frac = if end > start {
            ((pos - start) / (end - start)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some((i, frac))
    }
}

/// Which app is feeding the NOW PLAYING band. The same card/art/facts pipeline
/// serves both; only the labels, art source, and facts prompt differ.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum MediaSource {
    #[default]
    Music,
    Tv,
}

/// Extra metadata that only applies when watching (source == Tv): enough to label
/// the card nicely and to ground the trivia prompt in the right film/series.
#[derive(Clone, Default)]
pub struct WatchMeta {
    /// Series name for an episode; empty for a movie.
    pub show: String,
    pub season: u32,
    pub episode: u32,
    pub year: String,
    pub genre: String,
    pub director: String,
    /// Raw TV media kind ("movie", "TV show", "home video", …) for tuning labels.
    pub kind: String,
    /// Poster image URL (from iTunes) for streamed films the TV app won't supply
    /// artwork for; the artwork collector downloads + decodes it like album art.
    pub poster_url: String,
}

#[derive(Clone)]
pub struct MusicStats {
    pub running: bool,
    pub playing: bool,
    pub track: String,
    pub artist: String,
    pub album: String,
    pub duration: f64,
    /// Player position sampled at `sampled_at`. Interpolated in `position()`.
    pub base_pos: f64,
    pub sampled_at: Instant,
    /// Set true after the first AppleScript poll completes, so the UI doesn't
    /// flash the pulse before the real music state is known.
    pub polled: bool,
    /// Music vs. TV — selects the card's wording, art app, and facts prompt.
    pub source: MediaSource,
    /// Populated only when `source == Tv`.
    pub watch: WatchMeta,
}

impl Default for MusicStats {
    fn default() -> Self {
        Self {
            running: false,
            playing: false,
            track: String::new(),
            artist: String::new(),
            album: String::new(),
            duration: 0.0,
            base_pos: 0.0,
            sampled_at: Instant::now(),
            polled: false,
            source: MediaSource::Music,
            watch: WatchMeta::default(),
        }
    }
}

impl MusicStats {
    /// Smoothly-interpolated playback position for *this* render frame.
    pub fn position(&self) -> f64 {
        if self.playing {
            (self.base_pos + self.sampled_at.elapsed().as_secs_f64()).min(self.duration.max(0.0))
        } else {
            self.base_pos
        }
    }
    /// Stable identity for art/facts caching. Includes the source + series/episode
    /// so a movie can't collide with a same-named song, and each episode is its own
    /// cache entry.
    pub fn track_id(&self) -> String {
        match self.source {
            MediaSource::Music => format!("{}|{}|{}", self.artist, self.track, self.album),
            MediaSource::Tv => format!(
                "tv|{}|{}|S{}E{}",
                self.watch.show, self.track, self.watch.season, self.watch.episode
            ),
        }
    }
    pub fn is_tv(&self) -> bool {
        matches!(self.source, MediaSource::Tv)
    }
}

/// Downscaled album-art thumbnail (RGB, row-major). Rendered as half-blocks.
#[derive(Clone, Default)]
pub struct AlbumArt {
    pub track_id: String,
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 3]>,
}

impl AlbumArt {
    /// Nearest-neighbour sample at normalized (u, v) in 0..1.
    pub fn sample(&self, u: f32, v: f32) -> Option<[u8; 3]> {
        if self.px.is_empty() || self.w == 0 || self.h == 0 {
            return None;
        }
        let x = ((u.clamp(0.0, 0.999) * self.w as f32) as usize).min(self.w - 1);
        let y = ((v.clamp(0.0, 0.999) * self.h as f32) as usize).min(self.h - 1);
        self.px.get(y * self.w + x).copied()
    }

    /// Box-average the source pixels covering the normalized rect [u0,u1)×[v0,v1).
    /// This is the downscale that keeps a grainy photo legible: every output cell
    /// integrates its whole footprint instead of picking one aliased source pixel.
    pub fn sample_area(&self, u0: f32, v0: f32, u1: f32, v1: f32) -> Option<[u8; 3]> {
        if self.px.is_empty() || self.w == 0 || self.h == 0 {
            return None;
        }
        let w = self.w as f32;
        let h = self.h as f32;
        // Pixel-space footprint, clamped to the image, at least one pixel wide/tall.
        let x0 = (u0.clamp(0.0, 1.0) * w).floor() as usize;
        let y0 = (v0.clamp(0.0, 1.0) * h).floor() as usize;
        let x1 = ((u1.clamp(0.0, 1.0) * w).ceil() as usize).max(x0 + 1).min(self.w);
        let y1 = ((v1.clamp(0.0, 1.0) * h).ceil() as usize).max(y0 + 1).min(self.h);
        let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
        for y in y0..y1 {
            let row = y * self.w;
            for x in x0..x1 {
                if let Some(p) = self.px.get(row + x) {
                    r += p[0] as u32;
                    g += p[1] as u32;
                    b += p[2] as u32;
                    n += 1;
                }
            }
        }
        if n == 0 {
            return self.sample((u0 + u1) * 0.5, (v0 + v1) * 0.5);
        }
        Some([(r / n) as u8, (g / n) as u8, (b / n) as u8])
    }
}

/// Cross-fade duration for the album-art palette swap on a track change. Long
/// enough to read as a glide, short enough to feel responsive.
const THEME_FADE: Duration = Duration::from_millis(450);

/// The dynamic, album-art-derived accent palette. Holds where the accents are
/// fading *from* (`current`) and *to* (`target`), plus when the fade began, so
/// the render loop can serve a per-frame RGB lerp off `eased()` — never a hard
/// cut. The artwork collector retargets it (capturing the live eased value as
/// the new `current`) whenever a new cover yields different dominant colors.
#[derive(Clone)]
pub struct DynamicTheme {
    pub source_track_id: String,        // cover this target was derived from
    pub current: [(u8, u8, u8); 3],     // accent, cyan, pink we're fading FROM
    pub target: [(u8, u8, u8); 3],      // accent, cyan, pink we're fading TO
    pub blend_start: Instant,           // when the active cross-fade began
}

impl Default for DynamicTheme {
    fn default() -> Self {
        // Seed both ends to the house synthwave accents so the first frames are
        // on-brand and a no-art session never tints.
        let base = [
            crate::theme::ACCENT_BASE,
            crate::theme::CYAN_BASE,
            crate::theme::PINK_BASE,
        ];
        Self {
            source_track_id: String::new(),
            current: base,
            target: base,
            blend_start: Instant::now(),
        }
    }
}

impl DynamicTheme {
    /// Begin a cross-fade toward `target`, anchoring `current` at wherever the
    /// fade currently sits so retargeting mid-fade never pops.
    pub fn retarget(&mut self, source_track_id: String, target: [(u8, u8, u8); 3]) {
        self.current = self.eased();
        self.target = target;
        self.source_track_id = source_track_id;
        self.blend_start = Instant::now();
    }

    /// The cross-faded accent triple for *this* frame (ease-out cubic). Once the
    /// fade completes it pins to `target`, so steady-state is allocation- and
    /// drift-free.
    pub fn eased(&self) -> [(u8, u8, u8); 3] {
        let raw = (self.blend_start.elapsed().as_secs_f32() / THEME_FADE.as_secs_f32()).clamp(0.0, 1.0);
        // Ease-out cubic: fast then settle — reads as a smooth glide, not a ramp.
        let t = 1.0 - (1.0 - raw).powi(3);
        let lerp = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
            (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
        };
        [
            lerp(self.current[0], self.target[0]),
            lerp(self.current[1], self.target[1]),
            lerp(self.current[2], self.target[2]),
        ]
    }
}

/// One upcoming track in Apple Music's queue (next-up from the current playlist).
#[derive(Clone, Default)]
pub struct QueueTrack {
    pub track: String,
    pub artist: String,
    pub duration: f64,
}

/// The next few tracks Apple Music will play (best-effort, by current-playlist
/// order — AppleScript can't read the true dynamic "Up Next" list). Recomputed
/// when the current track changes.
#[derive(Clone, Default)]
pub struct Queue {
    pub fresh: bool,
    pub source_track_id: String, // current track this queue was derived from
    pub items: Vec<QueueTrack>,  // up to 3 upcoming tracks
}

/// Curated "interesting facts" about the current track/album/artist for the
/// LINER NOTES card. Generated off-thread (Claude when a key is present, else a
/// Wikipedia extract) and cached per track so it never regenerates on seek/pause.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MusicFacts {
    pub track_id: String,   // which track these belong to
    pub lines: Vec<String>, // one fact per entry
    pub note: String,       // status when empty: "gathering…" / "no notes"
    pub source: String,     // "claude" | "wikipedia" (shown faintly in the title)
}

#[derive(Clone, Default)]
pub struct GitStats {
    pub fresh: bool,
    pub ok: bool,
    pub branch: String,
    pub dirty: u32,
    pub untracked: u32,
    pub staged: u32,
    #[allow(dead_code)] // collected git state, kept for the card's display options
    pub ahead: i32,
    #[allow(dead_code)] // collected git state, kept for the card's display options
    pub behind: i32,
    pub last_hash: String,
    pub last_msg: String,
    pub last_rel: String,
    pub commits_today: u32,
    /// Pull requests authored by the user on this repo (all states).
    pub pr_count: u32,
    /// Repo directory name (shown before the branch, e.g. "overseer").
    pub repo: String,
    /// Branch activity vs origin/main: lines added/removed across the branch's
    /// commits, commit count, and merge-commit count ("merges to main").
    pub loc_added: u32,
    pub loc_removed: u32,
    pub branch_commits: u32,
    pub merges_main: u32,
}

#[derive(Clone, Default)]
pub struct Weather {
    pub fresh: bool,
    pub location: String,
    pub temp_f: i32,
    pub feels_f: i32,
    pub desc: String,
    pub icon: String,
    pub hi_f: i32,
    pub lo_f: i32,
    pub humidity: i32,
    pub wind_mph: i32,
    pub wind_dir: String,
    pub precip_chance: i32,
    pub uv: i32,
    pub pressure_mb: i32,
    pub sunrise: String,
    pub sunset: String,
    /// Next ~12 hourly forecast temps (°F) for the tiny jazz temp strip.
    pub temp_strip: Vec<u64>,
}

/// One conversation (chat), newest-active first in `Messages.items`. Grouped like
/// the iPhone Messages list: one row per chat, previewing its most recent message.
#[derive(Clone, Default)]
pub struct MessageItem {
    pub chat_id: i64,        // conversation id (chat.ROWID) — mark-read / reply target
    pub rowid: i64,          // latest message's ROWID in this conversation
    pub sender: String,      // contact name · group-chat name · pretty handle
    #[allow(dead_code)] // raw 1:1 address; superseded by `guid` as the reply target, kept for display/debug
    pub handle: String,      // 1:1 sender address (phone/email); empty for group chats
    pub guid: String,        // chat.guid (e.g. "SMS;-;+1…" / "iMessage;-;…" / group) — reply target
    pub preview: String,     // latest message: real text, summarized, or truncated
    pub full_text: String,   // untruncated latest text (summarize input / send context)
    pub ts_unix: f64,        // unix seconds of the latest message
    pub rel: String,         // "2m" "1h" "yesterday"
    pub is_rich: bool,       // attributedBody-only → "[rich message]" marker
    pub unread: bool,        // conversation has >=1 unread inbound message
    #[allow(dead_code)] // outbound flag, populated for preview prefixing decisions
    pub from_me: bool,       // latest message is outbound → preview prefixed "You: "
    pub is_shortcode: bool,  // 5-6 digit shortcode (e.g. 32665) — excluded from badge
}

/// One Discord voice channel that currently has at least one person in it.
#[derive(Clone, Default)]
pub struct VoiceChannel {
    pub name: String,          // channel name (no leading glyph)
    pub members: Vec<String>,  // display names currently connected
}

/// One Discord text channel row (newest-active first), iMessage/Signal-style.
#[derive(Clone, Default)]
pub struct TextChannel {
    pub name: String,     // channel name (rendered with a leading '#')
    pub author: String,   // last message author's display name
    pub preview: String,  // last message text, flattened
    pub rel: String,      // "2m" "1h" "yesterday"
    pub unread: bool,     // channel has unread messages for the bot
}

/// Discord card data (written by the spawn_discord collector). Voice presence
/// arrives over the gateway; text channels' last messages are polled over REST.
#[derive(Clone, Default)]
pub struct Discord {
    pub fresh: bool,                 // first poll/connect completed
    pub available: bool,             // bot token present + gateway/REST reachable
    pub voice: Vec<VoiceChannel>,    // only channels with someone in them (else empty)
    pub text: Vec<TextChannel>,      // recent text channels w/ last message
    pub voice_join_at: Option<Instant>, // when someone last JOINED voice → 20s border shimmer
    pub voice_speaking: bool,           // gateway SSRC speaking events → bright border shimmer
    pub voice_speaking_tap: bool,       // local Core Audio tap of Discord.app (post-E2EE) → speaking
    pub voice_e2ee_blocked: bool,       // Discord rejected the voice listener with DAVE/E2EE (4017)
}

/// mac-doctor / syswatch triage agent status (written by spawn_doctor). The
/// watchdog samples cheap metrics 24/7 and, on a threshold breach, fires a
/// diagnosis run (local Ollama triage → optional Claude escalation) that writes
/// an incident row. This mirrors that state onto the dashboard.
#[derive(Clone, Default)]
pub struct Doctor {
    pub available: bool,        // syswatch.db found / readable
    pub running: bool,          // diagnose.lock present → a run is in flight right now
    pub step: String,           // live step from the log tail while running (e.g. "local triage…")
    pub trigger: String,        // what breached (flattened trigger_reasons of the active/last run)
    pub last_title: String,     // latest incident's one-line verdict
    pub last_outcome: String,   // resolved | mitigated | no-action-needed | needs-user | unresolved
    pub last_severity: String,  // info | warn | critical
    pub last_model: String,     // qwen2.5:14b | sonnet | …
    pub last_actions: Vec<String>, // commands the agent ran on the last incident (may be empty)
    pub last_rel: String,       // "12m" since the last run completed
    pub today_cost: f64,        // sum of Claude escalation cost today (USD)
    pub incidents_total: u64,   // lifetime incident count
}

/// One group of Hammerspoon keybindings (e.g. "Apps · Hyper+key") with its rows.
#[derive(Clone, Default)]
pub struct KeyGroup {
    pub name: String,
    pub binds: Vec<(String, String)>, // (keys, description), cheat-sheet order
}

/// Hammerspoon keybind cheat sheet, mirrored from the live config. Hammerspoon
/// exports its self-documenting `doc` registry to JSON on every reload; the
/// spawn_keybinds collector reads it so this card always matches the real binds.
#[derive(Clone, Default)]
pub struct Keybinds {
    pub available: bool,        // the exported JSON was found + parsed
    pub hyper: String,          // how "Hyper" is produced (e.g. "hold Caps Lock")
    pub groups: Vec<KeyGroup>,  // in cheat-sheet display order
}

/// iMessage card data (written by the spawn_messages collector).
#[derive(Clone, Default)]
pub struct Messages {
    pub fresh: bool,        // first poll completed
    pub available: bool,    // chat.db readable (Full Disk Access granted)?
    pub unread_count: u32,  // conversations w/ recent unread inbound (excl. shortcodes)
    pub items: Vec<MessageItem>, // recent conversations, newest-active first
}

/// Phase of the iMessage card's interaction animation.
#[derive(Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum MsgPhase {
    #[default]
    Idle,
    /// Marking the focused message read + sliding the marker to the next unread.
    Advancing,
    /// The inline reply input is wiping open.
    Opening,
    /// The reply is shimmering away + the input collapsing.
    Sending,
    /// Esc cancel: the input wipes closed.
    Closing,
}


/// iMessage interaction state. Lives on AppState so the pure render fn can read
/// it; mutated only by the main.rs event loop. Animations are interpolated each
/// frame off `anim_start` — never a discrete flip — so motion is buttery.
#[derive(Clone)]
pub struct MsgUi {
    pub active: bool,                 // is the iMessage card focused?
    pub focus_chat_id: Option<i64>,   // focused conversation (click/'m'); identity, not index
    pub last_key_at: Option<Instant>, // double-press window for 'm'
    pub last_click_at: Option<Instant>, // double-click window for the mouse
    pub last_click_chat: Option<i64>, // chat the last click landed on (double-click must match)
    pub composing: bool,              // inline reply input open?
    pub draft: String,                // reply being typed
    pub phase: MsgPhase,              // current transition
    pub anim_start: Option<Instant>,  // when the current transition began
    pub send_failed_at: Option<Instant>, // osascript nonzero → border flash
}

impl Default for MsgUi {
    fn default() -> Self {
        Self {
            active: false,
            focus_chat_id: None,
            last_key_at: None,
            last_click_at: None,
            last_click_chat: None,
            composing: false,
            draft: String::new(),
            phase: MsgPhase::Idle,
            anim_start: None,
            send_failed_at: None,
        }
    }
}

impl MsgUi {
    /// Animation progress 0..1 over `dur`; None when no transition is running.
    pub fn progress(&self, dur: Duration) -> Option<f32> {
        let start = self.anim_start?;
        let p = (start.elapsed().as_secs_f32() / dur.as_secs_f32()).clamp(0.0, 1.0);
        Some(p)
    }
    /// Is any motion (transition or live caret blink) in flight?
    pub fn animating(&self) -> bool {
        self.composing
            || self.phase != MsgPhase::Idle
            || self.send_failed_at.is_some()
    }
}

#[derive(Clone, Default)]
pub struct UsageStats {
    pub fresh: bool,
    pub today_input: u64,
    pub today_output: u64,
    pub today_cache_read: u64,
    pub today_cache_write: u64,
    pub today_cost: f64,
    pub today_messages: u64,
    pub top_model: String,
    pub sessions_today: u64,
    /// Rolling-window token totals (input+output+cache) for the de-cluttered card.
    pub tokens_7d: u64,
    pub tokens_30d: u64,
    /// Distinct Claude Code sessions in the last 30 days.
    pub sessions_30d: u64,
    /// Per-hour token activity for the last 24h, for the burn graph.
    pub hourly: Vec<u64>,
}

/// What a live Claude Code session is doing this very moment — derived from the
/// tail of its JSONL transcript. Drives the icon + color in the ROBOTS feed.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ActionKind {
    /// Turn finished (`stop_reason: end_turn`) — the robot is waiting on a human.
    #[default]
    Idle,
    /// Extended-thinking block is the latest thing on the wire.
    Think,
    /// A shell command (`Bash`) is running.
    Run,
    /// Editing/writing a file (`Edit`/`Write`/`MultiEdit`/`NotebookEdit`).
    Edit,
    /// Reading or searching the codebase (`Read`/`Grep`/`Glob`).
    Read,
    /// Reaching out to the web (`WebFetch`/`WebSearch`).
    Web,
    /// Spawning a sub-agent (`Task`/`Agent`).
    Agent,
    /// Any other tool call (incl. MCP) in flight.
    Tool,
    /// Streaming a text answer right now (no `end_turn` yet).
    Respond,
}

/// One Claude Code session that has touched its transcript within the live
/// window. Read cheaply from the file's tail by the fast `spawn_live_sessions`
/// thread so the ROBOTS card can show a realtime "who's working" feed.
#[derive(Clone, Default)]
pub struct LiveSession {
    #[allow(dead_code)] // session identity, populated by the collector for future use
    pub session_id: String,
    /// Basename of the session's cwd, e.g. `battlestation`.
    pub project: String,
    /// Git branch the session is on, if the transcript recorded one.
    pub branch: String,
    /// Short model name in use (`Opus`/`Sonnet`/`Haiku`).
    pub model: String,
    /// Human-readable label of what it's doing right now.
    pub action: String,
    pub kind: ActionKind,
    /// Seconds since the session's last transcript event (mtime / last ts).
    pub age_secs: f64,
}

/// The realtime roster of working Claude Code sessions, newest-active first.
#[derive(Clone, Default)]
pub struct LiveSessions {
    pub fresh: bool,
    pub sessions: Vec<LiveSession>,
}

/// Synopsis + key credits for what's being watched — fills the (repurposed)
/// LYRICS card while a movie/show plays. Synopsis is Wikipedia's lead; credits
/// come from Claude (Wikipedia-parse fallback).
#[derive(Clone, Default)]
pub struct WatchInfo {
    /// Identity of the title this describes, so the UI only shows a matching one.
    pub track_id: String,
    pub synopsis: String,
    pub director: String,
    pub cast: Vec<String>,
    /// Producers (and exec producers) for the credits page of the watch card.
    pub producers: Vec<String>,
    /// Status line while gathering (e.g. "gathering synopsis…").
    pub note: String,
}

/// The whole world, behind one mutex. Cheap to clone for a render snapshot.
#[derive(Clone)]
pub struct AppState {
    pub system: SystemStats,
    pub silicon: SiliconStats,
    /// Timestamped Apple-Silicon metrics for smooth, delayed gauges. Indices:
    /// [gpu%, all_power, cpu_temp, gpu_temp, cpu_power, gpu_power, sys_power].
    pub silicon_samples: VecDeque<(Instant, Vec<f32>)>,
    /// Timestamped top-process snapshots for delay-interpolated playback: the
    /// proc_panel eases each process's cpu%/mem and slides rows toward their new
    /// rank between samples, keyed by name (same delay-interpolation pattern).
    pub proc_samples: VecDeque<(Instant, Vec<ProcSample>)>,
    /// Timestamped log-spaced audio spectrum bands (each 0..1) captured from a
    /// real Core Audio process tap of the system output. Fed to the NOW PLAYING
    /// / LYRICS spectrum the same delay-interpolated way the EQ plays back, so
    /// the bars glide between FFT frames instead of stepping. Empty until the
    /// audio thread produces a frame; `audio_live` flips true once it does.
    pub audio_samples: VecDeque<(Instant, Vec<f32>)>,
    /// Timestamped resource metrics for the RESOURCES wave — [mem %, net down
    /// KB/s, net up KB/s, disk I/O KB/s] — played back delay-interpolated so the
    /// wave glides per-frame instead of stepping at 1 Hz.
    pub res_samples: VecDeque<(Instant, Vec<f32>)>,
    /// True once the audio tap is capturing (the visualizer then reflects real
    /// sound); false keeps the honest synthetic resting flourish.
    pub audio_live: bool,
    pub music: MusicStats,
    pub lyrics: Lyrics,
    /// How many distinct tracks are sitting in the lyrics miss log waiting to be
    /// reconciled — surfaced as the "N missing" badge on the LYRICS card.
    pub lyrics_misses: usize,
    /// Decoded art is up to ART_THUMB² (~768 KB). Held behind an `Arc` so the
    /// per-frame AppState snapshot clone in the event loop (issue #16) is a
    /// refcount bump, not a 768 KB copy on the render hot path.
    pub album_art: Arc<AlbumArt>,
    /// Album-art-derived accent palette, cross-faded per frame (issue #8).
    pub dynamic_theme: DynamicTheme,
    pub facts: MusicFacts,
    pub queue: Queue,
    pub git: GitStats,
    pub weather: Weather,
    pub usage: UsageStats,
    /// Realtime feed of currently-working Claude Code sessions (ROBOTS card).
    pub live: LiveSessions,
    /// Synopsis + credits for the current movie/show (repurposed LYRICS card).
    pub watch_info: WatchInfo,
    pub messages: Messages,
    pub msg_ui: MsgUi,
    pub signal: Messages, // Signal Desktop conversations (read-only; reuses Messages shape)
    pub discord: Discord, // Discord voice presence + recent text channels
    pub doctor: Doctor,   // mac-doctor / syswatch triage agent status
    pub keybinds: Keybinds, // Hammerspoon keybind cheat sheet
    pub started: Instant,
    /// Whether the QUEUE card is currently "open" (has tracks to show). When it
    /// goes empty the card smoothly collapses and the LYRICS card expands to fill
    /// the width; `queue_toggle_at` stamps the last open↔closed flip so the render
    /// can ease the split. Driven by `settle_queue_anim` each frame.
    pub queue_open: bool,
    pub queue_toggle_at: Instant,
    /// Whether overseer's KEYBINDS card is shown. Toggled by Hyper+H via a
    /// flag file Hammerspoon writes; `keybinds_toggle_at` stamps the flip so the
    /// card eases open/closed (same mechanism as the QUEUE collapse).
    pub keybinds_visible: bool,
    pub keybinds_toggle_at: Instant,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            system: SystemStats::default(),
            silicon: SiliconStats::default(),
            res_samples: VecDeque::new(),
            silicon_samples: VecDeque::new(),
            proc_samples: VecDeque::new(),
            audio_samples: VecDeque::new(),
            audio_live: false,
            music: MusicStats::default(),
            lyrics: Lyrics::default(),
            lyrics_misses: 0,
            album_art: Arc::new(AlbumArt::default()),
            dynamic_theme: DynamicTheme::default(),
            facts: MusicFacts::default(),
            queue: Queue::default(),
            git: GitStats::default(),
            weather: Weather::default(),
            usage: UsageStats::default(),
            live: LiveSessions::default(),
            watch_info: WatchInfo::default(),
            messages: Messages::default(),
            msg_ui: MsgUi::default(),
            signal: Messages::default(),
            discord: Discord::default(),
            doctor: Doctor::default(),
            keybinds: Keybinds::default(),
            started: Instant::now(),
            queue_open: false,
            // Stamped in the past so a cold start reads as a settled, collapsed
            // queue (no flash of an empty QUEUE card before the first poll).
            queue_toggle_at: Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now),
            keybinds_visible: true,
            keybinds_toggle_at: Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now),
        }
    }
}
