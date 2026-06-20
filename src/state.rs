//! Shared application state. Collector threads write; the render loop reads a snapshot.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Ring-buffer history for sparklines. Fixed capacity, cheap push.
#[derive(Clone)]
pub struct History {
    pub data: Vec<u64>,
    cap: usize,
}

impl History {
    pub fn new(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), cap }
    }
    pub fn push(&mut self, v: u64) {
        if self.data.len() == self.cap {
            self.data.remove(0);
        }
        self.data.push(v);
    }
    pub fn last(&self) -> u64 {
        self.data.last().copied().unwrap_or(0)
    }
}

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
    pub top_procs: Vec<(String, f32, u64, u64)>,
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
    pub fn track_id(&self) -> String {
        format!("{}|{}|{}", self.artist, self.track, self.album)
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
    pub ahead: i32,
    pub behind: i32,
    pub last_hash: String,
    pub last_msg: String,
    pub last_rel: String,
    pub commits_today: u32,
    /// Pull requests authored by the user on this repo (all states).
    pub pr_count: u32,
    /// Repo directory name (shown before the branch, e.g. "battlestation").
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
    pub handle: String,      // 1:1 reply target (phone/email); empty for group chats
    pub preview: String,     // latest message: real text, summarized, or truncated
    pub full_text: String,   // untruncated latest text (summarize input / send context)
    pub ts_unix: f64,        // unix seconds of the latest message
    pub rel: String,         // "2m" "1h" "yesterday"
    pub is_rich: bool,       // attributedBody-only → "[rich message]" marker
    pub unread: bool,        // conversation has >=1 unread inbound message
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
pub enum MsgPhase {
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

impl Default for MsgPhase {
    fn default() -> Self {
        MsgPhase::Idle
    }
}

/// iMessage interaction state. Lives on AppState so the pure render fn can read
/// it; mutated only by the main.rs event loop. Animations are interpolated each
/// frame off `anim_start` — never a discrete flip — so motion is buttery.
#[derive(Clone)]
pub struct MsgUi {
    pub active: bool,                 // is the iMessage card focused?
    pub queue_pos: usize,             // index into the unread queue (focused row)
    pub last_key_at: Option<Instant>, // double-press window for 'm'
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
            queue_pos: 0,
            last_key_at: None,
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
    pub month_cost: f64,
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

/// The whole world, behind one mutex. Cheap to clone for a render snapshot.
#[derive(Clone)]
pub struct AppState {
    pub system: SystemStats,
    pub silicon: SiliconStats,
    /// Timestamped per-core CPU samples for delayed, interpolated playback.
    pub cpu_samples: VecDeque<(Instant, Vec<f32>)>,
    /// Timestamped total network throughput (bytes/sec) for the net visualizer.
    pub net_samples: VecDeque<(Instant, f32)>,
    /// Timestamped Apple-Silicon metrics for smooth, delayed gauges. Indices:
    /// [gpu%, all_power, cpu_temp, gpu_temp, cpu_power, gpu_power, sys_power].
    pub silicon_samples: VecDeque<(Instant, Vec<f32>)>,
    /// Timestamped top-process snapshots for delay-interpolated playback: the
    /// proc_panel eases each process's cpu%/mem and slides rows toward their new
    /// rank between samples, keyed by name (same pattern as cpu_samples).
    pub proc_samples: VecDeque<(Instant, Vec<(String, f32, u64, u64)>)>,
    pub music: MusicStats,
    pub lyrics: Lyrics,
    /// How many distinct tracks are sitting in the lyrics miss log waiting to be
    /// reconciled — surfaced as the "N missing" badge on the LYRICS card.
    pub lyrics_misses: usize,
    pub album_art: AlbumArt,
    /// Album-art-derived accent palette, cross-faded per frame (issue #8).
    pub dynamic_theme: DynamicTheme,
    pub facts: MusicFacts,
    pub queue: Queue,
    pub git: GitStats,
    pub weather: Weather,
    pub usage: UsageStats,
    pub messages: Messages,
    pub msg_ui: MsgUi,
    pub signal: Messages, // Signal Desktop conversations (read-only; reuses Messages shape)
    pub discord: Discord, // Discord voice presence + recent text channels
    pub doctor: Doctor,   // mac-doctor / syswatch triage agent status
    pub keybinds: Keybinds, // Hammerspoon keybind cheat sheet
    pub cpu_hist: History,
    pub gpu_hist: History,
    pub power_hist: History,
    pub net_rx_hist: History,
    pub net_tx_hist: History,
    /// Memory-used percent (0..100) history — a band of the MEM·DISK·NET wave.
    pub mem_hist: History,
    /// Disk I/O rate (KB/s, read+written across all procs) — a wave band.
    pub disk_io_hist: History,
    /// Disk free space (percent of root, 0..100) — a wave band.
    pub disk_free_hist: History,
    pub started: Instant,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            system: SystemStats::default(),
            silicon: SiliconStats::default(),
            cpu_samples: VecDeque::new(),
            net_samples: VecDeque::new(),
            silicon_samples: VecDeque::new(),
            proc_samples: VecDeque::new(),
            music: MusicStats::default(),
            lyrics: Lyrics::default(),
            lyrics_misses: 0,
            album_art: AlbumArt::default(),
            dynamic_theme: DynamicTheme::default(),
            facts: MusicFacts::default(),
            queue: Queue::default(),
            git: GitStats::default(),
            weather: Weather::default(),
            usage: UsageStats::default(),
            messages: Messages::default(),
            msg_ui: MsgUi::default(),
            signal: Messages::default(),
            discord: Discord::default(),
            doctor: Doctor::default(),
            keybinds: Keybinds::default(),
            cpu_hist: History::new(120),
            gpu_hist: History::new(120),
            power_hist: History::new(120),
            net_rx_hist: History::new(120),
            net_tx_hist: History::new(120),
            mem_hist: History::new(120),
            disk_io_hist: History::new(120),
            disk_free_hist: History::new(120),
            started: Instant::now(),
        }
    }
}
