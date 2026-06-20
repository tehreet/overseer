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
    pub music: MusicStats,
    pub lyrics: Lyrics,
    pub album_art: AlbumArt,
    pub git: GitStats,
    pub weather: Weather,
    pub usage: UsageStats,
    pub messages: Messages,
    pub msg_ui: MsgUi,
    pub signal: Messages, // Signal Desktop conversations (read-only; reuses Messages shape)
    pub cpu_hist: History,
    pub gpu_hist: History,
    pub power_hist: History,
    pub net_rx_hist: History,
    pub net_tx_hist: History,
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
            music: MusicStats::default(),
            lyrics: Lyrics::default(),
            album_art: AlbumArt::default(),
            git: GitStats::default(),
            weather: Weather::default(),
            usage: UsageStats::default(),
            messages: Messages::default(),
            msg_ui: MsgUi::default(),
            signal: Messages::default(),
            cpu_hist: History::new(120),
            gpu_hist: History::new(120),
            power_hist: History::new(120),
            net_rx_hist: History::new(120),
            net_tx_hist: History::new(120),
            started: Instant::now(),
        }
    }
}
