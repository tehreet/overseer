//! Background data collectors. Each runs on its own thread and writes into the
//! shared `AppState` behind a mutex. None of them block the render loop.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Datelike, Local, Timelike};

use crate::lyrics;
use crate::state::AppState;

type Shared = Arc<Mutex<AppState>>;

// ---------------------------------------------------------------------------
// System metrics (sysinfo): CPU, memory, disk, network, load, uptime.
// ---------------------------------------------------------------------------
pub fn spawn_system(shared: Shared) {
    thread::spawn(move || {
        use sysinfo::{Disks, Networks, ProcessesToUpdate, System};
        let mut sys = System::new();
        let mut networks = Networks::new_with_refreshed_list();
        let hostname = System::host_name().unwrap_or_else(|| "mac-studio".into());
        let os = format!(
            "{} {}",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default()
        );
        let mut last_net = Instant::now();

        loop {
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            let nproc = sys.refresh_processes(ProcessesToUpdate::All, true);

            networks.refresh(true);
            let dt = last_net.elapsed().as_secs_f64().max(0.001);
            last_net = Instant::now();
            let (mut rx, mut tx) = (0u64, 0u64);
            for (_n, data) in networks.iter() {
                rx += data.received();
                tx += data.transmitted();
            }
            let rx_bps = rx as f64 / dt;
            let tx_bps = tx as f64 / dt;

            let disks = Disks::new_with_refreshed_list();
            let (mut d_total, mut d_avail) = (0u64, 0u64);
            for d in disks.iter() {
                if d.mount_point().to_string_lossy() == "/" {
                    d_total = d.total_space();
                    d_avail = d.available_space();
                }
            }

            let per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
            let overall = sys.global_cpu_usage();
            let load = System::load_average();

            let mut procs: Vec<(String, f32, u64, u64)> = sys
                .processes()
                .values()
                .map(|p| {
                    (
                        p.name().to_string_lossy().to_string(),
                        p.cpu_usage(),
                        p.memory(),
                        p.run_time(),
                    )
                })
                .collect();
            procs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            procs.truncate(6);

            {
                let mut s = shared.lock().unwrap();
                s.system.hostname = hostname.clone();
                s.system.os = os.clone();
                s.system.cpu_overall = overall;
                s.system.per_core = per_core;
                s.system.load = (load.one, load.five, load.fifteen);
                s.system.mem_used = sys.used_memory();
                s.system.mem_total = sys.total_memory();
                s.system.swap_used = sys.used_swap();
                s.system.swap_total = sys.total_swap();
                s.system.disk_used = d_total.saturating_sub(d_avail);
                s.system.disk_total = d_total;
                s.system.net_rx_bps = rx_bps;
                s.system.net_tx_bps = tx_bps;
                s.system.uptime_secs = System::uptime();
                s.system.proc_count = nproc;
                s.system.top_procs = procs;
                // Timestamped sample for the EQ's delayed, interpolated playback.
                let sample = s.system.per_core.clone();
                let now = Instant::now();
                s.cpu_samples.push_back((now, sample));
                while s.cpu_samples.len() > 16 {
                    s.cpu_samples.pop_front();
                }
                s.net_samples.push_back((now, (rx_bps + tx_bps) as f32));
                while s.net_samples.len() > 16 {
                    s.net_samples.pop_front();
                }
                s.net_rx_hist.push((rx_bps / 1024.0) as u64);
                s.net_tx_hist.push((tx_bps / 1024.0) as u64);
                // If macmon isn't feeding silicon CPU%, mirror sysinfo's.
                if !s.silicon.fresh {
                    s.cpu_hist.push(overall as u64);
                }
            }

            thread::sleep(Duration::from_millis(1000));
        }
    });
}

// ---------------------------------------------------------------------------
// Apple Silicon metrics: stream `macmon pipe` as newline-delimited JSON.
// ---------------------------------------------------------------------------
pub fn spawn_macmon(shared: Shared) {
    thread::spawn(move || loop {
        let child = Command::new("macmon")
            .args(["pipe", "-i", "1000"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(_) => {
                // macmon missing — leave silicon stale and retry later.
                thread::sleep(Duration::from_secs(10));
                continue;
            }
        };
        let Some(stdout) = child.stdout.take() else {
            thread::sleep(Duration::from_secs(5));
            continue;
        };
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let getf = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            let pair1 = |k: &str| {
                v.get(k)
                    .and_then(|a| a.get(1))
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as f32
            };
            let pair0 = |k: &str| {
                v.get(k)
                    .and_then(|a| a.get(0))
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as u32
            };
            let mut s = shared.lock().unwrap();
            s.silicon.fresh = true;
            s.silicon.cpu_pct = getf("cpu_usage_pct") * 100.0;
            s.silicon.gpu_pct = pair1("gpu_usage") * 100.0;
            s.silicon.gpu_freq_mhz = pair0("gpu_usage");
            s.silicon.all_power_w = getf("all_power");
            s.silicon.cpu_power_w = getf("cpu_power");
            s.silicon.gpu_power_w = getf("gpu_power");
            s.silicon.sys_power_w = getf("sys_power");
            s.silicon.ane_power_w = getf("ane_power");
            s.silicon.ecpu_pct = pair1("ecpu_usage") * 100.0;
            s.silicon.ecpu_freq_mhz = pair0("ecpu_usage");
            s.silicon.pcpu_pct = pair1("pcpu_usage") * 100.0;
            s.silicon.pcpu_freq_mhz = pair0("pcpu_usage");
            if let Some(t) = v.get("temp") {
                s.silicon.cpu_temp_c =
                    t.get("cpu_temp_avg").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                s.silicon.gpu_temp_c =
                    t.get("gpu_temp_avg").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            }
            let (cpu_v, gpu_v, pwr_v) = (
                s.silicon.cpu_pct as u64,
                s.silicon.gpu_pct as u64,
                s.silicon.all_power_w as u64,
            );
            s.cpu_hist.push(cpu_v);
            s.gpu_hist.push(gpu_v);
            s.power_hist.push(pwr_v);
            // Timestamped sample for smooth, delayed gauges.
            let si = &s.silicon;
            let sample = vec![
                si.gpu_pct,
                si.all_power_w,
                si.cpu_temp_c,
                si.gpu_temp_c,
                si.cpu_power_w,
                si.gpu_power_w,
                si.sys_power_w,
                si.ecpu_pct,
                si.pcpu_pct,
                si.ane_power_w,
            ];
            s.silicon_samples.push_back((Instant::now(), sample));
            while s.silicon_samples.len() > 16 {
                s.silicon_samples.pop_front();
            }
        }
        // Pipe closed (macmon exited) — restart.
        let _ = child.wait();
        thread::sleep(Duration::from_secs(2));
    });
}

// ---------------------------------------------------------------------------
// Apple Music: poll now-playing + position via AppleScript. Guarded so it
// never launches Music if it isn't already running.
// ---------------------------------------------------------------------------
const MUSIC_SCRIPT: &str = r#"
if application "Music" is running then
  tell application "Music"
    try
      set s to player state as string
      set t to current track
      return s & "\t" & (name of t) & "\t" & (artist of t) & "\t" & (album of t) & "\t" & (duration of t) & "\t" & (player position)
    on error
      return "STOPPED"
    end try
  end tell
else
  return "NOTRUNNING"
end if
"#;

/// One-shot music probe for diagnostics. Returns (running, playing, track,
/// artist, album, duration, position).
pub fn probe_music() -> (bool, bool, String, String, String, f64, f64) {
    let out = Command::new("osascript").arg("-e").arg(MUSIC_SCRIPT).output();
    if let Ok(o) = out {
        let raw = String::from_utf8_lossy(&o.stdout);
        let line = raw.trim();
        if line == "NOTRUNNING" {
            return (false, false, String::new(), String::new(), String::new(), 0.0, 0.0);
        }
        if line == "STOPPED" || line.is_empty() {
            return (true, false, String::new(), String::new(), String::new(), 0.0, 0.0);
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() >= 6 {
            return (
                true,
                f[0].eq_ignore_ascii_case("playing"),
                f[1].to_string(),
                f[2].to_string(),
                f[3].to_string(),
                f[4].parse().unwrap_or(0.0),
                f[5].parse().unwrap_or(0.0),
            );
        }
    }
    (false, false, String::new(), String::new(), String::new(), 0.0, 0.0)
}

pub fn spawn_music(shared: Shared) {
    thread::spawn(move || loop {
        let out = Command::new("osascript").arg("-e").arg(MUSIC_SCRIPT).output();
        if let Ok(o) = out {
            let raw = String::from_utf8_lossy(&o.stdout);
            let line = raw.trim();
            let mut s = shared.lock().unwrap();
            if line == "NOTRUNNING" {
                s.music.running = false;
                s.music.playing = false;
            } else if line == "STOPPED" || line.is_empty() {
                s.music.running = true;
                s.music.playing = false;
            } else {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() >= 6 {
                    let new_playing = f[0].eq_ignore_ascii_case("playing");
                    let track = f[1].to_string();
                    let artist = f[2].to_string();
                    let album = f[3].to_string();
                    let duration: f64 = f[4].parse().unwrap_or(0.0);
                    let polled: f64 = f[5].parse().unwrap_or(0.0);

                    // Where our interpolated clock *thinks* we are right now.
                    let same_track =
                        s.music.track == track && s.music.artist == artist && s.music.album == album;
                    let was_playing = s.music.playing;
                    let predicted = if was_playing {
                        s.music.base_pos + s.music.sampled_at.elapsed().as_secs_f64()
                    } else {
                        s.music.base_pos
                    };
                    // Snap on track change / play-state change / seek; otherwise
                    // slew gently toward the truth so the wipe never hitches.
                    let snap = !same_track
                        || !was_playing
                        || !new_playing
                        || (polled - predicted).abs() > 1.5;
                    let new_base = if snap {
                        polled
                    } else {
                        predicted + (polled - predicted) * 0.25
                    };

                    s.music.running = true;
                    s.music.playing = new_playing;
                    s.music.track = track;
                    s.music.artist = artist;
                    s.music.album = album;
                    s.music.duration = duration;
                    s.music.base_pos = new_base;
                    s.music.sampled_at = Instant::now();
                }
            }
            s.music.polled = true; // real music state is now known
        }
        thread::sleep(Duration::from_millis(500));
    });
}

/// Watches for track changes and fetches lyrics off the network thread so the
/// HTTP call never stalls anything else.
pub fn spawn_lyrics(shared: Shared) {
    thread::spawn(move || {
        // Pooled, keep-alive agent → no fresh TLS handshake per fetch.
        // LRCLIB can be slow under load; let slow-but-valid responses land so
        // the disk cache captures them (UI shows "loading…" meanwhile).
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .timeout_read(Duration::from_secs(13))
            .build();
        let mut cache: std::collections::HashMap<String, crate::state::Lyrics> = Default::default();
        let mut current = String::new();
        loop {
            let info = {
                let s = shared.lock().unwrap();
                if s.music.running && !s.music.track.is_empty() {
                    Some((
                        s.music.track_id(),
                        s.music.artist.clone(),
                        s.music.track.clone(),
                        s.music.album.clone(),
                        s.music.duration,
                    ))
                } else {
                    None
                }
            };
            if let Some((id, artist, track, album, dur)) = info {
                if id != current {
                    current = id.clone();
                    // Warm the in-memory cache from disk on first sight.
                    if !cache.contains_key(&id) {
                        if let Some(disk) = lyrics::cache_load(&id) {
                            cache.insert(id.clone(), disk);
                        }
                    }
                    if let Some(cached) = cache.get(&id) {
                        // Instant: in-memory or disk cache hit.
                        let mut s = shared.lock().unwrap();
                        if s.music.track_id() == id {
                            s.lyrics = cached.clone();
                        }
                    } else {
                        // Clear stale lyrics immediately so we never show the
                        // wrong song while the new fetch is in flight.
                        {
                            let mut s = shared.lock().unwrap();
                            if s.music.track_id() == id {
                                s.lyrics = crate::state::Lyrics {
                                    lines: Vec::new(),
                                    synced: false,
                                    track_id: id.clone(),
                                    note: "loading lyrics…".into(),
                                };
                            }
                        }
                        let lyr = lyrics::fetch(&agent, &artist, &track, &album, dur, &id);
                        lyrics::cache_save(&id, &lyr); // persist synced for instant replays
                        cache.insert(id.clone(), lyr.clone());
                        let mut s = shared.lock().unwrap();
                        if s.music.track_id() == id {
                            s.lyrics = lyr;
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(200));
        }
    });
}

// ---------------------------------------------------------------------------
// Album art: dump current track artwork to a temp PNG/JPEG, decode + downscale.
// ---------------------------------------------------------------------------
const ART_THUMB: u32 = 160;

fn artwork_script(path: &str) -> String {
    format!(
        r#"
tell application "Music"
  if not (exists current track) then return "NOTRACK"
  try
    set d to data of artwork 1 of current track
  on error
    return "NOART"
  end try
end tell
set fp to POSIX file "{path}"
try
  set f to open for access fp with write permission
  set eof f to 0
  write d to f
  close access f
on error
  try
    close access fp
  end try
  return "WRITEERR"
end try
return "OK"
"#
    )
}

pub fn spawn_artwork(shared: Shared) {
    thread::spawn(move || {
        let path = std::env::temp_dir().join("studioboard_art.dat");
        let path_str = path.to_string_lossy().to_string();
        let mut current = String::new();
        loop {
            let id = {
                let s = shared.lock().unwrap();
                if s.music.running && !s.music.track.is_empty() {
                    Some(s.music.track_id())
                } else {
                    None
                }
            };
            if let Some(id) = id {
                if id != current {
                    current = id.clone();
                    // Warm from the disk cache first so a song heard once shows its
                    // cover instantly, with no AppleScript/decode on replay.
                    let art = if let Some(cached) = load_art_cache(&id) {
                        cached
                    } else {
                        let out = Command::new("osascript")
                            .arg("-e")
                            .arg(artwork_script(&path_str))
                            .output();
                        let ok = out
                            .as_ref()
                            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "OK")
                            .unwrap_or(false);
                        if ok {
                            let decoded = decode_art(&path, &id);
                            save_art_cache(&id, &decoded);
                            decoded
                        } else {
                            crate::state::AlbumArt { track_id: id.clone(), ..Default::default() }
                        }
                    };
                    let mut s = shared.lock().unwrap();
                    if s.music.track_id() == id {
                        s.album_art = art;
                    }
                }
            }
            thread::sleep(Duration::from_millis(300));
        }
    });
}

/// One-shot artwork probe for diagnostics: dump + decode the current art.
pub fn probe_artwork() -> Option<(usize, usize, usize, [u8; 3])> {
    let path = std::env::temp_dir().join("studioboard_art.dat");
    let path_str = path.to_string_lossy().to_string();
    let out = Command::new("osascript")
        .arg("-e")
        .arg(artwork_script(&path_str))
        .output()
        .ok()?;
    if String::from_utf8_lossy(&out.stdout).trim() != "OK" {
        return None;
    }
    let art = decode_art(&path, "probe");
    let center = art.sample(0.5, 0.5)?;
    Some((art.w, art.h, art.px.len(), center))
}

/// Album art for the off-screen visual-verify path (`--cells` / `--snapshot`):
/// decode the most recent real artwork dump if present, else a radial gradient.
pub fn sample_album_art(id: String) -> crate::state::AlbumArt {
    let dump = std::env::temp_dir().join("studioboard_art.dat");
    if dump.exists() {
        let art = decode_art(&dump, &id);
        if !art.px.is_empty() {
            return art;
        }
    }
    let dim = ART_THUMB as usize;
    let mut px = Vec::with_capacity(dim * dim);
    for y in 0..dim {
        for x in 0..dim {
            px.push([(x * 255 / dim) as u8, (y * 255 / dim) as u8, 180]);
        }
    }
    crate::state::AlbumArt { track_id: id, w: dim, h: dim, px }
}

/// Persist the downscaled RGB thumb to `~/.cache/studioboard/art/<hash>.bin`.
/// Format: little-endian u32 `w`, u32 `h`, then `w*h*3` raw RGB bytes. Skips
/// empty thumbs so a failed decode isn't cached as a permanent blank.
fn save_art_cache(id: &str, art: &crate::state::AlbumArt) {
    if art.px.is_empty() || art.w == 0 || art.h == 0 {
        return;
    }
    let mut buf = Vec::with_capacity(8 + art.px.len() * 3);
    buf.extend_from_slice(&(art.w as u32).to_le_bytes());
    buf.extend_from_slice(&(art.h as u32).to_le_bytes());
    for p in &art.px {
        buf.extend_from_slice(p);
    }
    crate::cache::put_bytes("art", id, "bin", &buf);
}

/// Load a cached album-art thumb, if present. Fails soft to `None` on a missing,
/// truncated, or size-mismatched file so the caller regenerates from AppleScript.
fn load_art_cache(id: &str) -> Option<crate::state::AlbumArt> {
    let buf = crate::cache::get_bytes("art", id, "bin")?;
    if buf.len() < 8 {
        return None;
    }
    let w = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let h = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
    let body = &buf[8..];
    if w == 0 || h == 0 || body.len() != w * h * 3 {
        return None; // truncated/corrupt — regenerate
    }
    let px = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    Some(crate::state::AlbumArt { track_id: id.to_string(), w, h, px })
}

fn decode_art(path: &std::path::Path, id: &str) -> crate::state::AlbumArt {
    use image::imageops::FilterType;
    let mut art = crate::state::AlbumArt { track_id: id.to_string(), ..Default::default() };
    // Decode from bytes so format is detected by magic, not file extension
    // (Music writes a raw PNG/JPEG to a .dat path).
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(img) = image::load_from_memory(&bytes) {
            let small = img.resize_exact(ART_THUMB, ART_THUMB, FilterType::Lanczos3).to_rgb8();
            art.w = small.width() as usize;
            art.h = small.height() as usize;
            art.px = small.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        }
    }
    art
}

// ---------------------------------------------------------------------------
// QUEUE: the next few tracks Apple Music will play. AppleScript can't read the
// true dynamic "Up Next" list, so we read the current playlist and take the
// tracks after the current one (correct for in-order playback; shuffle reorders).
// ---------------------------------------------------------------------------
const QUEUE_SCRIPT: &str = r#"
if application "Music" is running then
  tell application "Music"
    try
      set cp to current playlist
      set ct to current track
      set i to index of ct
      set n to count of tracks of cp
      set out to ""
      repeat with k from (i + 1) to (i + 3)
        if k is less than or equal to n then
          set tr to track k of cp
          set out to out & (name of tr) & "\t" & (artist of tr) & "\t" & (duration of tr) & "\n"
        end if
      end repeat
      return out
    on error
      return "NOQUEUE"
    end try
  end tell
else
  return "NOTRUNNING"
end if
"#;

pub fn spawn_queue(shared: Shared) {
    thread::spawn(move || loop {
        let want = {
            let s = shared.lock().unwrap();
            if s.music.running && !s.music.track.is_empty() {
                Some(s.music.track_id())
            } else {
                None
            }
        };
        if let Some(id) = want {
            // Only re-query when the current track changes (the queue shifts as
            // playback advances) — cheap and avoids hammering osascript.
            let stale = {
                let s = shared.lock().unwrap();
                s.queue.source_track_id != id || !s.queue.fresh
            };
            if stale {
                let items = read_queue();
                let mut s = shared.lock().unwrap();
                if s.music.track_id() == id {
                    s.queue = crate::state::Queue { fresh: true, source_track_id: id, items };
                }
            }
        }
        thread::sleep(Duration::from_millis(1200));
    });
}

fn read_queue() -> Vec<crate::state::QueueTrack> {
    let out = Command::new("osascript").arg("-e").arg(QUEUE_SCRIPT).output();
    let Ok(o) = out else { return Vec::new() };
    let raw = String::from_utf8_lossy(&o.stdout);
    let body = raw.trim();
    if body == "NOTRUNNING" || body == "NOQUEUE" || body.is_empty() {
        return Vec::new();
    }
    body.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 2 && !f[0].trim().is_empty() {
                Some(crate::state::QueueTrack {
                    track: f[0].trim().to_string(),
                    artist: f[1].trim().to_string(),
                    duration: f.get(2).and_then(|d| d.trim().parse().ok()).unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LINER NOTES: interesting facts about the current track/album/artist. Claude
// (Haiku) writes punchy trivia when ANTHROPIC_API_KEY is present; otherwise we
// fall back to a Wikipedia extract so the card is never empty. Cached per track
// in-memory AND on disk (~/.cache/studioboard/facts/) so it never regenerates on
// seek/pause and a song heard once loads its facts instantly across restarts —
// with zero network/LLM calls on replay.
// ---------------------------------------------------------------------------
const FACTS_UA: &str = "studioboard/0.1 (https://github.com/tehreet/battlestation)";

pub fn spawn_facts(shared: Shared) {
    thread::spawn(move || {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .timeout_read(Duration::from_secs(20))
            .build();
        let mut cache: std::collections::HashMap<String, crate::state::MusicFacts> =
            Default::default();
        let mut current = String::new();
        loop {
            let info = {
                let s = shared.lock().unwrap();
                if s.music.running && !s.music.track.is_empty() {
                    Some((
                        s.music.track_id(),
                        s.music.artist.clone(),
                        s.music.track.clone(),
                        s.music.album.clone(),
                    ))
                } else {
                    None
                }
            };
            if let Some((id, artist, track, album)) = info {
                if id != current {
                    current = id.clone();
                    // Warm from the in-memory cache first, then disk (survives
                    // restarts), before spending a network/LLM call.
                    let warm = cache.get(&id).cloned().or_else(|| load_facts_cache(&id));
                    if let Some(mut hit) = warm {
                        hit.track_id = id.clone();
                        cache.insert(id.clone(), hit.clone());
                        let mut s = shared.lock().unwrap();
                        if s.music.track_id() == id {
                            s.facts = hit;
                        }
                    } else {
                        // Show a status line while we gather, scoped to this track.
                        {
                            let mut s = shared.lock().unwrap();
                            if s.music.track_id() == id {
                                s.facts = crate::state::MusicFacts {
                                    track_id: id.clone(),
                                    note: "gathering liner notes…".into(),
                                    ..Default::default()
                                };
                            }
                        }
                        let facts = build_facts(&agent, &artist, &track, &album, &id);
                        cache.insert(id.clone(), facts.clone());
                        // Persist only real results — skip empty/note-only so a
                        // transient failure never becomes a sticky cache miss.
                        save_facts_cache(&id, &facts);
                        let mut s = shared.lock().unwrap();
                        if s.music.track_id() == id {
                            s.facts = facts;
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(300));
        }
    });
}

/// Load liner-note facts for a track from the persistent disk cache, if a real
/// (non-empty) entry exists. Corrupt/partial JSON fails soft to `None`.
fn load_facts_cache(id: &str) -> Option<crate::state::MusicFacts> {
    let facts: crate::state::MusicFacts = crate::cache::get_json("facts", id)?;
    if facts.lines.is_empty() {
        None // empty/note-only entries are never written, but stay defensive
    } else {
        Some(facts)
    }
}

/// Persist liner-note facts to the disk cache. Skips empty/note-only results so a
/// transient generation failure isn't cached as a permanent miss.
fn save_facts_cache(id: &str, facts: &crate::state::MusicFacts) {
    if facts.lines.is_empty() {
        return;
    }
    crate::cache::put_json("facts", id, facts);
}

/// One-shot facts probe for diagnostics (`studioboard --facts`).
pub fn probe_facts(artist: &str, track: &str, album: &str) -> crate::state::MusicFacts {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(4))
        .timeout_read(Duration::from_secs(20))
        .build();
    let id = format!("{artist}|{track}|{album}");
    build_facts(&agent, artist, track, album, &id)
}

/// Resolve the Anthropic API key without an interactive prompt. Tries the
/// `ANTHROPIC_API_KEY` env var first, then 1Password via the `op` CLI (zero-
/// prompt when the service-account token is in the Keychain). The op item must
/// live in a vault the active credential can read — override the lookup with
/// `OP_ANTHROPIC_VAULT` / `OP_ANTHROPIC_ITEM` / `OP_ANTHROPIC_FIELD` if needed.
fn anthropic_key() -> Option<String> {
    if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    let vault = std::env::var("OP_ANTHROPIC_VAULT").unwrap_or_else(|_| "Claude Code".into());
    let item = std::env::var("OP_ANTHROPIC_ITEM").unwrap_or_else(|_| "Claude Anthropic API Key".into());
    let field = std::env::var("OP_ANTHROPIC_FIELD").unwrap_or_else(|_| "notesPlain".into());
    let out = Command::new("op")
        .args(["item", "get", &item, "--vault", &vault, "--fields", &field, "--reveal"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let k = String::from_utf8_lossy(&out.stdout).trim().trim_matches('"').to_string();
    if k.is_empty() {
        None
    } else {
        Some(k)
    }
}

pub fn build_facts(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
    id: &str,
) -> crate::state::MusicFacts {
    let mut out = crate::state::MusicFacts { track_id: id.to_string(), ..Default::default() };
    if let Some(key) = anthropic_key() {
        if let Some(lines) = facts_via_claude(agent, &key, artist, track, album) {
            if !lines.is_empty() {
                out.lines = lines;
                out.source = "claude".into();
                return out;
            }
        }
    }
    if let Some(lines) = facts_via_wikipedia(agent, artist, track, album) {
        if !lines.is_empty() {
            out.lines = lines;
            out.source = "wikipedia".into();
            return out;
        }
    }
    out.note = "no liner notes found".into();
    out
}

/// Ask Haiku for a handful of short, surprising, *specific* facts. Best-effort:
/// returns None on any failure so the Wikipedia fallback takes over.
fn facts_via_claude(
    agent: &ureq::Agent,
    key: &str,
    artist: &str,
    track: &str,
    album: &str,
) -> Option<Vec<String>> {
    let system = "You are a music historian writing punchy liner-note trivia for a \
now-playing dashboard. Output ONLY the facts — one per line, no numbering, no \
bullets, no preamble, no caveats, no sign-off. NEVER refuse and NEVER add \
commentary about your confidence. If you are unsure of a hyper-specific detail, \
give a fact you ARE confident about (the artist, the era, the album's place in \
their catalog, cultural impact). Every line must be a concrete, surprising fact \
under 16 words.";
    let prompt = format!(
        "Four liner-note facts about the song \"{track}\" from the album \"{album}\" \
         by {artist}. Favor production, samples/interpolations, hidden references, \
         chart or record feats, and studio lore — what a superfan would geek out over."
    );
    let body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 400,
        "system": system,
        "messages": [{ "role": "user", "content": prompt }]
    });
    let resp: serde_json::Value = agent
        .post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(body)
        .ok()?
        .into_json()
        .ok()?;
    let text = resp.get("content")?.get(0)?.get("text")?.as_str()?;
    if looks_like_refusal(text) {
        return None; // let the Wikipedia fallback take over
    }
    let lines: Vec<String> = text
        .lines()
        .filter_map(clean_fact)
        .filter(|l| !looks_like_refusal(l))
        .take(6)
        .collect();
    if lines.len() < 2 {
        None // too thin to be worth showing — fall back
    } else {
        Some(lines)
    }
}

/// True when model output reads as a hedge/refusal rather than facts.
fn looks_like_refusal(s: &str) -> bool {
    let l = s.to_lowercase();
    const TELLS: [&str; 9] = [
        "i'm not confident",
        "i need to be honest",
        "i should acknowledge",
        "i'd recommend checking",
        "i appreciate",
        "i don't have",
        "i cannot",
        "i can't provide",
        "as an ai",
    ];
    TELLS.iter().any(|t| l.contains(t))
}

/// Strip leading bullets / numbering / whitespace from a model-emitted fact line.
fn clean_fact(raw: &str) -> Option<String> {
    let t = raw.trim().trim_start_matches(|c: char| {
        c == '-' || c == '*' || c == '•' || c == '·' || c.is_ascii_digit() || c == '.' || c == ')' || c == ' '
    });
    let t = t.trim();
    if t.len() < 4 {
        None
    } else {
        Some(t.to_string())
    }
}

/// Keyless fallback: stitch a few sentences from the song's and artist's
/// Wikipedia extracts into standalone facts.
fn facts_via_wikipedia(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
) -> Option<Vec<String>> {
    let mut facts: Vec<String> = Vec::new();
    // The song article first (most specific); try a couple of title forms.
    let song = wiki_extract(agent, track)
        .or_else(|| wiki_extract(agent, &format!("{track} (song)")))
        .or_else(|| wiki_extract(agent, album));
    if let Some(ex) = song {
        facts.extend(sentences(&ex).into_iter().take(3));
    }
    // Then a line or two about the artist.
    if let Some(ex) = wiki_extract(agent, artist) {
        facts.extend(sentences(&ex).into_iter().take(2));
    }
    facts.truncate(5);
    if facts.is_empty() {
        None
    } else {
        Some(facts)
    }
}

/// Fetch the lead extract for a Wikipedia page title (REST summary API). Skips
/// disambiguation pages and empty/missing results.
fn wiki_extract(agent: &ureq::Agent, title: &str) -> Option<String> {
    if title.trim().is_empty() {
        return None;
    }
    let enc: String = title
        .trim()
        .chars()
        .map(|c| match c {
            ' ' => "_".to_string(),
            'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.' | '(' | ')' => c.to_string(),
            _ => c.to_string().bytes().map(|b| format!("%{b:02X}")).collect(),
        })
        .collect();
    let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{enc}");
    let v: serde_json::Value = agent
        .get(&url)
        .set("User-Agent", FACTS_UA)
        .call()
        .ok()?
        .into_json()
        .ok()?;
    if v.get("type").and_then(|x| x.as_str()) == Some("disambiguation") {
        return None;
    }
    let ex = v.get("extract").and_then(|x| x.as_str())?.trim();
    if ex.is_empty() {
        None
    } else {
        Some(ex.to_string())
    }
}

/// Split prose into trimmed, sentence-ish chunks (naive but fine for extracts).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        if c == '.' {
            // End a sentence on ". " or end-of-text, but not on a decimal like 9.11.
            let next_space = chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).map(|n| n.is_ascii_digit()).unwrap_or(false);
            if next_space && !(prev_digit && next_digit) {
                let s = cur.trim().trim_end_matches('.').trim().to_string();
                if s.len() > 12 {
                    out.push(s);
                }
                cur.clear();
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Git: pulse of the battlestation repo (branch, dirty, ahead/behind, last).
// ---------------------------------------------------------------------------
pub fn spawn_git(shared: Shared) {
    let repo = std::env::var("STUDIOBOARD_REPO").ok().unwrap_or_else(|| {
        dirs::home_dir()
            .map(|h| h.join("workspace/battlestation").to_string_lossy().to_string())
            .unwrap_or_default()
    });
    thread::spawn(move || loop {
        let g = scan_git(&repo);
        {
            let mut s = shared.lock().unwrap();
            s.git = g;
        }
        thread::sleep(Duration::from_secs(10));
    });
}

fn git(repo: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(repo).args(args).output().ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

fn scan_git(repo: &str) -> crate::state::GitStats {
    use crate::state::GitStats;
    let mut g = GitStats { fresh: true, ..Default::default() };
    // The most-recently-active LOCAL branch (by last commit date), plus that
    // branch's last commit — branch, short hash, subject, relative age — in one
    // for-each-ref call. \x1f = unit separator so commit subjects stay intact.
    let fmt = "--format=%(refname:short)\x1f%(objectname:short)\x1f%(contents:subject)\x1f%(committerdate:relative)";
    let Some(line) = git(
        repo,
        &["for-each-ref", "--sort=-committerdate", "refs/heads", fmt, "--count=1"],
    ) else {
        return g; // not a repo / git missing
    };
    if line.trim().is_empty() {
        return g;
    }
    g.ok = true;
    g.repo = std::path::Path::new(repo)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut p = line.split('\x1f');
    g.branch = p.next().unwrap_or("").trim().to_string();
    g.last_hash = p.next().unwrap_or("").to_string();
    g.last_msg = p.next().unwrap_or("").to_string();
    g.last_rel = p.next().unwrap_or("").trim().to_string();

    // Commits on that branch since midnight + working-tree dirtiness (for the hot border).
    if let Some(n) = git(repo, &["rev-list", "--count", "--since=midnight", &g.branch]) {
        g.commits_today = n.trim().parse().unwrap_or(0);
    }
    if let Some(porc) = git(repo, &["status", "--porcelain"]) {
        for l in porc.lines() {
            g.dirty += 1;
            if l.starts_with("??") {
                g.untracked += 1;
            } else if l.chars().next().map(|c| c != ' ').unwrap_or(false) {
                g.staged += 1;
            }
        }
    }
    // Branch activity vs the main line: commits/loc/merges this branch carries
    // that aren't yet on origin/main (its own work). Falls back gracefully.
    if let Some(base) = ["origin/main", "origin/master", "main", "master"]
        .into_iter()
        .find(|r| git(repo, &["rev-parse", "--verify", "--quiet", r]).map(|h| !h.trim().is_empty()).unwrap_or(false))
    {
        let range = format!("{base}..HEAD");
        if let Some(n) = git(repo, &["rev-list", "--count", &range]) {
            g.branch_commits = n.trim().parse().unwrap_or(0);
        }
        if let Some(n) = git(repo, &["rev-list", "--count", "--merges", &range]) {
            g.merges_main = n.trim().parse().unwrap_or(0);
        }
        if let Some(stat) = git(repo, &["diff", "--numstat", &format!("{base}...HEAD")]) {
            for l in stat.lines() {
                let mut it = l.split('\t');
                g.loc_added += it.next().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0);
                g.loc_removed += it.next().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0);
            }
        }
    }
    // PRs authored by the user (all states). Best-effort — needs gh + auth.
    g.pr_count = gh_pr_count(repo).unwrap_or(0);
    g
}

/// Count of pull requests the user has authored on this repo (any state), via
/// the `gh` CLI run inside the repo. None if gh is missing/unauthed/offline.
fn gh_pr_count(repo: &str) -> Option<u32> {
    let out = Command::new("gh")
        .args([
            "pr", "list", "--author", "@me", "--state", "all", "--limit", "200", "--json",
            "number", "--jq", "length",
        ])
        .current_dir(repo)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

// ---------------------------------------------------------------------------
// Weather: wttr.in JSON, IP-geolocated, refreshed every 15 min.
// ---------------------------------------------------------------------------
pub fn spawn_weather(shared: Shared) {
    thread::spawn(move || loop {
        if let Some(w) = fetch_weather() {
            let mut s = shared.lock().unwrap();
            s.weather = w;
        }
        thread::sleep(Duration::from_secs(900));
    });
}

fn fetch_weather() -> Option<crate::state::Weather> {
    use crate::state::Weather;
    // Pin the location: Fond du Lac, WI 54937 (43.7730, -88.4471). Querying
    // wttr.in by explicit coords keeps the dependency-light path while landing
    // the right city (IP-geolocation otherwise lands on the wrong town).
    let v: serde_json::Value = ureq::get("https://wttr.in/43.7730,-88.4471?format=j1")
        .set("User-Agent", "curl/8")
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let cur = v.get("current_condition")?.get(0)?;
    let today = v.get("weather")?.get(0)?;
    let gi = |o: &serde_json::Value, k: &str| -> i32 {
        o.get(k).and_then(|x| x.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0)
    };
    let gs = |o: &serde_json::Value, k: &str| -> String {
        o.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
    };
    let desc = cur
        .get("weatherDesc")
        .and_then(|d| d.get(0))
        .and_then(|d| d.get("value"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    // Highest rain chance across today's hourly slots.
    let precip_chance = today
        .get("hourly")
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|h| h.get("chanceofrain").and_then(|x| x.as_str()).and_then(|s| s.parse::<i32>().ok()))
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);

    let astro = today.get("astronomy").and_then(|a| a.get(0));
    let (sunrise, sunset) = astro
        .map(|a| (gs(a, "sunrise"), gs(a, "sunset")))
        .unwrap_or_default();

    // Next ~12 hourly temps (°F) across today + tomorrow for the jazz strip.
    let mut temp_strip: Vec<u64> = Vec::new();
    if let Some(days) = v.get("weather").and_then(|w| w.as_array()) {
        for day in days.iter().take(2) {
            if let Some(hrs) = day.get("hourly").and_then(|h| h.as_array()) {
                for hr in hrs {
                    if let Some(t) = hr.get("tempF").and_then(|x| x.as_str()).and_then(|s| s.parse::<i64>().ok()) {
                        temp_strip.push(t.max(0) as u64);
                    }
                }
            }
        }
    }
    temp_strip.truncate(12);

    Some(Weather {
        fresh: true,
        icon: weather_icon(&desc),
        location: "Fond du Lac, WI".into(),
        temp_f: gi(cur, "temp_F"),
        feels_f: gi(cur, "FeelsLikeF"),
        humidity: gi(cur, "humidity"),
        hi_f: gi(today, "maxtempF"),
        lo_f: gi(today, "mintempF"),
        wind_mph: gi(cur, "windspeedMiles"),
        wind_dir: gs(cur, "winddir16Point"),
        precip_chance,
        uv: gi(cur, "uvIndex"),
        pressure_mb: gi(cur, "pressure"),
        sunrise: fmt_clock12_to_24(&sunrise),
        sunset: fmt_clock12_to_24(&sunset),
        temp_strip,
        desc,
    })
}

/// wttr.in emits astronomy times like "06:21 AM" / "08:14 PM". Normalize to a
/// fixed-width 24h "HH:MM" so the right-aligned weather sun row never jitters.
fn fmt_clock12_to_24(s: &str) -> String {
    let s = s.trim();
    let (time, ap) = match s.rsplit_once(' ') {
        Some((t, ap)) => (t, ap.to_uppercase()),
        None => return s.to_string(),
    };
    let (h, m) = match time.split_once(':') {
        Some((h, m)) => (h.parse::<i32>().unwrap_or(0), m),
        None => return s.to_string(),
    };
    let h24 = match ap.as_str() {
        "PM" if h != 12 => h + 12,
        "AM" if h == 12 => 0,
        _ => h,
    };
    format!("{h24:02}:{m}")
}

fn weather_icon(desc: &str) -> String {
    let d = desc.to_lowercase();
    let i = if d.contains("thunder") {
        "⛈"
    } else if d.contains("snow") || d.contains("blizzard") || d.contains("sleet") {
        "❄"
    } else if d.contains("rain") || d.contains("drizzle") || d.contains("shower") {
        "🌧"
    } else if d.contains("fog") || d.contains("mist") {
        "🌫"
    } else if d.contains("overcast") {
        "☁"
    } else if d.contains("cloud") || d.contains("partly") {
        "⛅"
    } else if d.contains("sun") || d.contains("clear") {
        "☀"
    } else {
        "🌡"
    };
    i.to_string()
}

// ---------------------------------------------------------------------------
// Claude Code usage: roll up token usage + estimated cost from ~/.claude logs.
// ---------------------------------------------------------------------------

/// Per-million-token pricing, USD: (input, output, cache_read, cache_write_5m).
/// API-equivalent only — the user is on a Claude subscription, not API billing.
/// cache_read = 0.1×input, cache_write(5m) = 1.25×input. Current as of 2026-06.
fn pricing(model: &str) -> (f64, f64, f64, f64) {
    let m = model.to_lowercase();
    if m.contains("opus") {
        (5.0, 25.0, 0.5, 6.25)
    } else if m.contains("haiku") {
        (1.0, 5.0, 0.1, 1.25)
    } else if m.contains("sonnet") {
        (3.0, 15.0, 0.3, 3.75)
    } else {
        (3.0, 15.0, 0.3, 3.75)
    }
}

pub fn spawn_usage(shared: Shared) {
    thread::spawn(move || loop {
        if let Some(stats) = scan_usage() {
            let mut s = shared.lock().unwrap();
            s.usage = stats;
        }
        thread::sleep(Duration::from_secs(8));
    });
}

fn scan_usage() -> Option<crate::state::UsageStats> {
    use crate::state::UsageStats;
    let base = dirs::home_dir()?.join(".claude").join("projects");
    let now = Local::now();
    let today = now.date_naive();
    let month = (now.year(), now.month());

    let d7 = now - chrono::Duration::days(7);
    let d30 = now - chrono::Duration::days(30);

    let mut st = UsageStats { fresh: true, ..Default::default() };
    let mut model_counts: std::collections::HashMap<String, u64> = Default::default();
    let mut sessions = std::collections::HashSet::new();
    let mut sessions_30d = std::collections::HashSet::new();
    let mut hourly = [0u64; 24];
    // Resumed sessions / sidechains replay the same assistant message across
    // multiple JSONL files; count each request once or the totals balloon.
    let mut seen_req: std::collections::HashSet<String> = Default::default();

    // Only files touched within the last 31 days can hold rows for our windows
    // (today / 7d / 30d). One day of slack covers timezone edges.
    let scan_cutoff = now.timestamp() - 31 * 86400;

    let walk = walk_jsonl(&base);
    for path in walk {
        if let Ok(meta) = std::fs::metadata(&path) {
            if let Ok(modt) = meta.modified() {
                if let Ok(dur) = modt.duration_since(std::time::UNIX_EPOCH) {
                    if (dur.as_secs() as i64) < scan_cutoff {
                        continue;
                    }
                }
            }
        }
        let Ok(content) = std::fs::read_to_string(&path) else { continue };
        for line in content.lines() {
            if !line.contains("\"usage\"") {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let msg = v.get("message");
            let usage = msg.and_then(|m| m.get("usage")).or_else(|| v.get("usage"));
            let Some(u) = usage else { continue };
            // Dedup on requestId (fall back to message.id) so replayed rows from
            // resumed sessions/sidechains aren't double-counted.
            if let Some(rid) = v
                .get("requestId")
                .and_then(|x| x.as_str())
                .or_else(|| msg.and_then(|m| m.get("id")).and_then(|x| x.as_str()))
            {
                if !seen_req.insert(rid.to_string()) {
                    continue;
                }
            }
            let ts = v.get("timestamp").and_then(|t| t.as_str()).unwrap_or("");
            let Ok(dt) = DateTime::parse_from_rfc3339(ts) else { continue };
            let local = dt.with_timezone(&Local);
            // Older than our widest window (30d) — nothing to do with it.
            if local < d30 {
                continue;
            }
            let model = msg
                .and_then(|m| m.get("model"))
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let g = |k: &str| u.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
            let inp = g("input_tokens");
            let outp = g("output_tokens");
            let cread = g("cache_read_input_tokens");
            let cwrite = g("cache_creation_input_tokens");
            let tokens = inp + outp + cread + cwrite;
            let (pi, po, pcr, pcw) = pricing(&model);
            let cost = inp as f64 / 1e6 * pi
                + outp as f64 / 1e6 * po
                + cread as f64 / 1e6 * pcr
                + cwrite as f64 / 1e6 * pcw;

            // Rolling 30-day window (always reached, since we skipped older rows).
            st.tokens_30d += tokens;
            if let Some(sid) = v.get("sessionId").and_then(|x| x.as_str()) {
                sessions_30d.insert(sid.to_string());
            }
            if local >= d7 {
                st.tokens_7d += tokens;
            }
            if (local.year(), local.month()) == month {
                st.month_cost += cost; // kept for the footer's "Claude today/$" readout
            }

            if local.date_naive() == today {
                st.today_input += inp;
                st.today_output += outp;
                st.today_cache_read += cread;
                st.today_cache_write += cwrite;
                st.today_cost += cost;
                st.today_messages += 1;
                *model_counts.entry(model).or_insert(0) += 1;
                if let Some(sid) = v.get("sessionId").and_then(|x| x.as_str()) {
                    sessions.insert(sid.to_string());
                }
                let h = local.hour() as usize;
                hourly[h] += tokens; // token activity per hour
            }
        }
    }

    st.top_model = model_counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(m, _)| short_model(&m))
        .unwrap_or_default();
    st.sessions_today = sessions.len() as u64;
    st.sessions_30d = sessions_30d.len() as u64;
    st.hourly = hourly.to_vec();
    Some(st)
}

fn short_model(m: &str) -> String {
    let l = m.to_lowercase();
    if l.contains("opus") {
        "Opus".into()
    } else if l.contains("sonnet") {
        "Sonnet".into()
    } else if l.contains("haiku") {
        "Haiku".into()
    } else {
        m.to_string()
    }
}

// ---------------------------------------------------------------------------
// iMessage: read recent inbound messages from ~/Library/Messages/chat.db via
// the `sqlite3` CLI (read-only + immutable — zero new deps, FDA grants access).
// Replies are sent back through osascript. Long previews are summarized via the
// Anthropic Haiku API when ANTHROPIC_API_KEY is present (best-effort, cached).
// ---------------------------------------------------------------------------

/// Recent conversations, newest-active first — one row per chat, previewing the
/// chat's latest message (either direction), exactly like the iPhone Messages
/// list. We join each chat to its single most-recent message and carry an
/// unread-inbound count for the dot/badge.
///
/// `text` is flattened (newlines/tabs → spaces) in SQL so a multi-line body can't
/// split a TSV row. `quote(attributedBody)` emits one `X'..'` hex literal, so the
/// BLOB likewise can't break parsing. Hex is only fetched when `text` is empty
/// (modern macOS stores the real string in attributedBody for ~9% of messages).
const MSG_SQL: &str = "\
SELECT c.ROWID AS chat_id, \
       m.ROWID AS msg_rowid, \
       (m.date/1000000000.0 + 978307200) AS ts, \
       m.is_from_me, \
       COALESCE(c.display_name,'') AS display_name, \
       COALESCE(c.chat_identifier,'') AS chat_ident, \
       replace(replace(replace(COALESCE(m.text,''), char(10),' '), char(13),' '), char(9),' ') AS text, \
       CASE WHEN (m.text IS NULL OR m.text='') AND m.attributedBody IS NOT NULL \
            THEN quote(m.attributedBody) ELSE '' END AS ab, \
       (SELECT count(*) FROM chat_message_join j2 JOIN message m2 ON m2.ROWID=j2.message_id \
        WHERE j2.chat_id=c.ROWID AND m2.is_from_me=0 AND m2.is_read=0) AS unread_n, \
       COALESCE(sh.id,'') AS sender_handle \
FROM chat c \
JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID \
JOIN message m ON m.ROWID = cmj.message_id \
LEFT JOIN handle sh ON sh.ROWID = m.handle_id \
JOIN (SELECT j.chat_id AS cid, MAX(mm.date) AS maxd \
      FROM chat_message_join j JOIN message mm ON mm.ROWID=j.message_id \
      GROUP BY j.chat_id) latest ON latest.cid=c.ROWID AND m.date=latest.maxd \
ORDER BY m.date DESC \
LIMIT 40;";

/// "Recent" window for the unread badge: a conversation only counts toward the
/// badge if its latest message landed within this many seconds. Keeps stale
/// never-cleared read-flags from inflating the count.
const UNREAD_RECENT_SECS: f64 = 30.0 * 86400.0;

/// How many recent conversations the card shows (a glance, not a full inbox).
const SHOWN_CONVERSATIONS: usize = 5;

/// Beyond this many display chars a preview is summarized (if a key is present)
/// or smart-truncated to keep the card readable.
const PREVIEW_BUDGET: usize = 96;

pub fn spawn_messages(shared: Shared) {
    thread::spawn(move || {
        let Some(home) = dirs::home_dir() else { return };
        let db = home.join("Library/Messages/chat.db");
        // Read-only (NOT immutable): chat.db is a live WAL database. `immutable=1`
        // tells SQLite to ignore the -wal file, so freshly-received messages that
        // haven't been checkpointed into the main db yet stay invisible and the
        // card shows a stale snapshot. `mode=ro` reads the WAL in place (no 627MB
        // copy, no write lock on Messages.app) so the latest messages show.
        let uri = format!("file:{}?mode=ro", db.display());

        // Network agent + per-ROWID summary cache, reused across polls.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .timeout_read(Duration::from_secs(12))
            .build();
        let mut summary_cache: std::collections::HashMap<i64, String> = Default::default();

        // Contact names change rarely; load once and refresh every ~50 polls.
        let mut contacts = load_contacts();
        let mut poll = 0u32;

        loop {
            poll = poll.wrapping_add(1);
            if poll % 50 == 0 {
                let fresh = load_contacts();
                if !fresh.is_empty() {
                    contacts = fresh;
                }
            }
            match read_messages(&uri, &contacts) {
                Some((mut items, unread)) => {
                    // Best-effort summarization of long previews (gated on key).
                    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                        if !key.is_empty() {
                            for it in items.iter_mut() {
                                if it.is_rich {
                                    continue;
                                }
                                if it.full_text.chars().count() <= PREVIEW_BUDGET {
                                    continue;
                                }
                                if let Some(cached) = summary_cache.get(&it.rowid) {
                                    it.preview = cached.clone();
                                    continue;
                                }
                                if let Some(sum) = summarize(&agent, &key, &it.full_text) {
                                    summary_cache.insert(it.rowid, sum.clone());
                                    it.preview = sum;
                                }
                            }
                        }
                    }
                    let mut s = shared.lock().unwrap();
                    s.messages.fresh = true;
                    s.messages.available = true;
                    s.messages.unread_count = unread;
                    s.messages.items = items;
                    // Keep the focus queue position in bounds as rows change.
                    let n = s.messages.items.iter().filter(|i| i.unread).count();
                    if s.msg_ui.queue_pos >= n.max(1) {
                        s.msg_ui.queue_pos = n.saturating_sub(1);
                    }
                }
                None => {
                    let mut s = shared.lock().unwrap();
                    s.messages.fresh = true;
                    s.messages.available = false;
                }
            }
            thread::sleep(Duration::from_secs(8));
        }
    });
}

/// `--diag-msg`: exercise the real iMessage path (contacts + query + parse +
/// contact filter) and print exactly where it breaks — so a frozen card can be
/// told apart from "query errored on this schema", "no contacts loaded", or
/// "rows returned but everything got filtered out".
pub fn diag_messages() {
    let Some(home) = dirs::home_dir() else {
        println!("no home dir");
        return;
    };
    let db = home.join("Library/Messages/chat.db");
    let uri = format!("file:{}?mode=ro", db.display()); // read WAL (see spawn_messages)
    println!("studioboard --diag-msg\n");
    println!("chat.db: {}", db.display());

    // 1) Raw query: does MSG_SQL run against THIS chat.db schema at all?
    match Command::new("sqlite3")
        .args(["-separator", "\t", "-newline", "\n", &uri, MSG_SQL])
        .output()
    {
        Ok(o) => {
            let rows = String::from_utf8_lossy(&o.stdout).lines().count();
            let err = String::from_utf8_lossy(&o.stderr);
            println!("  raw query : ok={}  rows={}", o.status.success(), rows);
            if !err.trim().is_empty() {
                println!("  raw stderr: {}", err.trim());
            }
        }
        Err(e) => println!("  raw query : could not run sqlite3: {e}"),
    }

    // 2) Contacts (AddressBook). Empty => every 1:1 chat is dropped by the filter.
    let contacts = load_contacts();
    println!("\ncontacts loaded: {}", contacts.len());
    for (k, v) in contacts.iter().take(3) {
        println!("  {k} -> {v}");
    }

    // 3) Full path: query + parse + contact filter (what the card actually shows).
    match read_messages(&uri, &contacts) {
        Some((items, unread)) => {
            println!("\nread_messages: {} shown, badge unread={}", items.len(), unread);
            for it in &items {
                println!(
                    "  [{}] {} — {}",
                    if it.unread { "•" } else { " " },
                    it.sender,
                    it.preview
                );
            }
            if items.is_empty() {
                println!("  → rows returned but ALL were filtered out: no 1:1 chat resolved to a");
                println!("    contact, and no named group chats. (contacts unreadable, or the");
                println!("    contacts-only filter is too strict for your inbox.)");
            }
        }
        None => {
            println!("\nread_messages: None — query failed or chat.db unreadable (see raw stderr).")
        }
    }
}

/// Run the conversation query; return (items newest-active-first, unread_count).
/// `contacts` maps normalized handles → display names (empty if no AddressBook
/// access). None if chat.db can't be read so the panel shows a graceful hint.
fn read_messages(
    uri: &str,
    contacts: &std::collections::HashMap<String, String>,
) -> Option<(Vec<crate::state::MessageItem>, u32)> {
    use crate::state::MessageItem;
    let out = Command::new("sqlite3")
        .args(["-separator", "\t", "-newline", "\n", uri, MSG_SQL])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("authorization denied") || stderr.contains("unable to open") {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);

    let now = Local::now().timestamp() as f64;
    let mut items = Vec::new();
    let mut seen_chats = std::collections::HashSet::new();
    for line in body.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 10 {
            continue;
        }
        let chat_id: i64 = f[0].parse().unwrap_or(0);
        // A chat can tie on MAX(date) across two messages; keep the first only.
        if !seen_chats.insert(chat_id) {
            continue;
        }
        let rowid: i64 = f[1].parse().unwrap_or(0);
        let ts: f64 = f[2].parse().unwrap_or(0.0);
        let from_me = f[3].trim() == "1";
        let display_name = f[4].trim().to_string();
        let chat_ident = f[5].to_string();
        let text = f[6].to_string();
        let ab = f[7];
        let unread_n: u32 = f[8].trim().parse().unwrap_or(0);
        let sender_handle = f[9].trim();

        let (body_text, is_rich) = if !text.trim().is_empty() {
            (text, false)
        } else if !ab.is_empty() {
            match decode_attributed_body(ab) {
                Some(t) if !t.trim().is_empty() => (t, false),
                _ => ("[rich message]".to_string(), true),
            }
        } else {
            ("[rich message]".to_string(), true)
        };

        // Group chats: GUID-style chat_identifier ("chat3889…"); reply target n/a.
        let is_group = chat_ident.starts_with("chat");
        // Shortcode: a short all-digits 1:1 sender (32665, 91703 — banks, spam).
        let digit_count = chat_ident.chars().filter(|c| c.is_ascii_digit()).count();
        let all_digits = !chat_ident.is_empty()
            && chat_ident.chars().all(|c| c.is_ascii_digit() || c == '+');
        let is_shortcode = !is_group && all_digits && digit_count > 0 && digit_count <= 6;

        // Contacts-only: keep 1:1 chats that resolve to a real AddressBook
        // contact, plus named group chats. Drops shortcodes, unknown numbers,
        // and business/notification SMS senders (Google, Coinbase, verif codes…).
        let contact = if is_group {
            None
        } else {
            contacts.get(&norm_handle(&chat_ident)).cloned()
        };
        let named_group = is_group && !display_name.is_empty();
        if contact.is_none() && !named_group {
            continue;
        }

        let sender = if is_group {
            display_name
        } else {
            contact.clone().unwrap_or_else(|| pretty_handle(&chat_ident))
        };

        let preview = {
            let p = smart_preview(&body_text, PREVIEW_BUDGET);
            if from_me {
                format!("You: {p}")
            } else if is_group && !sender_handle.is_empty() {
                // iPhone-style "<who>: <message>" prefix for group previews.
                let who = contacts
                    .get(&norm_handle(sender_handle))
                    .map(|n| n.split_whitespace().next().unwrap_or(n).to_string())
                    .unwrap_or_else(|| pretty_handle(sender_handle));
                format!("{who}: {p}")
            } else {
                p
            }
        };
        let unread = unread_n > 0;
        items.push(MessageItem {
            chat_id,
            rowid,
            sender,
            handle: if is_group { String::new() } else { chat_ident },
            preview,
            full_text: body_text,
            ts_unix: ts,
            rel: fmt_rel((now - ts).max(0.0)),
            is_rich,
            unread,
            from_me,
            is_shortcode,
        });
    }

    // Just the most recent few conversations — the card is a glance, not an inbox.
    items.truncate(SHOWN_CONVERSATIONS);
    let unread = unread_badge_count(&items, now);
    Some((items, unread))
}

/// Badge count: conversations with an unread inbound message whose latest activity
/// is recent, excluding shortcode/notification senders. Matches what a person
/// would actually call "unread" (not every stale never-cleared flag).
fn unread_badge_count(items: &[crate::state::MessageItem], now: f64) -> u32 {
    items
        .iter()
        .filter(|i| i.unread && !i.is_shortcode && (now - i.ts_unix) <= UNREAD_RECENT_SECS)
        .count() as u32
}

/// Make a raw handle (phone/email) friendlier when no contact name is known:
/// email local-part, or a US-formatted phone like (920) 555-1212.
fn pretty_handle(h: &str) -> String {
    if h.is_empty() {
        return "Unknown".into();
    }
    if let Some((user, _dom)) = h.split_once('@') {
        return user.to_string();
    }
    let digits: String = h.chars().filter(|c| c.is_ascii_digit()).collect();
    let d = if digits.len() == 11 && digits.starts_with('1') {
        &digits[1..]
    } else {
        &digits
    };
    if d.len() == 10 {
        format!("({}) {}-{}", &d[0..3], &d[3..6], &d[6..10])
    } else {
        h.to_string()
    }
}

/// Normalize a handle for contact lookup: emails lowercased; phones reduced to
/// their last 10 digits so +1 / spacing / punctuation variants all match.
fn norm_handle(h: &str) -> String {
    let h = h.trim();
    if h.contains('@') {
        return h.to_lowercase();
    }
    let digits: String = h.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() > 10 {
        digits[digits.len() - 10..].to_string()
    } else {
        digits
    }
}

/// Build a normalized-handle → contact-name map from the macOS AddressBook
/// sources (same Full Disk Access that unlocks chat.db). Best-effort: a missing
/// or unreadable AddressBook just yields an empty map (we fall back to handles).
fn load_contacts() -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    let Some(home) = dirs::home_dir() else { return map };
    let root = home.join("Library/Application Support/AddressBook/Sources");
    let Ok(sources) = std::fs::read_dir(&root) else { return map };

    // name = "First Last", falling back to organization; key = normalized handle.
    const PHONE_SQL: &str = "SELECT COALESCE(r.ZFIRSTNAME,''), COALESCE(r.ZLASTNAME,''), \
        COALESCE(r.ZORGANIZATION,''), p.ZFULLNUMBER FROM ZABCDRECORD r \
        JOIN ZABCDPHONENUMBER p ON p.ZOWNER=r.Z_PK WHERE p.ZFULLNUMBER IS NOT NULL;";
    const EMAIL_SQL: &str = "SELECT COALESCE(r.ZFIRSTNAME,''), COALESCE(r.ZLASTNAME,''), \
        COALESCE(r.ZORGANIZATION,''), e.ZADDRESS FROM ZABCDRECORD r \
        JOIN ZABCDEMAILADDRESS e ON e.ZOWNER=r.Z_PK WHERE e.ZADDRESS IS NOT NULL;";

    for src in sources.flatten() {
        let db = src.path().join("AddressBook-v22.abcddb");
        if !db.exists() {
            continue;
        }
        let uri = format!("file:{}?immutable=1", db.display());
        for sql in [PHONE_SQL, EMAIL_SQL] {
            let Ok(out) = Command::new("sqlite3")
                .args(["-separator", "\t", "-newline", "\n", &uri, sql])
                .output()
            else {
                continue;
            };
            if !out.status.success() {
                continue;
            }
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 4 {
                    continue;
                }
                let name = format!("{} {}", f[0].trim(), f[1].trim());
                let name = name.trim();
                let name = if name.is_empty() { f[2].trim() } else { name };
                let key = norm_handle(f[3]);
                if name.is_empty() || key.is_empty() {
                    continue;
                }
                // First write wins so the earlier source isn't clobbered by junk.
                map.entry(key).or_insert_with(|| name.to_string());
            }
        }
    }
    map
}

/// Best-effort decode of an Apple typedstream `attributedBody` blob delivered as
/// a `X'..'` SQL hex literal. We don't pull a typedstream crate (dependency-light
/// rule); instead we hex-decode and extract the longest printable UTF-8 run,
/// which is the message body in practice. Returns None if nothing sane is found.
fn decode_attributed_body(hexlit: &str) -> Option<String> {
    // Strip the leading X' and trailing '.
    let inner = hexlit
        .strip_prefix("X'")
        .or_else(|| hexlit.strip_prefix("x'"))?
        .strip_suffix('\'')?;
    if inner.len() < 4 || inner.len() % 2 != 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(inner.len() / 2);
    let hb = inner.as_bytes();
    let hexval = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut i = 0;
    while i + 1 < hb.len() {
        let hi = hexval(hb[i])?;
        let lo = hexval(hb[i + 1])?;
        bytes.push((hi << 4) | lo);
        i += 2;
    }

    // typedstream marks the body string with the class name "NSString"; the
    // readable run begins shortly after. Find it, else fall back to the longest
    // printable run anywhere in the blob.
    let body = if let Some(pos) = find_subseq(&bytes, b"NSString") {
        longest_printable_run(&bytes[pos + 8..])
    } else {
        longest_printable_run(&bytes)
    };
    body.filter(|s| s.chars().count() >= 2)
}

fn find_subseq(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A printable run that's actually typedstream scaffolding (class names, attribute
/// keys), not the message body — e.g. `__kIMBaseWritingDirectionAttributeName`,
/// `NSDictionary`. Rejected so a formatting-only message reads as "[rich message]"
/// instead of leaking metadata, and so a real body run is preferred over these.
fn is_typedstream_artifact(s: &str) -> bool {
    let t = s.trim();
    t.contains("kIM")
        || t.contains("AttributeName")
        || t.contains("streamtyped")
        || t.starts_with("NS")
        || t.starts_with("__")
        || t.starts_with('$') // attachment/transfer GUID placeholder ($<UUID>)
        || t == "+"
        || is_uuid_like(t)
}

/// A bare UUID (optionally `$`-prefixed) — a message-part / attachment id, never
/// readable body. Pattern: 32 hex digits split by dashes (8-4-4-4-12).
fn is_uuid_like(s: &str) -> bool {
    let s = s.strip_prefix('$').unwrap_or(s);
    let stripped: String = s.chars().filter(|&c| c != '-').collect();
    stripped.len() == 32 && stripped.chars().all(|c| c.is_ascii_hexdigit())
}

/// Scan a byte buffer for the longest contiguous run of printable UTF-8 text,
/// ignoring typedstream scaffolding runs (class names / attribute keys).
fn longest_printable_run(bytes: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(bytes);
    let printable = |c: char| c == ' ' || (!c.is_control() && c != '\u{fffd}');
    let mut best: Option<&str> = None;
    let mut start = 0usize;
    let mut active = false;
    for (idx, ch) in s.char_indices() {
        if printable(ch) {
            if !active {
                active = true;
                start = idx;
            }
            continue;
        }
        if active {
            active = false;
            let run = &s[start..idx];
            if !is_typedstream_artifact(run)
                && best.map(|b| b.trim().len()).unwrap_or(0) < run.trim().len()
            {
                best = Some(run);
            }
        }
    }
    if active {
        let run = &s[start..];
        if !is_typedstream_artifact(run)
            && best.map(|b| b.trim().len()).unwrap_or(0) < run.trim().len()
        {
            best = Some(run);
        }
    }
    best.map(|b| b.trim().to_string())
}

/// Single-line smart truncation: collapse newlines, cap to `max` chars with an
/// ellipsis. The always-on fallback when no summarization key is present.
fn smart_preview(text: &str, max: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' || c == '\t' { ' ' } else { c })
        .collect();
    let flat = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

/// Relative-time label from a delta in seconds: "now" "2m" "1h" "yesterday" "3d".
fn fmt_rel(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s < 45 {
        "now".into()
    } else if s < 3600 {
        format!("{}m", (s / 60).max(1))
    } else if s < 86400 {
        format!("{}h", s / 3600)
    } else if s < 172800 {
        "yd".into()
    } else {
        format!("{}d", s / 86400)
    }
}

/// AppleScript string escaping: backslash + double-quote, to stop the reply text
/// from breaking the script (and basic injection hygiene).
fn applescript_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Send an iMessage reply via osascript, fire-and-forget so the event loop never
/// blocks. Targets the iMessage service buddy for the given handle.
pub fn send_imessage(handle: &str, body: &str) {
    let script = format!(
        "tell application \"Messages\"\n\
         set targetService to 1st account whose service type = iMessage\n\
         set targetBuddy to participant \"{h}\" of targetService\n\
         send \"{b}\" to targetBuddy\n\
         end tell",
        h = applescript_escape(handle),
        b = applescript_escape(body),
    );
    let _ = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Mark every unread inbound message in one conversation read, persisting to
/// chat.db so the next poll doesn't resurrect the unread state. Fire-and-forget
/// and tightly scoped (one chat, inbound only); a busy DB just no-ops this round.
pub fn mark_chat_read(chat_id: i64) {
    let Some(home) = dirs::home_dir() else { return };
    let db = home.join("Library/Messages/chat.db");
    let now_unix = Local::now().timestamp();
    let now_ns = (now_unix - 978307200) * 1_000_000_000;
    let sql = format!(
        "PRAGMA busy_timeout=3000; \
         UPDATE message SET is_read=1, date_read={now_ns} \
         WHERE is_from_me=0 AND is_read=0 AND ROWID IN \
           (SELECT message_id FROM chat_message_join WHERE chat_id={chat_id});"
    );
    let _ = Command::new("sqlite3")
        .arg(db.as_os_str())
        .arg(sql)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Summarize a long message to <=8 words via Anthropic Haiku. Best-effort:
/// returns None on any failure so the truncation fallback stays in control.
fn summarize(agent: &ureq::Agent, key: &str, text: &str) -> Option<String> {
    let body = serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 60,
        "messages": [{
            "role": "user",
            "content": format!(
                "Summarize this text message in 8 words or fewer, no quotes, no preamble:\n\n{text}"
            )
        }]
    });
    let resp: serde_json::Value = agent
        .post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(body)
        .ok()?
        .into_json()
        .ok()?;
    let s = resp
        .get("content")?
        .get(0)?
        .get("text")?
        .as_str()?
        .trim()
        .trim_matches('"')
        .to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Shallow recursive walk collecting *.jsonl paths.
fn walk_jsonl(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk_jsonl(&p));
        } else if p.extension().map(|x| x == "jsonl").unwrap_or(false) {
            out.push(p);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Signal Desktop: read recent conversations from the SQLCipher-encrypted
// db.sqlite. The DB key is wrapped (Chromium safeStorage "v10") with a secret in
// the macOS Keychain; we unwrap it via `security` + `openssl` (PBKDF2-HMAC-SHA1 →
// AES-128-CBC), then read with the `sqlcipher` CLI. All shell-outs, mirroring the
// iMessage collector — no new crates. Read-only: Signal Desktop exposes no send
// API, so this card never writes (no reply / mark-read).
// ---------------------------------------------------------------------------

/// Latest real message per active conversation, newest-active first. Names come
/// from Signal's own profile data; group previews carry the sender's first name.
const SIGNAL_SQL: &str = "\
SELECT COALESCE(NULLIF(TRIM(c.name),''), NULLIF(TRIM(c.profileFullName),''), \
                NULLIF(TRIM(COALESCE(c.profileName,'')||' '||COALESCE(c.profileFamilyName,'')),''), \
                c.e164, 'Unknown') AS who, \
       c.type AS ctype, \
       m.type AS dir, \
       (m.sent_at/1000.0) AS ts, \
       (SELECT count(*) FROM messages mu WHERE mu.conversationId=c.id \
        AND mu.type='incoming' AND mu.seenStatus=1) AS unread_n, \
       COALESCE(NULLIF(TRIM(sc.profileFullName),''), NULLIF(TRIM(sc.name),''), '') AS src, \
       m.hasVisualMediaAttachments AS vis, \
       m.hasAttachments AS att, \
       replace(replace(replace(COALESCE(m.body,''), char(10),' '), char(13),' '), char(9),' ') AS body \
FROM conversations c \
JOIN messages m ON m.rowid = (SELECT rowid FROM messages mm \
     WHERE mm.conversationId=c.id AND mm.type IN ('incoming','outgoing') \
     ORDER BY mm.received_at DESC LIMIT 1) \
LEFT JOIN conversations sc ON sc.serviceId = m.sourceServiceId \
WHERE c.active_at IS NOT NULL \
ORDER BY c.active_at DESC \
LIMIT 40;";

pub fn spawn_signal(shared: Shared) {
    thread::spawn(move || {
        let Some(home) = dirs::home_dir() else { return };
        let src_db = home.join("Library/Application Support/Signal/sql/db.sqlite");

        // Derive the SQLCipher key once; re-derive only after a failed read (in
        // case the keychain secret rotated). Without it the card shows "locked".
        let mut key: Option<String> = None;

        loop {
            if key.is_none() {
                key = signal_db_key();
            }
            let result = key.as_ref().and_then(|k| {
                if !src_db.exists() {
                    return None;
                }
                // Read the live DB in place WITH its WAL (mode=ro), not a -wal-less
                // copy: Signal leaves freshly-received messages in db.sqlite-wal until
                // checkpoint, so a plain copy reads a stale snapshot (old previews +
                // "all read"). mode=ro sees the latest, no lock on Signal.
                let uri = format!("file:{}?mode=ro", src_db.display());
                read_signal(&uri, k)
            });
            let failed = result.is_none();
            {
                let mut s = shared.lock().unwrap();
                s.signal.fresh = true;
                match result {
                    Some((items, unread)) => {
                        s.signal.available = true;
                        s.signal.unread_count = unread;
                        s.signal.items = items;
                    }
                    None => s.signal.available = false,
                }
            }
            if failed {
                key = None; // re-derive next round (torn copy or rotated secret)
            }
            thread::sleep(Duration::from_secs(15));
        }
    });
}

/// Run the conversation query against the SQLCipher DB copy; parse into the same
/// `MessageItem` shape the iMessage card uses. None if the key/DB/tool is unusable.
fn read_signal(db_copy: &str, key: &str) -> Option<(Vec<crate::state::MessageItem>, u32)> {
    use crate::state::MessageItem;
    let script = format!("PRAGMA key=\"x'{key}'\";{SIGNAL_SQL}");
    let out = Command::new(resolved_tool("sqlcipher"))
        .args(["-separator", "\t", "-newline", "\n", db_copy, &script])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let now = Local::now().timestamp() as f64;
    let mut items = Vec::new();
    for line in body.lines() {
        // The PRAGMA emits a lone "ok" line (no tabs) → caught by the field guard.
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 {
            continue;
        }
        let who = f[0].to_string();
        let is_group = f[1] == "group";
        let from_me = f[2] == "outgoing";
        let ts: f64 = f[3].parse().unwrap_or(0.0);
        let unread_n: u32 = f[4].trim().parse().unwrap_or(0);
        let src = f[5].trim();
        let has_vis = f[6].trim() == "1";
        let has_att = f[7].trim() == "1";
        let raw_body = f[8].to_string();

        let (text, is_rich) = if !raw_body.trim().is_empty() {
            (raw_body, false)
        } else if has_vis {
            ("[photo]".to_string(), true)
        } else if has_att {
            ("[attachment]".to_string(), true)
        } else {
            ("[message]".to_string(), true)
        };

        let preview = {
            let p = smart_preview(&text, PREVIEW_BUDGET);
            if from_me {
                format!("You: {p}")
            } else if is_group && !src.is_empty() {
                let who_first = src.split_whitespace().next().unwrap_or(src);
                format!("{who_first}: {p}")
            } else {
                p
            }
        };

        items.push(MessageItem {
            chat_id: 0,
            rowid: 0,
            sender: who,
            handle: String::new(), // read-only: no reply target
            preview,
            full_text: text,
            ts_unix: ts,
            rel: fmt_rel((now - ts).max(0.0)),
            is_rich,
            unread: unread_n > 0,
            from_me,
            is_shortcode: false,
        });
    }
    items.truncate(SHOWN_CONVERSATIONS);
    let unread = unread_badge_count(&items, now);
    Some((items, unread))
}

/// `--diag-signal`: dump how recent INCOMING Signal messages are flagged so we can
/// see which column/value actually marks "unread" on this Signal Desktop schema
/// (the unread badge depends on getting this right).
pub fn diag_signal() {
    let Some(home) = dirs::home_dir() else {
        println!("no home");
        return;
    };
    let src_db = home.join("Library/Application Support/Signal/sql/db.sqlite");
    println!("studioboard --diag-signal\n");
    let Some(key) = signal_db_key() else {
        println!("could not derive Signal key (keychain/openssl).");
        return;
    };
    if !src_db.exists() {
        println!("Signal db not found: {}", src_db.display());
        return;
    }
    // Read the LIVE db in place with the WAL (mode=ro), not a -wal-less copy.
    let uri = format!("file:{}?mode=ro", src_db.display());
    let run = |label: &str, sql: &str| {
        let script = format!("PRAGMA key=\"x'{key}'\";{sql}");
        println!("--- {label} ---");
        match Command::new(resolved_tool("sqlcipher"))
            .args(["-separator", "\t", "-newline", "\n", &uri, &script])
            .output()
        {
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr);
                if !err.trim().is_empty() {
                    println!("  stderr: {}", err.trim());
                }
                for l in String::from_utf8_lossy(&o.stdout).lines() {
                    if l == "ok" {
                        continue; // PRAGMA echo
                    }
                    println!("  {l}");
                }
            }
            Err(e) => println!("  sqlcipher failed: {e}"),
        }
    };
    run(
        "incoming counts by seenStatus / readStatus",
        "SELECT 'seen='||COALESCE(seenStatus,-1), count(*) FROM messages WHERE type='incoming' GROUP BY seenStatus \
         UNION ALL \
         SELECT 'read='||COALESCE(readStatus,-1), count(*) FROM messages WHERE type='incoming' GROUP BY readStatus;",
    );
    run(
        "12 most-recent incoming: body | readStatus | seenStatus | when",
        "SELECT substr(replace(COALESCE(body,'[media]'),char(10),' '),1,30), \
                COALESCE(readStatus,-1), COALESCE(seenStatus,-1), \
                datetime(received_at/1000,'unixepoch','localtime') \
         FROM messages WHERE type='incoming' ORDER BY received_at DESC LIMIT 12;",
    );
}

/// Unwrap Signal's SQLCipher key: read `encryptedKey` from config.json, fetch the
/// Keychain secret, PBKDF2-derive the AES-128 key, and AES-CBC-decrypt — all via
/// the `security` and `openssl` CLIs. Returns the 64-hex-char key, or None.
fn signal_db_key() -> Option<String> {
    let home = dirs::home_dir()?;
    let base = home.join("Library/Application Support/Signal");
    let cfg = std::fs::read_to_string(base.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&cfg).ok()?;
    let enc = v.get("encryptedKey")?.as_str()?;
    // Strip the "v10" prefix (3 bytes = 6 hex chars) and hex-decode the ciphertext.
    let cipher = hex_decode(enc.get(6..)?)?;
    let cipher_path = std::env::temp_dir().join("studioboard-signal-cipher.bin");
    std::fs::write(&cipher_path, &cipher).ok()?;

    // Keychain secret Chromium's safeStorage used to wrap the DB key.
    let pw_out = Command::new("security")
        .args(["find-generic-password", "-ws", "Signal Safe Storage"])
        .output()
        .ok()?;
    if !pw_out.status.success() {
        return None;
    }
    let pw = String::from_utf8_lossy(&pw_out.stdout);
    let pw = pw.trim();
    if pw.is_empty() {
        return None;
    }

    // PBKDF2-HMAC-SHA1(pw, "saltysalt", 1003) → 16-byte AES-128 key. macOS system
    // openssl is LibreSSL (no `kdf` subcommand), so we resolve the Homebrew one.
    let openssl = resolved_tool("openssl");
    let kdf = Command::new(&openssl)
        .args([
            "kdf",
            "-keylen",
            "16",
            "-binary",
            "-kdfopt",
            "digest:SHA1",
            "-kdfopt",
            &format!("pass:{pw}"),
            "-kdfopt",
            "salt:saltysalt",
            "-kdfopt",
            "iter:1003",
            "PBKDF2",
        ])
        .output()
        .ok()?;
    if !kdf.status.success() || kdf.stdout.len() != 16 {
        return None;
    }
    let aes_hex = hex_encode(&kdf.stdout);

    // AES-128-CBC decrypt (IV = 16×0x20); plaintext is the hex key + PKCS7 pad.
    let iv = "20".repeat(16);
    let dec = Command::new(&openssl)
        .args([
            "enc",
            "-aes-128-cbc",
            "-d",
            "-K",
            &aes_hex,
            "-iv",
            &iv,
            "-nopad",
            "-in",
            cipher_path.to_str()?,
        ])
        .output()
        .ok()?;
    if !dec.status.success() {
        return None;
    }
    let plain = String::from_utf8_lossy(&dec.stdout);
    let key: String = plain.chars().filter(|c| c.is_ascii_hexdigit()).take(64).collect();
    (key.len() == 64).then_some(key)
}

/// Prefer a Homebrew-installed CLI (Apple/Intel paths) over PATH lookup — the
/// system openssl is LibreSSL and lacks `kdf`, and sqlcipher isn't a system tool.
fn resolved_tool(bin: &str) -> String {
    for p in [format!("/opt/homebrew/bin/{bin}"), format!("/usr/local/bin/{bin}")] {
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    bin.to_string()
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let b = s.as_bytes();
    let val = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i + 1 < b.len() {
        out.push((val(b[i])? << 4) | val(b[i + 1])?);
        i += 2;
    }
    Some(out)
}

fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for &x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Discord: live voice presence over the gateway (sync tungstenite, one thread)
// + recent text channels' last message over REST (another thread). Bot token and
// guild id live in the Keychain ("studioboard-discord-bot" / "-guild"). Read-only.
// ---------------------------------------------------------------------------

const DISCORD_INTENTS: u64 = 1 | (1 << 7); // GUILDS | GUILD_VOICE_STATES
const DISCORD_TEXT_SHOWN: usize = 3; // text channels surfaced on the card

/// Only these voice channels are shown (when occupied); everything else ignored.
const DISCORD_VOICE_CHANNELS: &[&str] = &["200 club", "grind time", "Back 2 Work", "WHERE STREAM"];
/// Only these text channels are surfaced, newest-active first.
const DISCORD_TEXT_CHANNELS: &[&str] = &["actual-degenery", "battlestation", "normies"];

/// Case-insensitive membership test for the channel allowlists.
fn name_allowed(list: &[&str], name: &str) -> bool {
    list.iter().any(|x| x.eq_ignore_ascii_case(name))
}

fn discord_token() -> Option<String> {
    keychain_secret("studioboard-discord-bot")
}
fn discord_guild() -> Option<String> {
    keychain_secret("studioboard-discord-guild")
}
fn keychain_secret(service: &str) -> Option<String> {
    let out = Command::new("security")
        .args(["find-generic-password", "-ws", service])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

pub fn spawn_discord(shared: Shared) {
    // Voice presence (gateway) and text channels (REST) run independently.
    {
        let sh = shared.clone();
        thread::spawn(move || discord_text_loop(sh));
    }
    thread::spawn(move || discord_voice_loop(shared));
}

// ----- text channels: poll each accessible channel's latest message over REST --

fn discord_text_loop(shared: Shared) {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(12))
        .build();
    loop {
        let cfg = discord_token().zip(discord_guild());
        if let Some((tok, guild)) = cfg {
            if let Some(text) = fetch_text_channels(&agent, &tok, &guild) {
                let mut s = shared.lock().unwrap();
                s.discord.text = text;
                s.discord.fresh = true;
                s.discord.available = true;
            }
        } else {
            let mut s = shared.lock().unwrap();
            s.discord.fresh = true;
            s.discord.available = false;
        }
        thread::sleep(Duration::from_secs(15));
    }
}

/// The N most-recently-active text channels the bot can read, each with its last
/// message (author + text). Channels the bot can't see (403) are skipped.
fn fetch_text_channels(
    agent: &ureq::Agent,
    tok: &str,
    guild: &str,
) -> Option<Vec<crate::state::TextChannel>> {
    use crate::state::TextChannel;
    let auth = format!("Bot {tok}");
    let chans: serde_json::Value = agent
        .get(&format!("https://discord.com/api/v10/guilds/{guild}/channels"))
        .set("Authorization", &auth)
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let arr = chans.as_array()?;

    // Allowlisted text channels (type 0), most-recently-active first.
    let mut text: Vec<(&serde_json::Value, u64)> = arr
        .iter()
        .filter(|c| c["type"].as_u64() == Some(0))
        .filter(|c| name_allowed(DISCORD_TEXT_CHANNELS, c["name"].as_str().unwrap_or("")))
        .map(|c| {
            let last = c["last_message_id"].as_str().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            (c, last)
        })
        .filter(|(_, last)| *last > 0)
        .collect();
    text.sort_by(|a, b| b.1.cmp(&a.1));

    let now = Local::now().timestamp() as f64;
    let mut rows: Vec<(f64, TextChannel)> = Vec::new();
    // Only probe the few most-recent channels — enough to fill the card cheaply.
    for (c, _) in text.into_iter().take(DISCORD_TEXT_SHOWN + 4) {
        let cid = c["id"].as_str().unwrap_or("");
        let name = c["name"].as_str().unwrap_or("?").to_string();
        let resp = agent
            .get(&format!("https://discord.com/api/v10/channels/{cid}/messages?limit=1"))
            .set("Authorization", &auth)
            .call();
        let Ok(r) = resp else { continue }; // 403 / no access → skip
        let Ok(msgs) = r.into_json::<serde_json::Value>() else { continue };
        let Some(m) = msgs.as_array().and_then(|a| a.first()) else { continue };
        let author = m["author"]["global_name"]
            .as_str()
            .or_else(|| m["author"]["username"].as_str())
            .unwrap_or("?")
            .to_string();
        let raw = m["content"].as_str().unwrap_or("");
        let preview = if !raw.trim().is_empty() {
            smart_preview(raw, PREVIEW_BUDGET)
        } else if let Some(att) = m["attachments"].as_array().and_then(|a| a.first()) {
            // Media post with no caption — label by attachment kind.
            let ct = att["content_type"].as_str().unwrap_or("");
            if ct.starts_with("image/") {
                "[image]".to_string()
            } else if ct.starts_with("video/") {
                "[video]".to_string()
            } else if ct.starts_with("audio/") {
                "[audio]".to_string()
            } else {
                "[attachment]".to_string()
            }
        } else if m["sticker_items"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            "[sticker]".to_string()
        } else if m["embeds"].as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            "[link]".to_string()
        } else {
            "[no text]".to_string()
        };
        let ts = m["timestamp"]
            .as_str()
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|d| d.timestamp() as f64)
            .unwrap_or(0.0);
        rows.push((
            ts,
            TextChannel { name, author, preview, rel: fmt_rel((now - ts).max(0.0)), unread: false },
        ));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Some(rows.into_iter().take(DISCORD_TEXT_SHOWN).map(|(_, r)| r).collect())
}

// ----- voice presence: a persistent gateway connection -----------------------

fn discord_voice_loop(shared: Shared) {
    loop {
        let _ = discord_gateway_session(&shared);
        // Connection dropped — clear stale presence and reconnect after a beat.
        {
            let mut s = shared.lock().unwrap();
            s.discord.voice.clear();
        }
        thread::sleep(Duration::from_secs(5));
    }
}

fn discord_gateway_session(shared: &Shared) -> Result<(), String> {
    use std::collections::HashMap;
    use tungstenite::{connect, Message};

    let tok = discord_token().ok_or("no token")?;
    let guild = discord_guild().ok_or("no guild")?;
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(8))
        .build();
    let (mut sock, _resp) =
        connect("wss://gateway.discord.gg/?v=10&encoding=json").map_err(|e| e.to_string())?;

    // Timeout reads so the heartbeat can fire even when no events are arriving.
    if let tungstenite::stream::MaybeTlsStream::Rustls(s) = sock.get_mut() {
        let _ = s.sock.set_read_timeout(Some(Duration::from_millis(800)));
    }

    let mut hb_interval = Duration::from_secs(41);
    let mut last_hb = Instant::now();
    let mut seq: Option<u64> = None;
    // voice presence: user_id -> channel_id
    let mut who: HashMap<String, String> = HashMap::new();
    // voice channel id -> name
    let mut chan: HashMap<String, String> = HashMap::new();
    // user_id -> display name (cached; resolved from the event's member object, or
    // a REST lookup when GUILD_CREATE voice states omit it).
    let mut names: HashMap<String, String> = HashMap::new();

    loop {
        if last_hb.elapsed() >= hb_interval {
            let hb = serde_json::json!({ "op": 1, "d": seq });
            sock.send(Message::Text(hb.to_string().into())).map_err(|e| e.to_string())?;
            last_hb = Instant::now();
        }

        let txt = match sock.read() {
            Ok(Message::Text(t)) => t.to_string(),
            Ok(Message::Close(_)) => return Err("closed".into()),
            Ok(_) => continue,
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) =>
            {
                continue
            }
            Err(e) => return Err(e.to_string()),
        };
        let v: serde_json::Value = match serde_json::from_str(&txt) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(s) = v["s"].as_u64() {
            seq = Some(s);
        }
        match v["op"].as_u64() {
            Some(10) => {
                // HELLO → adopt heartbeat interval, identify.
                if let Some(ms) = v["d"]["heartbeat_interval"].as_u64() {
                    hb_interval = Duration::from_millis(ms);
                }
                let identify = serde_json::json!({
                    "op": 2,
                    "d": {
                        "token": tok,
                        "intents": DISCORD_INTENTS,
                        "properties": {"os":"macos","browser":"studioboard","device":"studioboard"}
                    }
                });
                sock.send(Message::Text(identify.to_string().into())).map_err(|e| e.to_string())?;
                last_hb = Instant::now();
            }
            Some(1) => {
                // Server asked us to heartbeat immediately.
                let hb = serde_json::json!({ "op": 1, "d": seq });
                sock.send(Message::Text(hb.to_string().into())).map_err(|e| e.to_string())?;
                last_hb = Instant::now();
            }
            Some(7) | Some(9) => return Err("reconnect requested".into()),
            Some(0) => match v["t"].as_str().unwrap_or("") {
                "READY" => {
                    let mut s = shared.lock().unwrap();
                    s.discord.fresh = true;
                    s.discord.available = true;
                }
                "GUILD_CREATE" if v["d"]["id"].as_str() == Some(guild.as_str()) => {
                    chan.clear();
                    who.clear();
                    if let Some(chs) = v["d"]["channels"].as_array() {
                        for c in chs {
                            if c["type"].as_u64() == Some(2) {
                                chan.insert(
                                    c["id"].as_str().unwrap_or("").to_string(),
                                    c["name"].as_str().unwrap_or("?").to_string(),
                                );
                            }
                        }
                    }
                    if let Some(states) = v["d"]["voice_states"].as_array() {
                        for st in states {
                            apply_voice_state(&mut who, &mut names, st);
                        }
                    }
                    publish_voice(shared, &who, &chan, &mut names, &agent, &tok, &guild, false);
                }
                "VOICE_STATE_UPDATE" if v["d"]["guild_id"].as_str() == Some(guild.as_str()) => {
                    apply_voice_state(&mut who, &mut names, &v["d"]);
                    publish_voice(shared, &who, &chan, &mut names, &agent, &tok, &guild, true);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

/// Fold one voice state into the user→channel map (channel_id null ⇒ disconnected).
/// Caches the display name when the event carries a `member` object (VOICE_STATE_
/// UPDATE does; the GUILD_CREATE snapshot usually doesn't — resolved later by REST).
fn apply_voice_state(
    who: &mut std::collections::HashMap<String, String>,
    names: &mut std::collections::HashMap<String, String>,
    st: &serde_json::Value,
) {
    let Some(uid) = st["user_id"].as_str() else { return };
    if let Some(n) = member_display_name(&st["member"]) {
        names.insert(uid.to_string(), n);
    }
    match st["channel_id"].as_str() {
        Some(cid) if !cid.is_empty() => {
            who.insert(uid.to_string(), cid.to_string());
        }
        _ => {
            who.remove(uid);
        }
    }
}

/// Display name from a member object: server nick › global name › username.
fn member_display_name(member: &serde_json::Value) -> Option<String> {
    member["nick"]
        .as_str()
        .or_else(|| member["user"]["global_name"].as_str())
        .or_else(|| member["user"]["username"].as_str())
        .map(|s| s.to_string())
}

/// Rebuild the card's occupied-voice-channel list, resolving any unknown member
/// names via a cached REST lookup (works without the privileged Members intent).
fn publish_voice(
    shared: &Shared,
    who: &std::collections::HashMap<String, String>,
    chan: &std::collections::HashMap<String, String>,
    names: &mut std::collections::HashMap<String, String>,
    agent: &ureq::Agent,
    tok: &str,
    guild: &str,
    detect_joins: bool, // true for live VOICE_STATE_UPDATE; false for the GUILD_CREATE snapshot
) {
    use crate::state::VoiceChannel;
    use std::collections::BTreeMap;
    // Group members by channel (allowlisted only), ordered by name for stability.
    let mut by_chan: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (uid, cid) in who.iter() {
        let cname = chan.get(cid).cloned().unwrap_or_else(|| "voice".to_string());
        if !name_allowed(DISCORD_VOICE_CHANNELS, &cname) {
            continue; // ignore voice channels we don't care about
        }
        if !names.contains_key(uid) {
            if let Some(n) = fetch_member_name(agent, tok, guild, uid) {
                names.insert(uid.clone(), n);
            }
        }
        let who_name = names.get(uid).cloned().unwrap_or_else(|| "someone".to_string());
        by_chan.entry(cname).or_default().push(who_name);
    }
    let voice: Vec<VoiceChannel> = by_chan
        .into_iter()
        .map(|(name, mut members)| {
            members.sort();
            VoiceChannel { name, members }
        })
        .collect();
    let mut s = shared.lock().unwrap();
    // A new name present now that wasn't connected before = a join → 20s shimmer.
    // Only on live updates; the GUILD_CREATE snapshot would otherwise fire on every
    // (re)connect.
    if detect_joins {
        let prev: std::collections::HashSet<String> = s
            .discord
            .voice
            .iter()
            .flat_map(|c| c.members.iter().cloned())
            .collect();
        let joined = voice
            .iter()
            .flat_map(|c| &c.members)
            .any(|m| !prev.contains(m));
        if joined {
            s.discord.voice_join_at = Some(Instant::now());
        }
    }
    s.discord.voice = voice;
}

/// REST: GET /guilds/{guild}/members/{uid} → display name. None on any failure.
fn fetch_member_name(agent: &ureq::Agent, tok: &str, guild: &str, uid: &str) -> Option<String> {
    let v: serde_json::Value = agent
        .get(&format!("https://discord.com/api/v10/guilds/{guild}/members/{uid}"))
        .set("Authorization", &format!("Bot {tok}"))
        .call()
        .ok()?
        .into_json()
        .ok()?;
    member_display_name(&v)
}

// ----------------------------------------------------------------------------
// mac-doctor / syswatch: the on-call triage agent's live status.
//
// The watchdog writes everything to ~/Library/Application Support/syswatch:
//   • diagnose.lock  — a dir that exists only while a diagnosis is in flight
//   • syswatch.log   — human-readable "[diagnose] <step>" lines as a run proceeds
//   • syswatch.db    — one `incidents` row per completed run (verdict, cost, …)
// We poll all three read-only (zero locking contention; WAL handles it).
// ----------------------------------------------------------------------------

const DOCTOR_SQL: &str = "\
SELECT COALESCE(severity,''), COALESCE(outcome,''), COALESCE(model,''), \
REPLACE(REPLACE(COALESCE(title,''),char(9),' '),char(10),' '), \
REPLACE(REPLACE(COALESCE(trigger_reasons,''),char(9),' '),char(10),' · '), \
COALESCE(cost_usd,0), \
REPLACE(REPLACE(COALESCE(actions_taken,'[]'),char(9),' '),char(10),' '), \
epoch, (SELECT COUNT(*) FROM incidents), \
(SELECT COALESCE(ROUND(SUM(cost_usd),2),0) FROM incidents WHERE date(ts)=date('now','localtime')) \
FROM incidents ORDER BY id DESC LIMIT 1;";

fn syswatch_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join("Library/Application Support/syswatch")
}

/// Last `n` lines of a (possibly large) text file, read from a tail window so we
/// never slurp the whole log. None if the file is missing.
fn tail_lines(path: &std::path::Path, n: usize) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let start = len.saturating_sub(16 * 1024);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();
    Some(lines[lines.len().saturating_sub(n)..].join("\n"))
}

fn rel_short(secs: i64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

pub fn spawn_doctor(shared: Shared) {
    thread::spawn(move || loop {
        let snap = read_doctor();
        {
            let mut s = shared.lock().unwrap();
            s.doctor = snap;
        }
        thread::sleep(Duration::from_millis(1500));
    });
}

fn read_doctor() -> crate::state::Doctor {
    use crate::state::Doctor;
    let dir = syswatch_dir();
    let db = dir.join("syswatch.db");
    let mut d = Doctor::default();
    if !db.exists() {
        return d; // syswatch not installed → card shows a graceful hint
    }
    d.available = true;
    d.running = dir.join("diagnose.lock").exists();

    // Live step + (while running) the breach that triggered the run, from the log.
    if let Some(tail) = tail_lines(&dir.join("syswatch.log"), 120) {
        for line in tail.lines().rev() {
            let Some(ix) = line.find("[diagnose]") else { continue };
            let msg = line[ix + "[diagnose]".len()..].trim();
            if d.step.is_empty() {
                d.step = msg.to_string();
            }
            if d.trigger.is_empty() {
                if let Some(r) = msg.split("reasons=").nth(1) {
                    d.trigger = r.trim().to_string();
                }
            }
            if !d.step.is_empty() && !d.trigger.is_empty() {
                break;
            }
        }
    }

    // Latest incident + lifetime/today aggregates, one round-trip.
    if let Some(out) = Command::new("sqlite3")
        .args(["-separator", "\t", "-newline", "\n", &db.to_string_lossy(), DOCTOR_SQL])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let body = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = body.lines().next() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 10 {
                d.last_severity = f[0].trim().to_string();
                d.last_outcome = f[1].trim().to_string();
                d.last_model = f[2].trim().to_string();
                d.last_title = f[3].trim().to_string();
                if d.trigger.is_empty() {
                    d.trigger = f[4].trim().to_string();
                }
                d.last_actions =
                    serde_json::from_str::<Vec<String>>(f[6].trim()).unwrap_or_default();
                let epoch: i64 = f[7].trim().parse().unwrap_or(0);
                d.incidents_total = f[8].trim().parse().unwrap_or(0);
                d.today_cost = f[9].trim().parse().unwrap_or(0.0);
                if epoch > 0 {
                    d.last_rel = rel_short((Local::now().timestamp() - epoch).max(0));
                }
            }
        }
    }

    // Preview override: STUDIOBOARD_FAKE_DOCTOR=running forces the diagnosing
    // state live (overlaid on real incident data) so the in-flight card can be
    // seen without waiting for an actual threshold breach.
    if std::env::var("STUDIOBOARD_FAKE_DOCTOR").map(|v| v == "running").unwrap_or(false) {
        d.available = true;
        d.running = true;
        if d.step.is_empty() {
            d.step = "local triage (qwen2.5:14b)…".to_string();
        }
        if d.trigger.is_empty() {
            d.trigger = "runaway: rustc at 356% ≥ 220%".to_string();
        }
    }
    d
}

// ----------------------------------------------------------------------------
// Hammerspoon keybinds: mirror the live cheat sheet.
//
// init.lua exports its self-documenting `doc` registry to
// ~/Library/Application Support/studioboard/keybinds.json on every reload. We
// read + parse it (poll on mtime) so the KEYBINDS card always matches the real
// bindings with no second list to maintain.
// ----------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct KbBind {
    group: String,
    keys: String,
    desc: String,
}
#[derive(serde::Deserialize)]
struct KbExport {
    #[serde(default)]
    hyper: String,
    #[serde(default)]
    group_order: Vec<String>,
    #[serde(default, rename = "groupOrder")]
    group_order_camel: Vec<String>,
    #[serde(default)]
    binds: Vec<KbBind>,
}

fn keybinds_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join("Library/Application Support/studioboard/keybinds.json")
}

pub fn spawn_keybinds(shared: Shared) {
    thread::spawn(move || loop {
        let snap = read_keybinds();
        {
            let mut s = shared.lock().unwrap();
            s.keybinds = snap;
        }
        thread::sleep(Duration::from_secs(3));
    });
}

fn read_keybinds() -> crate::state::Keybinds {
    use crate::state::{KeyGroup, Keybinds};
    let mut kb = Keybinds::default();
    let Ok(raw) = std::fs::read_to_string(keybinds_path()) else {
        return kb; // Hammerspoon hasn't exported yet → card shows a hint
    };
    let Ok(export) = serde_json::from_str::<KbExport>(&raw) else {
        return kb;
    };
    kb.available = true;
    kb.hyper = export.hyper;

    // Group the flat bind list, preserving first-seen row order within a group.
    let mut order: Vec<String> = if !export.group_order_camel.is_empty() {
        export.group_order_camel
    } else {
        export.group_order
    };
    let mut groups: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
    for b in export.binds {
        if !order.contains(&b.group) {
            order.push(b.group.clone()); // any group missing from the order list
        }
        groups.entry(b.group).or_default().push((b.keys, b.desc));
    }
    kb.groups = order
        .into_iter()
        .filter_map(|name| {
            groups.remove(&name).map(|binds| KeyGroup { name, binds })
        })
        .filter(|g| !g.binds.is_empty())
        .collect();
    kb
}
