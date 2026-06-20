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
const ART_THUMB: u32 = 64;

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
                    let out = Command::new("osascript")
                        .arg("-e")
                        .arg(artwork_script(&path_str))
                        .output();
                    let ok = out
                        .as_ref()
                        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "OK")
                        .unwrap_or(false);
                    let art = if ok {
                        decode_art(&path, &id)
                    } else {
                        crate::state::AlbumArt { track_id: id.clone(), ..Default::default() }
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

fn decode_art(path: &std::path::Path, id: &str) -> crate::state::AlbumArt {
    use image::imageops::FilterType;
    let mut art = crate::state::AlbumArt { track_id: id.to_string(), ..Default::default() };
    // Decode from bytes so format is detected by magic, not file extension
    // (Music writes a raw PNG/JPEG to a .dat path).
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(img) = image::load_from_memory(&bytes) {
            let small = img.resize_exact(ART_THUMB, ART_THUMB, FilterType::Triangle).to_rgb8();
            art.w = small.width() as usize;
            art.h = small.height() as usize;
            art.px = small.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        }
    }
    art
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

/// Recent inbound messages, newest first. `quote(attributedBody)` emits a single
/// `X'..'` hex literal so no embedded tab/newline from the BLOB can break TSV
/// parsing. We only fetch the hex when `text` is empty (modern macOS stores the
/// real string in attributedBody for ~9% of messages).
const MSG_SQL: &str = "\
SELECT m.ROWID, \
       (m.date/1000000000.0 + 978307200) AS ts, \
       m.is_read, \
       COALESCE(h.id,'') AS handle, \
       COALESCE(m.text,'') AS text, \
       CASE WHEN (m.text IS NULL OR m.text='') AND m.attributedBody IS NOT NULL \
            THEN quote(m.attributedBody) ELSE '' END AS ab \
FROM message m \
LEFT JOIN handle h ON m.handle_id = h.ROWID \
WHERE m.is_from_me = 0 \
ORDER BY m.date DESC \
LIMIT 40;";

const UNREAD_SQL: &str =
    "SELECT count(*) FROM message WHERE is_from_me=0 AND is_read=0;";

/// Beyond this many display chars a preview is summarized (if a key is present)
/// or smart-truncated to keep the card readable.
const PREVIEW_BUDGET: usize = 96;

pub fn spawn_messages(shared: Shared) {
    thread::spawn(move || {
        let Some(home) = dirs::home_dir() else { return };
        let db = home.join("Library/Messages/chat.db");
        let uri = format!("file:{}?immutable=1", db.display());

        // Network agent + per-ROWID summary cache, reused across polls.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .timeout_read(Duration::from_secs(12))
            .build();
        let mut summary_cache: std::collections::HashMap<i64, String> = Default::default();

        loop {
            match read_messages(&uri) {
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

/// Run the two queries; return (items newest-first, unread_count). None if the
/// DB can't be read (no Full Disk Access) so the panel shows a graceful hint.
fn read_messages(uri: &str) -> Option<(Vec<crate::state::MessageItem>, u32)> {
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
    for line in body.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 6 {
            continue;
        }
        let rowid: i64 = f[0].parse().unwrap_or(0);
        let ts: f64 = f[1].parse().unwrap_or(0.0);
        let is_read: i64 = f[2].parse().unwrap_or(1);
        let handle = f[3].to_string();
        let text = f[4].to_string();
        let ab = f[5];

        let (full_text, is_rich) = if !text.trim().is_empty() {
            (text, false)
        } else if !ab.is_empty() {
            match decode_attributed_body(ab) {
                Some(t) if !t.trim().is_empty() => (t, false),
                _ => ("[rich message]".to_string(), true),
            }
        } else {
            ("[rich message]".to_string(), true)
        };

        let sender = pretty_handle(&handle);
        let preview = smart_preview(&full_text, PREVIEW_BUDGET);
        items.push(MessageItem {
            rowid,
            sender,
            handle,
            preview,
            full_text,
            ts_unix: ts,
            rel: fmt_rel((now - ts).max(0.0)),
            is_rich,
            unread: is_read == 0,
        });
    }

    // Unread count (cheap separate query; cap to keep the badge sane).
    let unread = Command::new("sqlite3")
        .args([uri, UNREAD_SQL])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u32>().ok())
        .unwrap_or_else(|| items.iter().filter(|i| i.unread).count() as u32);

    Some((items, unread))
}

/// Make a raw handle (phone/email) friendlier. We don't have AddressBook access
/// here; keep the email local-part or the last digits of a phone number so the
/// fixed sender column reads cleanly.
fn pretty_handle(h: &str) -> String {
    if h.is_empty() {
        return "Unknown".into();
    }
    if let Some((user, _dom)) = h.split_once('@') {
        return user.to_string();
    }
    h.to_string()
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

/// Scan a byte buffer for the longest contiguous run of printable UTF-8 text.
fn longest_printable_run(bytes: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(bytes);
    let mut best = "";
    let mut cur_start = 0usize;
    let mut in_run = false;
    let mut run_start = 0usize;
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let printable = |c: char| c == ' ' || (!c.is_control() && c != '\u{fffd}');
    for &(idx, ch) in &chars {
        if printable(ch) {
            if !in_run {
                in_run = true;
                run_start = idx;
                cur_start = idx;
            }
        } else if in_run {
            in_run = false;
            let run = &s[run_start..idx];
            if run.trim().len() > best.trim().len() {
                best = &s[cur_start..idx];
            }
        }
    }
    if in_run {
        let run = &s[run_start..];
        if run.trim().len() > best.trim().len() {
            best = run;
        }
    }
    let t = best.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
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
