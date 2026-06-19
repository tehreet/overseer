//! Time-synced lyrics via LRCLIB (https://lrclib.net) — free, open, no key.
//!
//! Speed strategy:
//!   * exact `/api/get` and fuzzy `/api/search` race **concurrently**; we
//!     return the instant *either* yields synced lyrics (no waiting on the
//!     slow one),
//!   * a shared pooled `ureq::Agent` avoids per-call TLS handshakes,
//!   * synced results are cached to disk (`~/.cache/studioboard/lyrics`), so a
//!     song heard once loads instantly forever — even across restarts.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::state::{LyricLine, Lyrics};

const UA: &str = "studioboard (https://github.com/local/studioboard)";
const FETCH_BUDGET: Duration = Duration::from_secs(13);

enum Msg {
    Get(Option<serde_json::Value>),
    Search(Option<Vec<LyricLine>>),
}

pub fn fetch(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
    duration: f64,
    track_id: &str,
) -> Lyrics {
    let mut out = Lyrics {
        lines: Vec::new(),
        synced: false,
        track_id: track_id.to_string(),
        note: String::new(),
    };
    if track.is_empty() {
        out.note = "no track".into();
        return out;
    }

    let (tx, rx) = mpsc::channel();
    {
        let (a, ar, tr, al, tx) =
            (agent.clone(), artist.to_string(), track.to_string(), album.to_string(), tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Get(api_get(&a, &ar, &tr, &al, duration)));
        });
    }
    {
        let (a, ar, tr) = (agent.clone(), artist.to_string(), track.to_string());
        std::thread::spawn(move || {
            let _ = tx.send(Msg::Search(search_synced(&a, &ar, &tr, duration)));
        });
    }

    // Take the first synced result from either source.
    let mut exact: Option<serde_json::Value> = None;
    let deadline = Instant::now() + FETCH_BUDGET;
    let mut received = 0;
    while received < 2 {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match rx.recv_timeout(deadline - now) {
            Ok(Msg::Get(v)) => {
                received += 1;
                if let Some(ev) = &v {
                    if ev.get("instrumental").and_then(|x| x.as_bool()).unwrap_or(false) {
                        out.note = "🎵 instrumental".into();
                        return out;
                    }
                    if let Some(lines) = synced_from(ev) {
                        out.lines = lines;
                        out.synced = true;
                        return out;
                    }
                }
                exact = v;
            }
            Ok(Msg::Search(v)) => {
                received += 1;
                if let Some(lines) = v {
                    out.lines = lines;
                    out.synced = true;
                    return out;
                }
            }
            Err(_) => break,
        }
    }

    // No synced anywhere — fall back to plain from the exact record.
    if let Some(ev) = &exact {
        if let Some(p) = ev.get("plainLyrics").and_then(|x| x.as_str()) {
            if !p.trim().is_empty() {
                out.lines = p
                    .lines()
                    .map(|l| LyricLine { t: -1.0, text: l.trim().to_string() })
                    .collect();
                out.note = "plain (no synced found)".into();
                return out;
            }
        }
    }
    out.note = "no lyrics found".into();
    out
}

fn api_get(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
    duration: f64,
) -> Option<serde_json::Value> {
    let dur = duration.round() as i64;
    agent
        .get("https://lrclib.net/api/get")
        .query("artist_name", artist)
        .query("track_name", track)
        .query("album_name", album)
        .query("duration", &dur.to_string())
        .set("User-Agent", UA)
        .call()
        .ok()?
        .into_json()
        .ok()
}

fn search_synced(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    duration: f64,
) -> Option<Vec<LyricLine>> {
    let arr: serde_json::Value = agent
        .get("https://lrclib.net/api/search")
        .query("track_name", track)
        .query("artist_name", artist)
        .set("User-Agent", UA)
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let results = arr.as_array()?;

    let mut best: Option<(f64, Vec<LyricLine>)> = None;
    for r in results {
        let synced = r.get("syncedLyrics").and_then(|x| x.as_str()).unwrap_or("");
        if synced.trim().is_empty() {
            continue;
        }
        let rdur = r.get("duration").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let dist = (rdur - duration).abs();
        let lines = parse_lrc(synced);
        if lines.is_empty() {
            continue;
        }
        match &best {
            Some((bdist, _)) if *bdist <= dist => {}
            _ => best = Some((dist, lines)),
        }
    }
    best.map(|(_, l)| l)
}

fn synced_from(v: &serde_json::Value) -> Option<Vec<LyricLine>> {
    let s = v.get("syncedLyrics").and_then(|x| x.as_str())?;
    if s.trim().is_empty() {
        return None;
    }
    let lines = parse_lrc(s);
    if lines.is_empty() {
        None
    } else {
        Some(lines)
    }
}

// --- disk cache ------------------------------------------------------------

fn cache_dir() -> Option<PathBuf> {
    let d = dirs::cache_dir()?.join("studioboard").join("lyrics");
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

fn cache_path(track_id: &str) -> Option<PathBuf> {
    let mut h = DefaultHasher::new();
    track_id.hash(&mut h);
    Some(cache_dir()?.join(format!("{:016x}.lrc", h.finish())))
}

/// Load synced lyrics for a track from disk, if present.
pub fn cache_load(track_id: &str) -> Option<Lyrics> {
    let path = cache_path(track_id)?;
    let body = std::fs::read_to_string(path).ok()?;
    let lines = parse_lrc(&body);
    if lines.is_empty() {
        return None;
    }
    Some(Lyrics { lines, synced: true, track_id: track_id.to_string(), note: String::new() })
}

/// Persist synced lyrics to disk as a standard `.lrc` file.
pub fn cache_save(track_id: &str, ly: &Lyrics) {
    if !ly.synced || ly.lines.is_empty() {
        return;
    }
    let Some(path) = cache_path(track_id) else { return };
    let mut body = String::new();
    for l in &ly.lines {
        let m = (l.t / 60.0).floor() as i64;
        let s = l.t - (m as f64) * 60.0;
        body.push_str(&format!("[{:02}:{:05.2}]{}\n", m, s, l.text));
    }
    let _ = std::fs::write(path, body);
}

/// Parse an LRC blob: lines like `[01:23.45] text`, possibly multiple tags
/// per line. Sorted by timestamp.
fn parse_lrc(s: &str) -> Vec<LyricLine> {
    let mut lines = Vec::new();
    for raw in s.lines() {
        let mut rest = raw;
        let mut stamps: Vec<f64> = Vec::new();
        while rest.starts_with('[') {
            if let Some(close) = rest.find(']') {
                let tag = &rest[1..close];
                if let Some(t) = parse_stamp(tag) {
                    stamps.push(t);
                }
                rest = &rest[close + 1..];
            } else {
                break;
            }
        }
        let text = rest.trim().to_string();
        for t in stamps {
            lines.push(LyricLine { t, text: text.clone() });
        }
    }
    lines.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    lines
}

/// Parse `mm:ss.xx` or `mm:ss` into seconds. Ignores non-time metadata tags.
fn parse_stamp(tag: &str) -> Option<f64> {
    let (m, rest) = tag.split_once(':')?;
    let minutes: f64 = m.trim().parse().ok()?;
    let seconds: f64 = rest.trim().parse().ok()?;
    Some(minutes * 60.0 + seconds)
}
