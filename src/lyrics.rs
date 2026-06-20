//! Time-synced lyrics via LRCLIB (https://lrclib.net) — free, open, no key.
//!
//! Speed strategy:
//!   * exact `/api/get` and fuzzy `/api/search` race **concurrently**; we
//!     return the instant *either* yields synced lyrics (no waiting on the
//!     slow one),
//!   * a shared pooled `ureq::Agent` avoids per-call TLS handshakes,
//!   * synced results are cached to disk (`~/.cache/studioboard/lyrics`), so a
//!     song heard once loads instantly forever — even across restarts.

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
            let _ = tx.send(Msg::Get(best_get(&a, &ar, &tr, &al, duration)));
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

/// LRCLIB's exact `/api/get` is precise but brittle: it 404s when the
/// artist/track/album don't match its record verbatim. Apple Music reports
/// titles like "FE!N (feat. Playboi Carti)" and albums like "BULLY - DELUXE",
/// neither of which LRCLIB stores — so we try the verbatim query first, then a
/// few progressively-cleaned variants (drop the feat, drop the album, use the
/// primary artist). First variant with synced/instrumental lyrics wins; a plain
/// record is kept as a last-ditch fallback.
fn best_get(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
    duration: f64,
) -> Option<serde_json::Value> {
    let primary = primary_artist(artist);
    let ct = clean_title(track);
    let ca = clean_album(album);
    let variants: [(&str, &str, &str); 4] = [
        (artist, track, album),       // exactly as Apple reports it
        (&primary, &ct, ""),          // cleaned title, no album (most forgiving)
        (&primary, &ct, &ca),         // cleaned title + cleaned album
        (artist, &ct, ""),            // full artist, cleaned title
    ];
    let mut seen: Vec<(String, String, String)> = Vec::new();
    let mut fallback: Option<serde_json::Value> = None;
    for (ar, tr, al) in variants {
        let key = (ar.to_string(), tr.to_string(), al.to_string());
        if seen.contains(&key) {
            continue; // skip variants that collapsed to an earlier one
        }
        seen.push(key);
        if let Some(v) = api_get(agent, ar, tr, al, duration) {
            if v.get("instrumental").and_then(|x| x.as_bool()).unwrap_or(false)
                || synced_from(&v).is_some()
            {
                return Some(v);
            }
            if fallback.is_none() {
                fallback = Some(v); // remember the first plain-only hit
            }
        }
    }
    fallback
}

fn api_get(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    album: &str,
    duration: f64,
) -> Option<serde_json::Value> {
    let dur = duration.round() as i64;
    let mut req = agent
        .get("https://lrclib.net/api/get")
        .query("artist_name", artist)
        .query("track_name", track)
        .query("duration", &dur.to_string())
        .set("User-Agent", UA);
    if !album.is_empty() {
        req = req.query("album_name", album);
    }
    req.call().ok()?.into_json().ok()
}

/// The primary performer: Apple joins collaborators with "&", ",", "feat.", "x",
/// etc.; LRCLIB usually files a track under just the lead artist, so we match on
/// that. Returns the original string when there is no separator.
fn primary_artist(artist: &str) -> String {
    let lower = artist.to_lowercase();
    let seps = [" feat.", " feat ", " featuring", " ft.", " ft ", " with ", " & ", ", ", " x ", " / "];
    let mut cut = artist.len();
    for sep in seps {
        if let Some(i) = lower.find(sep) {
            cut = cut.min(i);
        }
    }
    artist[..cut].trim().to_string()
}

/// Strip the noise Apple appends to titles that LRCLIB does not store: trailing
/// "(feat. …)" / "(with …)" / "(prod. …)" groups and " - <Remaster/Single/…>"
/// dash-suffixes. Album-style "(Deluxe)" parentheses are also dropped.
fn clean_title(track: &str) -> String {
    strip_dash_suffix(&strip_noise_parens(track)).trim().to_string()
}

/// Albums carry the same edition cruft ("BULLY - DELUXE", "Utopia (Deluxe)").
fn clean_album(album: &str) -> String {
    strip_dash_suffix(&strip_noise_parens(album)).trim().to_string()
}

/// Remove any "(…)" / "[…]" group whose contents look like a credit or edition
/// note (feat/with/prod/deluxe/remaster/version/etc.), keeping real parenthetical
/// title words intact.
fn strip_noise_parens(s: &str) -> String {
    const NOISE: [&str; 12] = [
        "feat", "ft", "featuring", "with", "prod", "deluxe", "remaster",
        "version", "edition", "bonus", "remix", "mix",
    ];
    let mut out = String::with_capacity(s.len());
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let close = match c {
            '(' => Some(')'),
            '[' => Some(']'),
            _ => None,
        };
        if let Some(cl) = close {
            let start = i + c.len_utf8();
            if let Some(rel) = s[start..].find(cl) {
                let inner = &s[start..start + rel];
                if NOISE.iter().any(|n| inner.to_lowercase().contains(n)) {
                    let skip_to = start + rel + cl.len_utf8();
                    while chars.peek().map_or(false, |&(j, _)| j < skip_to) {
                        chars.next();
                    }
                    continue; // drop the whole credit/edition group
                }
            }
        }
        out.push(c);
    }
    out
}

/// Drop a " - <noise>" suffix (Remastered 2011, Single, Deluxe, Mono, Live, …)
/// while leaving meaningful dashes in a title alone.
fn strip_dash_suffix(s: &str) -> String {
    if let Some(idx) = s.rfind(" - ") {
        let tail = s[idx + 3..].to_lowercase();
        const NOISE: [&str; 12] = [
            "remaster", "single", "deluxe", "version", "mono", "stereo", "edit",
            "mix", "bonus", "live", "edition", "anniversary",
        ];
        if NOISE.iter().any(|n| tail.contains(n)) {
            return s[..idx].to_string();
        }
    }
    s.to_string()
}

/// Fuzzy fallback when the exact get misses. Searches the cleaned title scoped
/// to the primary artist first (precise), then the cleaned title alone (broad),
/// and ranks synced candidates by artist agreement *then* duration closeness —
/// so a same-length cover by the wrong artist can't beat the real recording.
fn search_synced(
    agent: &ureq::Agent,
    artist: &str,
    track: &str,
    duration: f64,
) -> Option<Vec<LyricLine>> {
    let primary = primary_artist(artist);
    let ct = clean_title(track);
    // (track_query, artist_query) attempts, broadening as we go.
    let attempts: [(&str, &str); 2] = [(&ct, &primary), (&ct, "")];
    let mut seen: Vec<(String, String)> = Vec::new();
    for (tr, ar) in attempts {
        let key = (tr.to_string(), ar.to_string());
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        if let Some(lines) = search_once(agent, ar, tr, &primary, duration) {
            return Some(lines);
        }
    }
    None
}

fn search_once(
    agent: &ureq::Agent,
    artist_query: &str,
    track_query: &str,
    primary: &str,
    duration: f64,
) -> Option<Vec<LyricLine>> {
    let mut req = agent
        .get("https://lrclib.net/api/search")
        .query("track_name", track_query)
        .set("User-Agent", UA);
    if !artist_query.is_empty() {
        req = req.query("artist_name", artist_query);
    }
    let arr: serde_json::Value = req.call().ok()?.into_json().ok()?;
    let results = arr.as_array()?;

    let pl = primary.to_lowercase();
    // Lower score is better: artist mismatch is penalised far above any plausible
    // duration gap, so duration only breaks ties among same-artist candidates.
    let mut best: Option<(f64, Vec<LyricLine>)> = None;
    for r in results {
        let synced = r.get("syncedLyrics").and_then(|x| x.as_str()).unwrap_or("");
        if synced.trim().is_empty() {
            continue;
        }
        let lines = parse_lrc(synced);
        if lines.is_empty() {
            continue;
        }
        let rdur = r.get("duration").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let rartist = r.get("artistName").and_then(|x| x.as_str()).unwrap_or("").to_lowercase();
        let artist_ok = pl.is_empty() || rartist.contains(&pl) || pl.contains(&rartist);
        let score = (rdur - duration).abs() + if artist_ok { 0.0 } else { 10_000.0 };
        match &best {
            Some((bscore, _)) if *bscore <= score => {}
            _ => best = Some((score, lines)),
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
// Stored as standard `.lrc` files under `~/.cache/studioboard/lyrics/`, keyed by
// the shared track hash (see `crate::cache`) so a song heard once loads instantly
// forever — even across restarts.

const KIND: &str = "lyrics";

/// Load synced lyrics for a track from disk, if present.
pub fn cache_load(track_id: &str) -> Option<Lyrics> {
    let body = crate::cache::get_bytes(KIND, track_id, "lrc")?;
    let lines = parse_lrc(&String::from_utf8_lossy(&body));
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
    let mut body = String::new();
    for l in &ly.lines {
        let m = (l.t / 60.0).floor() as i64;
        let s = l.t - (m as f64) * 60.0;
        body.push_str(&format!("[{:02}:{:05.2}]{}\n", m, s, l.text));
    }
    crate::cache::put_bytes(KIND, track_id, "lrc", body.as_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_artist_splits() {
        assert_eq!(primary_artist("Kanye West & Lauryn Hill"), "Kanye West");
        assert_eq!(primary_artist("Travis Scott"), "Travis Scott");
        assert_eq!(primary_artist("Drake feat. Rihanna"), "Drake");
        assert_eq!(primary_artist("A, B, C"), "A");
        assert_eq!(primary_artist("Tyler, The Creator"), "Tyler"); // known caveat
    }

    #[test]
    fn clean_title_strips_credits_and_editions() {
        assert_eq!(clean_title("FE!N (feat. Playboi Carti)"), "FE!N");
        assert_eq!(clean_title("Song [feat. X] "), "Song");
        assert_eq!(clean_title("Tune - Remastered 2011"), "Tune");
        assert_eq!(clean_title("Track - Single"), "Track");
        // Real parentheses that aren't credit/edition noise survive (kept
        // conservative so "(Alive)" etc. are never mistaken for "(Live)").
        assert_eq!(clean_title("Marvins Room (Live)"), "Marvins Room (Live)");
        assert_eq!(clean_title("Hello (Acoustic)"), "Hello (Acoustic)");
        // Mix/edit suffixes are edition noise (broke "Shook Ones, Pt. II (Mixed)").
        assert_eq!(clean_title("Shook Ones, Pt. II (Mixed)"), "Shook Ones, Pt. II");
        assert_eq!(clean_title("Song (Club Mix)"), "Song");
        assert_eq!(clean_album("J.PERIOD Presents CLASS OF 95 (DJ Mix)"), "J.PERIOD Presents CLASS OF 95");
    }

    #[test]
    fn clean_album_strips_deluxe() {
        assert_eq!(clean_album("BULLY - DELUXE"), "BULLY");
        assert_eq!(clean_album("Utopia (Deluxe)"), "Utopia");
        assert_eq!(clean_album("Nothing Was the Same (Deluxe)"), "Nothing Was the Same");
    }

    #[test]
    fn strip_noise_parens_is_utf8_safe() {
        // Must not panic or corrupt multibyte input.
        assert_eq!(strip_noise_parens("¥$ café (feat. Ty)"), "¥$ café ");
        assert_eq!(strip_noise_parens("naïve"), "naïve");
    }
}

#[cfg(test)]
mod net_tests {
    use super::*;
    // Live LRCLIB call — run with: cargo test --release -- --ignored e2e
    #[test]
    #[ignore]
    fn e2e_feat_track_now_resolves() {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(4))
            .timeout_read(std::time::Duration::from_secs(13))
            .build();
        // Apple-style metadata that the OLD verbatim get 404s on.
        let ly = fetch(&agent, "Travis Scott", "FE!N (feat. Playboi Carti)", "UTOPIA", 191.0, "t1");
        println!("synced={} lines={} note={:?}", ly.synced, ly.lines.len(), ly.note);
        assert!(ly.synced && !ly.lines.is_empty(), "expected synced lyrics, got note={:?}", ly.note);
    }
}
