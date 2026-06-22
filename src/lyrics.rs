//! Time-synced lyrics via a small provider chain — all free, no key.
//!
//!   1. **LRCLIB** (https://lrclib.net) — primary; exact `/api/get` and fuzzy
//!      `/api/search` race **concurrently** and we return the instant *either*
//!      yields synced lyrics (no waiting on the slow one).
//!   2. **NetEase Cloud Music** (unofficial endpoints) — backup behind LRCLIB
//!      for the tracks its crowd-sourced catalog hasn't covered yet (e.g.
//!      "MTBTTF" by Clipse). Synced when NetEase has it, plain otherwise.
//!
//! Speed strategy:
//!   * a shared pooled `ureq::Agent` avoids per-call TLS handshakes,
//!   * synced results are cached to disk (`~/.cache/overseer/lyrics`), so a
//!     song heard once loads instantly forever — even across restarts.
//!
//! Misses don't vanish: every track we fail to resolve is appended to a JSONL
//! miss log (`~/.cache/overseer/misses.jsonl`) with the sources tried, so the
//! LYRICS card can surface a count and a reconcile pass can retry them later
//! (catalogs grow — a miss today may hit next week).

use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::state::{LyricLine, Lyrics};

const UA: &str = "overseer (https://github.com/local/overseer)";
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

    // Take the first synced result from either LRCLIB source. On a hit we clear
    // any logged miss for this track (it's resolved now) before returning.
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
                        clear_miss(track_id);
                        return out;
                    }
                    if let Some(lines) = synced_from(ev) {
                        out.lines = lines;
                        out.synced = true;
                        clear_miss(track_id);
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
                    clear_miss(track_id);
                    return out;
                }
            }
            Err(_) => break,
        }
    }

    // LRCLIB whiffed on synced. Try the backup source (NetEase) *before* settling
    // for LRCLIB's plain text — a synced NetEase hit reads far better than plain.
    if let Some(lines) = netease_synced(agent, artist, track, duration) {
        out.lines = lines;
        out.synced = true;
        out.note = "synced · netease".into();
        clear_miss(track_id);
        return out;
    }

    // No synced anywhere — fall back to plain from LRCLIB's exact record…
    if let Some(ev) = &exact {
        if let Some(p) = ev.get("plainLyrics").and_then(|x| x.as_str()) {
            if !p.trim().is_empty() {
                out.lines = p
                    .lines()
                    .map(|l| LyricLine { t: -1.0, text: l.trim().to_string() })
                    .collect();
                out.note = "plain (no synced found)".into();
                clear_miss(track_id);
                return out;
            }
        }
    }
    // …then plain from NetEase, the last shot before we record a real miss.
    if let Some(lines) = netease_plain(agent, artist, track) {
        out.lines = lines;
        out.note = "plain · netease".into();
        clear_miss(track_id);
        return out;
    }

    // Genuinely nothing. Don't drop it on the floor — log the miss (artist ·
    // title · album · when · sources tried) so it's countable and retryable.
    log_miss(artist, track, album, track_id, "lrclib,netease");
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

// --- NetEase Cloud Music backup ------------------------------------------
// LRCLIB is crowd-sourced and has real holes (e.g. Clipse · "MTBTTF"). NetEase
// has a huge synced catalog reachable via its unofficial, no-key web endpoints.
// We search for the track, take the best artist-matching song, then pull its
// synced `.lrc`. Same cleaning + duration/artist ranking we use for LRCLIB so a
// wrong-artist cover can't win.

const NETEASE_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) overseer";

/// Best synced lyrics from NetEase, or `None`.
fn netease_synced(agent: &ureq::Agent, artist: &str, track: &str, duration: f64) -> Option<Vec<LyricLine>> {
    let id = netease_song_id(agent, artist, track, duration)?;
    let lrc = netease_lyric_raw(agent, id)?;
    let lines = parse_lrc(&strip_lrc_metadata(&lrc));
    if lines.iter().any(|l| l.t >= 0.0) {
        // Keep only the timestamped (synced) lines.
        Some(lines.into_iter().filter(|l| l.t >= 0.0).collect())
    } else {
        None
    }
}

/// Plain (unsynced) NetEase lyrics as a last resort — every line `t == -1.0`.
fn netease_plain(agent: &ureq::Agent, artist: &str, track: &str) -> Option<Vec<LyricLine>> {
    let id = netease_song_id(agent, artist, track, 0.0)?;
    let lrc = netease_lyric_raw(agent, id)?;
    let lines: Vec<LyricLine> = strip_lrc_metadata(&lrc)
        .lines()
        .filter_map(|l| {
            // Strip any leading [..] tags, keep the bare text.
            let mut rest = l;
            while rest.starts_with('[') {
                match rest.find(']') {
                    Some(c) => rest = &rest[c + 1..],
                    None => break,
                }
            }
            let t = rest.trim();
            (!t.is_empty()).then(|| LyricLine { t: -1.0, text: t.to_string() })
        })
        .collect();
    (!lines.is_empty()).then_some(lines)
}

/// Resolve `artist`/`track` to the best NetEase song id. Ranks candidates by
/// artist agreement first, then duration closeness (NetEase reports ms).
fn netease_song_id(agent: &ureq::Agent, artist: &str, track: &str, duration: f64) -> Option<i64> {
    let primary = primary_artist(artist);
    let ct = clean_title(track);
    let query = format!("{primary} {ct}");
    let url = format!(
        "https://music.163.com/api/search/get?s={}&type=1&limit=8",
        urlencode(&query)
    );
    let v: serde_json::Value = agent
        .get(&url)
        .set("User-Agent", NETEASE_UA)
        .set("Referer", "https://music.163.com")
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let songs = v.get("result")?.get("songs")?.as_array()?;
    let pl = primary.to_lowercase();
    let mut best: Option<(f64, i64)> = None;
    for s in songs {
        let id = s.get("id").and_then(|x| x.as_i64())?;
        let rdur = s.get("duration").and_then(|x| x.as_f64()).unwrap_or(0.0) / 1000.0;
        let rartist = s
            .get("artists")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| a.get("name").and_then(|n| n.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
                    .to_lowercase()
            })
            .unwrap_or_default();
        let artist_ok = pl.is_empty() || rartist.contains(&pl) || pl.contains(&rartist);
        // Same scoring shape as LRCLIB: artist mismatch dwarfs any duration gap;
        // a zero duration (plain-only path) leaves artist as the sole signal.
        let dgap = if duration > 0.0 { (rdur - duration).abs() } else { 0.0 };
        let score = dgap + if artist_ok { 0.0 } else { 10_000.0 };
        match &best {
            Some((bscore, _)) if *bscore <= score => {}
            _ => best = Some((score, id)),
        }
    }
    best.map(|(_, id)| id)
}

/// Fetch the raw `.lrc` blob for a NetEase song id.
fn netease_lyric_raw(agent: &ureq::Agent, id: i64) -> Option<String> {
    let url = format!("https://music.163.com/api/song/lyric?id={id}&lv=1&kv=1&tv=-1");
    let v: serde_json::Value = agent
        .get(&url)
        .set("User-Agent", NETEASE_UA)
        .set("Referer", "https://music.163.com")
        .call()
        .ok()?
        .into_json()
        .ok()?;
    let lrc = v.get("lrc")?.get("lyric")?.as_str()?.to_string();
    (!lrc.trim().is_empty()).then_some(lrc)
}

/// Drop NetEase's non-lyric `.lrc` metadata lines (作词/作曲 credit tags, and the
/// "纯音乐，请欣赏" instrumental marker) so they don't pollute the karaoke wipe.
fn strip_lrc_metadata(lrc: &str) -> String {
    lrc.lines()
        .filter(|l| {
            let low = l.to_lowercase();
            !(low.contains("作词")
                || low.contains("作曲")
                || low.contains("编曲")
                || low.contains("纯音乐")
                || low.contains(": 作")
                || low.contains("by ：")
                || low.contains("制作人")
                || low.contains("混音"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Minimal percent-encoding for a query string (NetEase's search `s` param).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// --- miss log --------------------------------------------------------------
// Every track we can't resolve is appended to a JSONL file so misses are visible
// and retryable instead of silently set to "no lyrics found" and forgotten. The
// log is keyed by `track_id`; `clear_miss` removes a row the moment a later fetch
// (a new catalog upload, the reconcile pass) finally lands lyrics for it.

const MISS_FILE: &str = "misses.jsonl";

/// One logged lyrics miss.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Miss {
    pub track_id: String,
    pub artist: String,
    pub track: String,
    pub album: String,
    pub ts: i64,         // unix seconds, when first/last logged
    pub tried: String,   // comma-joined source list, e.g. "lrclib,netease"
    #[serde(default)]
    pub retries: u32,    // how many reconcile passes have re-tried this one
}

fn miss_path() -> Option<std::path::PathBuf> {
    Some(crate::cache::root()?.join(MISS_FILE))
}

/// Read every logged miss (deduped by `track_id`, last entry wins). Fails soft to
/// an empty list on any I/O / parse error.
pub fn load_misses() -> Vec<Miss> {
    let Some(p) = miss_path() else { return Vec::new() };
    let Ok(body) = std::fs::read_to_string(&p) else { return Vec::new() };
    let mut by_id: std::collections::HashMap<String, Miss> = Default::default();
    let mut order: Vec<String> = Vec::new();
    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(m) = serde_json::from_str::<Miss>(line) {
            if !by_id.contains_key(&m.track_id) {
                order.push(m.track_id.clone());
            }
            by_id.insert(m.track_id.clone(), m);
        }
    }
    order.into_iter().filter_map(|id| by_id.remove(&id)).collect()
}

/// How many distinct tracks are currently in the miss log.
pub fn miss_count() -> usize {
    load_misses().len()
}

/// Append a miss row (best-effort). De-dup happens on read, so re-logging the
/// same track just refreshes its timestamp/retry count.
pub fn log_miss(artist: &str, track: &str, album: &str, track_id: &str, tried: &str) {
    let Some(root) = crate::cache::root() else { return };
    let _ = std::fs::create_dir_all(&root);
    // Carry forward the prior retry count if this track was already logged.
    let retries = load_misses()
        .iter()
        .find(|m| m.track_id == track_id)
        .map(|m| m.retries)
        .unwrap_or(0);
    let m = Miss {
        track_id: track_id.to_string(),
        artist: artist.to_string(),
        track: track.to_string(),
        album: album.to_string(),
        ts: now_secs(),
        tried: tried.to_string(),
        retries,
    };
    if let Ok(line) = serde_json::to_string(&m) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(root.join(MISS_FILE)) {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Drop a track from the miss log — called the instant any source resolves it.
/// Rewrites the file without that `track_id` (no-op if it wasn't logged).
pub fn clear_miss(track_id: &str) {
    let Some(p) = miss_path() else { return };
    let misses = load_misses();
    if !misses.iter().any(|m| m.track_id == track_id) {
        return;
    }
    let kept: Vec<String> = misses
        .into_iter()
        .filter(|m| m.track_id != track_id)
        .filter_map(|m| serde_json::to_string(&m).ok())
        .collect();
    let body = if kept.is_empty() { String::new() } else { format!("{}\n", kept.join("\n")) };
    let _ = std::fs::write(p, body);
}

/// Bump the retry counter for a track after a reconcile pass re-queries it (kept
/// even when it's still a miss, so we can back off chronic offenders later).
pub fn bump_retry(track_id: &str) {
    let Some(p) = miss_path() else { return };
    let mut misses = load_misses();
    let mut changed = false;
    for m in &mut misses {
        if m.track_id == track_id {
            m.retries = m.retries.saturating_add(1);
            m.ts = now_secs();
            changed = true;
        }
    }
    if !changed {
        return;
    }
    let body: String = misses
        .into_iter()
        .filter_map(|m| serde_json::to_string(&m).ok())
        .collect::<Vec<_>>()
        .join("\n");
    let _ = std::fs::write(p, if body.is_empty() { body } else { format!("{body}\n") });
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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
// Stored as standard `.lrc` files under `~/.cache/overseer/lyrics/`, keyed by
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

    // Acceptance for issue #7: Clipse · "MTBTTF" must resolve (LRCLIB fuzzy or the
    // NetEase backup). Run with: cargo test --release -- --ignored e2e_mtbttf
    #[test]
    #[ignore]
    fn e2e_mtbttf_resolves() {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(4))
            .timeout_read(std::time::Duration::from_secs(13))
            .build();
        let ly = fetch(&agent, "Clipse", "MTBTTF", "Let God Sort Em Out", 156.0, "mtbttf");
        println!("synced={} lines={} note={:?}", ly.synced, ly.lines.len(), ly.note);
        if let Some(first) = ly.lines.iter().find(|l| !l.text.is_empty()) {
            println!("first line: {:?}", first.text);
        }
        assert!(!ly.lines.is_empty(), "expected lyrics for MTBTTF, got note={:?}", ly.note);
    }

    // NetEase backup in isolation: a track LRCLIB lacks but NetEase has synced.
    #[test]
    #[ignore]
    fn e2e_netease_backup() {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(4))
            .timeout_read(std::time::Duration::from_secs(13))
            .build();
        let lines = netease_synced(&agent, "Clipse", "MTBTTF", 156.0);
        println!("netease synced lines: {}", lines.as_ref().map(|l| l.len()).unwrap_or(0));
        assert!(lines.map(|l| !l.is_empty()).unwrap_or(false), "netease should have synced MTBTTF");
    }
}
