//! Persistent on-disk cache under `~/.cache/overseer/<kind>/`.
//!
//! A song heard once loads its lyrics, liner-note facts, and album art instantly
//! forever — with zero network/LLM calls on replay — because each is written to
//! disk keyed by a stable hash of the track id (`artist|track|album`). The hash
//! is shared across all kinds so keys stay consistent (see `track_key`).
//!
//! Everything here fails soft: a missing, corrupt, or partial entry simply
//! reads back as `None`, so the live collectors regenerate it instead of
//! crashing.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Root cache directory: `~/.cache/overseer`.
pub fn root() -> Option<PathBuf> {
    Some(dirs::cache_dir()?.join("overseer"))
}

/// A per-kind subdirectory (e.g. "lyrics" / "facts" / "art"), created on demand.
pub fn cache_dir(kind: &str) -> Option<PathBuf> {
    let d = root()?.join(kind);
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// Stable 16-hex-digit key for a track id (`artist|track|album`). Shared by every
/// cache kind so the same song maps to the same filename stem everywhere.
pub fn track_key(track_id: &str) -> String {
    let mut h = DefaultHasher::new();
    track_id.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Path to a cache entry: `~/.cache/overseer/<kind>/<hash>.<ext>`.
pub fn entry_path(kind: &str, track_id: &str, ext: &str) -> Option<PathBuf> {
    Some(cache_dir(kind)?.join(format!("{}.{}", track_key(track_id), ext)))
}

/// Read raw bytes for a cache entry, or `None` if absent/unreadable.
pub fn get_bytes(kind: &str, track_id: &str, ext: &str) -> Option<Vec<u8>> {
    std::fs::read(entry_path(kind, track_id, ext)?).ok()
}

/// Write raw bytes for a cache entry. Best-effort: silently ignores I/O errors.
pub fn put_bytes(kind: &str, track_id: &str, ext: &str, data: &[u8]) {
    if let Some(p) = entry_path(kind, track_id, ext) {
        let _ = std::fs::write(p, data);
    }
}

/// Read and JSON-decode a cache entry, failing soft to `None` on any error
/// (missing file, corrupt/partial JSON, schema drift).
pub fn get_json<T: serde::de::DeserializeOwned>(kind: &str, track_id: &str) -> Option<T> {
    let bytes = get_bytes(kind, track_id, "json")?;
    serde_json::from_slice(&bytes).ok()
}

/// JSON-encode and write a cache entry. Best-effort.
pub fn put_json<T: serde::Serialize>(kind: &str, track_id: &str, value: &T) {
    if let Ok(bytes) = serde_json::to_vec(value) {
        put_bytes(kind, track_id, "json", &bytes);
    }
}

/// Wipe every cache subdirectory and report what was removed, one line per kind
/// (used by `overseer --clear-cache`).
pub fn clear() -> Vec<String> {
    let mut report = Vec::new();
    let Some(root) = root() else {
        report.push("cache dir unavailable".into());
        return report;
    };
    if !root.exists() {
        report.push(format!("nothing to remove ({} does not exist)", root.display()));
        return report;
    }
    for kind in ["lyrics", "facts", "art"] {
        let dir = root.join(kind);
        let count = std::fs::read_dir(&dir)
            .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_file()).count())
            .unwrap_or(0);
        match std::fs::remove_dir_all(&dir) {
            Ok(_) => report.push(format!("removed {kind}/ ({count} entries)")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                report.push(format!("{kind}/ (empty)"))
            }
            Err(e) => report.push(format!("{kind}/ — error: {e}")),
        }
    }
    // The miss log lives at the cache ROOT, not under a kind subdir — clear it too.
    let miss = root.join(crate::lyrics::MISS_FILE);
    match std::fs::remove_file(&miss) {
        Ok(_) => report.push(format!("removed {}", crate::lyrics::MISS_FILE)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            report.push(format!("{} (empty)", crate::lyrics::MISS_FILE))
        }
        Err(e) => report.push(format!("{} — error: {e}", crate::lyrics::MISS_FILE)),
    }
    report
}
