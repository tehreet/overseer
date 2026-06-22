//! System-wide "Now Playing" via the private MediaRemote framework — the same
//! info Control Center shows, and the ONLY reliable way to see what the macOS TV
//! app is *streaming* (TV.app's AppleScript bridge reports a streamed title as
//! `missing value` with no position).
//!
//! The catch: macOS 15.4+ gates `MRMediaRemoteGetNowPlayingInfo` behind a private
//! entitlement, so an ad-hoc-signed binary (like ours) gets back `nil`. The query
//! still works from any **Apple-signed** process, so we run it through the system
//! `swift` interpreter (~0.25 s cold) and parse a tab-separated line — the same
//! shell-out pattern the Music/TV AppleScript collectors already use. Polled on a
//! background thread every couple of seconds; the position is interpolated locally
//! between polls, so the cadence stays cheap.
#![cfg(target_os = "macos")]

use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// A snapshot of the system now-playing item.
#[derive(Default, Clone)]
pub struct NowPlaying {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub duration: f64,
    pub elapsed: f64,
    pub playing: bool,
    /// True when the item is video (a movie/show) rather than audio.
    pub is_video: bool,
}

/// Swift that loads MediaRemote, reads the now-playing dict, and prints one
/// tab-separated line: `playing  isVideo  title  artist  album  duration  elapsed`.
/// `elapsed` is corrected to "now" using the info timestamp + playback rate.
const SWIFT_SRC: &str = r#"
import Foundation
let url = NSURL(fileURLWithPath: "/System/Library/PrivateFrameworks/MediaRemote.framework")
guard let b = CFBundleCreate(kCFAllocatorDefault, url),
      let p = CFBundleGetFunctionPointerForName(b, "MRMediaRemoteGetNowPlayingInfo" as CFString)
else { print("NONE"); exit(0) }
typealias Fn = @convention(c) (DispatchQueue, @escaping ([String: Any]?) -> Void) -> Void
let getInfo = unsafeBitCast(p, to: Fn.self)
func clean(_ s: String) -> String {
  return s.replacingOccurrences(of: "\t", with: " ").replacingOccurrences(of: "\n", with: " ")
}
getInfo(DispatchQueue.main) { info in
  guard let i = info else { print("NONE"); exit(0) }
  func str(_ k: String) -> String { return (i[k] as? String) ?? "" }
  func num(_ k: String) -> Double { return (i[k] as? NSNumber)?.doubleValue ?? 0 }
  let rate = num("kMRMediaRemoteNowPlayingInfoPlaybackRate")
  let playing = ((i["kMRMediaRemoteNowPlayingApplicationIsPlayingUserInfoKey"] as? NSNumber)?.boolValue) ?? (rate > 0)
  let strict = str("kMRMediaRemoteNowPlayingInfoStrictMediaType").lowercased()
  let mtype = str("kMRMediaRemoteNowPlayingInfoMediaType").lowercased()
  let isVideo = strict == "video" || mtype.contains("video")
  var elapsed = num("kMRMediaRemoteNowPlayingInfoElapsedTime")
  if playing, let ts = i["kMRMediaRemoteNowPlayingInfoTimestamp"] as? Date {
    elapsed += Date().timeIntervalSince(ts)
  }
  let fields = [
    playing ? "1" : "0",
    isVideo ? "1" : "0",
    clean(str("kMRMediaRemoteNowPlayingInfoTitle")),
    clean(str("kMRMediaRemoteNowPlayingInfoArtist")),
    clean(str("kMRMediaRemoteNowPlayingInfoAlbum")),
    String(num("kMRMediaRemoteNowPlayingInfoDuration")),
    String(elapsed),
  ]
  print(fields.joined(separator: "\t"))
  exit(0)
}
DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { print("NONE"); exit(0) }
CFRunLoopRun()
"#;

/// Cache the script to a temp file once so each poll is just `swift <file>`.
fn script_path() -> Option<&'static std::path::Path> {
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::env::temp_dir().join("overseer_nowplaying.swift");
        std::fs::write(&p, SWIFT_SRC).ok().map(|_| p)
    })
    .as_deref()
}

/// Run the helper and parse the current system now-playing. `None` when nothing
/// is loaded, the helper can't run (no `swift`), or MediaRemote returns nothing.
pub fn get() -> Option<NowPlaying> {
    let path = script_path()?;
    let out = Command::new("swift")
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let raw = String::from_utf8_lossy(&out.stdout);
    let line = raw.lines().next_back().unwrap_or("").trim();
    if line.is_empty() || line == "NONE" {
        return None;
    }
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() < 7 {
        return None;
    }
    let np = NowPlaying {
        playing: f[0] == "1",
        is_video: f[1] == "1",
        title: f[2].to_string(),
        artist: f[3].to_string(),
        album: f[4].to_string(),
        duration: f[5].parse().unwrap_or(0.0),
        elapsed: f[6].parse().unwrap_or(0.0),
    };
    if np.title.is_empty() && np.duration == 0.0 {
        return None;
    }
    Some(np)
}

/// Headless probe (`overseer --diag-np`): print the raw system now-playing.
pub fn diag() {
    println!("Probing MediaRemote via the swift helper …\n");
    match get() {
        None => println!("  (nothing playing, or the swift helper returned nothing)"),
        Some(np) => {
            println!("  title    : {}", np.title);
            println!("  artist   : {}", np.artist);
            println!("  album    : {}", np.album);
            println!("  playing  : {}", np.playing);
            println!("  is_video : {}", np.is_video);
            println!("  elapsed  : {:.1}s / {:.1}s", np.elapsed, np.duration);
        }
    }
}
