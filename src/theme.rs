//! Palette + small drawing helpers. A cohesive "synthwave graphite" look.

use ratatui::style::Color;
use std::sync::atomic::{AtomicU32, Ordering};

pub const BG: Color = Color::Rgb(13, 14, 22);
pub const PANEL_BORDER: Color = Color::Rgb(44, 48, 74);
pub const PANEL_BORDER_HOT: Color = Color::Rgb(108, 92, 231);
pub const TEXT: Color = Color::Rgb(206, 212, 235);
pub const DIM: Color = Color::Rgb(108, 116, 150);
pub const FAINT: Color = Color::Rgb(70, 76, 104);

/// House defaults — the on-brand synthwave hues the dynamic theme departs from
/// (and returns to when there's no album art). The lively accents (ACCENT/CYAN/
/// PINK) are exposed through the `accent()`/`cyan()`/`pink()` accessors below,
/// which serve a live, album-art-biased, cross-faded value each frame.
pub const ACCENT_BASE: (u8, u8, u8) = (157, 124, 255); // violet
pub const CYAN_BASE: (u8, u8, u8) = (86, 214, 255);
pub const PINK_BASE: (u8, u8, u8) = (255, 110, 199);

pub const GREEN: Color = Color::Rgb(94, 234, 160);
pub const YELLOW: Color = Color::Rgb(255, 209, 102);
#[allow(dead_code)]
pub const ORANGE: Color = Color::Rgb(255, 154, 90);
pub const RED: Color = Color::Rgb(255, 95, 109);

// ── Live (dynamic) accent store ─────────────────────────────────────────────
// The render loop pushes the per-frame cross-faded accent/cyan/pink here once,
// up front; every card then reads them through the accessors. Packed RGB in an
// atomic so the pure render fn stays lock-free and `apply_dynamic` is the only
// writer. Seeded to the house defaults so the very first frame is on-brand.
static LIVE_ACCENT: AtomicU32 = AtomicU32::new(pack(ACCENT_BASE));
static LIVE_CYAN: AtomicU32 = AtomicU32::new(pack(CYAN_BASE));
static LIVE_PINK: AtomicU32 = AtomicU32::new(pack(PINK_BASE));

const fn pack(c: (u8, u8, u8)) -> u32 {
    (c.0 as u32) << 16 | (c.1 as u32) << 8 | c.2 as u32
}
fn unpack(v: u32) -> (u8, u8, u8) {
    ((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

/// Live, album-art-biased violet accent (cross-faded each frame). Falls back to
/// the house violet when no art is present.
pub fn accent() -> Color {
    let (r, g, b) = unpack(LIVE_ACCENT.load(Ordering::Relaxed));
    Color::Rgb(r, g, b)
}
pub fn cyan() -> Color {
    let (r, g, b) = unpack(LIVE_CYAN.load(Ordering::Relaxed));
    Color::Rgb(r, g, b)
}
pub fn pink() -> Color {
    let (r, g, b) = unpack(LIVE_PINK.load(Ordering::Relaxed));
    Color::Rgb(r, g, b)
}

/// Push this frame's cross-faded accents into the live store. Called once at the
/// top of `ui::render` so every card downstream reads a coherent, glided value.
pub fn apply_dynamic(dt: &crate::state::DynamicTheme) {
    let [a, c, p] = dt.eased();
    LIVE_ACCENT.store(pack(a), Ordering::Relaxed);
    LIVE_CYAN.store(pack(c), Ordering::Relaxed);
    LIVE_PINK.store(pack(p), Ordering::Relaxed);
}

/// Funky synthwave ramp for a 0..1 value: blue -> violet -> pink -> white.
/// Used to give the Apple Silicon gauges + power wave their jazzy look. The
/// blue/violet/pink stops ride the *live* (album-biased) accents so the lively
/// bits re-tint with the cover too — white stays white so highlights read.
pub fn jazz(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let cy = rgb(cyan());
    let ac = rgb(accent());
    let pk = rgb(pink());
    let (r, g, b) = if t < 0.40 {
        lerp(cy, ac, t / 0.40) // blue -> violet
    } else if t < 0.75 {
        lerp(ac, pk, (t - 0.40) / 0.35) // violet -> pink
    } else {
        lerp(pk, (245, 245, 255), (t - 0.75) / 0.25) // pink -> white
    };
    Color::Rgb(r, g, b)
}

/// Cool gradient for the karaoke wipe: cyan -> violet -> pink across a line.
/// Also rides the live accents so the lyric wipe wears the album's colors.
pub fn wipe(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        lerp(rgb(cyan()), rgb(accent()), t / 0.5)
    } else {
        lerp(rgb(accent()), rgb(pink()), (t - 0.5) / 0.5)
    };
    Color::Rgb(r, g, b)
}

/// Unpack an `Rgb` Color back to a tuple (non-Rgb fall back to mid-grey).
fn rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (200, 200, 200),
    }
}

/// Blend two colors. `t=0` -> a, `t=1` -> b. Only meaningful for Rgb colors.
pub fn blend(a: Color, b: Color, t: f32) -> Color {
    let to = |c: Color| match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (200, 200, 200),
    };
    let (r, g, bl) = lerp(to(a), to(b), t.clamp(0.0, 1.0));
    Color::Rgb(r, g, bl)
}

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// A smooth horizontal bar using 1/8-cell block glyphs for sub-cell precision.
pub fn bar(frac: f32, width: usize) -> String {
    let frac = frac.clamp(0.0, 1.0) as f64;
    let total_eighths = (frac * width as f64 * 8.0).round() as usize;
    let full = total_eighths / 8;
    let rem = total_eighths % 8;
    let mut s = String::with_capacity(width * 3);
    for _ in 0..full.min(width) {
        s.push('█');
    }
    let mut drawn = full.min(width);
    if drawn < width && rem > 0 {
        s.push([' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'][rem]);
        drawn += 1;
    }
    for _ in drawn..width {
        s.push('░');
    }
    s
}

/// Vertical block glyph for a 0..1 fill fraction (fills from the bottom).
pub fn vblock(frac: f32) -> char {
    let g = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let idx = (frac.clamp(0.0, 1.0) * 8.0).round() as usize;
    g[idx.min(8)]
}

// ── Dynamic theme from album art ────────────────────────────────────────────
// Extract a small dominant-color set from the downscaled cover, then bias the
// ACCENT/CYAN/PINK accents toward it while clamping into the house jazzy
// purple/white/pink family — vivid, readable, never muddy/off-brand.

/// RGB (0..255) -> HSL with h in 0..360, s/l in 0..1.
fn rgb_to_hsl(c: (u8, u8, u8)) -> (f32, f32, f32) {
    let r = c.0 as f32 / 255.0;
    let g = c.1 as f32 / 255.0;
    let b = c.2 as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let d = max - min;
    if d.abs() < 1e-6 {
        return (0.0, 0.0, l); // grey
    }
    let s = d / (1.0 - (2.0 * l - 1.0).abs());
    let h = if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / d) + 2.0)
    } else {
        60.0 * (((r - g) / d) + 4.0)
    };
    ((h + 360.0) % 360.0, s, l)
}

/// HSL -> RGB (0..255).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = ((h % 360.0) + 360.0) % 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (((h / 60.0) % 2.0) - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let f = |v: f32| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (f(r), f(g), f(b))
}

/// Pull a tinted accent toward `base_hue` (the house hue family) so an album
/// that's, say, teal still lands as a *purple-leaning teal* rather than going
/// fully off-brand, then clamp saturation/lightness so it stays vivid + legible
/// on the near-black bg. `pull` is how strongly to drag the art hue home.
fn brandify(art: (u8, u8, u8), base: (u8, u8, u8), pull: f32, smin: f32, lmin: f32, lmax: f32) -> (u8, u8, u8) {
    let (ah, asat, al) = rgb_to_hsl(art);
    let (bh, ..) = rgb_to_hsl(base);
    // A near-grey cover carries no usable hue → just keep the house accent.
    if asat < 0.12 {
        return base;
    }
    // Hue-lerp the short way around the wheel toward the house hue.
    let delta = ((bh - ah + 540.0) % 360.0 - 180.0) * pull;
    let h = (ah + delta + 360.0) % 360.0;
    // Vivid but not neon; bright enough to read on graphite, not blown out.
    let s = asat.max(smin).min(0.95);
    // Bias lightness up off the floor so dark covers still read, then clamp.
    let l = (al * 0.6 + 0.35).clamp(lmin, lmax);
    hsl_to_rgb(h, s, l)
}

/// Median-cut the thumbnail into `want` representative colors, brightest +
/// most-saturated buckets first. Cheap: the thumb is tiny (ART_THUMB²).
fn dominant_colors(px: &[[u8; 3]], want: usize) -> Vec<(u8, u8, u8)> {
    // Drop near-black / near-white pixels: they're usually letterbox or paper
    // and don't carry the cover's identity.
    let mut pool: Vec<(u8, u8, u8)> = px
        .iter()
        .map(|p| (p[0], p[1], p[2]))
        .filter(|c| {
            let (_h, _s, l) = rgb_to_hsl(*c);
            l > 0.08 && l < 0.96
        })
        .collect();
    if pool.is_empty() {
        pool = px.iter().map(|p| (p[0], p[1], p[2])).collect();
    }
    if pool.is_empty() {
        return Vec::new();
    }
    // Each box = a slice of `pool`. Split the box with the widest channel at its
    // median repeatedly until we have `want` boxes (classic median cut).
    let mut boxes: Vec<Vec<(u8, u8, u8)>> = vec![pool];
    while boxes.len() < want {
        // Pick the box with the largest single-channel spread.
        let idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .max_by_key(|(_, b)| channel_spread(b))
            .map(|(i, _)| i);
        let Some(idx) = idx else { break };
        let mut b = boxes.swap_remove(idx);
        let ch = widest_channel(&b);
        b.sort_by_key(|c| match ch {
            0 => c.0,
            1 => c.1,
            _ => c.2,
        });
        let mid = b.len() / 2;
        let hi = b.split_off(mid);
        boxes.push(b);
        boxes.push(hi);
    }
    // Average each box, then order by "vividness" (sat × lightness) so the
    // liveliest swatch leads.
    let mut out: Vec<(u8, u8, u8)> = boxes
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let n = b.len() as u32;
            let (mut r, mut g, mut bl) = (0u32, 0u32, 0u32);
            for c in b {
                r += c.0 as u32;
                g += c.1 as u32;
                bl += c.2 as u32;
            }
            ((r / n) as u8, (g / n) as u8, (bl / n) as u8)
        })
        .collect();
    out.sort_by(|a, b| vividness(*b).partial_cmp(&vividness(*a)).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn vividness(c: (u8, u8, u8)) -> f32 {
    let (_h, s, l) = rgb_to_hsl(c);
    s * (1.0 - (l - 0.55).abs()) // saturated + mid-light reads best
}

fn channel_spread(b: &[(u8, u8, u8)]) -> u32 {
    let mut lo = (255u8, 255u8, 255u8);
    let mut hi = (0u8, 0u8, 0u8);
    for c in b {
        lo.0 = lo.0.min(c.0); lo.1 = lo.1.min(c.1); lo.2 = lo.2.min(c.2);
        hi.0 = hi.0.max(c.0); hi.1 = hi.1.max(c.1); hi.2 = hi.2.max(c.2);
    }
    let r = (hi.0 - lo.0) as u32;
    let g = (hi.1 - lo.1) as u32;
    let bl = (hi.2 - lo.2) as u32;
    r.max(g).max(bl)
}

fn widest_channel(b: &[(u8, u8, u8)]) -> u8 {
    let mut lo = (255u8, 255u8, 255u8);
    let mut hi = (0u8, 0u8, 0u8);
    for c in b {
        lo.0 = lo.0.min(c.0); lo.1 = lo.1.min(c.1); lo.2 = lo.2.min(c.2);
        hi.0 = hi.0.max(c.0); hi.1 = hi.1.max(c.1); hi.2 = hi.2.max(c.2);
    }
    let r = (hi.0 - lo.0) as u32;
    let g = (hi.1 - lo.1) as u32;
    let bl = (hi.2 - lo.2) as u32;
    if r >= g && r >= bl { 0 } else if g >= bl { 1 } else { 2 }
}

/// Derive the three on-brand accent targets (accent, cyan, pink) from album-art
/// pixels. Returns the house defaults when the art is empty or unusable, so the
/// theme cleanly relaxes back to synthwave between songs / with no cover.
pub fn theme_from_art(px: &[[u8; 3]]) -> ((u8, u8, u8), (u8, u8, u8), (u8, u8, u8)) {
    if px.is_empty() {
        return (ACCENT_BASE, CYAN_BASE, PINK_BASE);
    }
    let dom = dominant_colors(px, 5);
    let lead = dom.first().copied().unwrap_or(ACCENT_BASE);
    let second = dom.get(1).copied().unwrap_or(lead);
    // ACCENT (violet) is the anchor: drag the cover's lead hue mostly home so the
    // board still reads purple. CYAN/PINK take the secondary hue with a lighter
    // pull, so they pick up the cover's color while staying in the cool/warm
    // synthwave lanes.
    let accent = brandify(lead, ACCENT_BASE, 0.62, 0.45, 0.55, 0.80);
    let cyan = brandify(second, CYAN_BASE, 0.45, 0.55, 0.55, 0.82);
    let pink = brandify(lead, PINK_BASE, 0.40, 0.55, 0.60, 0.82);
    (accent, cyan, pink)
}

/// Braille-ish sparkline using vertical block glyphs.
pub fn spark(data: &[u64], width: usize) -> String {
    if data.is_empty() {
        return " ".repeat(width);
    }
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let slice: Vec<u64> = data.iter().rev().take(width).rev().copied().collect();
    let max = slice.iter().copied().max().unwrap_or(1).max(1);
    let mut s = String::new();
    for _ in slice.len()..width {
        s.push(' ');
    }
    for v in slice {
        let idx = ((v as f64 / max as f64) * 8.0).round() as usize;
        s.push(glyphs[idx.min(8)]);
    }
    s
}
