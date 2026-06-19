//! Palette + small drawing helpers. A cohesive "synthwave graphite" look.

use ratatui::style::Color;

pub const BG: Color = Color::Rgb(13, 14, 22);
pub const PANEL_BORDER: Color = Color::Rgb(44, 48, 74);
pub const PANEL_BORDER_HOT: Color = Color::Rgb(108, 92, 231);
pub const TEXT: Color = Color::Rgb(206, 212, 235);
pub const DIM: Color = Color::Rgb(108, 116, 150);
pub const FAINT: Color = Color::Rgb(70, 76, 104);

pub const ACCENT: Color = Color::Rgb(157, 124, 255); // violet
pub const CYAN: Color = Color::Rgb(86, 214, 255);
pub const PINK: Color = Color::Rgb(255, 110, 199);
pub const GREEN: Color = Color::Rgb(94, 234, 160);
pub const YELLOW: Color = Color::Rgb(255, 209, 102);
pub const ORANGE: Color = Color::Rgb(255, 154, 90);
pub const RED: Color = Color::Rgb(255, 95, 109);

/// Heat color for a 0..1 load value: green -> yellow -> orange -> red.
pub fn heat(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let k = t / 0.5;
        lerp((94, 234, 160), (255, 209, 102), k)
    } else {
        let k = (t - 0.5) / 0.5;
        lerp((255, 209, 102), (255, 95, 109), k)
    };
    Color::Rgb(r, g, b)
}

/// Funky synthwave ramp for a 0..1 value: blue -> violet -> pink -> white.
/// Used to give the Apple Silicon gauges + power wave their jazzy look.
pub fn jazz(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.40 {
        lerp((86, 214, 255), (157, 124, 255), t / 0.40) // blue -> violet
    } else if t < 0.75 {
        lerp((157, 124, 255), (255, 110, 199), (t - 0.40) / 0.35) // violet -> pink
    } else {
        lerp((255, 110, 199), (245, 245, 255), (t - 0.75) / 0.25) // pink -> white
    };
    Color::Rgb(r, g, b)
}

/// Cool gradient for the karaoke wipe: cyan -> violet -> pink across a line.
pub fn wipe(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        lerp((86, 214, 255), (157, 124, 255), t / 0.5)
    } else {
        lerp((157, 124, 255), (255, 110, 199), (t - 0.5) / 0.5)
    };
    Color::Rgb(r, g, b)
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
