//! All rendering. Pure function of (state snapshot, animation clock) -> frame.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use chrono::{Local, Timelike};

use crate::state::AppState;
use crate::theme as c;

/// How far behind real time the equalizer plays back. We render the value at
/// `now - DELAY`, which guarantees a "next" real sample exists to glide toward
/// — so the motion is smooth and faithful, just slightly delayed.
const EQ_DELAY: Duration = Duration::from_millis(1500);

/// Catmull-Rom spline through four sample points, evaluated at `u` in 0..1
/// between p1 and p2. Gives continuous velocity across samples = silky motion.
fn catmull(p0: f32, p1: f32, p2: f32, p3: f32, u: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3)
}

/// How wide a time window the scrolling area-graphs show, ending at
/// `now - EQ_DELAY`. We sample one delay-interpolated point per column, so the
/// curve glides continuously left at frame rate instead of stepping once per
/// real sample — honest data, buttery motion.
const GRAPH_WINDOW: Duration = Duration::from_millis(12_000);

/// Delayed Catmull-Rom value of channel `ch` from a multi-channel sample buffer
/// at `target` time (companion to `sampled_scalar` / `sampled_cores`).
fn sampled_channel(samples: &VecDeque<(Instant, Vec<f32>)>, ch: usize, target: Instant) -> f32 {
    let m = samples.len();
    if m == 0 {
        return 0.0;
    }
    let g = |k: usize| samples[k].1.get(ch).copied().unwrap_or(0.0);
    if m == 1 || target <= samples[0].0 {
        return g(0);
    }
    if target >= samples[m - 1].0 {
        return g(m - 1);
    }
    let mut i = m - 2;
    for j in 0..m - 1 {
        if samples[j].0 <= target && target < samples[j + 1].0 {
            i = j;
            break;
        }
    }
    let span = (samples[i + 1].0 - samples[i].0).as_secs_f32().max(1e-3);
    let u = ((target - samples[i].0).as_secs_f32() / span).clamp(0.0, 1.0);
    let i0 = i.saturating_sub(1);
    let i3 = (i + 2).min(m - 1);
    catmull(g(i0), g(i), g(i + 1), g(i3), u).max(0.0)
}

/// Build a `w`-wide series, one delay-interpolated point per column spanning
/// `GRAPH_WINDOW` and ending at `now - EQ_DELAY`. `sample` reads the buffer at a
/// target time; `norm` maps the raw value into 0..1 for plotting.
fn series<T>(buf: &T, w: usize, sample: impl Fn(&T, Instant) -> f32, norm: impl Fn(f32) -> f32) -> Vec<f32> {
    if w == 0 {
        return Vec::new();
    }
    let end = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let step = GRAPH_WINDOW.as_secs_f32() / w as f32;
    (0..w)
        .map(|x| {
            let back = Duration::from_secs_f32((w - 1 - x) as f32 * step);
            let target = end.checked_sub(back).unwrap_or(end);
            norm(sample(buf, target))
        })
        .collect()
}

/// Box-blur a 0..1 series a few times (≈ a Gaussian) so a bursty signal like
/// net throughput reads as one buttery curve instead of vertical cliffs between
/// 1 Hz samples. Edges replicate (clamp), so the ends don't sag.
fn smoothed(v: &[f32], radius: usize, passes: usize) -> Vec<f32> {
    let n = v.len();
    let mut cur = v.to_vec();
    if n < 3 || radius == 0 {
        return cur;
    }
    for _ in 0..passes {
        let src = cur.clone();
        for i in 0..n {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius).min(n - 1);
            let mut sum = 0.0f32;
            for &x in &src[lo..=hi] {
                sum += x;
            }
            cur[i] = sum / (hi - lo + 1) as f32;
        }
    }
    cur
}

/// How an area graph colours its fill.
#[derive(Clone, Copy)]
enum Fill {
    /// One hue, dim at the baseline → vivid at the crest.
    #[allow(dead_code)]
    Tint(ratatui::style::Color),
    /// Funky synthwave bands: blue→violet→pink→white up the height.
    Jazz,
}

/// Render a buttery, continuously-scrolling filled area graph into `area`.
/// `vals` holds one normalized value (0..1) per column, oldest→newest. The fill
/// is a vertical gradient — dim at the baseline, vivid at the crest — so spikes
/// glow; the top edge uses 1/8 vertical blocks for sub-cell smoothness.
fn area_graph(f: &mut Frame, area: Rect, vals: &[f32], fill: Fill) {
    use ratatui::style::Color;
    let w = area.width as usize;
    let h = area.height as usize;
    if w == 0 || h == 0 || vals.is_empty() {
        return;
    }
    // Blur the column series so every area graph glides — bursty signals (net)
    // and smoother ones (cpu/gpu/power) all read as one buttery curve, no cliffs.
    let vals = smoothed(vals, 2, 4);
    let vals = vals.as_slice();
    let mut lines: Vec<Line> = Vec::with_capacity(h);
    for row in 0..h {
        let r_bot = (h - 1 - row) as f32; // this cell spans rows [r_bot, r_bot+1)
        let crest = ((r_bot + 0.5) / h as f32).clamp(0.0, 1.0);
        // Run-length merge same-coloured cells so each row is a few spans.
        let mut spans: Vec<Span> = Vec::new();
        let mut run = String::new();
        let mut run_col: Option<Color> = None;
        for x in 0..w {
            let v = vals.get(x).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let fh = v * h as f32; // fill height in rows
            let lit = fh > r_bot;
            let ch = if !lit {
                ' '
            } else if fh >= r_bot + 1.0 {
                '█'
            } else {
                c::vblock(fh - r_bot)
            };
            let cell_col = if !lit {
                None
            } else {
                let base = match fill {
                    Fill::Tint(col) => col,
                    Fill::Jazz => c::jazz(crest),
                };
                Some(c::blend(c::BG, base, 0.28 + 0.72 * crest))
            };
            if cell_col == run_col {
                run.push(ch);
            } else {
                if !run.is_empty() {
                    let prev = std::mem::take(&mut run);
                    spans.push(match run_col {
                        Some(col) => Span::styled(prev, Style::default().fg(col)),
                        None => Span::raw(prev),
                    });
                }
                run.push(ch);
                run_col = cell_col;
            }
        }
        if !run.is_empty() {
            spans.push(match run_col {
                Some(col) => Span::styled(run, Style::default().fg(col)),
                None => Span::raw(run),
            });
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// Scalar version of the delayed Catmull-Rom interpolation (for net energy).
fn sampled_scalar(samples: &VecDeque<(Instant, f32)>, target: Instant) -> f32 {
    let m = samples.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 || target <= samples[0].0 {
        return samples[0].1;
    }
    if target >= samples[m - 1].0 {
        return samples[m - 1].1;
    }
    let mut i = m - 2;
    for j in 0..m - 1 {
        if samples[j].0 <= target && target < samples[j + 1].0 {
            i = j;
            break;
        }
    }
    let span = (samples[i + 1].0 - samples[i].0).as_secs_f32().max(1e-3);
    let u = ((target - samples[i].0).as_secs_f32() / span).clamp(0.0, 1.0);
    let i0 = i.saturating_sub(1);
    let i3 = (i + 2).min(m - 1);
    catmull(samples[i0].1, samples[i].1, samples[i + 1].1, samples[i3].1, u).max(0.0)
}

/// Interpolate per-core CPU values at `target` time from the sample buffer,
/// using a Catmull-Rom spline through the bracketing samples.
fn sampled_cores(samples: &VecDeque<(Instant, Vec<f32>)>, target: Instant) -> Vec<f32> {
    let m = samples.len();
    if m == 0 {
        return Vec::new();
    }
    if m == 1 || target <= samples[0].0 {
        return samples[0].1.clone();
    }
    if target >= samples[m - 1].0 {
        return samples[m - 1].1.clone();
    }
    // Segment [i, i+1] that brackets the target time.
    let mut i = m - 2;
    for j in 0..m - 1 {
        if samples[j].0 <= target && target < samples[j + 1].0 {
            i = j;
            break;
        }
    }
    let span = (samples[i + 1].0 - samples[i].0).as_secs_f32().max(1e-3);
    let u = ((target - samples[i].0).as_secs_f32() / span).clamp(0.0, 1.0);
    let i0 = i.saturating_sub(1);
    let i3 = (i + 2).min(m - 1);
    let n = samples[i].1.len();
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        let g = |k: usize| samples[k].1.get(c).copied().unwrap_or(0.0);
        out.push(catmull(g(i0), g(i), g(i + 1), g(i3), u).clamp(0.0, 100.0));
    }
    out
}

pub fn render(f: &mut Frame, s: &AppState, t: f64) {
    let area = f.area();
    f.render_widget(
        Block::default().style(Style::default().bg(c::BG)),
        area,
    );

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    footer(f, outer[1], s);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(outer[0]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),  // [0] CPU equalizer
            Constraint::Length(9),  // [1] MEM · DISK · NET
            Constraint::Length(11), // [2] APPLE SILICON
            Constraint::Length(9),  // [3] TOP PROCESSES (moved up, +uptime col, +header)
            Constraint::Length(10), // [4] ROBOTS WORKING (Claude tokens + git pulse, merged)
            Constraint::Length(7),  // [5] WEATHER       (jazzed, +data)
            // The two message cards size to their content (no trailing gap) and
            // shrink first under height pressure — keeping ROBOTS rigid above so
            // its bottom-row burn chart is never the thing that gets clipped.
            Constraint::Max(card_height(&s.messages, s.msg_ui.active)), // [6] iMESSAGE
            Constraint::Max(card_height(&s.signal, false)),             // [7] SIGNAL
            Constraint::Max(discord_height(&s.discord)),                // [8] DISCORD
            Constraint::Max(doctor_height(&s.doctor)),                  // [9] MAC-DOCTOR
            Constraint::Min(0),     // [10] slack absorbs leftover, keeping cards tight
        ])
        .split(body[0]);

    cpu_eq_panel(f, left[0], s);
    resources_panel(f, left[1], s);
    silicon_panel(f, left[2], s, t);
    proc_panel(f, left[3], s);
    robots_panel(f, left[4], s, t);
    weather_panel(f, left[5], s);
    messages_panel(f, left[6], s, t);
    signal_panel(f, left[7], s);
    discord_panel(f, left[8], s);
    doctor_panel(f, left[9], s, t);

    // Until the first music poll lands, show the (neutral) lyrics panel so we
    // never flash the wrong thing before the real state is known. After that:
    // actively playing → lyrics; paused/stopped/idle → live system graphs.
    let show_lyrics = !s.music.polled || (s.music.playing && !s.music.track.is_empty());

    // A taller NOW PLAYING card so the album cover renders as a real, legible
    // square (more half-block sub-pixels) instead of a postage stamp.
    const NP_H: u16 = 13;
    if show_lyrics {
        // Cap the lyric band to a tight 9-row block (7 inner rows) and hand the
        // freed vertical space to the live jazz system graphs — zero dead zone.
        const LYRICS_H: u16 = 9;
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(NP_H),
                Constraint::Length(LYRICS_H),
                Constraint::Min(0),
            ])
            .split(body[1]);
        now_playing_row(f, right[0], s, t);
        lyrics_row(f, right[1], s, t);
        if right[2].height >= 3 {
            keybinds_panel(f, right[2], s);
        }
    } else {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(NP_H), Constraint::Min(3)])
            .split(body[1]);
        now_playing_row(f, right[0], s, t);
        keybinds_panel(f, right[1], s);
    }
}

/// A horizontal gauge: a coloured filled portion, then plain background for the
/// remainder. The unfilled track is spaces (no glyph), so it vanishes into the
/// terminal background like the CPU bars' empty space — the spaces still hold
/// the width so trailing text stays aligned.
fn bar_spans(frac: f32, width: usize, color: ratatui::style::Color) -> Vec<Span<'static>> {
    let frac = frac.clamp(0.0, 1.0) as f64;
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    let full = (eighths / 8).min(width);
    let rem = eighths % 8;
    let mut fill = "█".repeat(full);
    let mut drawn = full;
    if drawn < width && rem > 0 {
        fill.push([' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'][rem]);
        drawn += 1;
    }
    vec![
        Span::styled(fill, Style::default().fg(color)),
        Span::raw(" ".repeat(width.saturating_sub(drawn))),
    ]
}

/// Render `text` with a soft sheen of light that glides smoothly across it — a
/// narrow bright band (toward pink/white) sweeps left→right over an otherwise
/// calm base colour, so stat readouts shimmer like brushed metal. `row` offsets
/// the sweep per line so stacked rows don't shimmer in lockstep. Per-character
/// truecolor blend → buttery, and the band wraps so it never pops.
fn shimmer_text(text: &str, t: f64, row: f32) -> Line<'static> {
    Line::from(shimmer_spans(text, t, row, c::DIM, 0.22, 0.07, false))
}

/// Per-character sheen sweep over `text` from `base` toward a pink-white glint.
/// `speed` sets how fast the band travels, `sigma` its width (bigger = broader,
/// more pronounced shimmer), `bold` keeps weight for titles. `row` phase-offsets
/// stacked lines so they don't shimmer in lockstep.
fn shimmer_spans(
    text: &str,
    t: f64,
    row: f32,
    base: Color,
    speed: f64,
    sigma: f32,
    bold: bool,
) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1);
    let head = ((t * speed + row as f64 * 0.18).rem_euclid(1.0)) as f32; // 0..1 sweep
    let sheen = c::jazz(0.9); // pink-white glint
    let mut spans = Vec::with_capacity(n);
    for (i, ch) in chars.iter().enumerate() {
        if *ch == ' ' {
            spans.push(Span::raw(" "));
            continue;
        }
        let p = i as f32 / n as f32;
        let mut d = (p - head).abs();
        if d > 0.5 {
            d = 1.0 - d; // wrap so the glint is continuous
        }
        let b = (-(d * d) / (2.0 * sigma * sigma)).exp(); // bright band 0..1
        let col = c::blend(base, sheen, b); // calm base -> glint
        let mut st = Style::default().fg(col);
        if bold {
            st = st.add_modifier(Modifier::BOLD);
        }
        spans.push(Span::styled(ch.to_string(), st));
    }
    spans
}

/// Pick which of `n` views to show and its visibility `alpha` (0..1) for a calm
/// cross-dissolve: hold a view fully, fade it to the background, fade the next
/// in. Per-frame truecolor blending makes this *sub-frame* smooth — unlike
/// cell-stepped scrolling, which can only ever jump a whole character at a time.
fn dissolve_phase(n: usize, t: f64, hold: f64, fade: f64) -> (usize, f32) {
    if n <= 1 {
        return (0, 1.0);
    }
    let half = (fade * 0.5).max(0.0001);
    let slot = hold + fade;
    let x = t.rem_euclid(n as f64 * slot);
    let idx = ((x / slot).floor() as usize) % n;
    let w = x - idx as f64 * slot;
    if w < hold {
        (idx, 1.0)
    } else if w < hold + half {
        (idx, 1.0 - ((w - hold) / half) as f32) // current view fades out
    } else {
        ((idx + 1) % n, ((w - hold - half) / half) as f32) // next view fades in
    }
}

/// Blend every span's foreground toward the background by `1-alpha` (alpha=1 →
/// untouched, alpha=0 → invisible) — the per-frame fade used by dissolves.
fn faded_lines(lines: Vec<Line<'static>>, alpha: f32) -> Vec<Line<'static>> {
    if alpha >= 0.999 {
        return lines;
    }
    let f = 1.0 - alpha.clamp(0.0, 1.0);
    lines
        .into_iter()
        .map(|ln| {
            let spans: Vec<Span<'static>> = ln
                .spans
                .into_iter()
                .map(|sp| {
                    let base = sp.style.fg.unwrap_or(c::TEXT);
                    let style = sp.style.fg(c::blend(base, c::BG, f));
                    Span::styled(sp.content, style)
                })
                .collect();
            Line::from(spans)
        })
        .collect()
}

/// A metadata line (title / artist · album) that shimmers, and — when it
/// overflows `width` — gently cross-dissolves between successive windows of the
/// text instead of cell-scrolling. `lead` is pinned at the left (transport icon).
fn cycle_dissolve(
    lead: Vec<Span<'static>>,
    lead_w: usize,
    segs: &[(&str, Color, bool)],
    width: usize,
    t: f64,
    row: f32,
) -> Line<'static> {
    let mut toks: Vec<(char, Color, bool)> = Vec::new();
    for (txt, col, bold) in segs {
        for ch in txt.chars() {
            toks.push((ch, *col, *bold));
        }
    }
    let avail = width.saturating_sub(lead_w).max(1);
    if toks.len() <= avail {
        return shimmer_window(lead, lead_w, &toks, width, t, row, 1.0);
    }
    // Window start offsets covering the whole string (slight overlap for context).
    let max_off = toks.len() - avail;
    let step = (avail * 3 / 4).max(1);
    let mut offs: Vec<usize> = Vec::new();
    let mut o = 0usize;
    loop {
        offs.push(o.min(max_off));
        if o >= max_off {
            break;
        }
        o += step;
    }
    let (i, alpha) = dissolve_phase(offs.len(), t, 2.8, 0.85);
    let off = offs[i];
    shimmer_window(lead, lead_w, &toks[off..off + avail], width, t, row, alpha)
}

/// Render a token window with the shimmer sweep and a dissolve `alpha`. The lead
/// (icon) stays fully opaque; only the text fades.
fn shimmer_window(
    lead: Vec<Span<'static>>,
    lead_w: usize,
    toks: &[(char, Color, bool)],
    width: usize,
    t: f64,
    row: f32,
    alpha: f32,
) -> Line<'static> {
    let sheen = c::jazz(0.9);
    let head = ((t * 0.40 + row as f64 * 0.18).rem_euclid(1.0)) as f32;
    let total_cols = width.max(1) as f32;
    let f = 1.0 - alpha.clamp(0.0, 1.0);
    let mut spans = lead;
    for (k, (ch, base, bold)) in toks.iter().enumerate() {
        if *ch == ' ' {
            spans.push(Span::raw(" "));
            continue;
        }
        let p = (lead_w + k) as f32 / total_cols;
        let mut d = (p - head).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        let b = (-(d * d) / (2.0 * 0.16 * 0.16)).exp();
        let col = c::blend(c::blend(*base, sheen, b), c::BG, f);
        let mut st = Style::default().fg(col);
        if *bold {
            st = st.add_modifier(Modifier::BOLD);
        }
        spans.push(Span::styled(ch.to_string(), st));
    }
    Line::from(spans)
}

fn panel(title: &str, hot: bool) -> Block<'_> {
    // Every card now wears the violet "hot" border — it reads as one cohesive,
    // lively board. `hot` is kept for callers but no longer dims the frame.
    let _ = hot;
    let border = c::PANEL_BORDER_HOT;
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border).add_modifier(Modifier::BOLD))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
        ))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG))
}

fn footer(f: &mut Frame, area: Rect, s: &AppState) {
    // Full-width status bar with a subtle background.
    f.render_widget(
        Block::default().style(Style::default().bg(c::PANEL_BORDER)),
        area,
    );
    let sys = &s.system;
    let si = &s.silicon;
    let memf = if sys.mem_total > 0 { sys.mem_used as f32 / sys.mem_total as f32 } else { 0.0 };
    let cpu = if si.fresh { si.cpu_pct } else { sys.cpu_overall };
    let ncpu = sys.per_core.len().max(1) as f64;

    let sep = || Span::styled("  │  ", Style::default().fg(c::FAINT).bg(c::PANEL_BORDER));
    let lbl = |t: &str| Span::styled(t.to_string(), Style::default().fg(c::DIM).bg(c::PANEL_BORDER));
    let val = |t: String, col| Span::styled(t, Style::default().fg(col).bg(c::PANEL_BORDER).add_modifier(Modifier::BOLD));
    // Jazz family for normal readouts; RED only when a real threshold is crossed.
    let temp_col = if si.cpu_temp_c > 80.0 { c::RED } else { c::jazz((si.cpu_temp_c - 30.0) / 70.0) };
    let load_alarm = sys.load.0 > ncpu;
    let cpu_col = if load_alarm { c::RED } else { c::jazz(cpu / 100.0) };

    let mut spans = vec![
        Span::styled(" q ", Style::default().fg(c::BG).bg(c::ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(" quit", Style::default().fg(c::DIM).bg(c::PANEL_BORDER)),
        sep(),
        lbl("CPU "),
        val(format!("{cpu:>3.0}%"), cpu_col),
    ];
    if si.fresh {
        spans.push(sep());
        spans.push(lbl("GPU "));
        spans.push(val(format!("{:>3.0}%", si.gpu_pct), c::jazz(si.gpu_pct / 100.0)));
        spans.push(sep());
        spans.push(lbl("PWR "));
        spans.push(val(format!("{:>5.1}W", si.all_power_w), c::jazz((si.all_power_w / 120.0).clamp(0.0, 1.0))));
        spans.push(sep());
        spans.push(lbl("TEMP "));
        spans.push(val(format!("{:>3.0}°", si.cpu_temp_c), temp_col));
    }
    spans.push(sep());
    spans.push(lbl("MEM "));
    spans.push(val(format!("{:>3.0}%", memf * 100.0), c::jazz(memf)));
    spans.push(sep());
    spans.push(lbl("NET "));
    spans.push(Span::styled(format!("▼{:>5}", fmt_rate_short(sys.net_rx_bps)), Style::default().fg(c::CYAN).bg(c::PANEL_BORDER)));
    spans.push(Span::styled(format!(" ▲{:>5}", fmt_rate_short(sys.net_tx_bps)), Style::default().fg(c::PINK).bg(c::PANEL_BORDER)));
    if s.usage.fresh {
        spans.push(sep());
        spans.push(lbl("Claude today "));
        spans.push(val(format!("${:>5.2}", s.usage.today_cost), c::GREEN));
    }
    spans.push(sep());
    spans.push(lbl("↑ "));
    spans.push(Span::styled(format!("{:>9}", fmt_dur(sys.uptime_secs)), Style::default().fg(c::TEXT).bg(c::PANEL_BORDER)));
    spans.push(sep());
    spans.push(Span::styled(format!("{:>3} procs", sys.proc_count), Style::default().fg(c::DIM).bg(c::PANEL_BORDER)));

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Half-width CPU equalizer: one vertical bar per core, height = that core's
/// real, delay-interpolated load. The Catmull-Rom playback (sampled at
/// `now - EQ_DELAY`, advanced every frame) makes the bars glide smoothly
/// between the 1 Hz samples — honest data, buttery motion. Green→red gradient.
fn cpu_eq_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let vals = sampled_cores(&s.cpu_samples, target);
    let n = vals.len();
    let overall = if n > 0 { vals.iter().sum::<f32>() / n as f32 } else { 0.0 };
    let load = s.system.load.0;
    let title = format!("CPU   {overall:>2.0}%   loadavg {load:.1}%");
    let block = panel(&title, overall > 80.0);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if n == 0 || inner.height < 2 || inner.width < 8 {
        return;
    }

    // Per-core heights = real, delay-interpolated load. No fabricated motion;
    // the smoothness is the continuous Catmull-Rom playback between samples.
    let heights: Vec<f32> = (0..n).map(|core| (vals[core] / 100.0).clamp(0.0, 1.0)).collect();

    let bar_rows = (inner.height as usize).saturating_sub(1); // reserve last row for labels
    let iw = inner.width as usize;
    let slot = (iw / n).max(2);
    let gap = if slot >= 6 { 2 } else { 1 };
    let bar_w = slot.saturating_sub(gap).max(1);
    let used = n * slot;
    let lead = iw.saturating_sub(used) / 2; // center the whole equalizer

    let mut lines: Vec<Line> = Vec::with_capacity(bar_rows + 1);
    for r in 0..bar_rows {
        let row_top = (bar_rows - r) as f32 / bar_rows as f32;
        let row_bot = (bar_rows - r - 1) as f32 / bar_rows as f32;
        let mid = (row_top + row_bot) * 0.5;
        let mut spans: Vec<Span> = Vec::with_capacity(n * 2 + 1);
        if lead > 0 {
            spans.push(Span::raw(" ".repeat(lead)));
        }
        for core in 0..n {
            let h = heights[core];
            let glyph = if h >= row_top {
                '█'
            } else if h > row_bot {
                c::vblock((h - row_bot) / (row_top - row_bot))
            } else {
                ' '
            };
            if glyph == ' ' {
                spans.push(Span::raw(" ".repeat(bar_w)));
            } else {
                // Synthwave spectrum: hue follows the core's REAL load (0.75*h)
                // with a gentle vertical lift (0.25*mid) so each column keeps a
                // legible bottom→top gradient. Idle cores stay blue, busy cores
                // climb violet→pink→white. No green/orange heat anywhere.
                let col = c::jazz((0.25 * mid + 0.75 * h).clamp(0.0, 1.0));
                let st = Style::default().fg(col);
                spans.push(Span::styled(glyph.to_string().repeat(bar_w), st));
            }
            if core < n {
                spans.push(Span::raw(" ".repeat(gap)));
            }
        }
        lines.push(Line::from(spans));
    }

    // Bottom row: faint core indices centered under each bar.
    let mut label_spans: Vec<Span> = Vec::with_capacity(n + 1);
    if lead > 0 {
        label_spans.push(Span::raw(" ".repeat(lead)));
    }
    for c in 0..n {
        let lab = format!("{c}");
        let padl = (bar_w.saturating_sub(lab.len())) / 2;
        let padr = bar_w.saturating_sub(lab.len()).saturating_sub(padl);
        label_spans.push(Span::styled(
            format!("{}{}{}", " ".repeat(padl), lab, " ".repeat(padr)),
            Style::default().fg(c::FAINT),
        ));
        label_spans.push(Span::raw(" ".repeat(gap)));
    }
    lines.push(Line::from(label_spans));

    f.render_widget(Paragraph::new(lines), inner);
}

fn resources_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("MEM · DISK · NET", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 || inner.width < 8 {
        return;
    }

    let memf = if s.system.mem_total > 0 {
        s.system.mem_used as f32 / s.system.mem_total as f32
    } else {
        0.0
    };
    let diskf = if s.system.disk_total > 0 {
        s.system.disk_used as f32 / s.system.disk_total as f32
    } else {
        0.0
    };
    let iw = inner.width as usize;
    let bw = iw.saturating_sub(30).clamp(8, 32);

    let mut lines: Vec<Line> = Vec::with_capacity(inner.height as usize);
    let mut mem_line = vec![Span::styled("mem  ", Style::default().fg(c::DIM))];
    mem_line.extend(bar_spans(memf, bw, c::jazz(memf)));
    mem_line.push(Span::styled(
        format!(" {:>3.0}%  {} / {}", memf * 100.0, fmt_bytes(s.system.mem_used), fmt_bytes(s.system.mem_total)),
        Style::default().fg(c::TEXT),
    ));
    lines.push(Line::from(mem_line));
    let mut disk_line = vec![Span::styled("disk ", Style::default().fg(c::DIM))];
    disk_line.extend(bar_spans(diskf, bw, c::jazz(diskf)));
    disk_line.push(Span::styled(
        format!(" {:>3.0}%  {} / {}", diskf * 100.0, fmt_bytes(s.system.disk_used), fmt_bytes(s.system.disk_total)),
        Style::default().fg(c::TEXT),
    ));
    lines.push(Line::from(disk_line));
    lines.push(Line::from(vec![
        Span::styled("net  ", Style::default().fg(c::DIM)),
        Span::styled("▼ ", Style::default().fg(c::CYAN)),
        Span::styled(format!("{:>10}", fmt_rate(s.system.net_rx_bps)), Style::default().fg(c::CYAN)),
        Span::styled("    ▲ ", Style::default().fg(c::PINK)),
        Span::styled(format!("{:>10}", fmt_rate(s.system.net_tx_bps)), Style::default().fg(c::PINK)),
    ]));

    // Top rows: the mem/disk/net text. Bottom rows: a real, continuously-
    // scrolling area graph of total throughput (delay-interpolated, log-scaled
    // ~1 KB/s..10 MB/s). Honest data, frame-smooth motion.
    let text_h = lines.len() as u16;
    f.render_widget(
        Paragraph::new(lines),
        Rect { x: inner.x, y: inner.y, width: inner.width, height: text_h },
    );

    let plot_h = inner.height.saturating_sub(text_h);
    if plot_h >= 1 {
        let plot = Rect { x: inner.x, y: inner.y + text_h, width: inner.width, height: plot_h };
        let vals = series(
            &s.net_samples,
            plot.width as usize,
            sampled_scalar,
            |bps| ((bps.max(1.0).log10() - 3.0) / 4.0).clamp(0.0, 1.0),
        );
        area_graph(f, plot, &vals, Fill::Jazz);
    }
}

fn silicon_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let hot = s.silicon.gpu_temp_c > 80.0 || s.silicon.cpu_temp_c > 90.0;
    let block = panel("APPLE SILICON", hot);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if !s.silicon.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("waiting for macmon…", Style::default().fg(c::DIM))),
            inner,
        );
        return;
    }

    // Smooth, delay-interpolated metrics so every gauge glides like CPU/net.
    let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let v = sampled_cores(&s.silicon_samples, target);
    let g = |i: usize, fallback: f32| v.get(i).copied().unwrap_or(fallback);
    let si = &s.silicon;
    let gpu = g(0, si.gpu_pct);
    let ctemp = g(2, si.cpu_temp_c);
    let gtemp = g(3, si.gpu_temp_c);
    let gpw = g(5, si.gpu_power_w);
    let spw = g(6, si.sys_power_w);
    let ecpu = g(7, si.ecpu_pct);
    let pcpu = g(8, si.pcpu_pct);

    // Gauges in a fixed-width column on the left; the smooth power-draw graph
    // fills everything to the right edge so it sits snug against the gauges
    // (no awkward gap) and reads the same width as the NET wave.
    // 34 (not 38) so the right-side stat block keeps ~22 cols: the strings
    // 'cpu 9.2W   ane 0.0W' need 21, and at bw=10 the gauge line still fits.
    let gauge_w = 34u16.min(inner.width);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(gauge_w), Constraint::Min(0)])
        .split(inner);
    let gauge_area = cols[0];
    let graph_area = cols[1];

    let bw = (gauge_area.width as usize).saturating_sub(24).clamp(8, 30);
    let tnorm = |t: f32| ((t - 30.0) / 70.0).clamp(0.0, 1.0); // 30–100 °C → 0–1
    let gpu_share = if spw > 0.1 { (gpw / spw * 100.0).clamp(0.0, 100.0) } else { 0.0 };

    // label, value bar (0..1), bar color, trailing text
    let gauge = |label: &str, frac: f32, col: ratatui::style::Color, tail: String| {
        let mut spans = vec![Span::styled(format!("{label:<6} "), Style::default().fg(c::DIM))];
        spans.extend(bar_spans(frac, bw, col));
        spans.push(Span::styled(tail, Style::default().fg(c::TEXT)));
        Line::from(spans)
    };

    // Jazzy synthwave bars: blue→violet→pink→white by fill, so the whole box
    // reads purple/pink/blue/white. Magnitude still scans (low=blue, high=white).
    let lines = vec![
        gauge("gpu", gpu / 100.0, c::jazz(gpu / 100.0), format!(" {:>4.0}%  {} MHz", gpu, si.gpu_freq_mhz)),
        gauge("e-cpu", ecpu / 100.0, c::jazz(ecpu / 100.0), format!(" {:>4.0}%  {} MHz", ecpu, si.ecpu_freq_mhz)),
        gauge("p-cpu", pcpu / 100.0, c::jazz(pcpu / 100.0), format!(" {:>4.0}%  {} MHz", pcpu, si.pcpu_freq_mhz)),
        {
            let pf = (spw / 120.0).clamp(0.0, 1.0);
            let mut spans = vec![Span::styled("power  ", Style::default().fg(c::DIM))];
            spans.extend(bar_spans(pf, bw, c::jazz(pf)));
            spans.push(Span::styled(format!(" {:>4.1}W", spw), Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)));
            Line::from(spans)
        },
        Line::from(""), // padding under the power bar
        Line::from(vec![
            Span::styled("       gpu ", Style::default().fg(c::FAINT)),
            Span::styled(format!("{gpw:.1} W", ), Style::default().fg(c::DIM)),
            Span::styled(format!("  ·  {gpu_share:.0}% of draw"), Style::default().fg(c::FAINT)),
        ]),
        Line::from(""),
        gauge("cpu °C", tnorm(ctemp), c::jazz(tnorm(ctemp)), format!(" {ctemp:>3.0}°C")),
        gauge("gpu °C", tnorm(gtemp), c::jazz(tnorm(gtemp)), format!(" {gtemp:>3.0}°C")),
    ];
    f.render_widget(Paragraph::new(lines), gauge_area);

    // Right side: a shimmering SoC power-breakdown stat block, then the live
    // system-power draw scrolling smoothly over GRAPH_WINDOW (jazz gradient).
    if graph_area.width > 6 {
        let g = graph_area;
        let si = &s.silicon;
        let stats = [
            format!("cpu {:>4.1}W   ane {:>4.1}W", si.cpu_power_w, si.ane_power_w),
            format!("gpu {:>4.1}W   pkg {:>4.1}W", si.gpu_power_w, si.all_power_w),
        ];
        for (i, line) in stats.iter().enumerate() {
            f.render_widget(
                Paragraph::new(shimmer_text(line, t, i as f32)),
                Rect { x: g.x, y: g.y + i as u16, width: g.width, height: 1 },
            );
        }
        let top = stats.len() as u16 + 1; // stats + one blank row
        let plot = Rect { x: g.x, y: g.y + top, width: g.width, height: g.height.saturating_sub(top) };
        let vals = series(
            &s.silicon_samples,
            plot.width as usize,
            |buf, t| sampled_channel(buf, 6, t),
            |w| (w / 120.0).clamp(0.0, 1.0),
        );
        area_graph(f, plot, &vals, Fill::Jazz);
    }
}

/// Merged "ROBOTS WORKING" card: left column = Claude token throughput
/// (today/week/month) + 30-day sessions + the burn bar chart; right column =
/// the most-recently-active local git branch's pulse. A shimmery jazz divider
/// sweeps between them.
fn robots_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let block = panel("ROBOTS WORKING", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let u = &s.usage;
    let g = &s.git;
    if inner.height < 2 || inner.width < 24 {
        return;
    }

    let h = inner.height as usize;
    // Left column kept narrow so the divider sits left and the commit message
    // (right column) gets the room.
    let lw = ((inner.width as usize * 34 / 100).max(16))
        .min((inner.width as usize).saturating_sub(24)) as u16;
    let div_x = inner.x + lw + 1; // 1-col gutter, then divider
    let rx = div_x + 2; // 1-col gutter after the divider
    let rw = (inner.x + inner.width).saturating_sub(rx);

    // ----- shimmery jazz divider: a base blue→pink gradient down the line with
    // a bright white-pink glint that sweeps downward over time -----
    let head = ((t * 0.5).rem_euclid(1.0)) as f32;
    for r in 0..h {
        let p = r as f32 / h.max(1) as f32;
        let mut d = (p - head).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        let glint = (-(d * d) / (2.0 * 0.13 * 0.13)).exp();
        let col = c::blend(c::jazz(0.22 + 0.58 * p), c::jazz(0.97), glint);
        f.render_widget(
            Paragraph::new(Span::styled("│", Style::default().fg(col))),
            Rect { x: div_x, y: inner.y + r as u16, width: 1, height: 1 },
        );
    }

    // ----- left: token windows + burn bar chart -----
    if !u.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("scanning ~/.claude…", Style::default().fg(c::DIM))),
            Rect { x: inner.x, y: inner.y, width: lw, height: 1 },
        );
    } else {
        let tot_today = u.today_input + u.today_output + u.today_cache_read + u.today_cache_write;
        let row = |label: &str, val: String, col| {
            Line::from(vec![
                Span::styled(format!("{label:<8} "), Style::default().fg(c::DIM)),
                Span::styled(val, Style::default().fg(col).add_modifier(Modifier::BOLD)),
            ])
        };
        let text = vec![
            row("today", fmt_tokens(tot_today), c::CYAN),
            row("week", fmt_tokens(u.tokens_7d), c::ACCENT),
            row("month", fmt_tokens(u.tokens_30d), c::PINK),
            row("sessions", format!("{}", u.sessions_30d), c::TEXT),
        ];
        let tlen = text.len() as u16;
        f.render_widget(
            Paragraph::new(text),
            Rect { x: inner.x, y: inner.y, width: lw, height: tlen.min(inner.height) },
        );
        // The loved burn bar chart, pinned to the bottom row of the left column.
        if inner.height > tlen {
            // "burn" label padded to the value-label width (8+1) so the bars
            // begin in the same column as the today/week/month numbers above.
            let mut burn = vec![Span::styled(format!("{:<8} ", "burn"), Style::default().fg(c::DIM))];
            // Window today's per-hour burn so it ENDS at the current hour — the
            // live hour is the rightmost bar. Showing the raw tail of the 24h
            // array would render the empty late-day/future hours (all blank).
            let today_so_far = if u.hourly.is_empty() {
                &u.hourly[..]
            } else {
                let hh = (Local::now().hour() as usize).min(u.hourly.len() - 1);
                &u.hourly[..=hh]
            };
            burn.extend(jazz_spark(today_so_far, (lw as usize).saturating_sub(9)));
            f.render_widget(
                Paragraph::new(Line::from(burn)),
                Rect { x: inner.x, y: inner.y + inner.height - 1, width: lw, height: 1 },
            );
        }
    }

    // ----- right: the active git branch's pulse -----
    if rw < 8 {
        return;
    }
    let rwn = rw as usize;
    if !g.fresh || !g.ok {
        let msg = if !g.fresh { "reading repo…" } else { "no git repo" };
        f.render_widget(
            Paragraph::new(Span::styled(msg, Style::default().fg(c::DIM))),
            Rect { x: rx, y: inner.y, width: rw, height: 1 },
        );
        return;
    }
    // Branch row sits flush at the right column's left edge (no glyph/pad) so the
    // repo/branch lines up exactly with the commit hash below it.
    let mut branch_spans: Vec<Span> = Vec::new();
    let mut used = 0usize;
    if !g.repo.is_empty() {
        let r = format!("{}/", g.repo);
        used += r.chars().count();
        branch_spans.push(Span::styled(r, Style::default().fg(c::DIM)));
    }
    branch_spans.push(Span::styled(
        truncate(&g.branch, rwn.saturating_sub(used).max(3)),
        Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
    ));
    let branch = Line::from(branch_spans);
    let hashw = g.last_hash.chars().count() + 1;
    let commit = Line::from(vec![
        Span::styled(format!("{} ", g.last_hash), Style::default().fg(c::CYAN)),
        Span::styled(truncate(&g.last_msg, rwn.saturating_sub(hashw)), Style::default().fg(c::TEXT)),
    ]);
    let age = Line::from(Span::styled(
        if g.last_rel.is_empty() { "—".to_string() } else { g.last_rel.clone() },
        Style::default().fg(c::FAINT),
    ));
    // Branch activity — one metric per line.
    let loc = Line::from(vec![
        Span::styled(format!("+{}", g.loc_added), Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" -{}", g.loc_removed), Style::default().fg(if g.loc_removed > 0 { c::RED } else { c::FAINT })),
        Span::styled(" loc", Style::default().fg(c::FAINT)),
    ]);
    let commits = Line::from(vec![
        Span::styled(format!("{}", g.branch_commits), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(" commits", Style::default().fg(c::DIM)),
    ]);
    let prs = Line::from(vec![
        Span::styled(format!("{}", g.pr_count), Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)),
        Span::styled(" PRs", Style::default().fg(c::DIM)),
    ]);
    let merges = Line::from(vec![
        Span::styled(
            format!("{}", g.merges_main),
            Style::default().fg(if g.merges_main > 0 { c::GREEN } else { c::TEXT }).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" merges → main", Style::default().fg(c::DIM)),
    ]);
    f.render_widget(
        Paragraph::new(vec![branch, commit, age, Line::from(""), loc, commits, prs, merges]),
        Rect { x: rx, y: inner.y, width: rw, height: inner.height },
    );
}

/// The top band of the right column: NOW PLAYING on the left, a LINER NOTES card
/// (interesting facts about the track/album/artist) on the right. Splitting the
/// old full-width player reclaims its dead space for something worth reading.
fn now_playing_row(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    // Narrow terminals: drop the facts card and let the player keep the row.
    if area.width < 64 {
        now_playing(f, area, s, t);
        return;
    }
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);
    now_playing(f, cols[0], s, t);
    facts_panel(f, cols[1], s, t);
}

/// The lyrics band: LYRICS on the left, the Apple Music QUEUE on the right —
/// the same 65/35 split as the player row above so the cards line up.
fn lyrics_row(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    if area.width < 64 {
        lyrics_panel(f, area, s, t);
        return;
    }
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);
    lyrics_panel(f, cols[0], s, t);
    queue_panel(f, cols[1], s);
}

/// QUEUE card: the next few tracks Apple Music will play, numbered, with the
/// artist + length beneath each. Best-effort from the current playlist order.
fn queue_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let q = &s.queue;
    let m = &s.music;
    let block = panel("QUEUE", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 6 || inner.height < 2 {
        return;
    }
    if !m.running || m.track.is_empty() || q.items.is_empty() {
        let note = if !m.running || m.track.is_empty() {
            "♫"
        } else if !q.fresh {
            "…"
        } else {
            "nothing up next"
        };
        f.render_widget(
            Paragraph::new(Span::styled(note.to_string(), Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    let w = inner.width as usize;
    let accents = [c::CYAN, c::PINK, c::GREEN];
    let mut lines: Vec<Line> = Vec::new();
    for (i, it) in q.items.iter().enumerate() {
        let col = accents[i % accents.len()];
        lines.push(Line::from(vec![
            Span::styled(format!("{}  ", i + 1), Style::default().fg(col).add_modifier(Modifier::BOLD)),
            Span::styled(truncate(&it.track, w.saturating_sub(3)), Style::default().fg(c::TEXT)),
        ]));
        let sub = if it.duration > 0.0 {
            format!("   {}  ·  {}", it.artist, fmt_clock(it.duration))
        } else {
            format!("   {}", it.artist)
        };
        lines.push(Line::from(Span::styled(truncate(&sub, w), Style::default().fg(c::DIM))));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The LINER NOTES card: a bulleted list of facts about the current track,
/// word-wrapped with a hanging indent. Title carries a faint source tag.
fn facts_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let fa = &s.facts;
    let block = panel("FACTS", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 6 || inner.height < 2 {
        return;
    }

    // No facts yet: show the status note (or a neutral glyph), centered.
    let m = &s.music;
    if fa.lines.is_empty() || fa.track_id != m.track_id() {
        let note = if !m.running || m.track.is_empty() {
            "♫"
        } else if fa.track_id != m.track_id() || fa.note.is_empty() {
            "gathering liner notes…"
        } else {
            &fa.note
        };
        f.render_widget(
            Paragraph::new(Span::styled(note.to_string(), Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let width = inner.width as usize;
    let max_rows = (inner.height as usize).max(1);
    let bullets = [c::CYAN, c::PINK, c::GREEN, c::YELLOW, c::ACCENT];

    // Render each fact to its wrapped, bulleted block of lines.
    let blocks: Vec<Vec<Line>> = fa
        .lines
        .iter()
        .enumerate()
        .map(|(i, fact)| {
            let color = bullets[i % bullets.len()];
            wrap_text(fact, width.saturating_sub(2).max(4))
                .into_iter()
                .enumerate()
                .map(|(j, seg)| {
                    if j == 0 {
                        Line::from(vec![
                            Span::styled("• ", Style::default().fg(color).add_modifier(Modifier::BOLD)),
                            Span::styled(seg, Style::default().fg(c::TEXT)),
                        ])
                    } else {
                        Line::from(vec![Span::raw("  "), Span::styled(seg, Style::default().fg(c::DIM))])
                    }
                })
                .collect()
        })
        .collect();

    // Pack whole facts into pages that fit the card height (never split a fact
    // across a page unless it alone is taller than the card).
    let mut pages: Vec<Vec<Line>> = Vec::new();
    let mut cur: Vec<Line> = Vec::new();
    for blk in blocks {
        if !cur.is_empty() && cur.len() + blk.len() > max_rows {
            pages.push(std::mem::take(&mut cur));
        }
        cur.extend(blk);
    }
    if !cur.is_empty() {
        pages.push(cur);
    }
    if pages.is_empty() {
        return;
    }

    // One page → static. Many → buttery cross-dissolve (no cell-stepped scroll).
    let (pi, alpha) = dissolve_phase(pages.len(), t, 3.4, 0.9);
    let page = pages.into_iter().nth(pi).unwrap_or_default();
    f.render_widget(Paragraph::new(faded_lines(page, alpha)), inner);
}

/// Greedy word-wrap to `width` columns (whitespace-delimited; long words are
/// hard-split so nothing overflows the card).
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > width {
            // Flush, then hard-split the over-long token.
            if !line.is_empty() {
                out.push(std::mem::take(&mut line));
            }
            let mut chunk = String::new();
            for ch in word.chars() {
                if chunk.chars().count() == width {
                    out.push(std::mem::take(&mut chunk));
                }
                chunk.push(ch);
            }
            line = chunk;
            continue;
        }
        let need = if line.is_empty() { word.chars().count() } else { line.chars().count() + 1 + word.chars().count() };
        if need > width {
            out.push(std::mem::take(&mut line));
            line.push_str(word);
        } else {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

fn now_playing(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let m = &s.music;
    let hot = m.playing;
    let block = panel("NOW PLAYING", hot);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if !m.running || m.track.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("Apple Music idle ♫", Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    // Square-ish album-art column on the left, inset so it breathes.
    let pad_v: u16 = 1;
    // Keep the cover square: a terminal cell is ~2:1, so a square needs cols ≈
    // 2×rows. When the card is narrow (e.g. split beside LINER NOTES) the width
    // budget wins and we shrink rows to match — never letting it go portrait —
    // while always reserving room for the text column.
    let max_rows = inner.height.saturating_sub(pad_v * 2).max(1);
    let side = (max_rows * 2).min(inner.width.saturating_sub(22)).max(2);
    let art_cols = side;
    let art_rows = (side / 2).max(1);
    let art_area = Rect { x: inner.x + 1, y: inner.y + pad_v, width: art_cols, height: art_rows };
    render_art(f, art_area, &s.album_art, m.track_id() == s.album_art.track_id);

    // Text column starts after the art + a small gap.
    let gap = art_cols + 3;
    let pos = m.position();
    let frac = if m.duration > 0.0 { (pos / m.duration) as f32 } else { 0.0 };
    let icon = if m.playing { "▶" } else { "❚❚" };

    // Text column width — needed before we lay out the (possibly scrolling) title.
    let textw = (inner.width.saturating_sub(gap)) as usize;

    // Title / artist / album shimmer; a line wider than the column marquee-scrolls
    // instead of truncating. The transport icon stays pinned at the left.
    let lead = vec![Span::styled(
        format!("{icon} "),
        Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD),
    )];
    let lead_w = icon.chars().count() + 1;
    let title = cycle_dissolve(lead, lead_w, &[(&m.track, c::TEXT, true)], textw, t, 0.0);
    let meta = cycle_dissolve(
        Vec::new(),
        0,
        &[
            (m.artist.as_str(), c::ACCENT, false),
            ("  ·  ", c::FAINT, false),
            (m.album.as_str(), c::DIM, false),
        ],
        textw,
        t,
        1.0,
    );

    // Progress bar stretches to the right edge of the card: only the two time
    // labels are reserved, the bar takes everything between them.
    let pfx = format!("{} ", fmt_clock(pos));
    let sfx = format!(" {}", fmt_clock(m.duration));
    let bw = textw.saturating_sub(pfx.chars().count() + sfx.chars().count()).max(8);
    let mut progress_spans = vec![Span::styled(pfx, Style::default().fg(c::CYAN))];
    progress_spans.extend(bar_spans(frac, bw, c::PINK));
    progress_spans.push(Span::styled(sfx, Style::default().fg(c::DIM)));

    // Transport word only — the dancing spectrum below carries the motion.
    let status = Line::from(Span::styled(
        if m.playing { "playing" } else { "paused" },
        Style::default().fg(c::FAINT),
    ));

    let tx = inner.x + gap;
    let tw = inner.width.saturating_sub(gap);
    let body = vec![title, meta, Line::from(""), Line::from(progress_spans), status];
    let body_h = body.len() as u16;

    // Text block sits at the top, sharing a baseline with the album art's top.
    let info = Rect { x: tx, y: inner.y + pad_v, width: tw, height: body_h };
    f.render_widget(Paragraph::new(body), info);

    // The spectrum spans the full text width and its bottom edge lines up with
    // the bottom of the album art — never overlapping the text block above it.
    const EQ_H: u16 = 2;
    let art_bottom = inner.y + pad_v + art_rows; // one past the last art row
    let eq_y = art_bottom.saturating_sub(EQ_H).max(info.y + body_h);
    if tw >= 4 && eq_y + EQ_H <= inner.y + inner.height {
        let eq_area = Rect { x: tx, y: eq_y, width: tw, height: EQ_H };
        f.render_widget(Paragraph::new(eq_bars(tw as usize, t, m.playing)), eq_area);
    }
}

/// A synthetic spectrum-analyzer flourish for the NOW PLAYING card. AppleScript
/// exposes no real audio levels, so this is an honest *visualizer*, not a
/// measurement: it only dances while music plays and settles to a flat resting
/// line when paused — it never pretends to be measured spectrum. Two cells tall
/// (16 height steps) with a fixed blue→pink positional gradient keeps the motion
/// buttery and calm (heights move; colours don't strobe).
fn eq_bars(n: usize, t: f64, playing: bool) -> Vec<Line<'static>> {
    spectrum(n, 2, t, playing)
}

/// `n` bars × `rows` cells tall. Bars fill from the bottom; 8 sub-levels per
/// cell. See `eq_bars` for the honesty note — this is a visualizer, not a real
/// FFT: it dances only while playing and rests flat when paused.
fn spectrum(n: usize, rows: usize, t: f64, playing: bool) -> Vec<Line<'static>> {
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let rows = rows.max(1);
    let tf = t as f32;
    let denom = (n as f32 - 1.0).max(1.0);
    let mut grid: Vec<Vec<Span>> = (0..rows).map(|_| Vec::with_capacity(n)).collect();
    for i in 0..n {
        let fi = i as f32;
        let h = if playing {
            // Bass on the left (slow, tall), treble on the right (fast, flickery)
            // — a few incommensurate sinusoids so the pattern never visibly loops.
            let speed = 1.7 + fi * 0.45;
            let a = (tf * speed + fi * 0.7).sin();
            let b = (tf * (speed * 0.5 + 1.3) + fi * 1.9).sin();
            let env = 0.6 + 0.4 * (tf * 0.8 + fi * 0.22).sin();
            let v = (0.5 + 0.5 * a) * 0.6 + (0.5 + 0.5 * b) * 0.4;
            (0.12 + v * env).clamp(0.05, 1.0)
        } else {
            0.06
        };
        let total = (h * (8 * rows) as f32).round() as usize;
        let col = c::jazz(0.18 + 0.62 * (fi / denom));
        for (r, line) in grid.iter_mut().enumerate() {
            let from_bottom = rows - 1 - r; // bottom cell fills first
            let level = total.saturating_sub(from_bottom * 8).min(8);
            line.push(Span::styled(glyphs[level].to_string(), Style::default().fg(col)));
        }
    }
    grid.into_iter().map(Line::from).collect()
}

/// Right-column filler when music isn't playing: three buttery, continuously-
/// scrolling area graphs of real SoC telemetry — CPU load, GPU load, and system
/// power over the last GRAPH_WINDOW. Honest data, frame-smooth motion (the same
/// delay-interpolated playback the gauges use).
fn system_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let si = &s.silicon;
    let cpu = if si.fresh { si.cpu_pct } else { s.system.cpu_overall };
    let title = format!("SYSTEM      cpu {cpu:>2.0}%    gpu {:>2.0}%    {:.0} W", si.gpu_pct, si.sys_power_w);
    let block = panel(&title, false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 3 || inner.width < 8 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Ratio(1, 3), Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)])
        .split(inner);

    let draw = |f: &mut Frame, r: Rect, label: &str, vals: Vec<f32>, fill: Fill| {
        if r.height < 2 {
            return;
        }
        f.render_widget(
            Paragraph::new(Span::styled(label.to_string(), Style::default().fg(c::FAINT))),
            Rect { x: r.x, y: r.y, width: r.width, height: 1 },
        );
        let plot = Rect { x: r.x, y: r.y + 1, width: r.width, height: r.height - 1 };
        area_graph(f, plot, &vals, fill);
    };

    let w = rows[0].width as usize;
    let cpu_vals = series(
        &s.cpu_samples,
        w,
        |buf, t| {
            let v = sampled_cores(buf, t);
            if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
        },
        |x| (x / 100.0).clamp(0.0, 1.0),
    );
    draw(f, rows[0], "cpu load", cpu_vals, Fill::Jazz);
    draw(
        f,
        rows[1],
        "gpu load",
        series(&s.silicon_samples, w, |buf, t| sampled_channel(buf, 0, t), |x| (x / 100.0).clamp(0.0, 1.0)),
        Fill::Jazz,
    );
    draw(
        f,
        rows[2],
        "power",
        series(&s.silicon_samples, w, |buf, t| sampled_channel(buf, 6, t), |x| (x / 120.0).clamp(0.0, 1.0)),
        Fill::Jazz,
    );
}

/// Render the album-art thumbnail as truecolor half-blocks (two vertical
/// pixels per cell via '▀' with fg=top, bg=bottom). Works in any truecolor
/// terminal — no image protocol needed.
fn render_art(f: &mut Frame, area: Rect, art: &crate::state::AlbumArt, matches: bool) {
    let w = area.width as usize;
    let h = area.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mut lines: Vec<Line> = Vec::with_capacity(h);
    let have = matches && !art.px.is_empty();
    // Each cell is one column wide and two half-block rows tall; box-average the
    // source over each sub-pixel's exact footprint so the downscale stays crisp.
    let cw = 1.0 / w as f32;
    let ch = 1.0 / (2.0 * h as f32);
    for row in 0..h {
        let mut spans = Vec::with_capacity(w);
        for col in 0..w {
            let u0 = col as f32 * cw;
            let vt0 = 2.0 * row as f32 * ch;
            let vb0 = (2.0 * row as f32 + 1.0) * ch;
            if have {
                let top = art.sample_area(u0, vt0, u0 + cw, vt0 + ch).unwrap_or([20, 20, 28]);
                let bot = art.sample_area(u0, vb0, u0 + cw, vb0 + ch).unwrap_or([20, 20, 28]);
                spans.push(Span::styled(
                    "▀",
                    Style::default()
                        .fg(ratatui::style::Color::Rgb(top[0], top[1], top[2]))
                        .bg(ratatui::style::Color::Rgb(bot[0], bot[1], bot[2])),
                ));
            } else {
                spans.push(Span::styled("░", Style::default().fg(c::FAINT)));
            }
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), area);
}

fn proc_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("TOP PROCESSES", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let procs = &s.system.top_procs;
    if procs.is_empty() || inner.height < 2 || inner.width < 8 {
        if !procs.is_empty() {
            // Too short to render rows gracefully — skip rather than clip.
            return;
        }
        f.render_widget(Paragraph::new(Span::styled("…", Style::default().fg(c::DIM))), inner);
        return;
    }
    // Column budget: cpu(6) gap(2) mem(6) gap(3) up(6) gap(2) = 25 before name.
    let namew = (inner.width as usize).saturating_sub(25).clamp(8, 40);
    let avail = inner.height as usize;

    // Always show a labeled header row so the columns are self-explanatory; the
    // labels right-align to the same columns as the data below them.
    let mut lines: Vec<Line> = Vec::with_capacity(avail);
    lines.push(Line::from(Span::styled(
        format!("{:>6}  {:>6}   {:>6}  {}", "cpu%", "mem", "uptime", "process"),
        Style::default().fg(c::DIM).add_modifier(Modifier::BOLD),
    )));
    let rows = avail.saturating_sub(lines.len()).min(procs.len());
    for (name, cpu, mem, uptime) in procs.iter().take(rows) {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:>5.1}%", cpu),
                Style::default().fg(c::jazz((*cpu / 100.0).min(1.0))).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {:>6}", fmt_bytes(*mem)), Style::default().fg(c::DIM)),
            Span::styled(format!("   {:>6}", fmt_dur_short(*uptime)), Style::default().fg(c::FAINT)),
            Span::styled(format!("  {}", truncate(name, namew)), Style::default().fg(c::TEXT)),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn weather_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let w = &s.weather;
    let block = panel("WEATHER", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if !w.fresh {
        f.render_widget(Paragraph::new(Span::styled("fetching…", Style::default().fg(c::DIM))), inner);
        return;
    }
    if inner.height < 2 || inner.width < 8 {
        return;
    }
    let iw = inner.width as usize;
    let h = inner.height as usize;

    // night-aware: a clear-sky sun becomes a moon once the local clock is past
    // sunset / before sunrise.
    let icon = night_icon(&w.icon, &w.sunrise, &w.sunset);

    // row 0 — big readout (icon · temp · desc · feels) on the left, sunrise/sunset flush-right.
    let left_txt = format!("{} {}°F  {}  feels {}°", icon, w.temp_f, w.desc, w.feels_f);
    let right_txt = format!("☀ {}  ☾ {}", w.sunrise, w.sunset);
    let pad = iw
        .saturating_sub(left_txt.chars().count() + right_txt.chars().count())
        .max(1);
    let big = Line::from(vec![
        Span::styled(format!("{} ", icon), Style::default().fg(c::YELLOW)),
        Span::styled(format!("{}°F", w.temp_f), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", w.desc), Style::default().fg(c::TEXT)),
        Span::styled(format!("  feels {}°", w.feels_f), Style::default().fg(c::DIM)),
        Span::raw(" ".repeat(pad)),
        Span::styled(
            format!("☀ {}  ☾ {}", w.sunrise, w.sunset),
            Style::default().fg(c::jazz(0.85)),
        ),
    ]);

    // row 1 — hi-lo (warm up = pink, cool down = cyan) / hum / UV.
    let detail = Line::from(vec![
        Span::styled(format!("↑{}°", w.hi_f), Style::default().fg(c::PINK)),
        Span::styled(" ", Style::default().fg(c::FAINT)),
        Span::styled(format!("↓{}°", w.lo_f), Style::default().fg(c::CYAN)),
        Span::styled(format!("   hum {}%", w.humidity), Style::default().fg(c::FAINT)),
        Span::styled("   UV ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}", w.uv), Style::default().fg(c::jazz((w.uv as f32 / 11.0).clamp(0.0, 1.0)))),
    ]);

    // row 2 — atmosphere: wind / chance of rain / barometric pressure.
    let atmos = Line::from(vec![
        Span::styled(format!("💨 {} {} mph", w.wind_dir, w.wind_mph), Style::default().fg(c::CYAN)),
        Span::styled("   ☔ rain ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}%", w.precip_chance), Style::default().fg(c::jazz((w.precip_chance as f32 / 100.0).clamp(0.0, 1.0)))),
        Span::styled(format!("   {} mb", w.pressure_mb), Style::default().fg(c::DIM)),
    ]);

    let mut lines: Vec<Line> = vec![big, detail, atmos];

    // bottom row: location flush-left, hourly temp strip (next ~12h of forecast
    // temps, jazz-colored) flush-right on the same line — preceded by one blank
    // line so it sits pinned to the card's bottom edge.
    let remaining = iw.saturating_sub(w.location.chars().count());
    let mut bottom = vec![Span::styled(w.location.clone(), Style::default().fg(c::FAINT))];
    if !w.temp_strip.is_empty() && remaining > 0 {
        bottom.extend(jazz_spark(&w.temp_strip, remaining));
    }
    if h > 3 {
        lines.push(Line::from(""));
    }
    lines.push(Line::from(bottom));

    f.render_widget(Paragraph::new(lines), inner);
}

/// Swap a clear-sky sun for a moon when the local clock is past sunset or before
/// sunrise. Any other condition icon (cloud, rain, fog, …) is returned unchanged.
fn night_icon(icon: &str, sunrise: &str, sunset: &str) -> String {
    use chrono::Timelike;
    if icon != "☀" {
        return icon.to_string();
    }
    let to_min = |s: &str| -> Option<u32> {
        let (h, m) = s.split_once(':')?;
        Some(h.trim().parse::<u32>().ok()? * 60 + m.trim().parse::<u32>().ok()?)
    };
    let now = chrono::Local::now();
    let now_min = now.hour() * 60 + now.minute();
    match (to_min(sunrise), to_min(sunset)) {
        (Some(sr), Some(ss)) if now_min < sr || now_min >= ss => "🌙".to_string(),
        _ => icon.to_string(),
    }
}

/// Exact rows a message card needs for its content (incl. borders) so the layout
/// can size it tight — no trailing gap below the last conversation. `active` adds
/// the focused iMESSAGE footer (separator + keybind hint).
fn card_height(m: &crate::state::Messages, active: bool) -> u16 {
    if !m.fresh || !m.available {
        return 4; // graceful gate message (1-2 lines + borders)
    }
    let n = m.items.len().clamp(1, 5) as u16;
    let footer = if active { 2 } else { 0 };
    2 + n + footer
}

/// iMESSAGE card: unread badge in the title, a list of recent inbound messages
/// (focus marker · sender · preview · rel-time · unread dot), and an inline reply
/// input that wipes open on a double-press. All motion is interpolated each
/// frame off `s.msg_ui.anim_start` — never a discrete flip.
fn messages_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    use crate::state::MsgPhase;
    let msgs = &s.messages;
    let ui = &s.msg_ui;

    // ----- title with unread badge / all-read tick -----
    let hot = ui.active;
    let _ = hot;
    let border = c::PANEL_BORDER_HOT;
    // Settle: pink "● n unread" crossfades to green "✓ all read" over 300ms.
    let badge_blend = match ui.phase {
        MsgPhase::Advancing if msgs.unread_count == 0 => {
            ui.progress(Duration::from_millis(300)).unwrap_or(1.0)
        }
        _ => {
            if msgs.unread_count == 0 {
                1.0
            } else {
                0.0
            }
        }
    };
    let mut title_spans = vec![Span::styled(
        " iMESSAGE ",
        Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
    )];
    if msgs.available && msgs.fresh {
        if msgs.unread_count > 0 {
            let dot = c::blend(c::PINK, c::GREEN, badge_blend);
            title_spans.push(Span::styled(" ● ", Style::default().fg(dot).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled(
                format!("{}", msgs.unread_count),
                Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" unread ", Style::default().fg(c::DIM)));
        } else {
            title_spans.push(Span::styled(" ✓ ", Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled("all read ", Style::default().fg(c::DIM)));
        }
    }

    // Failure flash: border pulses red, fade in 120ms / out 240ms.
    let border = if let Some(fa) = ui.send_failed_at {
        let e = fa.elapsed().as_secs_f32();
        let amp = if e < 0.12 {
            e / 0.12
        } else {
            (1.0 - (e - 0.12) / 0.24).clamp(0.0, 1.0)
        };
        c::blend(border, c::RED, amp)
    } else {
        border
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border).add_modifier(Modifier::BOLD))
        .title(Line::from(title_spans))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 || inner.width < 10 {
        return;
    }

    // ----- graceful gating states -----
    if !msgs.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("reading messages…", Style::default().fg(c::DIM))),
            inner,
        );
        return;
    }
    if !msgs.available {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("✉  can't read Messages", Style::default().fg(c::DIM))),
                Line::from(Span::styled(
                    "grant Full Disk Access to studioboard",
                    Style::default().fg(c::FAINT),
                )),
            ]),
            inner,
        );
        return;
    }
    if msgs.items.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("✉  inbox clear", Style::default().fg(c::FAINT)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let iw = inner.width as usize;
    let ih = inner.height as usize;

    // The composer (when open) consumes the card's reserved bottom slack so the
    // message rows above it never reflow. Reserve 1 row while opening/open.
    let composer_open = ui.composing || ui.phase == MsgPhase::Opening;
    let composer_rows = if composer_open { 1usize } else { 0 };
    // One row for the separator + keybind hint, only while focused and with room.
    let want_footer = ui.active && ih > msgs.items.len() + composer_rows + 1;
    let footer_rows = if want_footer { 2 } else { 0 };
    let list_rows = ih
        .saturating_sub(composer_rows)
        .saturating_sub(footer_rows);
    let shown = list_rows.min(msgs.items.len());

    // Index (within items) of the focused unread message, for the slide marker.
    let unread_indices: Vec<usize> = msgs
        .items
        .iter()
        .enumerate()
        .filter(|(_, m)| m.unread)
        .map(|(i, _)| i)
        .collect();
    let focus_idx = unread_indices.get(ui.queue_pos).copied();

    // Reserved column budget: marker(2) + sender(18) + gap(1) + reltime(4) + dot(2).
    let prevw = iw.saturating_sub(2 + 18 + 1 + 4 + 2).max(6);

    let mut lines: Vec<Line> = Vec::with_capacity(ih);
    for (i, m) in msgs.items.iter().take(shown).enumerate() {
        let is_focus = Some(i) == focus_idx;
        // Read/unread tone, crossfading on Advance for the focused row.
        let adv = if is_focus && ui.phase == MsgPhase::Advancing {
            ui.progress(Duration::from_millis(220)).unwrap_or(1.0)
        } else {
            0.0
        };
        let (sender_col, prev_col, marker_col, dot_col) = if m.unread {
            // unread → (read) as adv goes 0→1
            (
                c::blend(c::TEXT, c::DIM, adv),
                c::blend(c::TEXT, c::FAINT, adv),
                c::blend(c::PINK, c::FAINT, adv),
                c::blend(c::PINK, c::FAINT, adv),
            )
        } else {
            (c::DIM, c::FAINT, c::FAINT, c::FAINT)
        };

        // Focus marker (the only moving element); reserved width 2.
        let marker = if is_focus {
            Span::styled("▌ ", Style::default().fg(marker_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        let sender = pad_width(&m.sender, 18);
        let sender_style = if m.unread && adv < 0.5 {
            Style::default().fg(sender_col).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(sender_col)
        };
        let preview = pad_width(&m.preview, prevw);
        let prev_style = if m.is_rich {
            Style::default().fg(c::FAINT).add_modifier(Modifier::ITALIC)
        } else {
            Style::default().fg(prev_col)
        };
        let dot = if m.unread {
            Span::styled(" ●", Style::default().fg(dot_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            marker,
            Span::styled(sender, sender_style),
            Span::raw(" "),
            Span::styled(preview, prev_style),
            Span::styled(format!("{:>4}", truncate(&m.rel, 4)), Style::default().fg(c::FAINT)),
            dot,
        ]));
    }

    // ----- separator + keybind hint (only while focused) -----
    if want_footer {
        lines.push(Line::from(Span::styled("─".repeat(iw), Style::default().fg(c::FAINT))));
        lines.push(Line::from(Span::styled(
            "  m: read · mm: reply",
            Style::default().fg(c::FAINT),
        )));
    }

    // ----- inline reply composer (vertical-wipe open / close) -----
    if composer_open {
        // Wipe progress: Opening eases 0→1; Closing/Sending eases 1→0.
        let p = match ui.phase {
            MsgPhase::Opening => ui.progress(Duration::from_millis(180)).unwrap_or(1.0),
            MsgPhase::Closing => 1.0 - ui.progress(Duration::from_millis(180)).unwrap_or(1.0),
            _ => 1.0,
        };
        let e = p * p * (3.0 - 2.0 * p); // smoothstep
        let sender = focus_idx
            .and_then(|i| msgs.items.get(i))
            .map(|m| m.sender.clone())
            .unwrap_or_else(|| "message".into());

        // Blinking caret: sine on alpha (~1s period), DIM↔TEXT — soft, not hard.
        let blink = 0.5 + 0.5 * ((t * std::f64::consts::TAU).sin() as f32);
        let caret_col = c::blend(c::FAINT, c::TEXT, blink);

        let prompt_alpha = e; // text fades in with the wipe
        let mut spans = vec![
            Span::styled("↳ ", Style::default().fg(c::blend(c::BG, c::PINK, e))),
            Span::styled(
                format!("reply to {sender}  "),
                Style::default().fg(c::blend(c::BG, c::DIM, prompt_alpha)),
            ),
        ];
        if ui.phase == MsgPhase::Sending {
            // Shimmer the draft away as a send "whoosh".
            spans = vec![Span::styled("↳ ", Style::default().fg(c::PINK))];
            let head = ui.progress(Duration::from_millis(260)).unwrap_or(1.0);
            let chars: Vec<char> = ui.draft.chars().collect();
            let n = chars.len().max(1);
            for (ci, ch) in chars.iter().enumerate() {
                let cp = ci as f32 / n as f32;
                let d = (cp - head).abs();
                let b = (-(d * d) / (2.0 * 0.08 * 0.08)).exp();
                let col = c::blend(c::TEXT, c::jazz(0.88), b);
                spans.push(Span::styled(ch.to_string(), Style::default().fg(col)));
            }
        } else {
            // Live draft, left-truncated so the caret stays visible.
            let budget = iw.saturating_sub(2 + 9 + sender.chars().count() + 3).max(4);
            let draft = &ui.draft;
            let shown_draft: String = if draft.chars().count() > budget {
                let skip = draft.chars().count() - budget;
                draft.chars().skip(skip).collect()
            } else {
                draft.clone()
            };
            spans.push(Span::styled(
                shown_draft,
                Style::default().fg(c::blend(c::BG, c::TEXT, prompt_alpha)),
            ));
            spans.push(Span::styled("▏", Style::default().fg(caret_col)));
        }
        lines.push(Line::from(spans));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

/// SIGNAL card: identical row style to iMESSAGE (sender · preview · rel-time ·
/// unread dot) and unread-badge title, but read-only — Signal Desktop has no send
/// API, so there's no focus marker, reply composer, or mark-read interaction.
fn signal_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let sig = &s.signal;

    // ----- title with unread badge / all-read tick (mirrors iMESSAGE) -----
    let mut title_spans = vec![Span::styled(
        " SIGNAL ",
        Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
    )];
    if sig.available && sig.fresh {
        if sig.unread_count > 0 {
            title_spans.push(Span::styled(" ● ", Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled(
                format!("{}", sig.unread_count),
                Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" unread ", Style::default().fg(c::DIM)));
        } else {
            title_spans.push(Span::styled(" ✓ ", Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled("all read ", Style::default().fg(c::DIM)));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c::PANEL_BORDER_HOT).add_modifier(Modifier::BOLD))
        .title(Line::from(title_spans))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 || inner.width < 10 {
        return;
    }

    // ----- graceful gating states (mirror iMESSAGE) -----
    if !sig.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("reading Signal…", Style::default().fg(c::DIM))),
            inner,
        );
        return;
    }
    if !sig.available {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("✉  Signal locked", Style::default().fg(c::DIM))),
                Line::from(Span::styled(
                    "needs Keychain access + sqlcipher",
                    Style::default().fg(c::FAINT),
                )),
            ]),
            inner,
        );
        return;
    }
    if sig.items.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("✉  inbox clear", Style::default().fg(c::FAINT)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let iw = inner.width as usize;
    let ih = inner.height as usize;
    // Same reserved columns as iMESSAGE: marker(2)+sender(18)+gap(1)+rel(4)+dot(2).
    let prevw = iw.saturating_sub(2 + 18 + 1 + 4 + 2).max(6);
    let shown = ih.min(sig.items.len());

    let mut lines: Vec<Line> = Vec::with_capacity(ih);
    for m in sig.items.iter().take(shown) {
        let (sender_col, prev_col, dot_col) = if m.unread {
            (c::TEXT, c::TEXT, c::PINK)
        } else {
            (c::DIM, c::FAINT, c::FAINT)
        };
        let sender = pad_width(&m.sender, 18);
        let sender_style = if m.unread {
            Style::default().fg(sender_col).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(sender_col)
        };
        let preview = pad_width(&m.preview, prevw);
        let prev_style = if m.is_rich {
            Style::default().fg(c::FAINT).add_modifier(Modifier::ITALIC)
        } else {
            Style::default().fg(prev_col)
        };
        let dot = if m.unread {
            Span::styled(" ●", Style::default().fg(dot_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            Span::raw("  "), // marker column kept for alignment parity (no focus)
            Span::styled(sender, sender_style),
            Span::raw(" "),
            Span::styled(preview, prev_style),
            Span::styled(format!("{:>4}", truncate(&m.rel, 4)), Style::default().fg(c::FAINT)),
            dot,
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

const DISCORD_MAX_VOICE: usize = 3;
const DISCORD_MAX_TEXT: usize = 3;

/// Overlay an env-gated fake voice member so a join can be previewed live:
/// `STUDIOBOARD_FAKE_VOICE="200 club:Ghosty"` (or just a bare name → "200 club").
/// Returns the real list untouched when the env var is unset.
fn fake_voice(real: &[crate::state::VoiceChannel]) -> Vec<crate::state::VoiceChannel> {
    use crate::state::VoiceChannel;
    let mut voice = real.to_vec();
    let spec = match std::env::var("STUDIOBOARD_FAKE_VOICE") {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return voice,
    };
    let (chan, member) = match spec.split_once(':') {
        Some((c, m)) => (c.trim().to_string(), m.trim().to_string()),
        None => ("200 club".to_string(), spec),
    };
    if let Some(vc) = voice.iter_mut().find(|v| v.name.eq_ignore_ascii_case(&chan)) {
        vc.members.push(member);
        vc.members.sort();
    } else {
        voice.push(VoiceChannel { name: chan, members: vec![member] });
    }
    voice
}

/// Rows the Discord card needs: one row per occupied voice channel, then the
/// recent text channels — plus the two border rows. Hugs the content exactly
/// (no separator) so the bottom border flexes as channels come and go.
fn discord_height(d: &crate::state::Discord) -> u16 {
    if !d.fresh || !d.available {
        return 4; // gate message
    }
    let v = fake_voice(&d.voice).len().min(DISCORD_MAX_VOICE);
    let tx = d.text.len().min(DISCORD_MAX_TEXT);
    2 + (v + tx).max(1) as u16
}

/// DISCORD card: occupied voice channels (each with its members; the whole voice
/// block collapses when everyone's disconnected) above a few recent text channels
/// rendered iMessage/Signal-style (channel · "author: last message" · rel · dot).
fn discord_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let d = &s.discord;
    let voice = fake_voice(&d.voice); // real list, plus any env-gated preview join
    let in_voice: usize = voice.iter().map(|v| v.members.len()).sum();
    let unread_text = d.text.iter().filter(|t| t.unread).count();

    // ----- title with voice / unread badge -----
    let mut title_spans = vec![Span::styled(
        " DISCORD  ",
        Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
    )];
    if d.available && d.fresh {
        if in_voice > 0 {
            title_spans.push(Span::styled(
                format!("{in_voice}"),
                Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" in voice ", Style::default().fg(c::DIM)));
        } else if unread_text > 0 {
            title_spans.push(Span::styled(" ● ", Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled(
                format!("{unread_text}"),
                Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" unread ", Style::default().fg(c::DIM)));
        } else {
            title_spans.push(Span::styled(" ✓ ", Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled("idle ", Style::default().fg(c::DIM)));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c::PANEL_BORDER_HOT).add_modifier(Modifier::BOLD))
        .title(Line::from(title_spans))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 || inner.width < 10 {
        return;
    }

    // ----- gating -----
    if !d.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("connecting to Discord…", Style::default().fg(c::DIM))),
            inner,
        );
        return;
    }
    if !d.available {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("✉  Discord not configured", Style::default().fg(c::DIM))),
                Line::from(Span::styled(
                    "add bot token to Keychain",
                    Style::default().fg(c::FAINT),
                )),
            ]),
            inner,
        );
        return;
    }
    if voice.is_empty() && d.text.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("✉  all quiet", Style::default().fg(c::FAINT)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let iw = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Shared column geometry so voice + text rows line up into one clean table:
    // 2-space indent, an 18-cell name column, a 1-cell gap, then the body.
    let indent = 2usize;
    let namew = 18usize;
    let bodyx = indent + namew + 1;

    // ----- voice section (occupied channels only; collapsed when empty) -----
    // Voice channels read in calm cyan bold — distinct from the dim grey #text
    // rows below (and tied to the cyan "in voice" count), without the jarring
    // rainbow shimmer. The member list shares the text channels' preview column;
    // when too many people are in to fit, the tail collapses to "+N".
    let memw = iw.saturating_sub(bodyx).max(6);
    for vc in voice.iter().take(DISCORD_MAX_VOICE) {
        lines.push(Line::from(vec![
            Span::raw("  "), // column parity with #text + iMESSAGE/SIGNAL rows
            Span::styled(pad_width(&vc.name, namew), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
            Span::styled(fit_members(&vc.members, memw), Style::default().fg(c::DIM)),
        ]));
    }

    // ----- text channels (iMessage/Signal row style) -----
    let prevw = iw.saturating_sub(2 + 18 + 1 + 4 + 2).max(6);
    for tc in d.text.iter().take(DISCORD_MAX_TEXT) {
        let (name_col, prev_col, dot_col) = if tc.unread {
            (c::TEXT, c::TEXT, c::PINK)
        } else {
            (c::DIM, c::FAINT, c::FAINT)
        };
        let name = pad_width(&format!("#{}", tc.name), 18);
        let name_style = if tc.unread {
            Style::default().fg(name_col).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(name_col)
        };
        let body = if tc.author.is_empty() {
            tc.preview.clone()
        } else {
            let who = tc.author.split_whitespace().next().unwrap_or(&tc.author);
            format!("{who}: {}", tc.preview)
        };
        let preview = pad_width(&body, prevw);
        let dot = if tc.unread {
            Span::styled(" ●", Style::default().fg(dot_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            Span::raw("  "), // column parity with iMESSAGE/SIGNAL rows
            Span::styled(name, name_style),
            Span::raw(" "),
            Span::styled(preview, Style::default().fg(prev_col)),
            Span::styled(format!("{:>4}", truncate(&tc.rel, 4)), Style::default().fg(c::FAINT)),
            dot,
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

const DOC_SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Rows the MAC-DOCTOR card needs. Tight per-state so the box hugs its content:
/// a gate hint when syswatch isn't installed, 3 inner rows while a run is live
/// (step · breach · footer), 4 when idle (verdict · trigger · action · footer).
fn doctor_height(d: &crate::state::Doctor) -> u16 {
    if !d.available {
        return 4;
    }
    if d.running {
        5
    } else {
        6
    }
}

fn severity_color(sev: &str) -> ratatui::style::Color {
    match sev {
        "critical" => c::RED,
        "warn" => c::YELLOW,
        "info" => c::CYAN,
        _ => c::DIM,
    }
}

/// Friendly label for the raw "[diagnose] …" log line the agent is currently on.
fn pretty_step(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        "working…".into()
    } else if s.starts_with("start") {
        "starting triage…".into()
    } else if s.starts_with("verdict") {
        "forming verdict…".into()
    } else if s.starts_with("stored") || s == "done" {
        "wrapping up…".into()
    } else {
        s.into()
    }
}

/// Strip the "start — model=… reasons=" noise and a leading bullet so the breach
/// reads as a clean phrase.
fn clean_trigger(s: &str) -> String {
    let s = s.split("reasons=").last().unwrap_or(s).trim();
    s.trim_start_matches('•').trim().to_string()
}

/// MAC-DOCTOR card: live status of the syswatch threshold-triage agent — what
/// it's doing right now while a run is in flight, or the last verdict + what it
/// did when idle.
fn doctor_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let d = &s.doctor;
    let spin = DOC_SPINNER[((t * 12.0) as usize) % DOC_SPINNER.len()];

    // ----- title + status badge -----
    let mut title_spans = vec![Span::styled(
        " MAC-DOCTOR ",
        Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
    )];
    if d.available {
        if d.running {
            title_spans.push(Span::styled(format!(" {spin} "), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled("diagnosing ", Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD)));
        } else {
            title_spans.push(Span::styled(" ✓ ", Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled("watching ", Style::default().fg(c::DIM)));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c::PANEL_BORDER_HOT).add_modifier(Modifier::BOLD))
        .title(Line::from(title_spans))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 || inner.width < 10 {
        return;
    }

    if !d.available {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("⚕ syswatch not installed", Style::default().fg(c::DIM))),
                Line::from(Span::styled(
                    "the threshold watchdog isn't running here",
                    Style::default().fg(c::FAINT),
                )),
            ]),
            inner,
        );
        return;
    }

    let iw = inner.width as usize;
    let trigger = clean_trigger(&d.trigger);
    let mut lines: Vec<Line> = Vec::new();

    if d.running {
        // What it's doing right now.
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{spin} "), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
            Span::styled(
                fit_width(&pretty_step(&d.step), iw.saturating_sub(4)),
                Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD),
            ),
        ]));
        // Why it woke up.
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("breach  ", Style::default().fg(c::PINK)),
            Span::styled(fit_width(&trigger, iw.saturating_sub(10)), Style::default().fg(c::DIM)),
        ]));
    } else {
        // Verdict of the last run: severity dot + headline, age on the right.
        let sev = severity_color(&d.last_severity);
        let title = if d.last_title.is_empty() {
            "no incidents yet — all clear".to_string()
        } else {
            d.last_title.clone()
        };
        let rel = if d.last_rel.is_empty() { String::new() } else { format!("{:>5}", d.last_rel) };
        let titlew = iw.saturating_sub(2 + 2 + dwidth(&rel) + 1).max(6);
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("● ", Style::default().fg(sev).add_modifier(Modifier::BOLD)),
            Span::styled(pad_width(&title, titlew), Style::default().fg(c::TEXT)),
            Span::raw(" "),
            Span::styled(rel, Style::default().fg(c::FAINT)),
        ]));
        // Why it last woke up (or a heartbeat when nothing's fired).
        let (lbl, body) = if trigger.is_empty() {
            ("watching ", "cpu · mem · gpu · disk · thermal".to_string())
        } else {
            ("trigger  ", trigger.clone())
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(lbl, Style::default().fg(c::DIM)),
            Span::styled(fit_width(&body, iw.saturating_sub(2 + dwidth(lbl))), Style::default().fg(c::DIM)),
        ]));
        // What it did — concrete commands when it acted, else the outcome.
        let (atxt, acol) = if !d.last_actions.is_empty() {
            (format!("ran  {}", d.last_actions.join(", ")), c::GREEN)
        } else if d.last_outcome.is_empty() {
            ("standing by".to_string(), c::DIM)
        } else {
            (format!("outcome  {}", d.last_outcome), c::DIM)
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(fit_width(&atxt, iw.saturating_sub(2)), Style::default().fg(acol)),
        ]));
    }

    // Footer: backend model · today's Claude spend · lifetime incident count.
    let mut footer: Vec<Span> = vec![Span::raw("  ")];
    if !d.last_model.is_empty() {
        footer.push(Span::styled(d.last_model.clone(), Style::default().fg(c::ACCENT)));
        footer.push(Span::styled("  ·  ", Style::default().fg(c::FAINT)));
    }
    footer.push(Span::styled(
        format!("${:.2} today", d.today_cost),
        Style::default().fg(if d.today_cost > 0.0 { c::CYAN } else { c::DIM }),
    ));
    footer.push(Span::styled("  ·  ", Style::default().fg(c::FAINT)));
    footer.push(Span::styled(format!("{} incidents", d.incidents_total), Style::default().fg(c::DIM)));
    lines.push(Line::from(footer));

    f.render_widget(Paragraph::new(lines), inner);
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{t}…")
    }
}

/// Display width in terminal cells (emoji / CJK count as 2), so columns line up.
fn dwidth(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    s.width()
}

/// Truncate to a display width of `max` cells, appending '…' if anything is cut.
/// Width-aware so a wide glyph can't overrun the column by a cell.
fn fit_width(s: &str, max: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if dwidth(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0;
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > max.saturating_sub(1) {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

/// Lay out a voice channel's member list into `max` cells: comma-joined names,
/// and if they don't all fit, show as many as do plus a "+N" overflow tag so a
/// busy channel never silently hides who's in it.
fn fit_members(members: &[String], max: usize) -> String {
    if members.is_empty() {
        return String::new();
    }
    let full = members.join(", ");
    if dwidth(&full) <= max {
        return full;
    }
    for take in (1..members.len()).rev() {
        let s = format!("{} +{}", members[..take].join(", "), members.len() - take);
        if dwidth(&s) <= max {
            return s;
        }
    }
    fit_width(&format!("{} +{}", members[0], members.len() - 1), max)
}

/// `fit_width` then right-pad with spaces to exactly `width` display cells.
fn pad_width(s: &str, width: usize) -> String {
    let mut t = fit_width(s, width);
    let w = dwidth(&t);
    if w < width {
        t.push_str(&" ".repeat(width - w));
    }
    t
}

fn lyrics_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let synced = s.lyrics.synced;
    let block = panel("LYRICS", false);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let m = &s.music;
    let ly = &s.lyrics;

    // Hard guard: never show lyrics that don't belong to the current track.
    if !m.running || m.track.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("♫", Style::default().fg(c::FAINT)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    if ly.track_id != m.track_id() {
        f.render_widget(
            Paragraph::new(Span::styled("loading lyrics…", Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    if ly.lines.is_empty() {
        // Still fetching → quiet text; it's about to be replaced.
        if ly.note.contains("loading") || ly.note.is_empty() {
            let note = if ly.note.is_empty() { "…" } else { &ly.note };
            f.render_widget(
                Paragraph::new(Span::styled(note.to_string(), Style::default().fg(c::DIM)))
                    .alignment(Alignment::Center),
                inner,
            );
            return;
        }
        // Genuinely no lyrics: don't leave a lonely note in a big box — keep it
        // alive with a wide animated spectrum and a caption above it.
        let h = inner.height as usize;
        let viz_rows = h.saturating_sub(2).clamp(2, 4);
        let upper = h.saturating_sub(viz_rows);
        let caption_y = inner.y + (upper / 2).max(0) as u16;
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("♫  ", Style::default().fg(c::FAINT)),
                Span::styled(ly.note.clone(), Style::default().fg(c::DIM)),
                Span::styled("  ♫", Style::default().fg(c::FAINT)),
            ]))
            .alignment(Alignment::Center),
            Rect { x: inner.x, y: caption_y, width: inner.width, height: 1 },
        );
        if inner.width >= 4 && viz_rows >= 1 {
            let viz_y = inner.y + inner.height - viz_rows as u16;
            f.render_widget(
                Paragraph::new(spectrum(inner.width as usize, viz_rows, t, m.playing)),
                Rect { x: inner.x, y: viz_y, width: inner.width, height: viz_rows as u16 },
            );
        }
        return;
    }

    // Fixed tight band: show exactly 5 lines — prev2, prev1, ACTIVE, next1,
    // next2 — with the active line one row above true center so the eye rests
    // on it. The window's small height kills the old dead vertical space.
    let avail = inner.height as usize;
    let height = avail.min(5);
    let center = (height / 2).min(2);

    // Vertically center the 5-line band inside whatever inner height we got.
    let pad_top = (avail.saturating_sub(height)) / 2;
    let band = Rect {
        x: inner.x,
        y: inner.y + pad_top as u16,
        width: inner.width,
        height: height as u16,
    };

    if !synced {
        // Plain lyrics: gentle auto-scroll proportional to track progress.
        let pos = m.position();
        let frac = if m.duration > 0.0 { pos / m.duration } else { 0.0 };
        let active = ((ly.lines.len() as f64) * frac) as usize;
        let lines = window(ly, active, center, height, band.width as usize, None);
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), band);
        return;
    }

    let pos = m.position();
    let (active, frac) = ly.active(pos).unwrap_or((0, 0.0));
    // Smoothstep the per-line wipe so it eases in/out across line boundaries
    // instead of advancing perfectly linearly. karaoke() still sub-samples the
    // boundary glyph, so the motion stays buttery.
    let f32frac = frac as f32;
    let ef = f32frac * f32frac * (3.0 - 2.0 * f32frac);
    let lines = window(ly, active, center, height, band.width as usize, Some(ef));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), band);
}

/// Build the visible lyric window centered on `active`. If `wipe_frac` is set,
/// the active line gets the karaoke gradient wipe.
fn window(
    ly: &crate::state::Lyrics,
    active: usize,
    center: usize,
    height: usize,
    width: usize,
    wipe_frac: Option<f32>,
) -> Vec<Line<'static>> {
    let start = active as isize - center as isize;
    let mut out = Vec::with_capacity(height);
    for row in 0..height as isize {
        let idx = start + row;
        if idx < 0 || idx as usize >= ly.lines.len() {
            out.push(Line::from(""));
            continue;
        }
        let i = idx as usize;
        if ly.lines[i].text.is_empty() {
            out.push(Line::from(Span::styled("♪", Style::default().fg(c::FAINT))));
            continue;
        }
        // Truncate to the (possibly narrow) box so centered lines never clip at
        // the edges — long lyrics get a trailing ellipsis instead.
        let text = truncate(&ly.lines[i].text, width.max(4));
        if i == active {
            if let Some(frac) = wipe_frac {
                out.push(karaoke(&text, frac));
            } else {
                out.push(Line::from(Span::styled(
                    text,
                    Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
                )));
            }
        } else {
            let dist = (i as isize - active as isize).unsigned_abs();
            let col = match dist {
                1 => c::DIM,
                _ => c::FAINT,
            };
            out.push(Line::from(Span::styled(text, Style::default().fg(col))));
        }
    }
    out
}

/// The karaoke line: characters up to `frac` are lit with a cyan→violet→pink
/// gradient; the rest stay dim. The single boundary character is *blended*
/// between dim and lit by the sub-character fraction, so the wipe glides
/// smoothly instead of popping per glyph. Combined with the slewed playback
/// clock, this is the buttery part.
fn karaoke(text: &str, frac: f32) -> Line<'static> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1);
    let litf = (frac * n as f32).clamp(0.0, n as f32);
    let full = litf.floor() as usize;
    let partial = litf.fract();
    // The not-yet-wiped tail of the ACTIVE line glows brighter than the DIM
    // context lines (a calm cyan-violet), so the active lyric always reads as
    // "lit/current" even before the wipe reaches a given character. The boundary
    // glyph then blends from this pending tone to the vivid wipe colour.
    let pending = c::blend(c::DIM, c::ACCENT, 0.45);
    let mut spans = Vec::with_capacity(n);
    for (i, ch) in chars.iter().enumerate() {
        let p = i as f32 / n as f32;
        let style = if i < full {
            Style::default().fg(c::wipe(p)).add_modifier(Modifier::BOLD)
        } else if i == full {
            // Boundary glyph fades in as we sweep across it.
            Style::default()
                .fg(c::blend(pending, c::wipe(p), partial))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pending).add_modifier(Modifier::BOLD)
        };
        spans.push(Span::styled(ch.to_string(), style));
    }
    Line::from(spans)
}

/// A sparkline whose bars are coloured by their own height with the jazz ramp
/// (blue→violet→pink→white), giving a per-bar vertical-style gradient. Honest:
/// height = real data; colour is purely a function of that height.
fn jazz_spark(data: &[u64], width: usize) -> Vec<Span<'static>> {
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if data.is_empty() {
        return vec![Span::raw(" ".repeat(width))];
    }
    let slice: Vec<u64> = data.iter().rev().take(width).rev().copied().collect();
    let max = slice.iter().copied().max().unwrap_or(1).max(1);
    let mut spans: Vec<Span> = Vec::with_capacity(width);
    let lead = width.saturating_sub(slice.len());
    if lead > 0 {
        spans.push(Span::raw(" ".repeat(lead)));
    }
    for v in slice {
        let frac = (v as f32 / max as f32).clamp(0.0, 1.0);
        let idx = (frac * 8.0).round() as usize;
        let ch = glyphs[idx.min(8)];
        if ch == ' ' {
            spans.push(Span::raw(" "));
        } else {
            spans.push(Span::styled(ch.to_string(), Style::default().fg(c::jazz(frac))));
        }
    }
    spans
}

// --- formatting helpers ----------------------------------------------------

fn fmt_bytes(b: u64) -> String {
    let gb = b as f64 / 1_073_741_824.0;
    if gb >= 1.0 {
        format!("{gb:.1}G")
    } else {
        format!("{:.0}M", b as f64 / 1_048_576.0)
    }
}

fn fmt_rate(bps: f64) -> String {
    if bps >= 1_048_576.0 {
        format!("{:.1} MB/s", bps / 1_048_576.0)
    } else if bps >= 1024.0 {
        format!("{:.0} KB/s", bps / 1024.0)
    } else {
        format!("{bps:.0} B/s")
    }
}

fn fmt_rate_short(bps: f64) -> String {
    if bps >= 1_048_576.0 {
        format!("{:.1}M", bps / 1_048_576.0)
    } else if bps >= 1024.0 {
        format!("{:.0}K", bps / 1024.0)
    } else {
        format!("{bps:.0}B")
    }
}

fn fmt_tokens(t: u64) -> String {
    if t >= 1_000_000_000 {
        format!("{:.1}B", t as f64 / 1e9)
    } else if t >= 1_000_000 {
        format!("{:.1}M", t as f64 / 1e6)
    } else if t >= 1_000 {
        format!("{:.1}K", t as f64 / 1e3)
    } else {
        format!("{t}")
    }
}

fn fmt_clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

fn fmt_dur(secs: u64) -> String {
    let d = secs / 86400;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    if d > 0 {
        format!("{d}d {h}h {m}m")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else {
        format!("{m}m")
    }
}

/// Compact 2-component uptime, capped at 6 chars: "4d6h" "2h11m" "9m" "12s".
/// Reserved-width friendly so a process crossing a boundary never shoves.
fn fmt_dur_short(secs: u64) -> String {
    let d = secs / 86400;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if d > 0 {
        format!("{d}d{h}h")
    } else if h > 0 {
        format!("{h}h{m}m")
    } else if m > 0 {
        format!("{m}m")
    } else {
        format!("{s}s")
    }
}
