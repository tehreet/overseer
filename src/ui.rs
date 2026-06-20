//! All rendering. Pure function of (state snapshot, animation clock) -> frame.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

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
        .constraints([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    header(f, outer[0], s);
    footer(f, outer[2], s);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(outer[1]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),  // [0] CPU equalizer
            Constraint::Length(9),  // [1] MEM · DISK · NET
            Constraint::Length(11), // [2] APPLE SILICON
            Constraint::Length(9),  // [3] TOP PROCESSES (moved up, +uptime col, +header)
            Constraint::Length(9),  // [4] ROBOTS WORKING (Claude tokens + git pulse, merged)
            Constraint::Length(8),  // [5] WEATHER       (jazzed, +data)
            Constraint::Min(9),     // [6] iMESSAGE
        ])
        .split(body[0]);

    cpu_eq_panel(f, left[0], s);
    resources_panel(f, left[1], s);
    silicon_panel(f, left[2], s, t);
    proc_panel(f, left[3], s);
    robots_panel(f, left[4], s, t);
    weather_panel(f, left[5], s);
    messages_panel(f, left[6], s, t);

    // Until the first music poll lands, show the (neutral) lyrics panel so we
    // never flash the wrong thing before the real state is known. After that:
    // actively playing → lyrics; paused/stopped/idle → live system graphs.
    let show_lyrics = !s.music.polled || (s.music.playing && !s.music.track.is_empty());

    if show_lyrics {
        // Cap the lyric band to a tight 9-row block (7 inner rows) and hand the
        // freed vertical space to the live jazz system graphs — zero dead zone.
        const LYRICS_H: u16 = 9;
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9),
                Constraint::Length(LYRICS_H),
                Constraint::Min(0),
            ])
            .split(body[1]);
        now_playing(f, right[0], s, t);
        lyrics_panel(f, right[1], s);
        if right[2].height >= 3 {
            system_panel(f, right[2], s);
        }
    } else {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(9), Constraint::Min(3)])
            .split(body[1]);
        now_playing(f, right[0], s, t);
        system_panel(f, right[1], s);
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
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1);
    let head = ((t * 0.22 + row as f64 * 0.18).rem_euclid(1.0)) as f32; // 0..1 sweep
    let sheen = c::jazz(0.88); // pink-white glint
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
        let b = (-(d * d) / (2.0 * 0.07 * 0.07)).exp(); // narrow bright band 0..1
        let col = c::blend(c::DIM, sheen, b); // calm base -> glint
        spans.push(Span::styled(ch.to_string(), Style::default().fg(col)));
    }
    Line::from(spans)
}

fn panel(title: &str, hot: bool) -> Block<'_> {
    let border = if hot { c::PANEL_BORDER_HOT } else { c::PANEL_BORDER };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(c::ACCENT).add_modifier(Modifier::BOLD),
        ))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG))
}

fn header(f: &mut Frame, area: Rect, s: &AppState) {
    let clock = chrono::Local::now().format("%a %b %-d  %H:%M:%S").to_string();
    let left = Span::styled(
        "  STUDIOBOARD",
        Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
    );
    let mid = Span::styled(
        format!("  {}  ·  Apple M4 Max", s.system.hostname),
        Style::default().fg(c::DIM),
    );
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);
    f.render_widget(Paragraph::new(Line::from(vec![left, mid])), cols[0]);
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("{clock}  "),
            Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD),
        ))
        .alignment(Alignment::Right),
        cols[1],
    );
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
            burn.extend(jazz_spark(&u.hourly, (lw as usize).saturating_sub(9)));
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
    // Branch activity, two metrics per row: loc · commits, then PRs · merges.
    let loc = Line::from(vec![
        Span::styled(format!("+{}", g.loc_added), Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" -{}", g.loc_removed), Style::default().fg(if g.loc_removed > 0 { c::RED } else { c::FAINT })),
        Span::styled(" loc", Style::default().fg(c::FAINT)),
        Span::styled("   ·   ", Style::default().fg(c::FAINT)),
        Span::styled(format!("{}", g.branch_commits), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(" commits", Style::default().fg(c::DIM)),
    ]);
    let prm = Line::from(vec![
        Span::styled(format!("{}", g.pr_count), Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)),
        Span::styled(" PRs", Style::default().fg(c::DIM)),
        Span::styled("   ·   ", Style::default().fg(c::FAINT)),
        Span::styled(
            format!("{}", g.merges_main),
            Style::default().fg(if g.merges_main > 0 { c::GREEN } else { c::TEXT }).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" merges → main", Style::default().fg(c::DIM)),
    ]);
    f.render_widget(
        Paragraph::new(vec![branch, commit, age, Line::from(""), loc, prm]),
        Rect { x: rx, y: inner.y, width: rw, height: inner.height },
    );
}

fn now_playing(f: &mut Frame, area: Rect, s: &AppState, _t: f64) {
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
    let art_rows = inner.height.saturating_sub(pad_v * 2).max(1);
    let art_cols = (art_rows * 2).min(inner.width.saturating_sub(24)).max(1);
    let art_area = Rect { x: inner.x + 1, y: inner.y + pad_v, width: art_cols, height: art_rows };
    render_art(f, art_area, &s.album_art, m.track_id() == s.album_art.track_id);

    // Text column starts after the art + a 3-cell gap.
    let gap = art_cols + 4;
    let pos = m.position();
    let frac = if m.duration > 0.0 { (pos / m.duration) as f32 } else { 0.0 };
    let icon = if m.playing { "▶" } else { "❚❚" };

    let title = Line::from(vec![
        // Transport icon rides the jazz family, not green.
        Span::styled(format!("{icon} "), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(m.track.clone(), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
    ]);
    let meta = Line::from(vec![
        Span::styled(m.artist.clone(), Style::default().fg(c::ACCENT)),
        Span::styled("  ·  ", Style::default().fg(c::FAINT)),
        Span::styled(m.album.clone(), Style::default().fg(c::DIM)),
    ]);
    let textw = (inner.width.saturating_sub(gap)) as usize;
    let bw = textw.saturating_sub(16).clamp(8, 90);
    let mut progress_spans = vec![Span::styled(format!("{} ", fmt_clock(pos)), Style::default().fg(c::CYAN))];
    progress_spans.extend(bar_spans(frac, bw, c::PINK));
    progress_spans.push(Span::styled(format!(" {}", fmt_clock(m.duration)), Style::default().fg(c::DIM)));

    // Status row carries an honest, static EQ flourish (no time term, no fake
    // motion) plus the transport word — fills the column without lying.
    let status = Line::from(vec![
        Span::styled(
            if m.playing { "playing" } else { "paused" },
            Style::default().fg(c::FAINT),
        ),
        Span::styled("   ▁▂▃▃▂▁", Style::default().fg(c::FAINT)),
    ]);

    let body = vec![title, meta, Line::from(""), Line::from(progress_spans), status];

    // Vertically center the metadata block against the album art so the text and
    // art share a baseline — no dead rectangle below the text next to tall art.
    let nblock = body.len() as u16;
    let top_pad = inner.height.saturating_sub(nblock) / 2;
    let info = Rect {
        x: inner.x + gap,
        y: inner.y + top_pad,
        width: inner.width.saturating_sub(gap),
        height: inner.height.saturating_sub(top_pad),
    };
    f.render_widget(Paragraph::new(body), info);
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
    for row in 0..h {
        let mut spans = Vec::with_capacity(w);
        for col in 0..w {
            let u = (col as f32 + 0.5) / w as f32;
            let vt = (2.0 * row as f32 + 0.5) / (2.0 * h as f32);
            let vb = (2.0 * row as f32 + 1.5) / (2.0 * h as f32);
            if have {
                let top = art.sample(u, vt).unwrap_or([20, 20, 28]);
                let bot = art.sample(u, vb).unwrap_or([20, 20, 28]);
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

    // row 0 — big readout on the left, sunrise/sunset flush-right.
    let left_txt = format!("{} {}°F  {}", w.icon, w.temp_f, w.desc);
    let right_txt = format!("☀ {}  ☾ {}", w.sunrise, w.sunset);
    let pad = iw
        .saturating_sub(left_txt.chars().count() + right_txt.chars().count())
        .max(1);
    let big = Line::from(vec![
        Span::styled(format!("{} ", w.icon), Style::default().fg(c::YELLOW)),
        Span::styled(format!("{}°F", w.temp_f), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", w.desc), Style::default().fg(c::TEXT)),
        Span::raw(" ".repeat(pad)),
        Span::styled(
            format!("☀ {}  ☾ {}", w.sunrise, w.sunset),
            Style::default().fg(c::jazz(0.85)),
        ),
    ]);

    // row 1 — feels / hi-lo (warm up = pink, cool down = cyan) / hum / UV.
    let detail = Line::from(vec![
        Span::styled(format!("feels {}°  ", w.feels_f), Style::default().fg(c::DIM)),
        Span::styled(format!("↑{}°", w.hi_f), Style::default().fg(c::PINK)),
        Span::styled(" ", Style::default().fg(c::FAINT)),
        Span::styled(format!("↓{}°", w.lo_f), Style::default().fg(c::CYAN)),
        Span::styled(format!("   hum {}%", w.humidity), Style::default().fg(c::FAINT)),
        Span::styled("   UV ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}", w.uv), Style::default().fg(c::jazz((w.uv as f32 / 11.0).clamp(0.0, 1.0)))),
    ]);

    // row 2 — atmosphere: wind / precip chance / pressure.
    let atmos = Line::from(vec![
        Span::styled(format!("💨 {} {} mph", w.wind_dir, w.wind_mph), Style::default().fg(c::CYAN)),
        Span::styled("   ☔ ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}%", w.precip_chance), Style::default().fg(c::jazz((w.precip_chance as f32 / 100.0).clamp(0.0, 1.0)))),
        Span::styled(format!("   ◧ {} mb", w.pressure_mb), Style::default().fg(c::DIM)),
    ]);

    let mut lines: Vec<Line> = vec![big, detail, atmos];

    // row 3 — thin jazz rule for rhythm.
    if h > 4 {
        lines.push(Line::from(Span::styled("─".repeat(iw), Style::default().fg(c::FAINT))));
    }
    // row 4 — tiny hourly temp strip (real forecast temps, jazz-colored).
    if h > 5 && !w.temp_strip.is_empty() {
        let mut strip = vec![Span::styled("", Style::default())];
        strip.extend(jazz_spark(&w.temp_strip, iw));
        lines.push(Line::from(strip));
    }
    // last row — location.
    lines.push(Line::from(Span::styled(w.location.clone(), Style::default().fg(c::FAINT))));

    f.render_widget(Paragraph::new(lines), inner);
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
    let border = if hot { c::PANEL_BORDER_HOT } else { c::PANEL_BORDER };
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
        .border_style(Style::default().fg(border))
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
    // One row for the separator+footer summary when there's room.
    let want_footer = ih > msgs.items.len() + composer_rows + 1;
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

    // Reserved column budget: marker(2) + sender(16) + gap(1) + reltime(4) + dot(2).
    let prevw = iw.saturating_sub(2 + 16 + 1 + 4 + 2).max(6);

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
        let sender = format!("{:<16}", truncate(&m.sender, 16));
        let sender_style = if m.unread && adv < 0.5 {
            Style::default().fg(sender_col).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(sender_col)
        };
        let preview = truncate(&m.preview, prevw);
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
            Span::styled(format!("{preview:<width$}", width = prevw), prev_style),
            Span::styled(format!("{:>4}", truncate(&m.rel, 4)), Style::default().fg(c::FAINT)),
            dot,
        ]));
    }

    // ----- separator + tail summary -----
    if want_footer {
        lines.push(Line::from(Span::styled("─".repeat(iw), Style::default().fg(c::FAINT))));
        let earlier = msgs.items.len().saturating_sub(shown);
        let hint = if ui.active {
            "  m: read · mm: reply".to_string()
        } else {
            format!("read · {earlier} earlier")
        };
        lines.push(Line::from(Span::styled(hint, Style::default().fg(c::FAINT))));
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

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{t}…")
    }
}

fn lyrics_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let synced = s.lyrics.synced;
    let block = panel(if synced { "LYRICS ♪ synced" } else { "LYRICS" }, false);
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
        let note = if ly.note.is_empty() { "…" } else { &ly.note };
        f.render_widget(
            Paragraph::new(Span::styled(note.to_string(), Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
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
        let lines = window(ly, active, center, height, None);
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
    let lines = window(ly, active, center, height, Some(ef));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), band);
}

/// Build the visible lyric window centered on `active`. If `wipe_frac` is set,
/// the active line gets the karaoke gradient wipe.
fn window(
    ly: &crate::state::Lyrics,
    active: usize,
    center: usize,
    height: usize,
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
        let text = &ly.lines[i].text;
        if text.is_empty() {
            out.push(Line::from(Span::styled("♪", Style::default().fg(c::FAINT))));
            continue;
        }
        if i == active {
            if let Some(frac) = wipe_frac {
                out.push(karaoke(text, frac));
            } else {
                out.push(Line::from(Span::styled(
                    text.clone(),
                    Style::default().fg(c::PINK).add_modifier(Modifier::BOLD),
                )));
            }
        } else {
            let dist = (i as isize - active as isize).unsigned_abs();
            let col = match dist {
                1 => c::DIM,
                _ => c::FAINT,
            };
            out.push(Line::from(Span::styled(text.clone(), Style::default().fg(col))));
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
