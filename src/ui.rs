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

/// How an area graph colours its fill.
#[derive(Clone, Copy)]
enum Fill {
    /// Per-column flame: green→red by that column's own height.
    Heat,
    /// One hue, dim at the baseline → vivid at the crest.
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
                    Fill::Heat => c::heat(v),
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
            Constraint::Length(9),  // CPU equalizer
            Constraint::Length(9),  // MEM · DISK · NET (same span as CPU)
            Constraint::Length(11), // APPLE SILICON
            Constraint::Length(7),
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Min(6),
        ])
        .split(body[0]);

    cpu_eq_panel(f, left[0], s);
    resources_panel(f, left[1], s);
    silicon_panel(f, left[2], s, t);
    claude_panel(f, left[3], s);
    git_panel(f, left[4], s);
    weather_panel(f, left[5], s);
    proc_panel(f, left[6], s);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(3)])
        .split(body[1]);

    now_playing(f, right[0], s, t);
    // Until the first music poll lands, show the (neutral) lyrics panel so we
    // never flash the wrong thing before the real state is known. After that:
    // actively playing → lyrics; paused/stopped/idle → live system graphs.
    if !s.music.polled || (s.music.playing && !s.music.track.is_empty()) {
        lyrics_panel(f, right[1], s);
    } else {
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

    let sep = || Span::styled("  │  ", Style::default().fg(c::FAINT).bg(c::PANEL_BORDER));
    let lbl = |t: &str| Span::styled(t.to_string(), Style::default().fg(c::DIM).bg(c::PANEL_BORDER));
    let val = |t: String, col| Span::styled(t, Style::default().fg(col).bg(c::PANEL_BORDER).add_modifier(Modifier::BOLD));

    let mut spans = vec![
        Span::styled(" q ", Style::default().fg(c::BG).bg(c::ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(" quit", Style::default().fg(c::DIM).bg(c::PANEL_BORDER)),
        sep(),
        lbl("CPU "),
        val(format!("{cpu:>2.0}%"), c::heat(cpu / 100.0)),
    ];
    if si.fresh {
        spans.push(sep());
        spans.push(lbl("GPU "));
        spans.push(val(format!("{:>2.0}%", si.gpu_pct), c::heat(si.gpu_pct / 100.0)));
        spans.push(sep());
        spans.push(lbl("PWR "));
        spans.push(val(format!("{:.1}W", si.all_power_w), c::YELLOW));
        spans.push(sep());
        spans.push(lbl("TEMP "));
        spans.push(val(format!("{:.0}°", si.cpu_temp_c), c::heat((si.cpu_temp_c - 30.0) / 70.0)));
    }
    spans.push(sep());
    spans.push(lbl("MEM "));
    spans.push(val(format!("{:.0}%", memf * 100.0), c::heat(memf)));
    spans.push(sep());
    spans.push(lbl("NET "));
    spans.push(Span::styled(format!("▼{}", fmt_rate_short(sys.net_rx_bps)), Style::default().fg(c::GREEN).bg(c::PANEL_BORDER)));
    spans.push(Span::styled(format!(" ▲{}", fmt_rate_short(sys.net_tx_bps)), Style::default().fg(c::PINK).bg(c::PANEL_BORDER)));
    if s.usage.fresh {
        spans.push(sep());
        spans.push(lbl("Claude today "));
        spans.push(val(format!("${:.2}", s.usage.today_cost), c::GREEN));
    }
    spans.push(sep());
    spans.push(lbl("↑ "));
    spans.push(Span::styled(fmt_dur(sys.uptime_secs), Style::default().fg(c::TEXT).bg(c::PANEL_BORDER)));
    spans.push(sep());
    spans.push(Span::styled(format!("{} procs", sys.proc_count), Style::default().fg(c::DIM).bg(c::PANEL_BORDER)));

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
        let color = c::heat(mid);
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
                // Bright crest (no dark cap): the partial block already conveys
                // the sub-cell height; colour it the same as a full cell.
                let st = Style::default().fg(color);
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
    mem_line.extend(bar_spans(memf, bw, c::heat(memf)));
    mem_line.push(Span::styled(
        format!(" {:>3.0}%  {} / {}", memf * 100.0, fmt_bytes(s.system.mem_used), fmt_bytes(s.system.mem_total)),
        Style::default().fg(c::TEXT),
    ));
    lines.push(Line::from(mem_line));
    let mut disk_line = vec![Span::styled("disk ", Style::default().fg(c::DIM))];
    disk_line.extend(bar_spans(diskf, bw, c::heat(diskf)));
    disk_line.push(Span::styled(
        format!(" {:>3.0}%  {} / {}", diskf * 100.0, fmt_bytes(s.system.disk_used), fmt_bytes(s.system.disk_total)),
        Style::default().fg(c::TEXT),
    ));
    lines.push(Line::from(disk_line));
    lines.push(Line::from(vec![
        Span::styled("net  ", Style::default().fg(c::DIM)),
        Span::styled("▼ ", Style::default().fg(c::GREEN)),
        Span::styled(format!("{:>10}", fmt_rate(s.system.net_rx_bps)), Style::default().fg(c::GREEN)),
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
        area_graph(f, plot, &vals, Fill::Tint(c::CYAN));
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
    let gauge_w = 38u16.min(inner.width);
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
        gauge("gpu", gpu / 100.0, c::jazz(gpu / 100.0), format!(" {:>3.0}%  {} MHz", gpu, si.gpu_freq_mhz)),
        gauge("e-cpu", ecpu / 100.0, c::jazz(ecpu / 100.0), format!(" {:>3.0}%  {} MHz", ecpu, si.ecpu_freq_mhz)),
        gauge("p-cpu", pcpu / 100.0, c::jazz(pcpu / 100.0), format!(" {:>3.0}%  {} MHz", pcpu, si.pcpu_freq_mhz)),
        {
            let pf = (spw / 120.0).clamp(0.0, 1.0);
            let mut spans = vec![Span::styled("power  ", Style::default().fg(c::DIM))];
            spans.extend(bar_spans(pf, bw, c::jazz(pf)));
            spans.push(Span::styled(format!(" {:>4.1} W", spw), Style::default().fg(c::PINK).add_modifier(Modifier::BOLD)));
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

fn claude_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("CLAUDE CODE", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let u = &s.usage;

    if !u.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("scanning ~/.claude…", Style::default().fg(c::DIM))),
            inner,
        );
        return;
    }
    let tot_today = u.today_input + u.today_output + u.today_cache_read + u.today_cache_write;
    let lines = vec![
        Line::from(vec![
            Span::styled("today  ", Style::default().fg(c::DIM)),
            Span::styled(fmt_tokens(tot_today), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
            Span::styled(" tok   ", Style::default().fg(c::FAINT)),
            Span::styled(format!("${:.2}", u.today_cost), Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)),
            Span::styled(" est", Style::default().fg(c::FAINT)),
        ]),
        Line::from(vec![
            Span::styled("       in ", Style::default().fg(c::FAINT)),
            Span::styled(fmt_tokens(u.today_input), Style::default().fg(c::DIM)),
            Span::styled("  out ", Style::default().fg(c::FAINT)),
            Span::styled(fmt_tokens(u.today_output), Style::default().fg(c::DIM)),
            Span::styled("  cache ", Style::default().fg(c::FAINT)),
            Span::styled(fmt_tokens(u.today_cache_read + u.today_cache_write), Style::default().fg(c::DIM)),
        ]),
        Line::from(vec![
            Span::styled("sessions ", Style::default().fg(c::DIM)),
            Span::styled(format!("{}", u.sessions_today), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
            Span::styled("   msgs ", Style::default().fg(c::DIM)),
            Span::styled(format!("{}", u.today_messages), Style::default().fg(c::TEXT)),
            Span::styled("   ", Style::default().fg(c::FAINT)),
            Span::styled(u.top_model.clone(), Style::default().fg(c::PINK)),
        ]),
        Line::from(vec![
            Span::styled("burn   ", Style::default().fg(c::DIM)),
            Span::styled(c::spark(&u.hourly, 22), Style::default().fg(c::ORANGE)),
        ]),
        Line::from(vec![
            Span::styled("month  ", Style::default().fg(c::DIM)),
            Span::styled(format!("${:.2}", u.month_cost), Style::default().fg(c::YELLOW).add_modifier(Modifier::BOLD)),
            Span::styled(" est", Style::default().fg(c::FAINT)),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
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
    let art_rows = inner.height.saturating_sub(pad_v * 2).max(1);
    let art_cols = (art_rows * 2).min(inner.width.saturating_sub(24)).max(1);
    let art_area = Rect { x: inner.x + 1, y: inner.y + pad_v, width: art_cols, height: art_rows };
    render_art(f, art_area, &s.album_art, m.track_id() == s.album_art.track_id);

    // Text column starts after the art + a 3-cell gap.
    let gap = art_cols + 4;
    let info = Rect {
        x: inner.x + gap,
        y: inner.y,
        width: inner.width.saturating_sub(gap),
        height: inner.height,
    };
    let pos = m.position();
    let frac = if m.duration > 0.0 { (pos / m.duration) as f32 } else { 0.0 };
    let icon = if m.playing { "▶" } else { "❚❚" };
    let eq = if m.playing { eq_bars(t, 10) } else { "▁".repeat(10) };

    let title = Line::from(vec![
        Span::styled(format!("{icon} "), Style::default().fg(c::GREEN).add_modifier(Modifier::BOLD)),
        Span::styled(m.track.clone(), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(format!("   {eq}"), Style::default().fg(c::PINK)),
    ]);
    let meta = Line::from(vec![
        Span::styled(m.artist.clone(), Style::default().fg(c::ACCENT)),
        Span::styled("  ·  ", Style::default().fg(c::FAINT)),
        Span::styled(m.album.clone(), Style::default().fg(c::DIM)),
    ]);
    let bw = (info.width as usize).saturating_sub(16).clamp(8, 90);
    let mut progress_spans = vec![Span::styled(format!("{} ", fmt_clock(pos)), Style::default().fg(c::CYAN))];
    progress_spans.extend(bar_spans(frac, bw, c::PINK));
    progress_spans.push(Span::styled(format!(" {}", fmt_clock(m.duration)), Style::default().fg(c::DIM)));
    f.render_widget(
        Paragraph::new(vec![
            title,
            meta,
            Line::from(""),
            Line::from(progress_spans),
            Line::from(Span::styled(
                if m.playing { "playing" } else { "paused" },
                Style::default().fg(c::FAINT),
            )),
        ]),
        info,
    );
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

fn git_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let g = &s.git;
    let dirty = g.ok && g.dirty > 0;
    let block = panel("GIT · battlestation", dirty);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if !g.fresh {
        f.render_widget(Paragraph::new(Span::styled("reading repo…", Style::default().fg(c::DIM))), inner);
        return;
    }
    if !g.ok {
        f.render_widget(Paragraph::new(Span::styled("not a git repo", Style::default().fg(c::DIM))), inner);
        return;
    }
    let clean_col = if g.dirty == 0 { c::GREEN } else { c::YELLOW };
    let mut head = vec![
        Span::styled("", Style::default().fg(c::ORANGE)),
        Span::styled(g.branch.clone(), Style::default().fg(c::ORANGE).add_modifier(Modifier::BOLD)),
    ];
    if g.ahead > 0 {
        head.push(Span::styled(format!("  ↑{}", g.ahead), Style::default().fg(c::GREEN)));
    }
    if g.behind > 0 {
        head.push(Span::styled(format!("  ↓{}", g.behind), Style::default().fg(c::RED)));
    }
    let status = if g.dirty == 0 {
        Line::from(Span::styled("✓ clean", Style::default().fg(c::GREEN)))
    } else {
        Line::from(vec![
            Span::styled(format!("● {} dirty", g.dirty), Style::default().fg(clean_col).add_modifier(Modifier::BOLD)),
            Span::styled(format!("   {} staged  {} untracked", g.staged, g.untracked), Style::default().fg(c::DIM)),
        ])
    };
    let last = Line::from(vec![
        Span::styled(format!("{} ", g.last_hash), Style::default().fg(c::CYAN)),
        Span::styled(truncate(&g.last_msg, inner.width as usize - 12), Style::default().fg(c::TEXT)),
    ]);
    let rel = Line::from(vec![
        Span::styled(g.last_rel.clone(), Style::default().fg(c::FAINT)),
        Span::styled(format!("   {} commits today", g.commits_today), Style::default().fg(c::DIM)),
    ]);
    f.render_widget(Paragraph::new(vec![Line::from(head), status, last, rel]), inner);
}

fn proc_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("TOP PROCESSES", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let procs = &s.system.top_procs;
    if procs.is_empty() {
        f.render_widget(Paragraph::new(Span::styled("…", Style::default().fg(c::DIM))), inner);
        return;
    }
    let rows = (inner.height as usize).min(procs.len());
    let namew = (inner.width as usize).saturating_sub(20).clamp(8, 40);
    let mut lines = Vec::with_capacity(rows);
    for (name, cpu, mem) in procs.iter().take(rows) {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:>5.1}%", cpu),
                Style::default().fg(c::heat((*cpu / 100.0).min(1.0))).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {:>6}  ", fmt_bytes(*mem)), Style::default().fg(c::DIM)),
            Span::styled(truncate(name, namew), Style::default().fg(c::TEXT)),
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
    let big = Line::from(vec![
        Span::styled(format!("{} ", w.icon), Style::default().fg(c::YELLOW)),
        Span::styled(format!("{}°F", w.temp_f), Style::default().fg(c::CYAN).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", w.desc), Style::default().fg(c::TEXT)),
    ]);
    let detail = Line::from(vec![
        Span::styled(format!("feels {}°  ", w.feels_f), Style::default().fg(c::DIM)),
        Span::styled(format!("↑{}° ↓{}°  ", w.hi_f, w.lo_f), Style::default().fg(c::DIM)),
        Span::styled(format!("hum {}%", w.humidity), Style::default().fg(c::FAINT)),
    ]);
    let loc = Line::from(Span::styled(w.location.clone(), Style::default().fg(c::FAINT)));
    f.render_widget(Paragraph::new(vec![big, detail, loc]), inner);
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

    let height = inner.height as usize;
    let center = height / 2;

    if !synced {
        // Plain lyrics: gentle auto-scroll proportional to track progress.
        let pos = m.position();
        let frac = if m.duration > 0.0 { pos / m.duration } else { 0.0 };
        let active = ((ly.lines.len() as f64) * frac) as usize;
        let lines = window(ly, active, center, height, None);
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), inner);
        return;
    }

    let pos = m.position();
    let (active, frac) = ly.active(pos).unwrap_or((0, 0.0));
    let lines = window(ly, active, center, height, Some(frac as f32));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), inner);
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
                1 => c::TEXT,
                2 => c::DIM,
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
    let mut spans = Vec::with_capacity(n);
    for (i, ch) in chars.iter().enumerate() {
        let p = i as f32 / n as f32;
        let style = if i < full {
            Style::default().fg(c::wipe(p)).add_modifier(Modifier::BOLD)
        } else if i == full {
            // Boundary glyph fades in as we sweep across it.
            Style::default()
                .fg(c::blend(c::DIM, c::wipe(p), partial))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(c::DIM)
        };
        spans.push(Span::styled(ch.to_string(), style));
    }
    Line::from(spans)
}

/// Pseudo-random equalizer bars driven by sine waves of the animation clock.
fn eq_bars(t: f64, n: usize) -> String {
    let glyphs = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut s = String::new();
    for i in 0..n {
        let phase = i as f64 * 0.7;
        let v = (((t * 6.0 + phase).sin() * 0.5 + 0.5)
            * ((t * 9.3 + phase * 1.7).sin() * 0.5 + 0.5))
            .clamp(0.0, 1.0);
        let idx = (v * 7.0).round() as usize;
        s.push(glyphs[idx.min(7)]);
    }
    s
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
    if t >= 1_000_000 {
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
