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

/// Shared sine-flow pattern (0..1) for the equalizer panels. `phase` desyncs
/// each bar. Lower frequencies here = gentler, slower undulation across the
/// whole spectrum — tune these two numbers to taste.
fn eq_flow(t: f64, phase: f64) -> f32 {
    (((t * 1.6 + phase).sin() * 0.5 + 0.5) * ((t * 2.7 + phase * 1.7).sin() * 0.5 + 0.5)) as f32
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

    cpu_eq_panel(f, left[0], s, t);
    resources_panel(f, left[1], s, t);
    silicon_panel(f, left[2], s);
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
    // never flash the pulse before the real state is known. After that:
    // actively playing → lyrics; paused/stopped/idle → ambient system pulse.
    if !s.music.polled || (s.music.playing && !s.music.track.is_empty()) {
        lyrics_panel(f, right[1], s);
    } else {
        pulse_panel(f, right[1], s, t);
    }
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

/// Half-width CPU equalizer: one vertical bar per core, height = utilization.
/// Each core is a column that *flows* with the same sine motion as the
/// now-playing / net EQ, but its amplitude is that core's real (delay-
/// interpolated) load — busy cores dance tall, idle cores stay low. Smooth
/// like the music EQ, still honest to the data. Green→red gradient.
fn cpu_eq_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
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

    // Flowing per-core heights: amplitude = real load, motion = sine (like the
    // music EQ). Each core gets its own phase so the spectrum dances.
    let heights: Vec<f32> = (0..n)
        .map(|core| {
            let energy = (vals[core] / 100.0).clamp(0.0, 1.0);
            let amp = 0.08 + 0.92 * energy;
            let phase = core as f64 * 0.7;
            amp * (0.3 + 0.7 * eq_flow(t, phase))
        })
        .collect();

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
        let dim = c::blend(color, c::BG, 0.7);
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
                let st = Style::default().fg(if glyph == '█' { color } else { dim });
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

fn resources_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
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
    lines.push(Line::from(vec![
        Span::styled("mem  ", Style::default().fg(c::DIM)),
        Span::styled(c::bar(memf, bw), Style::default().fg(c::heat(memf))),
        Span::styled(
            format!(" {:>3.0}%  {} / {}", memf * 100.0, fmt_bytes(s.system.mem_used), fmt_bytes(s.system.mem_total)),
            Style::default().fg(c::TEXT),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("disk ", Style::default().fg(c::DIM)),
        Span::styled(c::bar(diskf, bw), Style::default().fg(c::heat(diskf))),
        Span::styled(
            format!(" {:>3.0}%  {} / {}", diskf * 100.0, fmt_bytes(s.system.disk_used), fmt_bytes(s.system.disk_total)),
            Style::default().fg(c::TEXT),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("net  ", Style::default().fg(c::DIM)),
        Span::styled("▼ ", Style::default().fg(c::GREEN)),
        Span::styled(format!("{:>10}", fmt_rate(s.system.net_rx_bps)), Style::default().fg(c::GREEN)),
        Span::styled("    ▲ ", Style::default().fg(c::PINK)),
        Span::styled(format!("{:>10}", fmt_rate(s.system.net_tx_bps)), Style::default().fg(c::PINK)),
    ]));

    // Net visualizer: a music-EQ-style spectrum whose energy swells/recedes
    // with real (delay-interpolated) throughput, but whose motion is sine-
    // smooth — so it flows like the now-playing EQ.
    let eq_rows = (inner.height as usize).saturating_sub(lines.len());
    if eq_rows >= 1 {
        let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
        let bps = sampled_scalar(&s.net_samples, target);
        let energy = ((bps.max(1.0).log10() - 3.0) / 4.0).clamp(0.0, 1.0); // ~1KB/s..10MB/s
        let amp = 0.18 + 0.82 * energy;

        let bar_w = 2usize;
        let gap = 1usize;
        let nbars = (iw + gap) / (bar_w + gap);
        for r in 0..eq_rows {
            let row_top = (eq_rows - r) as f32 / eq_rows as f32;
            let row_bot = (eq_rows - r - 1) as f32 / eq_rows as f32;
            let mid = (row_top + row_bot) * 0.5;
            let col = c::blend(c::GREEN, c::CYAN, mid);
            let dim = c::blend(col, c::BG, 0.7);
            let mut spans: Vec<Span> = Vec::with_capacity(nbars * 2);
            for b in 0..nbars {
                let phase = b as f64 * 0.6;
                let h = amp * (0.3 + 0.7 * eq_flow(t, phase));
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
                    spans.push(Span::styled(
                        glyph.to_string().repeat(bar_w),
                        Style::default().fg(if glyph == '█' { col } else { dim }),
                    ));
                }
                spans.push(Span::raw(" ".repeat(gap)));
            }
            lines.push(Line::from(spans));
        }
    }

    f.render_widget(Paragraph::new(lines), inner);
}

fn silicon_panel(f: &mut Frame, area: Rect, s: &AppState) {
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

    let bw = (inner.width as usize).saturating_sub(24).clamp(8, 30);
    let tnorm = |t: f32| ((t - 30.0) / 70.0).clamp(0.0, 1.0); // 30–100 °C → 0–1
    let gpu_share = if spw > 0.1 { (gpw / spw * 100.0).clamp(0.0, 100.0) } else { 0.0 };

    // label, value bar (0..1), bar color, trailing text
    let gauge = |label: &str, frac: f32, col: ratatui::style::Color, tail: String| {
        Line::from(vec![
            Span::styled(format!("{label:<6} "), Style::default().fg(c::DIM)),
            Span::styled(c::bar(frac, bw), Style::default().fg(col)),
            Span::styled(tail, Style::default().fg(c::TEXT)),
        ])
    };

    let lines = vec![
        gauge("gpu", gpu / 100.0, c::heat(gpu / 100.0), format!(" {:>3.0}%  {} MHz", gpu, si.gpu_freq_mhz)),
        gauge("e-cpu", ecpu / 100.0, c::heat(ecpu / 100.0), format!(" {:>3.0}%  {} MHz", ecpu, si.ecpu_freq_mhz)),
        gauge("p-cpu", pcpu / 100.0, c::heat(pcpu / 100.0), format!(" {:>3.0}%  {} MHz", pcpu, si.pcpu_freq_mhz)),
        Line::from(vec![
            Span::styled("power  ", Style::default().fg(c::DIM)),
            Span::styled(c::bar((spw / 120.0).clamp(0.0, 1.0), bw), Style::default().fg(c::YELLOW)),
            Span::styled(format!(" {:>4.1} W", spw), Style::default().fg(c::YELLOW).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""), // padding under the power bar
        Line::from(vec![
            Span::styled("       gpu ", Style::default().fg(c::FAINT)),
            Span::styled(format!("{gpw:.1} W", ), Style::default().fg(c::DIM)),
            Span::styled(format!("   ·   {gpu_share:.0}% of system draw"), Style::default().fg(c::FAINT)),
        ]),
        Line::from(""),
        gauge("cpu °C", tnorm(ctemp), c::heat(tnorm(ctemp)), format!(" {ctemp:>3.0}°C")),
        gauge("gpu °C", tnorm(gtemp), c::heat(tnorm(gtemp)), format!(" {gtemp:>3.0}°C")),
    ];
    f.render_widget(Paragraph::new(lines), inner);
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
    let progress_spans = vec![
        Span::styled(format!("{} ", fmt_clock(pos)), Style::default().fg(c::CYAN)),
        Span::styled(c::bar(frac, bw), Style::default().fg(c::PINK)),
        Span::styled(format!(" {}", fmt_clock(m.duration)), Style::default().fg(c::DIM)),
    ];
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

/// Ambient "system pulse": concentric sonar rings expanding from the center,
/// their brightness and speed driven by live CPU + GPU + network energy. Fills
/// the right side whenever music isn't playing.
fn pulse_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    use ratatui::style::Color;

    // Smooth, delay-interpolated energy from the same buffers the EQs use.
    let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let cpuv = sampled_cores(&s.cpu_samples, target);
    let cpu = if cpuv.is_empty() {
        s.system.cpu_overall
    } else {
        cpuv.iter().sum::<f32>() / cpuv.len() as f32
    };
    let gpu = sampled_cores(&s.silicon_samples, target)
        .first()
        .copied()
        .unwrap_or(s.silicon.gpu_pct);
    let netbps = sampled_scalar(&s.net_samples, target);
    let net_e = ((netbps.max(1.0).log10() - 3.0) / 4.0).clamp(0.0, 1.0);
    let activity = (cpu / 100.0 * 0.45 + gpu / 100.0 * 0.35 + net_e * 0.20).clamp(0.0, 1.0);
    let energy = 0.32 + 0.68 * activity;

    let title = format!("SYSTEM PULSE      cpu {cpu:>2.0}%    gpu {gpu:>2.0}%");
    let block = panel(&title, false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let (w, h) = (inner.width as i32, inner.height as i32);
    if w < 4 || h < 2 {
        return;
    }

    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;
    let maxd = ((cx * cx) + (cy * 2.0) * (cy * 2.0)).sqrt().max(1.0);
    let speed = 1.6 + 4.5 * energy; // rings expand faster when busy
    let density = 0.6f32; // ring spacing
    let tf = t as f32;
    let quant = |c: u8| c & 0xF0; // coarsen colours so runs merge → fewer spans

    let mut lines: Vec<Line> = Vec::with_capacity(h as usize);
    for y in 0..h {
        let mut spans: Vec<Span> = Vec::new();
        let mut run = String::new();
        let mut run_col: Option<Color> = None;
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = (y as f32 - cy) * 2.0; // cells are ~2× taller than wide
            let d = (dx * dx + dy * dy).sqrt();
            let wave = (d * density - tf * speed).sin();
            let b = ((wave - 0.15) / 0.85).clamp(0.0, 1.0);
            let inten = b * b * energy;
            let cell_col = if inten < 0.05 {
                None
            } else {
                let rc = c::wipe((d / maxd).clamp(0.0, 1.0));
                match c::blend(c::BG, rc, inten.min(1.0)) {
                    Color::Rgb(r, g, bl) => Some(Color::Rgb(quant(r), quant(g), quant(bl))),
                    other => Some(other),
                }
            };
            let ch = if cell_col.is_none() { ' ' } else { '█' };
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
    f.render_widget(Paragraph::new(lines), inner);
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
