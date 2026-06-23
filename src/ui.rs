//! All rendering. Pure function of (state snapshot, animation clock) -> frame.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::state::{AppState, ProcSample};
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
/// at `target` time.
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
    // Ping-pong between `cur` and one reusable scratch buffer, swapping roles each
    // pass — so each pass reads from the previous result and writes into the other
    // buffer (no per-pass clone). `scratch` starts as a sized placeholder.
    let mut scratch = vec![0.0f32; n];
    for _ in 0..passes {
        // `cur` is the source this pass; `scratch` receives the blurred output.
        for (i, out) in scratch.iter_mut().enumerate().take(n) {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius).min(n - 1);
            let mut sum = 0.0f32;
            for &x in &cur[lo..=hi] {
                sum += x;
            }
            *out = sum / (hi - lo + 1) as f32;
        }
        std::mem::swap(&mut cur, &mut scratch); // result is now in `cur` again
    }
    cur
}

/// Render the RESOURCES wave: the live channels **stacked** into one smooth,
/// cohesive area that scrolls per-frame (the inputs are delay-interpolated
/// upstream, so the curve glides instead of stepping at 1 Hz). Each channel is a
/// shade up the jazz ramp — violet → pink → white, the "four shades of pink" —
/// piled bottom→top, so the silhouette is the combined activity. A dim→vivid
/// vertical glow plus a 1/8-block top edge keep it buttery; the stack is
/// normalised so the busiest column just kisses the top and the wave owns the card.
fn stacked_wave(f: &mut Frame, area: Rect, bands: &[(ratatui::style::Color, Vec<f32>)]) {
    use ratatui::style::Color;
    let w = area.width as usize;
    let h = area.height as usize;
    if w == 0 || h == 0 || bands.is_empty() {
        return;
    }
    // Blur each channel so bursty signals read as one buttery curve, not cliffs.
    let bands: Vec<(Color, Vec<f32>)> =
        bands.iter().map(|(col, v)| (*col, smoothed(v, 2, 4))).collect();
    // Per-column stack totals → scale so the tallest column ~fills the plot.
    let mut totals = vec![0.0f32; w];
    for (x, total) in totals.iter_mut().enumerate().take(w) {
        for (_, v) in &bands {
            *total += v.get(x).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
    }
    let peak = totals.iter().copied().fold(0.0f32, f32::max).max(0.8);
    let scale = (h as f32 * 0.94) / peak;

    let mut lines: Vec<Line> = Vec::with_capacity(h);
    for row in 0..h {
        let r_bot = (h - 1 - row) as f32; // this cell spans rows [r_bot, r_bot+1)
        let crest = ((r_bot + 0.5) / h as f32).clamp(0.0, 1.0); // dim base → vivid top
        let mut spans: Vec<Span> = Vec::new();
        let mut run = String::new();
        let mut run_col: Option<Color> = None;
        for (x, &col_total) in totals.iter().enumerate().take(w) {
            let total = col_total * scale; // stack top, in rows
            let (ch, col) = if total <= r_bot {
                (' ', None) // above the wave
            } else {
                // Which shade owns this cell? Probe the cell's middle (clamped just
                // inside the stack top) and find the band whose slice contains it.
                let probe = (r_bot + 0.5).min(total - 1e-3).max(0.0);
                let mut acc = 0.0f32;
                let mut shade = bands.last().map(|b| b.0).unwrap_or(c::BG);
                for (col, v) in &bands {
                    let br = v.get(x).copied().unwrap_or(0.0).clamp(0.0, 1.0) * scale;
                    if probe < acc + br {
                        shade = *col;
                        break;
                    }
                    acc += br;
                }
                // Sub-cell smoothing only on the very top edge of the stack.
                let ch = if total < r_bot + 1.0 { c::vblock(total - r_bot) } else { '█' };
                (ch, Some(c::blend(c::BG, shade, 0.42 + 0.58 * crest)))
            };
            if col == run_col {
                run.push(ch);
            } else {
                if !run.is_empty() {
                    let prev = std::mem::take(&mut run);
                    spans.push(match run_col {
                        Some(c) => Span::styled(prev, Style::default().fg(c)),
                        None => Span::raw(prev),
                    });
                }
                run.push(ch);
                run_col = col;
            }
        }
        if !run.is_empty() {
            spans.push(match run_col {
                Some(c) => Span::styled(run, Style::default().fg(c)),
                None => Span::raw(run),
            });
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// Row heights for the 11-slot left column given the available height `avail`.
///
/// The six system cards [0..6) are rigid — ROBOTS' bottom-row burn chart is never
/// clipped. The four messaging cards [6..10) size to their content when there's
/// room, but keep a guaranteed floor so they never collapse to zero under height
/// pressure (the old `Max(..)` constraints shrank them all the way to nothing on
/// shorter terminals, which made iMessage/Signal/Discord silently disappear).
/// Order of sacrifice under pressure: trailing slack → messaging content above its
/// floor (bottom-up) → WEATHER gives a couple rows → and only an extreme-tiny
/// terminal clips further (bottom-up, last resort).
fn left_card_heights(avail: u16, s: &AppState) -> [u16; 8] {
    // [0..4) system (rigid): resources (big — its 6-lane wave is the show, with CPU
    // folded in as its own lane), proc, robots, weather. [4..7) messaging (content):
    // iMessage, Signal, Discord. [7] slack. (CPU and mac-doctor no longer own cards —
    // CPU folds into RESOURCES, doctor folds into ROBOTS.)
    let mut h = [16u16, 9, 10, 7, 0, 0, 0, 0];
    let want = [
        card_height(&s.messages, s.msg_ui.active),
        card_height(&s.signal, false),
        discord_height(&s.discord),
    ];
    for (i, &w) in want.iter().enumerate() {
        h[4 + i] = w;
    }

    let sum: u16 = h.iter().sum();
    if sum <= avail {
        h[7] = avail - sum; // slack absorbs the remainder; cards stay content-tight
        return h;
    }
    let mut deficit = sum - avail;

    // 1) shrink messaging cards from content toward a small floor, bottom-up so the
    //    most important (iMESSAGE) keeps its content longest.
    for i in [6usize, 5, 4] {
        if deficit == 0 {
            break;
        }
        let floor = want[i - 4].min(5); // title + unread badge + a line or two
        let take = h[i].saturating_sub(floor).min(deficit);
        h[i] -= take;
        deficit -= take;
    }
    // 2) let WEATHER, then the big RESOURCES wave, give rows before sacrificing floors.
    for i in [3usize, 0] {
        if deficit == 0 {
            break;
        }
        let take = h[i].saturating_sub(if i == 0 { 9 } else { 5 }).min(deficit);
        h[i] -= take;
        deficit -= take;
    }
    // 3) extreme-tiny terminal: clip bottom-up through the floors, then the system
    //    cards, so the column never overflows.
    for i in [6usize, 5, 4, 3, 2, 1, 0] {
        if deficit == 0 {
            break;
        }
        let take = h[i].min(deficit);
        h[i] -= take;
        deficit -= take;
    }
    h
}

/// Clickable iMessage conversation rows captured during a render, so a mouse
/// click — handled in the event loop, which only holds the render *snapshot* and
/// so can't read hitboxes back out of shared state — can be mapped to a chat_id.
/// Rebuilt every frame by `messages_panel`.
#[derive(Default)]
pub struct MsgHit {
    rows: Vec<(u16, i64)>, // (screen y, chat_id) for each drawn conversation row
    x0: u16,
    x1: u16, // inclusive screen-x range of the card's clickable area
}

impl MsgHit {
    fn clear(&mut self) {
        self.rows.clear();
        self.x0 = 0;
        self.x1 = 0;
    }
    /// chat_id at a click position, if it lands on a conversation row.
    pub fn chat_at(&self, col: u16, row: u16) -> Option<i64> {
        if col < self.x0 || col > self.x1 {
            return None;
        }
        self.rows.iter().find(|(y, _)| *y == row).map(|(_, id)| *id)
    }
}

pub fn render(f: &mut Frame, s: &AppState, t: f64, hit: &mut MsgHit) {
    hit.clear();
    // Push this frame's cross-faded, album-biased accents into the live palette
    // store up front so every card below reads a coherent, glided value (#8).
    c::apply_dynamic(&s.dynamic_theme);

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

    // Heights are computed (not fixed constraints) so the four messaging cards keep
    // a guaranteed floor and never collapse to zero on shorter terminals.
    let lh = left_card_heights(body[0].height, s);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints(lh.map(Constraint::Length))
        .split(body[0]);

    resources_panel(f, left[0], s);
    proc_panel(f, left[1], s);
    robots_panel(f, left[2], s, t);
    weather_panel(f, left[3], s);
    messages_panel(f, left[4], s, t, hit);
    signal_panel(f, left[5], s, t);
    discord_panel(f, left[6], s, t);

    // Until the first music poll lands, show the (neutral) lyrics panel so we
    // never flash the wrong thing before the real state is known. After that:
    // actively playing → lyrics; paused/stopped/idle → live system graphs.
    let show_lyrics = !s.music.polled || (s.music.playing && !s.music.track.is_empty());

    // A taller NOW PLAYING card so the album cover renders as a real, legible
    // square (more half-block sub-pixels) instead of a postage stamp.
    const NP_H: u16 = 13;
    // KEYBINDS sizes to its content so its box flexes as binds come/go; the
    // trailing Min(0) is the bottom-right slack reserved for the next card.
    // KEYBINDS height eases to 0 when hidden via Hyper+H, so the card glides shut
    // and the bottom-right slack opens up (then back when shown).
    let kbh_full = keybinds_height(&s.keybinds, body[1].width);
    let kbh = (kbh_full as f32 * keybinds_open_frac(s)).round() as u16;
    if show_lyrics {
        // Cap the lyric band to a tight 9-row block (7 inner rows).
        const LYRICS_H: u16 = 9;
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(NP_H),
                Constraint::Length(LYRICS_H),
                Constraint::Max(kbh),
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
            .constraints([Constraint::Length(NP_H), Constraint::Max(kbh), Constraint::Min(0)])
            .split(body[1]);
        now_playing_row(f, right[0], s, t);
        if right[1].height >= 3 {
            keybinds_panel(f, right[1], s);
        }
    }
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
    // Run-length merge adjacent chars that resolve to the same colour into one
    // Span (spaces stay unstyled runs of their own) — same per-char output, fewer
    // allocations.
    let mut spans: Vec<Span> = Vec::new();
    let mut run = String::new();
    let mut run_col: Option<Color> = None; // None == an unstyled (space) run
    let flush = |run: &mut String, run_col: Option<Color>, spans: &mut Vec<Span>| {
        if run.is_empty() {
            return;
        }
        let prev = std::mem::take(run);
        spans.push(match run_col {
            Some(col) => {
                let mut st = Style::default().fg(col);
                if bold {
                    st = st.add_modifier(Modifier::BOLD);
                }
                Span::styled(prev, st)
            }
            None => Span::raw(prev),
        });
    };
    for (i, ch) in chars.iter().enumerate() {
        let cell_col = if *ch == ' ' {
            None
        } else {
            let p = i as f32 / n as f32;
            let mut d = (p - head).abs();
            if d > 0.5 {
                d = 1.0 - d; // wrap so the glint is continuous
            }
            let b = (-(d * d) / (2.0 * sigma * sigma)).exp(); // bright band 0..1
            Some(c::blend(base, sheen, b)) // calm base -> glint
        };
        if cell_col != run_col {
            flush(&mut run, run_col, &mut spans);
            run_col = cell_col;
        }
        run.push(*ch);
    }
    flush(&mut run, run_col, &mut spans);
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
    // Run-length merge adjacent tokens that resolve to the same (colour, bold) into
    // one Span; spaces are unstyled runs. Same per-char output, fewer allocations.
    let mut run = String::new();
    let mut run_key: Option<(Color, bool)> = None; // None == an unstyled (space) run
    let flush = |run: &mut String, run_key: Option<(Color, bool)>, spans: &mut Vec<Span>| {
        if run.is_empty() {
            return;
        }
        let prev = std::mem::take(run);
        spans.push(match run_key {
            Some((col, bold)) => {
                let mut st = Style::default().fg(col);
                if bold {
                    st = st.add_modifier(Modifier::BOLD);
                }
                Span::styled(prev, st)
            }
            None => Span::raw(prev),
        });
    };
    for (k, (ch, base, bold)) in toks.iter().enumerate() {
        let key = if *ch == ' ' {
            None
        } else {
            let p = (lead_w + k) as f32 / total_cols;
            let mut d = (p - head).abs();
            if d > 0.5 {
                d = 1.0 - d;
            }
            let b = (-(d * d) / (2.0 * 0.16 * 0.16)).exp();
            let col = c::blend(c::blend(*base, sheen, b), c::BG, f);
            Some((col, *bold))
        };
        if key != run_key {
            flush(&mut run, run_key, &mut spans);
            run_key = key;
        }
        run.push(*ch);
    }
    flush(&mut run, run_key, &mut spans);
    Line::from(spans)
}

/// A single line of `segs` that shimmers, and — when it overflows `width` —
/// marquee-scrolls smoothly left across the box. The scroll is frame-paced off
/// `t`: it eases out of a hold at the start, glides through the text, eases into
/// a hold at the end, then bounces back — no cell-stepped jitter, the offset is
/// smoothed by a cosine so the sweep accelerates and decelerates buttery-soft.
/// `row` phase-offsets the sheen so stacked lines don't shimmer in lockstep.
fn marquee(segs: &[(&str, Color, bool)], width: usize, t: f64, row: f32) -> Line<'static> {
    let mut toks: Vec<(char, Color, bool)> = Vec::new();
    for (txt, col, bold) in segs {
        for ch in txt.chars() {
            toks.push((ch, *col, *bold));
        }
    }
    let width = width.max(1);
    if toks.len() <= width {
        return shimmer_window(Vec::new(), 0, &toks, width, t, row, 1.0);
    }

    // How far we must travel, in cells. A pad of blanks at the tail lets the line
    // fully clear the box before it bounces back, so the end never looks clipped.
    let pad = 3usize;
    let travel = (toks.len() + pad).saturating_sub(width) as f64;

    // One full cycle = scroll out (scan) + hold at each end. The phase ping-pongs
    // 0→1→0 so the text slides left, pauses, and slides home — never wrapping with
    // a visible pop. `triangle` is the raw 0..1 sweep; `eased` smooths its ends.
    let hold = 2.2f64; // seconds parked at each extreme
    let scan = (travel / 6.0).max(2.0); // seconds to traverse (≈6 cells/s)
    let period = 2.0 * (scan + hold);
    let x = t.rem_euclid(period);
    let raw = if x < hold {
        0.0
    } else if x < hold + scan {
        (x - hold) / scan
    } else if x < hold + scan + hold {
        1.0
    } else {
        1.0 - (x - hold - scan - hold) / scan
    };
    // Cosine ease-in-out so the start and end of every slide are gentle.
    let eased = 0.5 - 0.5 * (raw.clamp(0.0, 1.0) * std::f64::consts::PI).cos();
    let off = (eased * travel).round() as usize;
    let end = (off + width).min(toks.len());
    let window = &toks[off..end];
    shimmer_window(Vec::new(), 0, window, width, t, row, 1.0)
}

/// A progress bar that doesn't just sit there — a bright sheen glides across the
/// *filled* portion on top of a position-graded jazz ramp, so it reads like lit
/// glass. Sub-cell fill via 1/8-block glyphs keeps the leading edge smooth as the
/// track plays; the unfilled track is a faint rail. Built on `jazz`/`blend` so it
/// stays dead-on the synthwave palette and animates buttery off the `t` clock.
fn shimmer_bar(frac: f32, width: usize, t: f64) -> Vec<Span<'static>> {
    let frac = frac.clamp(0.0, 1.0) as f64;
    let width = width.max(1);
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    let full = (eighths / 8).min(width);
    let rem = eighths % 8;
    // A narrow bright band sweeps across the bar; it wraps so it never pops.
    let head = (t * 0.45).rem_euclid(1.0) as f32;
    let sigma = 0.10f32;
    let sheen = c::jazz(0.92); // pink-white glint
    let cell = |i: usize, glyph: char| -> Span<'static> {
        let p = (i as f32 + 0.5) / width as f32;
        // Position-graded base so the bar itself ramps violet→pink across its run.
        let base = c::jazz(0.30 + 0.55 * p);
        let mut d = (p - head).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        let b = (-(d * d) / (2.0 * sigma * sigma)).exp();
        let col = c::blend(base, sheen, b);
        Span::styled(glyph.to_string(), Style::default().fg(col).add_modifier(Modifier::BOLD))
    };
    let mut spans = Vec::with_capacity(width);
    for i in 0..full {
        spans.push(cell(i, '█'));
    }
    let mut drawn = full;
    if drawn < width && rem > 0 {
        spans.push(cell(full, ['█', '▏', '▎', '▍', '▌', '▋', '▊', '▉'][rem]));
        drawn += 1;
    }
    if drawn < width {
        // Faint rail for the unplayed remainder.
        spans.push(Span::styled("░".repeat(width - drawn), Style::default().fg(c::FAINT)));
    }
    spans
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
            Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
        ))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG))
}

/// Box-drawing glyphs we recolor for the shimmer (rounded + any inner tees).
fn is_box_glyph(s: &str) -> bool {
    matches!(
        s,
        "─" | "│" | "╭" | "╮" | "╰" | "╯" | "┌" | "┐" | "└" | "┘" | "├" | "┤" | "┬" | "┴" | "┼"
    )
}

/// Repaint a card's already-drawn border so a bright jazz gradient sweeps around
/// the perimeter (animated by `t`) — a hard-to-miss "needs attention" shimmer for
/// unread messages / a voice join / a doctor run. Title text and inner content are
/// left untouched (only box-drawing glyphs are recolored). Call AFTER rendering the
/// card's block, gated on the card's alert condition.
/// Sweep a jazz gradient around a card's border. `speed` sets how fast the band
/// travels (turns/sec ×2); `glow` (0..1) blends the sweep toward white for a
/// brighter, hotter shimmer — used when a Discord voice channel is actively
/// talking. The defaults (0.55, 0.0) are the calm attention shimmer.
fn shimmer_border(f: &mut Frame, area: Rect, t: f64, speed: f32, glow: f32) {
    if area.width < 2 || area.height < 2 {
        return;
    }
    let buf = f.buffer_mut();
    let (left, right) = (area.left(), area.right() - 1);
    let (top, bottom) = (area.top(), area.bottom() - 1);
    let (w, h) = (area.width as f32, area.height as f32);
    let perim = 2.0 * (w - 1.0) + 2.0 * (h - 1.0);
    for y in top..=bottom {
        for x in left..=right {
            if x != left && x != right && y != top && y != bottom {
                continue; // interior cell
            }
            let Some(cell) = buf.cell_mut((x, y)) else {
                continue;
            };
            if !is_box_glyph(cell.symbol()) {
                continue; // leave the title / badge text alone
            }
            // Perimeter index, clockwise from the top-left corner.
            let (cx, cy) = ((x - left) as f32, (y - top) as f32);
            let idx = if y == top {
                cx
            } else if x == right {
                (w - 1.0) + cy
            } else if y == bottom {
                (w - 1.0) + (h - 1.0) + ((w - 1.0) - cx)
            } else {
                2.0 * (w - 1.0) + (h - 1.0) + ((h - 1.0) - cy)
            };
            // Two color cycles around the box, sweeping at `speed` turns/sec.
            let phase = ((idx / perim) * 2.0 - t as f32 * speed).rem_euclid(1.0);
            let base = c::jazz(phase);
            cell.fg = if glow > 0.0 {
                c::blend(base, ratatui::style::Color::Rgb(245, 245, 255), glow.clamp(0.0, 1.0))
            } else {
                base
            };
            cell.modifier.insert(Modifier::BOLD);
        }
    }
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
        Span::styled(" q ", Style::default().fg(c::BG).bg(c::accent()).add_modifier(Modifier::BOLD)),
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
    spans.push(Span::styled(format!("▼{:>5}", fmt_rate_short(sys.net_rx_bps)), Style::default().fg(c::cyan()).bg(c::PANEL_BORDER)));
    spans.push(Span::styled(format!(" ▲{:>5}", fmt_rate_short(sys.net_tx_bps)), Style::default().fg(c::pink()).bg(c::PANEL_BORDER)));
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

/// One RESOURCES wave channel: (shade, label, 0..1 normaliser, res_samples slot).
type WaveChannel = (Color, &'static str, fn(f32) -> f32, usize);

fn resources_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("RESOURCES", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 || inner.width < 8 {
        return;
    }

    // The wave's six channels: jazz-ramp shade (cyan→violet→pink→white), label, the
    // 0..1 normaliser, and which res_samples slot it reads. The key and the wave both
    // derive from this list, so the colours in the legend always match. CPU leads on
    // the cyan end — its own lane + shade, folded in from the retired CPU card.
    let lograte: fn(f32) -> f32 = |kbps| (kbps.max(0.5).log10() / 4.0).clamp(0.0, 1.0);
    let pct: fn(f32) -> f32 = |p| (p / 100.0).clamp(0.0, 1.0);
    let channels: [WaveChannel; 6] = [
        (Color::Rgb(86, 214, 255), "cpu", pct, 5), // fixed cyan (CYAN_BASE) — no album drift
        (c::jazz(0.42), "mem", pct, 0),            // memory used
        (c::jazz(0.56), "gpu", pct, 4),            // GPU utilization
        (c::jazz(0.70), "net ↓", lograte, 1),      // network down
        (c::jazz(0.84), "net ↑", lograte, 2),      // network up
        (c::jazz(0.97), "disk", lograte, 3),       // disk I/O
    ];

    // --- Key: one row, a colour swatch per wave shade + what it means. Clean and
    // quiet (swatch bold in the shade, label dim) so the wave stays the show.
    let mut key: Vec<Span> = Vec::new();
    for (i, (col, name, _, _)) in channels.iter().enumerate() {
        if i > 0 {
            key.push(Span::raw("   "));
        }
        key.push(Span::styled("█ ", Style::default().fg(*col).add_modifier(Modifier::BOLD)));
        key.push(Span::styled(*name, Style::default().fg(c::DIM)));
    }
    let text_h = 1u16;
    f.render_widget(
        Paragraph::new(Line::from(key)),
        Rect { x: inner.x, y: inner.y, width: inner.width, height: text_h },
    );

    // --- The wave: BIG. It owns every remaining row of the card. The four live
    // channels stack into one smooth, cohesive area in four shades of pink. Inputs
    // are frame-interpolated (delayed Catmull-Rom from res_samples) so the curve
    // glides continuously instead of stepping at the 1 Hz collector tick — the
    // buttery motion the original net wave had. A blank gap row sits under the key.
    let plot_h = inner.height.saturating_sub(text_h + 1);
    if plot_h >= 1 {
        let plot = Rect {
            x: inner.x,
            y: inner.y + text_h + 1,
            width: inner.width,
            height: plot_h,
        };
        let pw = plot.width as usize;
        let plot_bands: Vec<(ratatui::style::Color, Vec<f32>)> = channels
            .iter()
            .map(|(col, _, norm, ch)| {
                let ch = *ch;
                (*col, series(&s.res_samples, pw, move |b, tt| sampled_channel(b, ch, tt), *norm))
            })
            .collect();
        stacked_wave(f, plot, &plot_bands);
    }
}

/// Merged "ROBOTS WORKING" card: left = Claude token throughput
/// (today/week/month/sessions) crowned by a per-hour burn chart for today;
/// right = the most-recently-active local git branch's pulse. The two halves
/// are grouped by a soft, static gradient rule rather than a hard split.
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
    // The live "who's working" feed is now the card's left half, so give it a
    // touch more room than the old burn chart needed — enough for one clean line
    // per robot (icon · project · action · age) — while the git pulse keeps the
    // right.
    let lw = ((inner.width as usize * 42 / 100).max(18))
        .min((inner.width as usize).saturating_sub(22)) as u16;
    let div_x = inner.x + lw + 1; // 1-col gutter, then divider
    let rx = div_x + 2; // 1-col gutter after the divider
    let rw = (inner.x + inner.width).saturating_sub(rx);

    // ----- soft grouping rule: a calm, static blue→violet gradient down the
    // line, drawn with the light "│" glyph and kept dim so it reads as a gentle
    // seam between the two halves rather than a jarring hard split.
    for r in 0..h {
        let p = r as f32 / h.max(1) as f32;
        let col = c::blend(c::FAINT, c::jazz(0.18 + 0.30 * p), 0.45);
        f.render_widget(
            Paragraph::new(Span::styled("╎", Style::default().fg(col))),
            Rect { x: div_x, y: inner.y + r as u16, width: 1, height: 1 },
        );
    }

    // ----- left: realtime "who's working" feed on top, token windows beneath --
    // The live session feed is the card's anchor now (replacing the old per-hour
    // burn chart); the rolling-window token totals are condensed to two compact
    // rows pinned at the bottom so the feed gets the room to show every robot.
    let wlen = 2u16; // today/week on one row, month/sessions on the next
    let feed_h = inner.height.saturating_sub(wlen);
    if feed_h >= 1 {
        live_feed(
            f,
            Rect { x: inner.x, y: inner.y, width: lw, height: feed_h },
            &s.live,
            &s.doctor,
            t,
        );
    }

    if !u.fresh {
        f.render_widget(
            Paragraph::new(Span::styled("scanning ~/.claude…", Style::default().fg(c::DIM))),
            Rect { x: inner.x, y: inner.y + feed_h, width: lw, height: 1 },
        );
    } else {
        let tot_today = u.today_input + u.today_output + u.today_cache_read + u.today_cache_write;
        // One label:value cell. Two cells per line keeps the totals to two rows.
        let cell = |label: &str, val: String, col| {
            vec![
                Span::styled(format!("{label} "), Style::default().fg(c::DIM)),
                Span::styled(val, Style::default().fg(col).add_modifier(Modifier::BOLD)),
            ]
        };
        let gap = Span::raw("  ");
        let mut l1 = cell("today", fmt_tokens(tot_today), c::cyan());
        l1.push(gap.clone());
        l1.extend(cell("wk", fmt_tokens(u.tokens_7d), c::accent()));
        let mut l2 = cell("month", fmt_tokens(u.tokens_30d), c::pink());
        l2.push(gap);
        l2.extend(cell("sess", format!("{}", u.sessions_30d), c::TEXT));
        let wy = inner.y + inner.height.saturating_sub(wlen);
        f.render_widget(
            Paragraph::new(vec![Line::from(l1), Line::from(l2)]),
            Rect { x: inner.x, y: wy, width: lw, height: wlen.min(inner.height) },
        );
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
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    ));
    let branch = Line::from(branch_spans);
    let hashw = g.last_hash.chars().count() + 1;
    let commit = Line::from(vec![
        Span::styled(format!("{} ", g.last_hash), Style::default().fg(c::cyan())),
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
        Span::styled(" loc today", Style::default().fg(c::FAINT)),
    ]);
    let commits = Line::from(vec![
        Span::styled(format!("{}", g.commits_today), Style::default().fg(c::TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(" commits today", Style::default().fg(c::DIM)),
    ]);
    let prs = Line::from(vec![
        Span::styled(format!("{}", g.pr_count), Style::default().fg(c::pink()).add_modifier(Modifier::BOLD)),
        Span::styled(" PRs", Style::default().fg(c::DIM)),
    ]);
    let merges = Line::from(vec![
        Span::styled(
            format!("{}", g.merges_main),
            Style::default().fg(if g.merges_main > 0 { c::GREEN } else { c::TEXT }).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" merges today", Style::default().fg(c::DIM)),
    ]);
    f.render_widget(
        Paragraph::new(vec![branch, commit, age, Line::from(""), loc, commits, prs, merges]),
        Rect { x: rx, y: inner.y, width: rw, height: inner.height },
    );
}

/// The glyph + base color for a live session's current action, so the kind reads
/// at a glance: a shell prompt for a command, a pencil for an edit, etc.
fn kind_glyph(k: crate::state::ActionKind) -> (&'static str, Color) {
    use crate::state::ActionKind as K;
    match k {
        K::Idle => ("○", c::FAINT),
        K::Think => ("✶", c::accent()),
        K::Run => ("❯", c::cyan()),
        K::Edit => ("✎", c::GREEN),
        K::Read => ("▤", c::accent()),
        K::Web => ("◍", c::cyan()),
        K::Agent => ("✦", c::pink()),
        K::Tool => ("⚙", c::cyan()),
        K::Respond => ("✺", c::pink()),
    }
}

/// Compact age: seconds under a minute, then minutes. Live sessions are by
/// definition young, so this never needs hours.
fn fmt_age(secs: f64) -> String {
    if secs < 1.0 {
        "now".to_string()
    } else if secs < 60.0 {
        format!("{}s", secs as u64)
    } else {
        format!("{}m", (secs / 60.0) as u64)
    }
}

/// Realtime feed for the ROBOTS card: one glide-y line per live Claude Code
/// session — icon (what it's doing), project, action label, and age — newest-
/// active on top (the collector sorts that way), so a session that just moved
/// rises to the top and pushes the others down. Capped at the 5 most recent. The
/// icon shimmers toward bright on fresh activity and fades as a session goes
/// quiet, so the whole feed breathes with the work.
fn live_feed(
    f: &mut Frame,
    area: Rect,
    l: &crate::state::LiveSessions,
    d: &crate::state::Doctor,
    t: f64,
) {
    if area.height == 0 || area.width < 6 {
        return;
    }
    let w = area.width as usize;
    let tt = t as f32;

    // mac-doctor folds in here: when it's actively triaging (or an incident needs
    // the user), it rides the TOP of the feed as a special shimmering alert row
    // instead of owning a whole card. Idle → it isn't shown at all.
    let mut row = 0u16;
    if doctor_alert(d) {
        f.render_widget(
            Paragraph::new(doctor_feed_line(d, w, t)),
            Rect { x: area.x, y: area.y, width: area.width, height: 1 },
        );
        row = 1;
    }

    if !l.fresh {
        if row == 0 {
            f.render_widget(
                Paragraph::new(Span::styled("scanning robots…", Style::default().fg(c::DIM))),
                Rect { height: 1, ..area },
            );
        }
        return;
    }
    if l.sessions.is_empty() {
        if row == 0 {
            f.render_widget(
                Paragraph::new(Span::styled("no active sessions", Style::default().fg(c::DIM))),
                Rect { height: 1, ..area },
            );
        }
        return;
    }

    // The most-recent sessions, filling from the top below any doctor row.
    let cap = (area.height.saturating_sub(row) as usize).min(5);
    for (i, sess) in l.sessions.iter().take(cap).enumerate() {
        f.render_widget(
            Paragraph::new(session_line(sess, w, tt, i)),
            Rect { x: area.x, y: area.y + row + i as u16, width: area.width, height: 1 },
        );
    }
}

/// Should mac-doctor show in the ROBOTS feed? Only when it's actually doing
/// something the user cares about — a triage run in flight, or a last incident
/// that still needs a human. Otherwise it stays out of the way entirely.
fn doctor_alert(d: &crate::state::Doctor) -> bool {
    d.available && (d.running || d.last_outcome == "needs-user" || d.last_outcome == "unresolved")
}

/// mac-doctor's row in the ROBOTS feed. The whole line shimmers (the only feed
/// row that does) so a system alert reads instantly as different from a Claude
/// session; severity sets the base color, and it sweeps fast + bold when urgent.
fn doctor_feed_line(d: &crate::state::Doctor, w: usize, t: f64) -> Line<'static> {
    let base = match d.last_severity.as_str() {
        "critical" => c::RED,
        "warn" => c::pink(),
        _ => c::accent(),
    };
    let action = if d.running {
        if d.step.is_empty() { "investigating…".to_string() } else { d.step.clone() }
    } else if !d.trigger.is_empty() {
        d.trigger.clone()
    } else if !d.last_title.is_empty() {
        d.last_title.clone()
    } else {
        "needs a look".to_string()
    };
    let age = if d.running {
        "now".to_string()
    } else if d.last_rel.is_empty() {
        "—".to_string()
    } else {
        d.last_rel.clone()
    };
    let age_w = age.chars().count();
    let icon_w = 2; // glyph + trailing space

    let proj = "mac-doctor";
    let middle = w.saturating_sub(icon_w + age_w + 1);
    let proj_budget = dwidth(proj).min(middle.saturating_sub(4).max(1));
    let projf = fit_width(proj, proj_budget.max(1));
    let mut rem = middle.saturating_sub(dwidth(&projf) + 1);
    let actionf = fit_width(&action, rem.max(1));
    rem = rem.saturating_sub(dwidth(&actionf));

    // Only an actively-running triage shimmers (urgent traveling band, severity-
    // tinted). A past incident that merely "needs a look" reads as plain, calm text
    // — present in the feed but visually like a quiet session, never nagging.
    let label = format!("✚ {projf} {actionf}");
    let mut spans = if d.running {
        shimmer_spans(&label, t, 0.0, base, 1.1, 0.10, true)
    } else {
        vec![Span::styled(label, Style::default().fg(c::DIM))]
    };
    if rem > 0 {
        spans.push(Span::raw(" ".repeat(rem)));
    }
    spans.push(Span::styled(format!(" {age}"), Style::default().fg(c::FAINT)));
    Line::from(spans)
}

/// One robot's line: `{icon} {project}  {action}            {age}`. The icon
/// pulses in its kind color (fading as the session ages); the age is right-
/// aligned and the action fills whatever's left.
fn session_line(s: &crate::state::LiveSession, w: usize, t: f32, idx: usize) -> Line<'static> {
    use crate::state::ActionKind as K;
    let idle = matches!(s.kind, K::Idle);
    let (glyph, kcol) = kind_glyph(s.kind);

    // Recency 1.0 (fresh) → 0.0 (going quiet over ~12s): drives both the fade and
    // how lively the pulse is, so a hot robot sparkles and a stalled one settles.
    let recency = (1.0 - s.age_secs as f32 / 12.0).clamp(0.0, 1.0);
    let icon_col = if idle {
        c::FAINT
    } else {
        // Dim toward FAINT as it ages, then shimmer toward TEXT on the beat. Phase
        // offset per row so the feed twinkles rather than blinking in lockstep.
        let base = c::blend(kcol, c::FAINT, 0.5 * (1.0 - recency));
        let pulse = 0.5 + 0.5 * (t * 3.0 + idx as f32 * 0.8).sin();
        c::blend(base, c::TEXT, 0.4 * pulse * (0.3 + 0.7 * recency))
    };

    // Age string sits flush right; reserve its width plus a leading gap.
    let age = fmt_age(s.age_secs);
    let age_w = age.chars().count();
    let icon_w = 2; // glyph + trailing space

    let mut spans = vec![Span::styled(
        format!("{glyph} "),
        Style::default().fg(icon_col).add_modifier(if recency > 0.5 && !idle {
            Modifier::BOLD
        } else {
            Modifier::empty()
        }),
    )];

    // Middle = project + action (+ optional model), sharing the room left after
    // the icon and the right-aligned age. The action is the interesting part, so
    // the project takes a capped slice and the action gets the rest.
    let middle = w.saturating_sub(icon_w + age_w + 1);
    let proj_budget = (middle * 9 / 20).clamp(6, 16).min(middle.saturating_sub(4));
    let proj = fit_width(&s.project, proj_budget.max(1));
    let mut rem = middle.saturating_sub(dwidth(&proj) + 1); // 1-col gap after project
    let action = fit_width(&s.action, rem.max(1));
    rem = rem.saturating_sub(dwidth(&action));

    // Show the model dim before the age when there's comfortable room (wide
    // terminals); it quietly drops on narrow ones.
    let model = (!s.model.is_empty() && rem >= dwidth(&s.model) + 2).then(|| {
        rem -= dwidth(&s.model) + 2;
        s.model.clone()
    });

    spans.push(Span::styled(
        proj,
        Style::default()
            .fg(if idle { c::DIM } else { c::accent() })
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        format!(" {action}"),
        Style::default().fg(if idle { c::FAINT } else { c::TEXT }),
    ));
    if let Some(m) = model {
        spans.push(Span::styled(format!(" ·{m}"), Style::default().fg(c::FAINT)));
    }
    if rem > 0 {
        spans.push(Span::raw(" ".repeat(rem)));
    }
    spans.push(Span::styled(format!(" {age}"), Style::default().fg(c::FAINT)));
    Line::from(spans)
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

/// How much of the QUEUE card is on-screen: 1.0 = fully shown (35% width), 0.0 =
/// fully collapsed (LYRICS owns the whole row). Smoothstep-eased off the
/// `queue_toggle_at` stamp set by `settle_queue_anim`, so the width glides.
fn queue_open_frac(s: &AppState) -> f32 {
    const DUR: f32 = 0.38; // matches main::QUEUE_ANIM
    let p = (s.queue_toggle_at.elapsed().as_secs_f32() / DUR).clamp(0.0, 1.0);
    let eased = p * p * (3.0 - 2.0 * p); // smoothstep
    if s.queue_open { eased } else { 1.0 - eased }
}

/// 0→1 eased visibility of the KEYBINDS card, driven by Hyper+H (the flag file
/// the keybinds collector polls). Mirrors `queue_open_frac` so the card glides
/// open/closed instead of popping.
fn keybinds_open_frac(s: &AppState) -> f32 {
    const DUR: f32 = 0.36; // matches main::KEYBINDS_ANIM (360ms)
    let p = (s.keybinds_toggle_at.elapsed().as_secs_f32() / DUR).clamp(0.0, 1.0);
    let eased = p * p * (3.0 - 2.0 * p); // smoothstep
    if s.keybinds_visible { eased } else { 1.0 - eased }
}

/// The lyrics band: LYRICS on the left, the Apple Music QUEUE on the right. When
/// there's nothing up next the QUEUE smoothly collapses and LYRICS expands to
/// fill the whole row (and back when a queue appears) — the width glides via
/// `queue_open_frac`.
fn lyrics_row(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    if area.width < 64 {
        lyrics_panel(f, area, s, t);
        return;
    }
    let frac = queue_open_frac(s);
    // The queue's full slot is 35% of the row; it shrinks to nothing as it closes.
    let q_full = (area.width as f32 * 0.35).round().max(1.0) as u16;
    let shown = (q_full as f32 * frac).round() as u16;
    let lyrics_w = area.width.saturating_sub(shown).max(1);
    let lyrics_area = Rect { x: area.x, y: area.y, width: lyrics_w, height: area.height };
    lyrics_panel(f, lyrics_area, s, t);
    // Draw the queue only while it's wide enough to hold its border + content; the
    // last few columns collapse as bare background so there's no empty-box flash.
    if shown >= 12 {
        let queue_area = Rect { x: area.x + lyrics_w, y: area.y, width: shown, height: area.height };
        queue_panel(f, queue_area, s);
    }
}

/// QUEUE card: the next few tracks Apple Music will play, numbered, with the
/// artist + length beneath each. Best-effort from the current playlist order.
fn queue_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let q = &s.queue;
    let m = &s.music;

    // Title with an on-palette count badge — same custom-span construction the
    // discord card uses, so the queued-track "3" reads as cyan (not off-palette).
    let mut title_spans = vec![Span::styled(
        " QUEUE  ",
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    let queued = q.items.len();
    if m.running && !m.track.is_empty() && queued > 0 {
        title_spans.push(Span::styled(
            format!("{queued}"),
            Style::default().fg(c::cyan()).add_modifier(Modifier::BOLD),
        ));
        title_spans.push(Span::styled(" up next ", Style::default().fg(c::DIM)));
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
    let accents = [c::cyan(), c::pink(), c::GREEN];
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
            if m.is_tv() { "gathering trivia…" } else { "gathering liner notes…" }
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
    let bullets = [c::cyan(), c::pink(), c::GREEN, c::YELLOW, c::accent()];

    // A slow, *uniform* sheen: every line breathes together (no travelling glint,
    // no per-line phase offset) so the facts read as a calm, nicely-tinted block
    // that shimmers just a little. Text sits in a soft violet-white; the breath
    // nudges it toward a gentle glint and back.
    let glow = 0.5 + 0.5 * (t * 0.6).sin() as f32; // 0..1, slow + uniform
    let base = c::blend(c::TEXT, c::accent(), 0.16); // soft violet-white — every line
    let sheen = c::blend(c::TEXT, c::cyan(), 0.35); // gentle glint target
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
                    let col = c::blend(base, sheen, 0.14 * glow); // a little, uniform
                    let mut spans = if j == 0 {
                        vec![Span::styled("• ", Style::default().fg(color).add_modifier(Modifier::BOLD))]
                    } else {
                        vec![Span::raw("  ")]
                    };
                    spans.push(Span::styled(seg, Style::default().fg(col)));
                    Line::from(spans)
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
    // Movie/TV trivia is denser and worth dwelling on — hold each page much
    // longer so a deep cast/production fact has time to land before it dissolves.
    let hold = if m.is_tv() { 9.0 } else { 3.4 };
    let (pi, alpha) = dissolve_phase(pages.len(), t, hold, 0.9);
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
    let tv = m.is_tv();
    let block = panel(if tv { "NOW WATCHING" } else { "NOW PLAYING" }, hot);
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

    // Album/poster column on the left, inset so it breathes. We size it to the
    // art's TRUE aspect ratio: a square album cover renders cols ≈ 2×rows (a
    // terminal cell is ~2:1, and a half-block pixel is ~square), while a portrait
    // movie poster renders taller-than-wide so it actually looks like the cover.
    let pad_v: u16 = 1;
    let max_rows = inner.height.saturating_sub(pad_v * 2).max(1);
    // Width reserved for the text column beside the art; the art takes the rest.
    let max_cols = inner.width.saturating_sub(24).max(2);
    // aspect = image width / height (1.0 album, ~0.67 movie poster). Fall back to
    // square until the real artwork has been decoded.
    let aspect = if s.album_art.w > 0 && s.album_art.h > 0 {
        (s.album_art.w as f32 / s.album_art.h as f32).clamp(0.4, 2.5)
    } else {
        1.0
    };
    // Fill the card height, then derive width from aspect; if that overflows the
    // width budget, clamp width and back-solve the rows so it never distorts.
    let mut art_rows = max_rows;
    let mut art_cols = (aspect * 2.0 * art_rows as f32).round() as u16;
    if art_cols > max_cols {
        art_cols = max_cols;
        art_rows = ((art_cols as f32 / (aspect * 2.0)).round() as u16).max(1);
    }
    art_cols = art_cols.max(2);
    art_rows = art_rows.max(1);
    let art_area = Rect { x: inner.x + 1, y: inner.y + pad_v, width: art_cols, height: art_rows };
    render_art(f, art_area, &s.album_art, m.track_id() == s.album_art.track_id);

    // Text column starts after the art + a small gap.
    let gap = art_cols + 3;
    let pos = m.position();
    let frac = if m.duration > 0.0 { (pos / m.duration) as f32 } else { 0.0 };

    // Text column width — needed before we lay out the (possibly scrolling) title.
    let textw = (inner.width.saturating_sub(gap)) as usize;

    // Two text lines, source-aware. Music: "Artist - Song" / album. TV episode:
    // "Show  S2·E5" / episode title. Movie: title / "year · genre · director".
    // A line wider than the column smoothly marquee-scrolls instead of truncating.
    let se_tag = if tv && m.watch.season > 0 {
        format!("  S{}·E{}", m.watch.season, m.watch.episode)
    } else {
        String::new()
    };
    let movie_sub = if tv {
        [m.watch.year.as_str(), m.watch.genre.as_str(), m.watch.director.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("  ·  ")
    } else {
        String::new()
    };
    let (title, album) = if tv {
        if !m.watch.show.is_empty() {
            (
                marquee(
                    &[(m.watch.show.as_str(), c::accent(), true), (se_tag.as_str(), c::FAINT, false)],
                    textw, t, 0.0,
                ),
                marquee(&[(m.track.as_str(), c::TEXT, true)], textw, t, 1.0),
            )
        } else {
            (
                marquee(&[(m.track.as_str(), c::accent(), true)], textw, t, 0.0),
                marquee(&[(movie_sub.as_str(), c::DIM, false)], textw, t, 1.0),
            )
        }
    } else {
        (
            marquee(
                &[
                    (m.artist.as_str(), c::accent(), true),
                    (" - ", c::FAINT, false),
                    (m.track.as_str(), c::TEXT, true),
                ],
                textw, t, 0.0,
            ),
            marquee(&[(m.album.as_str(), c::DIM, false)], textw, t, 1.0),
        )
    };

    // Progress bar stretches to the right edge of the card: only the two time
    // labels are reserved, the bar takes everything between them. A sheen glides
    // across the filled portion so it reads like lit glass while the track plays.
    let pfx = format!("{} ", fmt_clock(pos));
    let sfx = format!(" {}", fmt_clock(m.duration));
    let bw = textw.saturating_sub(pfx.chars().count() + sfx.chars().count()).max(8);
    let mut progress_spans = vec![Span::styled(pfx, Style::default().fg(c::cyan()))];
    progress_spans.extend(shimmer_bar(frac, bw, t));
    progress_spans.push(Span::styled(sfx, Style::default().fg(c::DIM)));

    let tx = inner.x + gap;
    let tw = inner.width.saturating_sub(gap);
    // Bar sits directly under the album line (no spacer, no status word) — the
    // dancing spectrum below carries the rest of the motion.
    let body = vec![title, album, Line::from(progress_spans)];
    let body_h = body.len() as u16;

    // Text block sits at the top, sharing a baseline with the album art's top.
    let info = Rect { x: tx, y: inner.y + pad_v, width: tw, height: body_h };
    f.render_widget(Paragraph::new(body), info);

    // The spectrum spans the full text width and its bottom edge lines up with
    // the bottom of the album art — never overlapping the text block above it.
    // Music only: a film's audio isn't ours to visualize, so NOW WATCHING drops
    // the EQ entirely and lets the synopsis/trivia carry the card.
    const EQ_H: u16 = 2;
    let art_bottom = inner.y + pad_v + art_rows; // one past the last art row
    let eq_y = art_bottom.saturating_sub(EQ_H).max(info.y + body_h);
    if !tv && tw >= 4 && eq_y + EQ_H <= inner.y + inner.height {
        let eq_area = Rect { x: tx, y: eq_y, width: tw, height: EQ_H };
        let real = real_spectrum(s, tw as usize);
        f.render_widget(Paragraph::new(eq_bars(tw as usize, t, m.playing, real.as_deref())), eq_area);
    }
}

/// Resample the captured audio spectrum (issue #13) to `n` column heights,
/// played back delay-interpolated like the EQ so the bars glide between FFT
/// frames instead of stepping. Returns `None` when the Core Audio tap isn't
/// live or its last frame is stale — in which case the caller falls back to the
/// honest synthetic flourish below. Each captured band is itself Catmull-Rom
/// interpolated through time, then linearly spread/averaged across `n` columns.
fn real_spectrum(s: &AppState, n: usize) -> Option<Vec<f32>> {
    if !s.audio_live || n == 0 {
        return None;
    }
    let samples = &s.audio_samples;
    let last = samples.back()?;
    // Bail to the synthetic dance if capture has gone quiet (paused/stopped) —
    // a flatlined real spectrum would read as a dead card.
    if last.0.elapsed() > Duration::from_millis(700) {
        return None;
    }
    let bands = last.1.len();
    if bands == 0 {
        return None;
    }
    // Time-smoothed band values via the same delayed Catmull-Rom playback the
    // gauges/EQ use: glides each band between FFT frames at frame rate.
    let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let vals: Vec<f32> = (0..bands).map(|b| sampled_channel(samples, b, target)).collect();

    // Spread `bands` source bands across `n` output columns. Box-blur a touch so
    // neighbouring bars share motion (buttery, never jagged).
    let out: Vec<f32> = (0..n)
        .map(|x| {
            let pos = x as f32 / (n.max(2) - 1) as f32 * (bands - 1) as f32;
            let i = pos.floor() as usize;
            let f = pos - i as f32;
            let a = vals[i.min(bands - 1)];
            let b = vals[(i + 1).min(bands - 1)];
            a + (b - a) * f
        })
        .collect();
    Some(smoothed(&out, 1, 2))
}

/// The NOW PLAYING spectrum. When the Core Audio tap is live (issue #13) this is
/// a *real* FFT of what's playing — `bars` carries the measured, glide-smoothed
/// band heights. When it isn't, it falls back to an honest synthetic visualizer:
/// it only dances while music plays and settles flat when paused, never
/// pretending to be measured spectrum. Two cells tall, blue→pink positional
/// gradient; heights move, colours don't strobe.
fn eq_bars(n: usize, t: f64, playing: bool, bars: Option<&[f32]>) -> Vec<Line<'static>> {
    spectrum(n, 2, t, playing, bars)
}

/// `n` bars × `rows` cells tall, filling from the bottom with 8 sub-levels per
/// cell. If `bars` is `Some`, those measured heights (0..1) are drawn directly
/// (real audio); otherwise the synthetic dance fills in — see `eq_bars`.
fn spectrum(n: usize, rows: usize, t: f64, playing: bool, bars: Option<&[f32]>) -> Vec<Line<'static>> {
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let rows = rows.max(1);
    let tf = t as f32;
    let denom = (n as f32 - 1.0).max(1.0);
    let mut grid: Vec<Vec<Span>> = (0..rows).map(|_| Vec::with_capacity(n)).collect();
    for i in 0..n {
        let fi = i as f32;
        let h = if let Some(b) = bars {
            // Real measured band height. A small floor keeps the baseline alive.
            (b.get(i).copied().unwrap_or(0.0)).clamp(0.0, 1.0).max(0.04)
        } else if playing {
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

/// One process row, eased for delayed playback: cpu%/mem glide between samples
/// and `rank` is a *fractional* position so rows slide toward their new slot
/// rather than snapping. Keyed by name across samples.
struct EasedProc {
    name: String,
    cpu: f32,
    mem: f32,
    uptime: u64,
    rank: f32, // fractional rank at the playback instant — sort key
    fade: f32, // 0..1 appearance weight (1 = fully present)
}

/// Build the delay-interpolated proc list at `target` time from `proc_samples`,
/// the same playback the gauges/EQ use. Membership is taken from the destination
/// (newer) bracketing sample so the list never flickers; each process's cpu%/mem
/// ease via Catmull-Rom and its row rank lerps so reorders slide. Processes that
/// just appeared/dropped fade in/out via `fade` instead of popping.
fn eased_procs(samples: &VecDeque<(Instant, Vec<ProcSample>)>, target: Instant) -> Vec<EasedProc> {
    let m = samples.len();
    if m == 0 {
        return Vec::new();
    }
    // Bracketing segment [i, i+1] around target (clamped to the ends).
    let (i, u) = if m == 1 || target <= samples[0].0 {
        (0usize, 0.0f32)
    } else if target >= samples[m - 1].0 {
        (m - 1, 0.0)
    } else {
        let mut idx = m - 2;
        for j in 0..m - 1 {
            if samples[j].0 <= target && target < samples[j + 1].0 {
                idx = j;
                break;
            }
        }
        let span = (samples[idx + 1].0 - samples[idx].0).as_secs_f32().max(1e-3);
        (idx, ((target - samples[idx].0).as_secs_f32() / span).clamp(0.0, 1.0))
    };
    let j = (i + 1).min(m - 1); // destination sample index
    // Rank/value lookups keyed by name. Rank = index in that sample (lower = top).
    let look = |idx: usize, name: &str| -> Option<(usize, &ProcSample)> {
        samples[idx].1.iter().enumerate().find(|(_, p)| p.0 == name)
    };
    // Catmull endpoints for value easing: one sample either side of [i, j].
    let i0 = i.saturating_sub(1);
    let i3 = (j + 1).min(m - 1);
    let dest = &samples[j].1;
    let n = dest.len();
    let mut out = Vec::with_capacity(n);
    for (rj, p) in dest.iter().enumerate() {
        let name = &p.0;
        // Value at each of the four spline points (fall back to nearest known).
        let cpu_at = |idx: usize| look(idx, name).map(|(_, q)| q.1).unwrap_or(p.1);
        let mem_at = |idx: usize| look(idx, name).map(|(_, q)| q.2 as f32).unwrap_or(p.2 as f32);
        let cpu = catmull(cpu_at(i0), cpu_at(i), cpu_at(j), cpu_at(i3), u).max(0.0);
        let mem = catmull(mem_at(i0), mem_at(i), mem_at(j), mem_at(i3), u).max(0.0);
        // Rank lerp: where this process sat in the *from* sample → its slot now.
        // Newcomers start one row below the bottom and slide up; `fade` mirrors it.
        let (rank, fade) = match look(i, name) {
            Some((ri, _)) => (ri as f32 + (rj as f32 - ri as f32) * u, 1.0),
            None => (n as f32 - (n as f32 - rj as f32) * u, u),
        };
        out.push(EasedProc { name: name.clone(), cpu, mem, uptime: p.3, rank, fade });
    }
    // Render in eased-rank order; as a process crosses a half-row its slot swaps,
    // producing the slide. Stable tie-break on name keeps equal ranks from jitter.
    out.sort_by(|a, b| {
        a.rank
            .partial_cmp(&b.rank)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

fn proc_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let block = panel("TOP PROCESSES", false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    // Delay-interpolated playback (now - EQ_DELAY) so cpu%/mem glide and rows
    // slide toward their new rank — identical pattern to the gauges/EQ.
    let target = Instant::now().checked_sub(EQ_DELAY).unwrap_or_else(Instant::now);
    let procs = eased_procs(&s.proc_samples, target);
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
    for p in procs.iter().take(rows) {
        // Fade newcomers/leavers in by dimming toward the background.
        let fade = p.fade.clamp(0.0, 1.0);
        let cpu_col = c::blend(c::BG, c::jazz((p.cpu / 100.0).min(1.0)), fade);
        let mem_col = c::blend(c::BG, c::DIM, fade);
        let up_col = c::blend(c::BG, c::FAINT, fade);
        let name_col = c::blend(c::BG, c::TEXT, fade);
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:>5.1}%", p.cpu),
                Style::default().fg(cpu_col).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {:>6}", fmt_bytes(p.mem as u64)), Style::default().fg(mem_col)),
            Span::styled(format!("   {:>6}", fmt_dur_short(p.uptime)), Style::default().fg(up_col)),
            Span::styled(format!("  {}", truncate(&p.name, namew)), Style::default().fg(name_col)),
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
        Span::styled(format!("{}°F", w.temp_f), Style::default().fg(c::cyan()).add_modifier(Modifier::BOLD)),
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
        Span::styled(format!("↑{}°", w.hi_f), Style::default().fg(c::pink())),
        Span::styled(" ", Style::default().fg(c::FAINT)),
        Span::styled(format!("↓{}°", w.lo_f), Style::default().fg(c::cyan())),
        Span::styled(format!("   hum {}%", w.humidity), Style::default().fg(c::FAINT)),
        Span::styled("   UV ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}", w.uv), Style::default().fg(c::jazz((w.uv as f32 / 11.0).clamp(0.0, 1.0)))),
    ]);

    // row 2 — atmosphere: wind / chance of rain / barometric pressure.
    let atmos = Line::from(vec![
        Span::styled(format!("💨 {} {} mph", w.wind_dir, w.wind_mph), Style::default().fg(c::cyan())),
        Span::styled("   ☔ rain ", Style::default().fg(c::DIM)),
        Span::styled(format!("{}%", w.precip_chance), Style::default().fg(c::jazz((w.precip_chance as f32 / 100.0).clamp(0.0, 1.0)))),
        Span::styled(format!("   {} mb", w.pressure_mb), Style::default().fg(c::DIM)),
    ]);

    let mut lines: Vec<Line> = vec![big, detail, atmos];

    // bottom: hourly temp chart (next ~12h of forecast temps, jazz-colored),
    // grown to fill the card's dead space — full inner width, as many rows tall as
    // the card has left over. A blank spacer row keeps the chart off the pressure
    // line so the graph isn't crammed against the text.
    let spacer = if h > 4 { 1 } else { 0 };
    let chart_h = h.saturating_sub(3 + spacer); // rows left below the info rows (+spacer)
    if chart_h >= 1 && !w.temp_strip.is_empty() && iw > 0 {
        if spacer == 1 {
            lines.push(Line::from(""));
        }
        for row in jazz_spark_rows(&w.temp_strip, iw, chart_h) {
            lines.push(Line::from(row));
        }
    }

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

/// Shared unread/all-read title badge for the iMESSAGE and SIGNAL cards so the
/// glyph, color and spacing stay congruent. `dot` lets the caller animate the
/// unread bullet color (iMESSAGE crossfades pink→violet on settle); Signal passes
/// a flat pink.
fn unread_badge(unread_count: u32, dot: ratatui::style::Color) -> Vec<Span<'static>> {
    if unread_count > 0 {
        vec![
            Span::styled(" ● ", Style::default().fg(dot).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{}", unread_count),
                Style::default().fg(c::pink()).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" unread ", Style::default().fg(c::DIM)),
        ]
    } else {
        vec![
            Span::styled(" ✓ ", Style::default().fg(c::accent()).add_modifier(Modifier::BOLD)),
            Span::styled("all read ", Style::default().fg(c::DIM)),
        ]
    }
}

/// iMESSAGE card: unread badge in the title, a list of recent inbound messages
/// (focus marker · sender · preview · rel-time · unread dot), and an inline reply
/// input that wipes open on a double-press. All motion is interpolated each
/// frame off `s.msg_ui.anim_start` — never a discrete flip.
fn messages_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64, hit: &mut MsgHit) {
    use crate::state::MsgPhase;
    let msgs = &s.messages;
    let ui = &s.msg_ui;

    // ----- title with unread badge / all-read tick -----
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
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    if msgs.available && msgs.fresh {
        let dot = c::blend(c::pink(), c::accent(), badge_blend);
        title_spans.extend(unread_badge(msgs.unread_count, dot));
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
    // Unread → shimmer the border until you've actually checked iMessage (but not
    // while the brief red send-fail flash is playing).
    let flashing = ui.send_failed_at.is_some_and(|fa| fa.elapsed().as_secs_f32() < 0.36);
    if msgs.available && msgs.unread_count > 0 && !flashing {
        shimmer_border(f, area, t, 0.55, 0.0);
    }

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
                    "grant Full Disk Access to overseer",
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

    // Clickable x-range of the card; the per-row y's are recorded as the rows are
    // drawn below, so the threaded composer's row-shift is reflected in the map.
    hit.x0 = inner.x;
    hit.x1 = inner.x.saturating_add(inner.width.saturating_sub(1));

    // Index (within items) of the focused conversation, for the slide marker —
    // now identity-based (focus_chat_id), so it works for read rows too.
    let focus_idx = ui
        .focus_chat_id
        .and_then(|id| msgs.items.iter().position(|m| m.chat_id == id));

    // Reserved column budget: marker(2) + sender(18) + gap(1) + reltime(4) + dot(2).
    let prevw = iw.saturating_sub(2 + 18 + 1 + 4 + 2).max(6);

    // ----- inline reply composer -----
    // Built once here, then threaded directly BENEATH the focused conversation in
    // the row loop, so it reads as a reply nested under that message. It unfurls
    // in (text fades up from BG over the Opening ease) and the rows below it slide
    // down to make room — a smooth "pop out below the message" rather than a fixed
    // box at the bottom of the card.
    let composer_line: Option<Line> = if composer_open {
        // Opening eases 0→1; Closing/Sending eases 1→0.
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
        // Soft blinking caret (~1s sine on alpha), DIM↔TEXT.
        let blink = 0.5 + 0.5 * ((t * std::f64::consts::TAU).sin() as f32);
        let caret_col = c::blend(c::FAINT, c::TEXT, blink);
        // Thread connector descending from the focused row's marker.
        let mut spans = vec![Span::styled(
            "╰▸ ",
            Style::default().fg(c::blend(c::BG, c::pink(), e)).add_modifier(Modifier::BOLD),
        )];
        if ui.phase == MsgPhase::Sending {
            // Shimmer the draft away as a send "whoosh".
            let head = ui.progress(Duration::from_millis(260)).unwrap_or(1.0);
            let chars: Vec<char> = ui.draft.chars().collect();
            let n = chars.len().max(1);
            for (ci, ch) in chars.iter().enumerate() {
                let cp = ci as f32 / n as f32;
                let d = (cp - head).abs();
                let b = (-(d * d) / (2.0 * 0.08 * 0.08)).exp();
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(c::blend(c::TEXT, c::jazz(0.88), b)),
                ));
            }
        } else if ui.draft.is_empty() {
            // Empty → faint prompt that fades in with the wipe.
            spans.push(Span::styled(
                format!("reply to {sender}…"),
                Style::default()
                    .fg(c::blend(c::BG, c::FAINT, e))
                    .add_modifier(Modifier::ITALIC),
            ));
            spans.push(Span::styled("▏", Style::default().fg(caret_col)));
        } else {
            // Live draft, left-truncated so the caret stays visible.
            let budget = iw.saturating_sub(3 + 2).max(4);
            let draft = &ui.draft;
            let shown_draft: String = if draft.chars().count() > budget {
                draft.chars().skip(draft.chars().count() - budget).collect()
            } else {
                draft.clone()
            };
            spans.push(Span::styled(shown_draft, Style::default().fg(c::blend(c::BG, c::TEXT, e))));
            spans.push(Span::styled("▏", Style::default().fg(caret_col)));
        }
        Some(Line::from(spans))
    } else {
        None
    };

    let mut lines: Vec<Line> = Vec::with_capacity(ih);
    let mut drawn = 0usize; // lines emitted so far (drives hit-test y + the insert)
    let mut composer_placed = composer_line.is_none();
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
                c::blend(c::pink(), c::FAINT, adv),
                c::blend(c::pink(), c::FAINT, adv),
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
        let dot = if m.unread {
            Span::styled(" ●", Style::default().fg(dot_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        let mut row_spans: Vec<Span> = vec![marker, Span::styled(sender, sender_style), Span::raw(" ")];
        if m.is_rich && m.unread {
            // A *new* picture/video preview ("[rich message]") gets a gentle
            // travelling sheen so it reads as special, settling to faint as the
            // row is marked read (adv 0→1). Once read it's plain, like before.
            let base = c::blend(c::pink(), c::FAINT, adv);
            row_spans.extend(shimmer_spans(&preview, t, i as f32, base, 0.5, 0.10, false));
        } else if m.is_rich {
            row_spans.push(Span::styled(preview, Style::default().fg(c::FAINT).add_modifier(Modifier::ITALIC)));
        } else {
            row_spans.push(Span::styled(preview, Style::default().fg(prev_col)));
        }
        row_spans.push(Span::styled(format!("{:>4}", truncate(&m.rel, 4)), Style::default().fg(c::FAINT)));
        row_spans.push(dot);
        hit.rows.push((inner.y + drawn as u16, m.chat_id));
        lines.push(Line::from(row_spans));
        drawn += 1;
        // Thread the reply composer directly beneath the focused conversation.
        if Some(i) == focus_idx && !composer_placed {
            if let Some(cl) = composer_line.clone() {
                lines.push(cl);
                drawn += 1;
            }
            composer_placed = true;
        }
    }

    // Fallback: if the focused row wasn't visible, append the composer at the end.
    if !composer_placed {
        if let Some(cl) = composer_line {
            lines.push(cl);
        }
    }

    // ----- separator + keybind hint (only while focused) -----
    if want_footer {
        lines.push(Line::from(Span::styled("─".repeat(iw), Style::default().fg(c::FAINT))));
        lines.push(Line::from(Span::styled(
            "  click: focus · dbl-click: reply · esc: close",
            Style::default().fg(c::FAINT),
        )));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

/// SIGNAL card: identical row style to iMESSAGE (sender · preview · rel-time ·
/// unread dot) and unread-badge title, but read-only — Signal Desktop has no send
/// API, so there's no focus marker, reply composer, or mark-read interaction.
fn signal_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let sig = &s.signal;

    // ----- title with unread badge / all-read tick (mirrors iMESSAGE) -----
    let mut title_spans = vec![Span::styled(
        " SIGNAL ",
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    if sig.available && sig.fresh {
        title_spans.extend(unread_badge(sig.unread_count, c::pink()));
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
    // Unread → shimmer the border until you've checked Signal.
    if sig.available && sig.unread_count > 0 {
        shimmer_border(f, area, t, 0.55, 0.0);
    }

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
    for (i, m) in sig.items.iter().take(shown).enumerate() {
        let (sender_col, prev_col, dot_col) = if m.unread {
            (c::TEXT, c::TEXT, c::pink())
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
        let dot = if m.unread {
            Span::styled(" ●", Style::default().fg(dot_col).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("  ")
        };
        let mut row_spans: Vec<Span> = vec![
            Span::raw("  "), // marker column kept for alignment parity (no focus)
            Span::styled(sender, sender_style),
            Span::raw(" "),
        ];
        if m.is_rich && m.unread {
            // An *unread* photo/attachment ("[photo]") gets the same gentle
            // travelling sheen as a fresh iMESSAGE picture, so it reads as a live,
            // highlighted message instead of being dimmed like a read one.
            row_spans.extend(shimmer_spans(&preview, t, i as f32, c::pink(), 0.5, 0.10, false));
        } else if m.is_rich {
            row_spans.push(Span::styled(preview, Style::default().fg(c::FAINT).add_modifier(Modifier::ITALIC)));
        } else {
            row_spans.push(Span::styled(preview, Style::default().fg(prev_col)));
        }
        row_spans.push(Span::styled(format!("{:>4}", truncate(&m.rel, 4)), Style::default().fg(c::FAINT)));
        row_spans.push(dot);
        lines.push(Line::from(row_spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

const DISCORD_MAX_VOICE: usize = 3;
const DISCORD_MAX_TEXT: usize = 3;

/// Overlay an env-gated fake voice member so a join can be previewed live:
/// `OVERSEER_FAKE_VOICE="200 club:Ghosty"` (or just a bare name → "200 club").
/// Returns the real list untouched when the env var is unset.
fn fake_voice(real: &[crate::state::VoiceChannel]) -> Vec<crate::state::VoiceChannel> {
    use crate::state::VoiceChannel;
    let mut voice = real.to_vec();
    let spec = match std::env::var("OVERSEER_FAKE_VOICE") {
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
fn discord_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let d = &s.discord;
    let voice = fake_voice(&d.voice); // real list, plus any env-gated preview join
    let in_voice: usize = voice.iter().map(|v| v.members.len()).sum();
    let unread_text = d.text.iter().filter(|t| t.unread).count();

    // ----- title with voice / unread badge -----
    let mut title_spans = vec![Span::styled(
        " DISCORD  ",
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    if d.available && d.fresh {
        if in_voice > 0 {
            title_spans.push(Span::styled(
                format!("{in_voice}"),
                Style::default().fg(c::cyan()).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" in voice ", Style::default().fg(c::DIM)));
        } else if unread_text > 0 {
            title_spans.push(Span::styled(" ● ", Style::default().fg(c::pink()).add_modifier(Modifier::BOLD)));
            title_spans.push(Span::styled(
                format!("{unread_text}"),
                Style::default().fg(c::pink()).add_modifier(Modifier::BOLD),
            ));
            title_spans.push(Span::styled(" unread ", Style::default().fg(c::DIM)));
        } else {
            title_spans.push(Span::styled(" ✓ ", Style::default().fg(c::accent()).add_modifier(Modifier::BOLD)));
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
    // Border shimmer, in priority order:
    //  • someone is TALKING in voice  → bright, fast, white-hot sweep
    //  • unread text / 20s after a voice JOIN → calm attention sweep
    // OVERSEER_FAKE_SPEAKING / _VOICE light these up for previewing.
    let voice_join = d.voice_join_at.is_some_and(|i| i.elapsed().as_secs_f64() < 20.0);
    let fake_join = std::env::var("OVERSEER_FAKE_VOICE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let fake_speaking = std::env::var("OVERSEER_FAKE_SPEAKING")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    // Light the border if EITHER detector hears talking: the bot-gateway SSRC events
    // OR the local Core Audio tap (which works past Discord's E2EE). They write
    // separate flags so one can't clobber the other to false.
    let speaking = d.voice_speaking || d.voice_speaking_tap || fake_speaking;
    if d.available && speaking {
        // Pulse the glow a touch so "talking" reads as alive, not just lit.
        let glow = 0.45 + 0.20 * (t * 5.0).sin().abs() as f32;
        shimmer_border(f, area, t, 1.6, glow);
    } else if d.available && (unread_text > 0 || voice_join || fake_join) {
        shimmer_border(f, area, t, 0.55, 0.0);
    }

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
            Span::styled(pad_width(&vc.name, namew), Style::default().fg(c::cyan()).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
            Span::styled(fit_members(&vc.members, memw), Style::default().fg(c::DIM)),
        ]));
    }

    // ----- text channels (iMessage/Signal row style) -----
    let prevw = iw.saturating_sub(2 + 18 + 1 + 4 + 2).max(6);
    for tc in d.text.iter().take(DISCORD_MAX_TEXT) {
        let (name_col, prev_col, dot_col) = if tc.unread {
            (c::TEXT, c::TEXT, c::pink())
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

/// The watching counterpart of the LYRICS card: titled with the film/show name,
/// it shows a synopsis with the director and top-billed cast beneath. No EQ.
fn watch_synopsis_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    let m = &s.music;
    let wi = &s.watch_info;
    let name = if !m.watch.show.is_empty() { &m.watch.show } else { &m.track };

    let title = Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(
            truncate(name, (area.width as usize).saturating_sub(4).max(4)),
            Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", Style::default()),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c::PANEL_BORDER_HOT).add_modifier(Modifier::BOLD))
        .title(title)
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 6 || inner.height < 2 {
        return;
    }

    // Still gathering, or nothing yet for this title → a quiet centered note.
    let ready = wi.track_id == m.track_id()
        && (!wi.synopsis.is_empty()
            || !wi.director.is_empty()
            || !wi.cast.is_empty()
            || !wi.producers.is_empty());
    if !ready {
        let note = if !wi.note.is_empty() { wi.note.as_str() } else { "gathering synopsis…" };
        f.render_widget(
            Paragraph::new(Span::styled(note.to_string(), Style::default().fg(c::DIM)))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }

    let width = inner.width as usize;
    let h = inner.height as usize;

    // A slow, uniform sheen so the block reads as calm, tinted text (matches FACTS).
    let glow = 0.5 + 0.5 * (t * 0.6).sin() as f32;
    let col = c::blend(c::blend(c::TEXT, c::accent(), 0.16), c::blend(c::TEXT, c::cyan(), 0.35), 0.14 * glow);

    // A labelled credit, wrapped with a hanging indent so a long bill doesn't run
    // off the right edge ("…, Natalie Portman, / Jake Lloyd"). All labels are the
    // same width so the values line up.
    let credit = |label: &'static str, value: String, vstyle: Style| -> Vec<Line<'static>> {
        let pad = label.chars().count();
        let indent = " ".repeat(pad);
        wrap_text(&value, width.saturating_sub(pad).max(4))
            .into_iter()
            .enumerate()
            .map(|(i, seg)| {
                let lead = if i == 0 {
                    Span::styled(label, Style::default().fg(c::DIM))
                } else {
                    Span::raw(indent.clone())
                };
                Line::from(vec![lead, Span::styled(seg, vstyle)])
            })
            .collect()
    };

    // Two screens that cross-dissolve: the synopsis (sized to fit), then the full
    // credits — director, the principal cast, and producers — filling the box.
    let mut pages: Vec<Vec<Line>> = Vec::new();

    // Page 1 — synopsis, trimmed to the card height with an ellipsis if long.
    if !wi.synopsis.is_empty() {
        let mut syn: Vec<Line> = wrap_text(&wi.synopsis, width)
            .into_iter()
            .map(|seg| Line::from(Span::styled(seg, Style::default().fg(col))))
            .collect();
        if syn.len() > h {
            syn.truncate(h.max(1));
            if let Some(last) = syn.last_mut() {
                last.spans.push(Span::styled(" …", Style::default().fg(c::DIM)));
            }
        }
        pages.push(syn);
    }

    // Page 2 — credits.
    let mut credits: Vec<Line> = Vec::new();
    if !wi.director.is_empty() {
        credits.extend(credit(
            "Director   ",
            wi.director.clone(),
            Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
        ));
    }
    if !wi.cast.is_empty() {
        credits.extend(credit("Starring   ", wi.cast.join(", "), Style::default().fg(c::TEXT)));
    }
    if !wi.producers.is_empty() {
        credits.extend(credit("Producers  ", wi.producers.join(", "), Style::default().fg(c::TEXT)));
    }
    if !credits.is_empty() {
        credits.truncate(h.max(1));
        pages.push(credits);
    }

    if pages.is_empty() {
        return;
    }

    // One screen → static. Two → a calm cross-dissolve flip every few seconds, so
    // the synopsis reads, then the credits, then back. No cell-stepped scroll.
    let (pi, alpha) = dissolve_phase(pages.len(), t, 8.0, 0.9);
    let page = pages.into_iter().nth(pi).unwrap_or_default();
    f.render_widget(Paragraph::new(faded_lines(page, alpha)), inner);
}

fn lyrics_panel(f: &mut Frame, area: Rect, s: &AppState, t: f64) {
    // Watching a movie/show → this card becomes the synopsis + credits, titled
    // with the film's name (no lyrics, no EQ).
    if s.music.is_tv() {
        watch_synopsis_panel(f, area, s, t);
        return;
    }
    let synced = s.lyrics.synced;
    // Plain LYRICS title — the un-resolved-count badge was removed at the user's
    // request; the reconcile pass still chases misses in the background silently.
    let title_spans = vec![Span::styled(
        " LYRICS  ",
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(c::PANEL_BORDER_HOT).add_modifier(Modifier::BOLD))
        .title(Line::from(title_spans))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(c::BG));
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
        let caption_y = inner.y + (upper / 2) as u16;
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
            let real = real_spectrum(s, inner.width as usize);
            f.render_widget(
                Paragraph::new(spectrum(inner.width as usize, viz_rows, t, m.playing, real.as_deref())),
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
    let width = width.max(4);
    let n_lines = ly.lines.len();
    if n_lines == 0 {
        return vec![Line::from(""); height];
    }

    // One *display row* of the band. A single lyric may wrap into several of
    // these; `cstart`/`total` carry the lyric's running character offset so the
    // karaoke wipe + gradient stay continuous as they flow across wrapped rows.
    struct DRow {
        lyric: usize,
        text: String,
        cstart: usize,
        total: usize,
        active: bool,
    }

    // Flatten the lyrics around `active` into wrapped display rows. The range is
    // padded enough on both sides to over-fill the band whatever the wrapping.
    let lo = active.saturating_sub(height + center);
    let hi = (active + height + 1).min(n_lines);
    let mut rows: Vec<DRow> = Vec::new();
    let mut active_start = 0usize;
    let mut active_h = 1usize;
    for i in lo..hi {
        let is_active = i == active;
        if is_active {
            active_start = rows.len();
        }
        let txt = &ly.lines[i].text;
        if txt.is_empty() {
            if is_active {
                active_h = 1;
            }
            rows.push(DRow { lyric: i, text: "♪".into(), cstart: 0, total: 1, active: is_active });
            continue;
        }
        let wrapped = wrap_text(txt, width);
        let total: usize = wrapped.iter().map(|r| r.chars().count()).sum::<usize>().max(1);
        if is_active {
            active_h = wrapped.len().max(1);
        }
        let mut cstart = 0usize;
        for r in wrapped {
            let rc = r.chars().count();
            rows.push(DRow { lyric: i, text: r, cstart, total, active: is_active });
            cstart += rc;
        }
    }

    // Anchor the active lyric's first row at `center`, then nudge so the whole
    // wrapped active block stays on-screen (prefer showing all of it — never
    // clip the line you're singing). If the active lyric alone is taller than
    // the band, top-align it so the wipe leads from the first row.
    let mut ds = active_start as isize - center as isize;
    if active_h <= height {
        let bottom = (active_start + active_h) as isize;
        if bottom > ds + height as isize {
            ds = bottom - height as isize;
        }
        if (active_start as isize) < ds {
            ds = active_start as isize;
        }
    } else {
        ds = active_start as isize;
    }

    let mut out = Vec::with_capacity(height);
    for r in 0..height as isize {
        let di = ds + r;
        if di < 0 || di as usize >= rows.len() {
            out.push(Line::from(""));
            continue;
        }
        let dr = &rows[di as usize];
        if dr.text == "♪" {
            out.push(Line::from(Span::styled("♪", Style::default().fg(c::FAINT))));
        } else if dr.active {
            if let Some(frac) = wipe_frac {
                out.push(karaoke_row(&dr.text, dr.cstart, dr.total, frac));
            } else {
                out.push(Line::from(Span::styled(
                    dr.text.clone(),
                    Style::default().fg(c::pink()).add_modifier(Modifier::BOLD),
                )));
            }
        } else {
            let dist = (dr.lyric as isize - active as isize).unsigned_abs();
            let col = if dist == 1 { c::DIM } else { c::FAINT };
            out.push(Line::from(Span::styled(dr.text.clone(), Style::default().fg(col))));
        }
    }
    out
}

/// One wrapped row of the active karaoke line. Characters lit up to `frac`
/// (measured across the *whole* lyric, not just this row) get the cyan→violet→
/// pink gradient; the rest stay a calm "pending" tone. `cstart` is this row's
/// character offset within the lyric and `total` its full length, so the wipe
/// and the gradient flow continuously from one wrapped row into the next. The
/// single boundary glyph is *blended* by the sub-character fraction, so the
/// wipe glides instead of popping per glyph — combined with the slewed playback
/// clock, this is the buttery part.
fn karaoke_row(text: &str, cstart: usize, total: usize, frac: f32) -> Line<'static> {
    let chars: Vec<char> = text.chars().collect();
    let total = total.max(1);
    let litf = (frac * total as f32).clamp(0.0, total as f32);
    let full = litf.floor() as usize;
    let partial = litf.fract();
    // The not-yet-wiped tail of the ACTIVE line glows brighter than the DIM
    // context lines (a calm cyan-violet), so the active lyric always reads as
    // "lit/current" even before the wipe reaches a given character.
    let pending = c::blend(c::DIM, c::accent(), 0.45);
    let mut spans = Vec::with_capacity(chars.len());
    for (i, ch) in chars.iter().enumerate() {
        let gi = cstart + i; // global char index within the whole lyric
        let p = gi as f32 / total as f32;
        let style = if gi < full {
            Style::default().fg(c::wipe(p)).add_modifier(Modifier::BOLD)
        } else if gi == full {
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

/// Multi-row jazz sparkline whose bars are coloured by their own height with the
/// jazz ramp (blue→violet→pink→white) and stand `rows` cells tall, so the bar for
/// each sample rises across several text rows for a bigger, easier-to-read chart.
/// Honest: height = real data; colour is purely a function of that height.
/// Returns one span-row per chart row, top-to-bottom, each `width` wide.
/// Each data point is stretched ~2 columns wide so the chart spans the full card.
fn jazz_spark_rows(data: &[u64], width: usize, rows: usize) -> Vec<Vec<Span<'static>>> {
    let glyphs = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let rows = rows.max(1);
    if data.is_empty() || width == 0 {
        return vec![vec![Span::raw(" ".repeat(width))]; rows];
    }
    // Stretch the samples so the chart fills the full width (≥1 column each).
    let mut cols: Vec<u64> = Vec::with_capacity(width);
    for i in 0..width {
        let idx = i * data.len() / width;
        cols.push(data[idx.min(data.len() - 1)]);
    }
    let max = cols.iter().copied().max().unwrap_or(1).max(1);
    let steps = rows * 8; // total sub-cell resolution across the column
    let mut out: Vec<Vec<Span>> = Vec::with_capacity(rows);
    for r in 0..rows {
        // Row 0 is the top of the chart; the bottom row is the base of the bars.
        let from_bottom = rows - 1 - r;
        let mut spans: Vec<Span> = Vec::with_capacity(width);
        for &v in &cols {
            let frac = (v as f32 / max as f32).clamp(0.0, 1.0);
            let filled = (frac * steps as f32).round() as usize;
            let cell = filled.saturating_sub(from_bottom * 8).min(8);
            let ch = glyphs[cell];
            if ch == ' ' {
                spans.push(Span::raw(" "));
            } else {
                spans.push(Span::styled(ch.to_string(), Style::default().fg(c::jazz(frac))));
            }
        }
        out.push(spans);
    }
    out
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

/// One renderable row of the KEYBINDS card.
#[derive(Clone)]
enum KbItem {
    Header(String),
    Bind(String, String),
    Blank,
}

const KB_KEYW: usize = 7;
const KB_COLGAP: usize = 2;
const KB_COL_MIN: usize = 30;

/// How many balanced columns fit in an inner width of `iw` cells.
fn keybinds_ncols(iw: usize) -> usize {
    ((iw + KB_COLGAP) / (KB_COL_MIN + KB_COLGAP)).clamp(1, 3)
}

/// Pack the keybind groups into `ncols` balanced columns. Each group is an
/// indivisible block (header + its rows) so a binding is never split from its
/// heading; columns aim for ~total/ncols rows, last column takes the rest.
fn keybinds_columns(kb: &crate::state::Keybinds, ncols: usize) -> Vec<Vec<KbItem>> {
    let blocks: Vec<Vec<KbItem>> = kb
        .groups
        .iter()
        .map(|g| {
            let mut blk = vec![KbItem::Header(g.name.clone())];
            for (keys, desc) in &g.binds {
                blk.push(KbItem::Bind(keys.clone(), desc.clone()));
            }
            blk
        })
        .collect();
    let total: usize = blocks.iter().map(|b| b.len()).sum();
    let target = total.div_ceil(ncols.max(1));
    let mut cols: Vec<Vec<KbItem>> = vec![Vec::new()];
    let mut ch = 0usize; // current column height
    for blk in blocks {
        let h = blk.len();
        let at_last = cols.len() == ncols;
        if !cols.last().unwrap().is_empty() && !at_last && ch + 1 + h > target {
            cols.push(Vec::new());
            ch = 0;
        }
        let col = cols.last_mut().unwrap();
        if !col.is_empty() {
            col.push(KbItem::Blank); // blank line between stacked groups
            ch += 1;
        }
        col.extend(blk);
        ch += h;
    }
    cols
}

/// Content height the KEYBINDS card wants for a card of outer width `outer_w`:
/// borders + the tallest balanced column, so the box flexes as binds come/go.
fn keybinds_height(kb: &crate::state::Keybinds, outer_w: u16) -> u16 {
    if !kb.available {
        return 4; // gate hint
    }
    if kb.groups.is_empty() {
        return 3;
    }
    let iw = (outer_w as usize).saturating_sub(4); // 2 borders + 2 padding
    let ncols = keybinds_ncols(iw);
    let tallest = keybinds_columns(kb, ncols).iter().map(|c| c.len()).max().unwrap_or(1);
    (tallest + 2).min(u16::MAX as usize) as u16
}

/// KEYBINDS card: a live mirror of the Hammerspoon cheat sheet (exported to JSON
/// on every reload). Groups + rows flow into balanced columns, and the box sizes
/// to its content so it expands/shrinks as bindings come and go.
fn keybinds_panel(f: &mut Frame, area: Rect, s: &AppState) {
    let kb = &s.keybinds;

    let mut title_spans = vec![Span::styled(
        " KEYBINDS ",
        Style::default().fg(c::accent()).add_modifier(Modifier::BOLD),
    )];
    if kb.available && !kb.hyper.is_empty() {
        title_spans.push(Span::styled(
            format!("· Hyper = {} ", kb.hyper),
            Style::default().fg(c::DIM),
        ));
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
    if inner.height < 2 || inner.width < 12 {
        return;
    }

    if !kb.available {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("⌘ waiting for Hammerspoon…", Style::default().fg(c::DIM))),
                Line::from(Span::styled(
                    "reload Hammerspoon (Ctrl+Alt+R) to populate",
                    Style::default().fg(c::FAINT),
                )),
            ]),
            inner,
        );
        return;
    }

    // Same column geometry the height calc uses, so the box hugs the content.
    let iw = inner.width as usize;
    let ncols = keybinds_ncols(iw);
    let colw = (iw - KB_COLGAP * (ncols - 1)) / ncols;
    let cols = keybinds_columns(kb, ncols);

    for (ci, col) in cols.iter().enumerate() {
        let cx = inner.x + (ci * (colw + KB_COLGAP)) as u16;
        let mut lines: Vec<Line> = Vec::with_capacity(col.len());
        for it in col {
            match it {
                KbItem::Blank => lines.push(Line::from("")),
                KbItem::Header(name) => lines.push(Line::from(Span::styled(
                    fit_width(name, colw),
                    Style::default().fg(c::pink()).add_modifier(Modifier::BOLD),
                ))),
                KbItem::Bind(keys, desc) => {
                    let dw = colw.saturating_sub(KB_KEYW + 1);
                    lines.push(Line::from(vec![
                        Span::styled(pad_width(keys, KB_KEYW), Style::default().fg(c::cyan()).add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::styled(fit_width(desc, dw), Style::default().fg(c::TEXT)),
                    ]));
                }
            }
        }
        f.render_widget(
            Paragraph::new(lines),
            Rect { x: cx, y: inner.y, width: colw as u16, height: inner.height },
        );
    }
}
