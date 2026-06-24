//! Real system-audio spectrum capture (issue #13).
//!
//! macOS 14.4+ exposes a **Core Audio process tap**: a private, virtual-device-
//! free tap of the whole output mix. We build a global stereo tap
//! (`CATapDescription` / `AudioHardwareCreateProcessTap`), wrap it in a private
//! aggregate device, install an IO proc, and in that realtime callback run a
//! windowed FFT (`rustfft`) over what's actually playing. The magnitudes are
//! folded into log-spaced bands and pushed — timestamped — into `AppState` so
//! the NOW PLAYING / LYRICS spectrum plays them back the same buttery, delay-
//! interpolated way the EQ gauges glide. No fabricated motion: when sound is
//! flowing, the bars *are* the sound.
//!
//! Everything here is macOS-only and lives behind `#[cfg(target_os = "macos")]`
//! at the call site; the capture path degrades to a no-op (leaving the honest
//! synthetic visualizer) if the tap can't be created in this environment.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use objc2_core_audio::{
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceIsStackedKey,
    kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey,
    kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, kAudioProcessPropertyBundleID,
    kAudioProcessPropertyIsRunning, kAudioProcessPropertyIsRunningInput,
    kAudioProcessPropertyIsRunningOutput, kAudioProcessPropertyPID, kAudioSubTapDriftCompensationKey,
    kAudioSubTapUIDKey, kAudioTapPropertyUID, AudioDeviceCreateIOProcID, AudioDeviceIOProcID,
    AudioDeviceStart, AudioDeviceStop, AudioDeviceDestroyIOProcID, AudioHardwareCreateAggregateDevice,
    AudioHardwareCreateProcessTap, AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, CATapDescription,
};
use objc2_core_audio_types::{AudioBufferList, AudioTimeStamp};
use objc2_core_foundation::{
    kCFBooleanTrue, CFArray, CFDictionary, CFRetained, CFString, CFType,
};
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSNumber};

use crate::state::AppState;

type Shared = Arc<Mutex<AppState>>;

/// FFT window length (power of two). At ~48 kHz this is a ~21 ms window — short
/// enough to feel instant, long enough for usable bass resolution.
const FFT_SIZE: usize = 1024;
/// How many log-spaced spectrum bands we render. Matches a tasteful, readable
/// bar count for the NOW PLAYING card.
const BANDS: usize = 28;

/// Shared realtime accumulator: the IO proc fills `window`, and every time it
/// has `FFT_SIZE` fresh mono samples it runs an FFT and publishes bands.
struct TapCtx {
    shared: Shared,
    fft: Arc<dyn rustfft::Fft<f32>>,
    hann: Vec<f32>,
    /// Triangular log-spaced band edges (bin indices) so each band sums a slice
    /// of the magnitude spectrum.
    edges: Vec<usize>,
    /// Rolling mono window we keep refilling; FFT fires once it's full.
    window: Mutex<Vec<f32>>,
    /// Smoothed bands carried between frames so the published series rises fast
    /// but falls gently — classic VU ballistics, no strobing.
    decay: Mutex<Vec<f32>>,
    /// Pre-allocated realtime scratch reused every `publish()` so the audio
    /// callback never allocates: the FFT in/out buffer, the rustfft internal
    /// scratch, and the per-frame bands. Behind one Mutex (only `publish` —
    /// itself on the IO proc — touches it) so the lock is uncontended.
    scratch: Mutex<PublishScratch>,
}

/// Reusable buffers for `publish()`, allocated once at tap setup so the
/// realtime IO callback stays allocation-free.
struct PublishScratch {
    fft_buf: Vec<rustfft::num_complex::Complex<f32>>,
    fft_scratch: Vec<rustfft::num_complex::Complex<f32>>,
    bands: Vec<f32>,
}

/// Live capture handle. Dropping it stops the IO proc and tears the tap +
/// aggregate device back down so we never leak HAL objects.
pub struct AudioCapture {
    agg_id: AudioObjectID,
    tap_id: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    // Kept alive for the lifetime of the IO proc; the proc holds a raw pointer
    // into this same allocation.
    _ctx: Arc<TapCtx>,
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        unsafe {
            if self.proc_id.is_some() {
                AudioDeviceStop(self.agg_id, self.proc_id);
                AudioDeviceDestroyIOProcID(self.agg_id, self.proc_id);
            }
            if self.agg_id != 0 {
                AudioHardwareDestroyAggregateDevice(self.agg_id);
            }
            if self.tap_id != 0 {
                AudioHardwareDestroyProcessTap(self.tap_id);
            }
        }
    }
}

fn prop_addr(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// Read a CFString property off an audio object (used for the tap's UID).
unsafe fn read_cfstring(obj: AudioObjectID, selector: u32) -> Option<CFRetained<CFString>> {
    let addr = prop_addr(selector);
    let mut out: *const CFString = std::ptr::null();
    let mut size = std::mem::size_of::<*const CFString>() as u32;
    let st = AudioObjectGetPropertyData(
        obj,
        NonNull::from(&addr),
        0,
        std::ptr::null(),
        NonNull::from(&mut size),
        NonNull::new(&mut out as *mut _ as *mut c_void)?,
    );
    if st != 0 || out.is_null() {
        return None;
    }
    // The Get returns a +1 retained CFStringRef; adopt it.
    Some(CFRetained::from_raw(NonNull::new_unchecked(out as *mut CFString)))
}

/// Build the precomputed FFT plan, Hann window, and log-spaced band edges.
fn make_ctx(shared: Shared) -> TapCtx {
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Hann window kills spectral leakage so tones land in clean bands.
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|n| {
            let x = std::f32::consts::PI * n as f32 / (FFT_SIZE as f32 - 1.0);
            x.sin() * x.sin()
        })
        .collect();

    // Log-spaced band edges across the audible-ish range (skip DC). We map band
    // boundaries geometrically from bin `lo` to `hi` so bass gets its own bars
    // and treble doesn't smear into one — the way a real EQ reads.
    let lo = 2usize; // ~94 Hz at 48k/1024
    let hi = FFT_SIZE / 2; // Nyquist bin
    let mut edges = Vec::with_capacity(BANDS + 1);
    for b in 0..=BANDS {
        let frac = b as f32 / BANDS as f32;
        let bin = (lo as f32) * ((hi as f32 / lo as f32).powf(frac));
        edges.push((bin.round() as usize).clamp(lo, hi));
    }

    let scratch = PublishScratch {
        fft_buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); FFT_SIZE],
        fft_scratch: vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()],
        bands: vec![0.0; BANDS],
    };

    TapCtx {
        shared,
        fft,
        hann,
        edges,
        window: Mutex::new(Vec::with_capacity(FFT_SIZE)),
        decay: Mutex::new(vec![0.0; BANDS]),
        scratch: Mutex::new(scratch),
    }
}

/// The realtime IO proc. `in_input_data` carries the tapped output as one or
/// more float buffers; we fold it to mono, refill the FFT window, and when full
/// publish a fresh band frame. Allocation-light and lock-brief so we never
/// glitch the realtime thread.
unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    in_input_data: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _out: NonNull<AudioBufferList>,
    _out_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    if client.is_null() {
        return 0;
    }
    let ctx = &*(client as *const TapCtx);
    let list = in_input_data.as_ref();
    let nbuf = list.mNumberBuffers as usize;
    if nbuf == 0 {
        return 0;
    }
    // mBuffers is a flexible array; walk it from the head pointer.
    let buffers = std::slice::from_raw_parts(list.mBuffers.as_ptr(), nbuf);

    // Fold all channels of the first buffer to mono. The tap is float32; each
    // buffer's mData holds `mDataByteSize / 4` floats (interleaved channels).
    let buf = &buffers[0];
    if buf.mData.is_null() {
        return 0;
    }
    let n_floats = (buf.mDataByteSize as usize) / std::mem::size_of::<f32>();
    if n_floats == 0 {
        return 0;
    }
    let ch = (buf.mNumberChannels as usize).max(1);
    let samples = std::slice::from_raw_parts(buf.mData as *const f32, n_floats);

    let mut win = ctx.window.lock().unwrap();
    let frames = n_floats / ch;
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += samples[f * ch + c];
        }
        win.push(acc / ch as f32);
        if win.len() >= FFT_SIZE {
            // Window is full — run the FFT and publish, then drop the buffer
            // (no overlap; the next callback refills it). Cheap and steady.
            publish(ctx, &win);
            win.clear();
        }
    }
    0
}

/// Hann-window the mono frame, FFT it, fold magnitudes into log-spaced bands,
/// apply rise-fast/fall-soft ballistics, normalize, and push a timestamped
/// frame into `AppState::audio_samples`.
fn publish(ctx: &TapCtx, win: &[f32]) {
    use rustfft::num_complex::Complex;
    // Reuse the pre-allocated scratch so this realtime callback never allocates.
    let mut scratch = ctx.scratch.lock().unwrap();
    let PublishScratch { fft_buf, fft_scratch, bands } = &mut *scratch;

    for i in 0..FFT_SIZE {
        fft_buf[i] = Complex::new(win[i] * ctx.hann[i], 0.0);
    }
    ctx.fft.process_with_scratch(fft_buf, fft_scratch);

    // Per-band energy = mean magnitude across its bin slice, then a perceptual
    // compress (sqrt) so quiet detail is visible without the loud parts pinning.
    for (b, band) in bands.iter_mut().enumerate().take(BANDS) {
        let lo = ctx.edges[b];
        let hi = ctx.edges[b + 1].max(lo + 1);
        let mut sum = 0.0f32;
        for bin in &fft_buf[lo..hi] {
            sum += bin.norm();
        }
        let mag = sum / (hi - lo) as f32;
        // A gentle tilt lifts treble bands (which carry less energy) so the top
        // bars don't sit dead — keeps the spectrum lively across its width.
        let tilt = 1.0 + 1.6 * (b as f32 / BANDS as f32);
        *band = (mag * tilt).sqrt();
    }

    // Normalize each frame to max(this frame's peak, a fixed floor of 0.35) so
    // the bars use the full height but loud passages don't clip flat. There's no
    // running ceiling here — temporal smoothing is the VU decay pass below.
    let peak = bands.iter().cloned().fold(0.0f32, f32::max);
    let norm = 1.0 / peak.max(0.35);
    for v in bands.iter_mut() {
        *v = (*v * norm).clamp(0.0, 1.0);
    }

    // Rise instantly, fall gently — VU ballistics so the spectrum feels alive
    // and never strobes between FFT frames (the delay-interpolated playback in
    // ui.rs then glides it the rest of the way).
    let mut decay = ctx.decay.lock().unwrap();
    for b in 0..BANDS {
        let prev = decay[b];
        decay[b] = if bands[b] > prev { bands[b] } else { prev * 0.82 + bands[b] * 0.18 };
        bands[b] = decay[b];
    }
    drop(decay);

    // Snapshot the smoothed bands into the shared ring (the scratch `bands`
    // buffer itself is retained for the next frame — only this small copy is
    // handed off, and the FFT in/out and rustfft scratch are never re-allocated).
    let frame = bands.clone();
    drop(scratch);

    let now = Instant::now();
    if let Ok(mut s) = ctx.shared.lock() {
        s.audio_live = true;
        s.audio_samples.push_back((now, frame));
        // Keep a short ring — same depth the EQ keeps for delayed playback.
        while s.audio_samples.len() > 16 {
            s.audio_samples.pop_front();
        }
    }
}

/// Try to stand up the process tap + aggregate device + IO proc. Returns the
/// live handle on success, or `None` if any HAL call fails (e.g. the OS is
/// pre-14.4, or a tap couldn't be created in this environment) — in which case
/// the caller keeps the honest synthetic visualizer.
pub fn start(shared: Shared) -> Option<AudioCapture> {
    unsafe {
        // 1. A private, unmuted, global stereo tap of the whole output mix —
        //    excluding no processes, so it hears everything that's playing.
        let empty: objc2::rc::Retained<NSArray<NSNumber>> = NSArray::new();
        let desc = CATapDescription::initStereoGlobalTapButExcludeProcesses(
            CATapDescription::alloc(),
            &empty,
        );
        desc.setName(&objc2_foundation::NSString::from_str("overseer-eq"));
        desc.setPrivate(true);
        desc.setMuteBehavior(objc2_core_audio::CATapMuteBehavior(0)); // CATapUnmuted

        let mut tap_id: AudioObjectID = 0;
        let st = AudioHardwareCreateProcessTap(Some(&desc), &mut tap_id as *mut AudioObjectID);
        if st != 0 || tap_id == 0 {
            return None;
        }

        // Guard so an early `return None` below still tears the tap down.
        let mut guard = AudioCapture { agg_id: 0, tap_id, proc_id: None, _ctx: Arc::new(make_ctx(shared.clone())) };

        // 2. The tap's UID — the aggregate device references the tap by UID.
        let tap_uid = read_cfstring(tap_id, kAudioTapPropertyUID)?;

        // 3. Build a private aggregate device whose sub-tap is our tap. Keys are
        //    C-string constants from the HAL; values a heterogeneous CF mix, so
        //    we assemble the dict from CFType pointers.
        let sub_tap: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioSubTapUIDKey), tap_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioSubTapDriftCompensationKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
        ])?;
        let tap_list: CFRetained<CFArray> = cf_array(&[sub_tap.as_ref() as &CFType as *const CFType])?;
        let agg_uid = CFString::from_str("com.overseer.eq.aggregate");
        let agg_name = CFString::from_str("overseer EQ");
        let agg_desc: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioAggregateDeviceUIDKey), agg_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceNameKey), agg_name.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceIsPrivateKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceIsStackedKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceTapListKey), tap_list.as_ref() as &CFType as *const CFType),
        ])?;

        let mut agg_id: AudioObjectID = 0;
        let st = AudioHardwareCreateAggregateDevice(&agg_desc, NonNull::from(&mut agg_id));
        if st != 0 || agg_id == 0 {
            return None;
        }
        guard.agg_id = agg_id;

        // 4. Install the IO proc, handing it a raw pointer into the Arc'd ctx.
        //    The Arc lives in `guard._ctx`, which outlives the running proc.
        let ctx_ptr = Arc::as_ptr(&guard._ctx) as *mut c_void;
        let mut proc_id: AudioDeviceIOProcID = None;
        let st = AudioDeviceCreateIOProcID(
            agg_id,
            Some(io_proc),
            ctx_ptr,
            NonNull::from(&mut proc_id),
        );
        if st != 0 || proc_id.is_none() {
            return None;
        }
        guard.proc_id = proc_id;

        if AudioDeviceStart(agg_id, proc_id) != 0 {
            return None;
        }

        Some(guard)
    }
}

/// Assemble an immutable CFArray of CF pointers using the default CFType
/// callbacks (so entries are retained/released for us). `None` if the HAL
/// refuses to allocate it — the caller degrades to the synthetic visualizer
/// rather than panicking (panic="abort" would take the whole TUI down).
fn cf_array(items: &[*const CFType]) -> Option<CFRetained<CFArray>> {
    use objc2_core_foundation::kCFTypeArrayCallBacks;
    let mut vals: Vec<*const c_void> = items.iter().map(|v| *v as *const c_void).collect();
    unsafe {
        CFArray::new(None, vals.as_mut_ptr(), items.len() as isize, &raw const kCFTypeArrayCallBacks)
    }
}

/// `--diag-audio`: stand up the tap, watch for a few band frames, and report
/// whether real capture is working — so the user can tell at a glance if the
/// EQ is measuring sound or falling back to the synthetic flourish.
pub fn diag() {
    println!("overseer --diag-audio\n");
    let shared: Shared = Arc::new(Mutex::new(AppState::default()));
    print!("creating Core Audio process tap… ");
    let cap = start(shared.clone());
    match cap {
        Some(_c) => {
            println!("ok");
            // Let the IO proc run; play audio now to see bands move.
            for i in 0..10 {
                std::thread::sleep(std::time::Duration::from_millis(300));
                let s = shared.lock().unwrap();
                let n = s.audio_samples.len();
                let peak = s
                    .audio_samples
                    .back()
                    .map(|(_, b)| b.iter().cloned().fold(0.0f32, f32::max))
                    .unwrap_or(0.0);
                println!("  t+{:>4}ms  frames={n:>2}  live={}  peak={peak:.3}", (i + 1) * 300, s.audio_live);
            }
            println!("\n→ Tap is capturing. If peak stays ~0, nothing is playing through the");
            println!("  default output (or audio is routed to a device the tap can't reach).");
        }
        None => {
            println!("FAILED");
            println!("\n→ Could not create the process tap / aggregate device. On macOS 14.4+");
            println!("  this usually means the app lacks Audio Capture (NSAudioCaptureUsage)");
            println!("  permission: System Settings → Privacy & Security → check the prompt");
            println!("  for this terminal, or run from an app bundle with the audio-input");
            println!("  entitlement. The EQ falls back to the synthetic visualizer meanwhile.");
        }
    }
}

/// CFString from a HAL C-string key constant.
fn cfstr(key: &std::ffi::CStr) -> CFRetained<CFString> {
    CFString::from_str(&key.to_string_lossy())
}

/// Assemble an immutable CFDictionary from (key, value) CF pointers using the
/// default CFType callbacks (retain/release the entries for us). `None` if the
/// HAL refuses to allocate it — the caller degrades to the synthetic visualizer
/// rather than panicking (panic="abort" would take the whole TUI down).
fn cf_dict(pairs: &[(CFRetained<CFString>, *const CFType)]) -> Option<CFRetained<CFDictionary>> {
    use objc2_core_foundation::{kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks};
    let mut keys: Vec<*const c_void> = pairs.iter().map(|(k, _)| k.as_ref() as *const CFString as *const c_void).collect();
    let mut vals: Vec<*const c_void> = pairs.iter().map(|(_, v)| *v as *const c_void).collect();
    unsafe {
        CFDictionary::new(
            None,
            keys.as_mut_ptr(),
            vals.as_mut_ptr(),
            pairs.len() as isize,
            &raw const kCFTypeDictionaryKeyCallBacks,
            &raw const kCFTypeDictionaryValueCallBacks,
        )
    }
}

// ===========================================================================
// Discord "who's talking" via a per-process audio tap.
//
// Discord's voice gateway now mandates DAVE/E2EE, so a bot can't read Speaking
// events anymore (close 4017). Instead we tap Discord.app's OWN audio output
// locally — the same Core Audio process-tap tech as the EQ, but scoped to just
// Discord's processes — and flip `discord.voice_speaking` whenever that audio is
// active, i.e. someone in your call is talking. No bot, no API, no E2EE. (This
// sees audio coming OUT of Discord; your own mic going INTO Discord isn't
// tappable per-process, so it reflects the people you're listening to.)
// ===========================================================================

/// RMS above this (≈ -52 dB) counts as "audio is flowing" → talking.
const VOICE_THRESH: f32 = 0.0025;
/// Hold the shimmer this long after the audio falls quiet, so the gaps between
/// words/sentences don't strobe the border off and on. Long enough to bridge a
/// breath mid-sentence; short enough that the border settles soon after the call
/// goes quiet.
const VOICE_HANG_MS: u64 = 800;

/// Realtime state for the Discord tap: a smoothed level + the current speaking
/// latch + when it was last loud (for the hang-off).
struct VoiceCtx {
    shared: Shared,
    level: Mutex<f32>,
    speaking: Mutex<bool>,
    loud_at: Mutex<Option<Instant>>,
    frames: std::sync::atomic::AtomicU64, // IO callbacks seen (diagnostic)
}

/// Live Discord-voice tap handle. Dropping it stops the IO proc and tears the
/// tap + aggregate device back down (same lifecycle as `AudioCapture`).
pub struct VoiceCapture {
    agg_id: AudioObjectID,
    tap_id: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    _ctx: Arc<VoiceCtx>,
}

impl VoiceCapture {
    /// Current smoothed RMS level the tap is seeing (for diagnostics).
    pub fn level(&self) -> f32 {
        *self._ctx.level.lock().unwrap()
    }
    /// Total IO callbacks seen so far (diagnostic: 0 ⇒ the tap is delivering nothing).
    #[allow(dead_code)] // diagnostic accessor, kept for debugging the audio tap
    pub fn frames(&self) -> u64 {
        self._ctx.frames.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for VoiceCapture {
    fn drop(&mut self) {
        unsafe {
            if self.proc_id.is_some() {
                AudioDeviceStop(self.agg_id, self.proc_id);
                AudioDeviceDestroyIOProcID(self.agg_id, self.proc_id);
            }
            if self.agg_id != 0 {
                AudioHardwareDestroyAggregateDevice(self.agg_id);
            }
            if self.tap_id != 0 {
                AudioHardwareDestroyProcessTap(self.tap_id);
            }
        }
    }
}

/// Realtime IO proc for the Discord tap: compute the buffer RMS, smooth it, and
/// latch `voice_speaking` on/off with a short hang so the border lights the
/// instant a voice comes through and settles a beat after it stops.
unsafe extern "C-unwind" fn voice_io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    in_input_data: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _out: NonNull<AudioBufferList>,
    _out_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    if client.is_null() {
        return 0;
    }
    let ctx = &*(client as *const VoiceCtx);
    ctx.frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let list = in_input_data.as_ref();
    let nbuf = list.mNumberBuffers as usize;
    if nbuf == 0 {
        return 0;
    }
    let buffers = std::slice::from_raw_parts(list.mBuffers.as_ptr(), nbuf);
    let buf = &buffers[0];
    if buf.mData.is_null() {
        return 0;
    }
    let n = (buf.mDataByteSize as usize) / std::mem::size_of::<f32>();
    if n == 0 {
        return 0;
    }
    let samples = std::slice::from_raw_parts(buf.mData as *const f32, n);
    let mut sumsq = 0.0f32;
    for &s in samples {
        sumsq += s * s;
    }
    let rms = (sumsq / n as f32).sqrt();

    let lvl = {
        let mut level = ctx.level.lock().unwrap();
        *level = *level * 0.6 + rms * 0.4; // smooth a touch so brief clicks don't trip it
        *level
    };

    let mut speaking = ctx.speaking.lock().unwrap();
    let mut loud_at = ctx.loud_at.lock().unwrap();
    let now = Instant::now();
    if lvl > VOICE_THRESH {
        *loud_at = Some(now);
        if !*speaking {
            *speaking = true;
            if let Ok(mut s) = ctx.shared.lock() {
                s.discord.voice_speaking_tap = true;
            }
        }
    } else if *speaking {
        let quiet_for = loud_at
            .map(|t| now.duration_since(t).as_millis() as u64)
            .unwrap_or(u64::MAX);
        if quiet_for >= VOICE_HANG_MS {
            *speaking = false;
            if let Ok(mut s) = ctx.shared.lock() {
                s.discord.voice_speaking_tap = false;
            }
        }
    }
    0
}

/// Every audio process object the HAL currently knows about.
unsafe fn process_object_list() -> Vec<AudioObjectID> {
    let addr = prop_addr(kAudioHardwarePropertyProcessObjectList);
    let sys = kAudioObjectSystemObject as AudioObjectID;
    let mut size: u32 = 0;
    if AudioObjectGetPropertyDataSize(sys, NonNull::from(&addr), 0, std::ptr::null(), NonNull::from(&mut size)) != 0
        || size == 0
    {
        return Vec::new();
    }
    let count = size as usize / std::mem::size_of::<AudioObjectID>();
    let mut objs = vec![0 as AudioObjectID; count];
    let Some(ptr) = NonNull::new(objs.as_mut_ptr() as *mut c_void) else {
        return Vec::new();
    };
    let mut size2 = size;
    if AudioObjectGetPropertyData(sys, NonNull::from(&addr), 0, std::ptr::null(), NonNull::from(&mut size2), ptr) != 0 {
        return Vec::new();
    }
    objs
}

/// Read a fixed-size u32 property (e.g. a process object's PID).
unsafe fn read_u32(obj: AudioObjectID, selector: u32) -> Option<u32> {
    let addr = prop_addr(selector);
    let mut out: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as u32;
    let st = AudioObjectGetPropertyData(
        obj,
        NonNull::from(&addr),
        0,
        std::ptr::null(),
        NonNull::from(&mut size),
        NonNull::new(&mut out as *mut _ as *mut c_void)?,
    );
    if st != 0 {
        return None;
    }
    Some(out)
}

/// PIDs of every running Discord process (main + all helpers), via `pgrep`.
fn discord_pids() -> std::collections::HashSet<u32> {
    std::process::Command::new("pgrep")
        .args(["-i", "discord"])
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .filter_map(|s| s.parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The audio process objects belonging to Discord — matched by PID against the
/// whole Discord process tree (not bundle id, which misses Chromium's separate
/// audio-service helper that actually renders the call). Empty when none.
unsafe fn discord_process_objects() -> Vec<AudioObjectID> {
    let pids = discord_pids();
    if pids.is_empty() {
        return Vec::new();
    }
    process_object_list()
        .into_iter()
        .filter(|&o| read_u32(o, kAudioProcessPropertyPID).map(|p| pids.contains(&p)).unwrap_or(false))
        .collect()
}

/// Stand up a private mono tap of just Discord's processes, an aggregate device,
/// and the level-detecting IO proc. None if Discord has no audio object right now
/// or any HAL call fails.
fn start_voice(shared: Shared, objs: &[AudioObjectID], global: bool) -> Option<VoiceCapture> {
    use objc2::rc::Retained;
    unsafe {
        let nums: Vec<Retained<NSNumber>> = objs.iter().map(|&o| NSNumber::new_u32(o)).collect();
        let arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&nums);
        let desc = if global {
            let empty: Retained<NSArray<NSNumber>> = NSArray::new();
            CATapDescription::initStereoGlobalTapButExcludeProcesses(CATapDescription::alloc(), &empty)
        } else {
            CATapDescription::initMonoMixdownOfProcesses(CATapDescription::alloc(), &arr)
        };
        desc.setName(&objc2_foundation::NSString::from_str("overseer-discord-voice"));
        desc.setPrivate(true);
        desc.setMuteBehavior(objc2_core_audio::CATapMuteBehavior(0)); // CATapUnmuted

        let mut tap_id: AudioObjectID = 0;
        if AudioHardwareCreateProcessTap(Some(&desc), &mut tap_id as *mut AudioObjectID) != 0 || tap_id == 0 {
            return None;
        }

        let ctx = Arc::new(VoiceCtx {
            shared,
            level: Mutex::new(0.0),
            speaking: Mutex::new(false),
            loud_at: Mutex::new(None),
            frames: std::sync::atomic::AtomicU64::new(0),
        });
        let ctx_ptr = Arc::as_ptr(&ctx) as *mut c_void;
        let mut guard = VoiceCapture { agg_id: 0, tap_id, proc_id: None, _ctx: ctx };

        let tap_uid = read_cfstring(tap_id, kAudioTapPropertyUID)?;
        let sub_tap: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioSubTapUIDKey), tap_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioSubTapDriftCompensationKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
        ])?;
        let tap_list: CFRetained<CFArray> = cf_array(&[sub_tap.as_ref() as &CFType as *const CFType])?;
        // Distinct aggregate UIDs per variant so two taps can coexist (the diag
        // runs a discord tap + a global tap at once; same UID would collide).
        let kind = if global { "global" } else { "discord" };
        let agg_uid = CFString::from_str(&format!("com.overseer.voicelevel.{kind}"));
        let agg_name = CFString::from_str(&format!("overseer voice {kind}"));
        let agg_desc: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioAggregateDeviceUIDKey), agg_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceNameKey), agg_name.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceIsPrivateKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceIsStackedKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceTapListKey), tap_list.as_ref() as &CFType as *const CFType),
        ])?;

        let mut agg_id: AudioObjectID = 0;
        if AudioHardwareCreateAggregateDevice(&agg_desc, NonNull::from(&mut agg_id)) != 0 || agg_id == 0 {
            return None;
        }
        guard.agg_id = agg_id;

        let mut proc_id: AudioDeviceIOProcID = None;
        if AudioDeviceCreateIOProcID(agg_id, Some(voice_io_proc), ctx_ptr, NonNull::from(&mut proc_id)) != 0
            || proc_id.is_none()
        {
            return None;
        }
        guard.proc_id = proc_id;

        if AudioDeviceStart(agg_id, proc_id) != 0 {
            return None;
        }
        Some(guard)
    }
}

/// `--diag-discord-audio`: list Discord's audio process objects, stand up the tap,
/// and print when the speaking latch flips while you talk in your call.
pub fn diag_voice() {
    println!("overseer --diag-discord-audio\n");
    let dpids = discord_pids();
    println!("Discord PIDs (pgrep): {dpids:?}");
    println!("all audio process objects (obj: pid bundle):");
    for o in unsafe { process_object_list() } {
        let pid = unsafe { read_u32(o, kAudioProcessPropertyPID) };
        let bundle = unsafe { read_cfstring(o, kAudioProcessPropertyBundleID) }.map(|b| b.to_string());
        let mark = pid.map(|p| dpids.contains(&p)).unwrap_or(false);
        println!("  {o}: pid={pid:?} bundle={bundle:?} {}", if mark { "← DISCORD" } else { "" });
    }
    println!();
    let objs = unsafe { discord_process_objects() };
    println!("→ tapping Discord objects: {objs:?}");
    let s1: Shared = Arc::new(Mutex::new(AppState::default()));
    let s2: Shared = Arc::new(Mutex::new(AppState::default()));
    let dcap = if objs.is_empty() { None } else { start_voice(s1, &objs, false) };
    let gcap = start_voice(s2, &[], true); // GLOBAL tap for comparison
    println!("  discord tap: {}", if dcap.is_some() {"up"} else {"DOWN"});
    println!("  global  tap: {}", if gcap.is_some() {"up"} else {"DOWN"});
    println!("→ talk in your Discord call — watching 20s. Also polling Discord's");
    println!("  per-process IsRunning/Input/Output flags to see if they track talking:\n");
    let mut dpeak = 0.0f32;
    let mut gpeak = 0.0f32;
    let mut last_flags = String::new();
    for i in 0..200 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let dl = dcap.as_ref().map(|c| c.level()).unwrap_or(0.0);
        let gl = gcap.as_ref().map(|c| c.level()).unwrap_or(0.0);
        dpeak = dpeak.max(dl);
        gpeak = gpeak.max(gl);
        // Aggregate the run-flags across all Discord audio objects (any = on).
        let mut run = false;
        let mut rin = false;
        let mut rout = false;
        for &o in &objs {
            run |= unsafe { read_u32(o, kAudioProcessPropertyIsRunning) }.unwrap_or(0) != 0;
            rin |= unsafe { read_u32(o, kAudioProcessPropertyIsRunningInput) }.unwrap_or(0) != 0;
            rout |= unsafe { read_u32(o, kAudioProcessPropertyIsRunningOutput) }.unwrap_or(0) != 0;
        }
        let flags = format!("running={run} input={rin} output={rout}");
        if flags != last_flags {
            println!("  [flags change] {flags}");
            last_flags = flags.clone();
        }
        if i % 10 == 9 {
            println!("  … global {gl:.5} (peak {gpeak:.5})   discord-tap {dl:.5}   {flags}");
        }
    }
    println!("\ndone. discord-tap peak {dpeak:.5} | global peak {gpeak:.5}");
}

/// Background manager: keep a Discord-scoped tap alive whenever Discord has audio
/// processes, rebuild it if Discord relaunches (its process objects change), and
/// clear the speaking flag when Discord goes away. Cheap 2 s poll; the tap itself
/// does the realtime work.
pub fn spawn_voice(shared: Shared) {
    // Discord-scoped process tap of the call audio → lights the DISCORD border
    // while the remote side is talking. The bot-gateway "Speaking" events this used
    // to ride are now walled off by Discord's mandatory DAVE/E2EE (voice close
    // 4017), and the local tap captures the call audio even through WaveLink
    // (verified via --diag-discord-audio). BUT: on a WaveLink rig, standing up this
    // second process-tap + private aggregate device alongside the always-on
    // spectrum tap disrupts CoreAudio routing — other apps (Chrome, Apple Music)
    // lose output while Discord keeps working. So keep it OPT-IN until that
    // interaction is understood: set OVERSEER_DISCORD_AUDIO=1 to enable.
    if std::env::var("OVERSEER_DISCORD_AUDIO").map(|v| v.trim().is_empty()).unwrap_or(true) {
        return;
    }
    std::thread::spawn(move || {
        let mut cap: Option<VoiceCapture> = None;
        let mut built_for: Vec<AudioObjectID> = Vec::new();
        loop {
            let objs = unsafe { discord_process_objects() };
            // Valid only if the live tap was built for exactly this object set —
            // compare as sorted sets so a relaunch (different IDs) rebuilds even
            // when it happens to overlap the old set.
            let still_valid = {
                let mut a = built_for.clone();
                let mut b = objs.clone();
                a.sort_unstable();
                b.sort_unstable();
                !a.is_empty() && a == b
            };
            if objs.is_empty() {
                if cap.take().is_some() {
                    if let Ok(mut s) = shared.lock() {
                        s.discord.voice_speaking_tap = false;
                    }
                }
                built_for.clear();
            } else if cap.is_none() || !still_valid {
                cap = start_voice(shared.clone(), &objs, false);
                built_for = if cap.is_some() { objs } else { Vec::new() };
                if cap.is_none() {
                    if let Ok(mut s) = shared.lock() {
                        s.discord.voice_speaking_tap = false;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    });
}
