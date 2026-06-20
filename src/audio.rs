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
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioSubTapDriftCompensationKey,
    kAudioSubTapUIDKey, kAudioTapPropertyUID, AudioDeviceCreateIOProcID, AudioDeviceIOProcID,
    AudioDeviceStart, AudioDeviceStop, AudioDeviceDestroyIOProcID, AudioHardwareCreateAggregateDevice,
    AudioHardwareCreateProcessTap, AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress, CATapDescription,
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

    TapCtx {
        shared,
        fft,
        hann,
        edges,
        window: Mutex::new(Vec::with_capacity(FFT_SIZE)),
        decay: Mutex::new(vec![0.0; BANDS]),
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
    let mut buf: Vec<Complex<f32>> = (0..FFT_SIZE)
        .map(|i| Complex::new(win[i] * ctx.hann[i], 0.0))
        .collect();
    ctx.fft.process(&mut buf);

    // Per-band energy = mean magnitude across its bin slice, then a perceptual
    // compress (sqrt) so quiet detail is visible without the loud parts pinning.
    let mut bands = vec![0.0f32; BANDS];
    for b in 0..BANDS {
        let lo = ctx.edges[b];
        let hi = ctx.edges[b + 1].max(lo + 1);
        let mut sum = 0.0f32;
        for k in lo..hi {
            sum += buf[k].norm();
        }
        let mag = sum / (hi - lo) as f32;
        // A gentle tilt lifts treble bands (which carry less energy) so the top
        // bars don't sit dead — keeps the spectrum lively across its width.
        let tilt = 1.0 + 1.6 * (b as f32 / BANDS as f32);
        bands[b] = (mag * tilt).sqrt();
    }

    // Normalize to a soft ceiling so the bars use the full height but loud
    // passages don't clip flat. The ceiling adapts slowly to the running peak.
    let peak = bands.iter().cloned().fold(0.0f32, f32::max).max(1e-4);
    let norm = 1.0 / peak.max(0.35);
    for v in &mut bands {
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

    let now = Instant::now();
    if let Ok(mut s) = ctx.shared.lock() {
        s.audio_live = true;
        s.audio_samples.push_back((now, bands));
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
        desc.setName(&objc2_foundation::NSString::from_str("studioboard-eq"));
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
        let tap_uid = match read_cfstring(tap_id, kAudioTapPropertyUID) {
            Some(u) => u,
            None => return None,
        };

        // 3. Build a private aggregate device whose sub-tap is our tap. Keys are
        //    C-string constants from the HAL; values a heterogeneous CF mix, so
        //    we assemble the dict from CFType pointers.
        let sub_tap: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioSubTapUIDKey), tap_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioSubTapDriftCompensationKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
        ]);
        let tap_list: CFRetained<CFArray> = cf_array(&[sub_tap.as_ref() as &CFType as *const CFType]);
        let agg_uid = CFString::from_str("com.studioboard.eq.aggregate");
        let agg_name = CFString::from_str("studioboard EQ");
        let agg_desc: CFRetained<CFDictionary> = cf_dict(&[
            (cfstr(kAudioAggregateDeviceUIDKey), agg_uid.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceNameKey), agg_name.as_ref() as &CFType as *const CFType),
            (cfstr(kAudioAggregateDeviceIsPrivateKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceIsStackedKey), kCFBooleanTrue.unwrap() as *const _ as *const CFType),
            (cfstr(kAudioAggregateDeviceTapListKey), tap_list.as_ref() as &CFType as *const CFType),
        ]);

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
/// callbacks (so entries are retained/released for us).
fn cf_array(items: &[*const CFType]) -> CFRetained<CFArray> {
    use objc2_core_foundation::kCFTypeArrayCallBacks;
    let mut vals: Vec<*const c_void> = items.iter().map(|v| *v as *const c_void).collect();
    unsafe {
        CFArray::new(None, vals.as_mut_ptr(), items.len() as isize, &raw const kCFTypeArrayCallBacks)
            .expect("CFArrayCreate")
    }
}

/// `--diag-audio`: stand up the tap, watch for a few band frames, and report
/// whether real capture is working — so the user can tell at a glance if the
/// EQ is measuring sound or falling back to the synthetic flourish.
pub fn diag() {
    println!("studioboard --diag-audio\n");
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
/// default CFType callbacks (retain/release the entries for us).
fn cf_dict(pairs: &[(CFRetained<CFString>, *const CFType)]) -> CFRetained<CFDictionary> {
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
        .expect("CFDictionaryCreate")
    }
}
