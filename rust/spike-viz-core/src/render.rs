use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::error::SpikeIOError;
use crate::events::SpikeEvents;

fn checked_size(a: usize, b: usize, name: &str) -> Result<usize, SpikeIOError> {
    a.checked_mul(b)
        .filter(|size| *size <= isize::MAX as usize)
        .ok_or_else(|| SpikeIOError::Invalid(format!("{name} is too large")))
}

fn grid_len(n_steps: usize, n_neurons: usize) -> Result<usize, SpikeIOError> {
    if n_steps < 1 || n_neurons < 1 {
        return Err(SpikeIOError::Invalid(
            "n_steps and n_neurons must be >= 1".to_string(),
        ));
    }
    let len = checked_size(n_steps, n_neurons, "dense grid")?;
    checked_size(len, std::mem::size_of::<f32>(), "dense grid byte size")?;
    Ok(len)
}

fn scaled_dimension(size: usize, scale: usize) -> Result<usize, SpikeIOError> {
    let dimension = checked_size(size, scale, "scaled raster dimension")?;
    // PNG dimensions must be in 1..=2^31-1 (PNG specification, section 11.2.2).
    if dimension == 0 || dimension > i32::MAX as usize {
        return Err(SpikeIOError::Invalid(
            "PNG dimensions must be in 1..=2^31-1".to_string(),
        ));
    }
    Ok(dimension)
}

fn pixel_len(width: usize, height: usize) -> Result<usize, SpikeIOError> {
    let len = checked_size(width, height, "scaled pixel count")?;
    checked_size(len, 3, "RGB byte size")?;
    Ok(len)
}

fn raster_geometry(
    n_steps: usize,
    n_neurons: usize,
    scale: usize,
) -> Result<(usize, usize, usize), SpikeIOError> {
    let len = grid_len(n_steps, n_neurons)?;
    if scale < 1 {
        return Err(SpikeIOError::Invalid(
            "scale must be an integer >= 1".to_string(),
        ));
    }
    let width = scaled_dimension(n_steps, scale)?;
    let height = scaled_dimension(n_neurons, scale)?;
    pixel_len(width, height)?;
    // (dense-grid length, image width, image height)
    Ok((len, width, height))
}

fn buffer<T>(len: usize) -> Result<Vec<T>, SpikeIOError> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|e| SpikeIOError::Invalid(format!("cannot allocate rendering buffer: {e}")))?;
    Ok(data)
}

/// Dense `[T, N]` float32 grid (time-major), the raster source of truth.
///
/// Mirrors the Python `sparse_to_dense` output. Data is row-major with
/// `data[t * n_neurons + n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGrid {
    n_steps: usize,
    n_neurons: usize,
    data: Vec<f32>,
}

impl DenseGrid {
    /// Build a dense grid from raw row-major data.
    ///
    /// Fails loud when the geometry is not at least 1x1 or the data length
    /// does not match `n_steps * n_neurons`.
    pub fn new(n_steps: usize, n_neurons: usize, data: Vec<f32>) -> Result<Self, SpikeIOError> {
        let len = grid_len(n_steps, n_neurons)?;
        if data.len() != len {
            return Err(SpikeIOError::Invalid(format!(
                "dense grid must hold n_steps * n_neurons = {} values, got {}",
                len,
                data.len()
            )));
        }
        Ok(Self {
            n_steps,
            n_neurons,
            data,
        })
    }

    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Row-major `[T, N]` data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Value at `(t, n)`.
    pub fn get(&self, t: usize, n: usize) -> f32 {
        self.data[t * self.n_neurons + n]
    }
}

/// Raster sparse events into a dense `[T, N]` float32 grid.
///
/// Port of the Python `sparse_to_dense` (`src/spike_viz/io.py`):
/// amplitudes accumulate into bins (duplicates add), `amp` defaults to 1.0,
/// and events outside the geometry fail loud instead of being dropped.
pub fn sparse_to_dense(
    events: &SpikeEvents,
    n_steps: usize,
    n_neurons: usize,
) -> Result<DenseGrid, SpikeIOError> {
    let len = grid_len(n_steps, n_neurons)?;
    let mut data = buffer(len)?;
    data.resize(len, 0.0f32);
    if events.is_empty() {
        return Ok(DenseGrid {
            n_steps,
            n_neurons,
            data,
        });
    }

    let t_min = events.t().iter().min().expect("non-empty");
    let t_max = events.t().iter().max().expect("non-empty");
    if *t_max >= n_steps as i64 || *t_min < 0 {
        return Err(SpikeIOError::Invalid(format!(
            "event time out of range [0, {n_steps}): min={t_min} max={t_max}"
        )));
    }
    let n_min = events.neuron_id().iter().min().expect("non-empty");
    let n_max = events.neuron_id().iter().max().expect("non-empty");
    if *n_max >= n_neurons as i64 || *n_min < 0 {
        return Err(SpikeIOError::Invalid(format!(
            "neuron_id out of range [0, {n_neurons}): min={n_min} max={n_max}"
        )));
    }

    for (t, n, amp) in events.iter() {
        data[t as usize * n_neurons + n as usize] += amp;
    }
    Ok(DenseGrid {
        n_steps,
        n_neurons,
        data,
    })
}

/// A rendered raster frame.
///
/// Axis convention matches the Python renderer: **time is horizontal**
/// (columns), **neuron is vertical** (rows). The unscaled pixel grid is
/// `[N, T]` row-major (`pixels[n * n_steps + t]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Raster {
    n_steps: usize,
    n_neurons: usize,
    scale: usize,
    /// Validated image width (`n_steps * scale`), within the PNG limits.
    width: usize,
    /// Validated image height (`n_neurons * scale`), within the PNG limits.
    height: usize,
    /// Pre-normalization peak of the frame (max bin value).
    peak: f32,
    /// Unscaled `[N, T]` grayscale pixels.
    pixels: Vec<u8>,
}

impl Raster {
    /// Image width in pixels: `n_steps * scale`.
    ///
    /// Infallible: the geometry is validated (against the PNG dimension
    /// limits) when the raster is rendered.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels: `n_neurons * scale`.
    ///
    /// Infallible: the geometry is validated (against the PNG dimension
    /// limits) when the raster is rendered.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Nearest-neighbor upscale factor.
    pub fn scale(&self) -> usize {
        self.scale
    }

    /// Unscaled grid geometry `(n_steps, n_neurons)`.
    pub fn grid_shape(&self) -> (usize, usize) {
        (self.n_steps, self.n_neurons)
    }

    /// Pre-normalization peak of the rendered frame.
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Unscaled `[N, T]` grayscale pixels, row-major (row = neuron, col = time).
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Nearest-neighbor scaled grayscale frame, row-major
    /// `[height, width]`. Returns `Invalid` when the allocation fails.
    pub fn scaled_pixels(&self) -> Result<Vec<u8>, SpikeIOError> {
        // Geometry was validated at render time; only allocation can fail.
        let mut out = buffer(pixel_len(self.width, self.height)?)?;
        for n in 0..self.n_neurons {
            for _ in 0..self.scale {
                for t in 0..self.n_steps {
                    let value = self.pixels[n * self.n_steps + t];
                    out.extend(std::iter::repeat_n(value, self.scale));
                }
            }
        }
        Ok(out)
    }

    /// Scaled RGB8 image bytes (grayscale replicated to three channels).
    ///
    /// Matches the PNG mode written by the Python renderer
    /// (`Image.fromarray(..., mode="L").convert("RGB")`).
    pub fn rgb_bytes(&self) -> Result<Vec<u8>, SpikeIOError> {
        let gray = self.scaled_pixels()?;
        let mut out = buffer(checked_size(gray.len(), 3, "RGB byte size")?)?;
        for value in gray {
            out.extend([value, value, value]);
        }
        Ok(out)
    }
}

/// Render sparse events to a dark-theme grayscale raster (CPU only).
///
/// Port of the Python `render_raster` (`src/spike_viz/render.py`), sparse
/// input path. Time is horizontal, neuron is vertical; width is
/// `n_steps * scale` and height is `n_neurons * scale`. The background is
/// black and spike intensity is normalized so the brightest bin in the grid
/// maps to full white.
///
/// Truth-preserving rules:
///
/// - Zero-amplitude events still represent a spike (axon-encoder polarity
///   false is exported as `amp = 0.0`), so they render as unit amplitude
///   instead of vanishing into the background.
/// - Duplicate `(t, neuron)` events accumulate.
/// - Non-finite or out-of-range values fail loud; nothing is clipped into
///   range silently except the final `[0, 1]` intensity clip.
pub fn render_raster(
    events: &SpikeEvents,
    n_steps: usize,
    n_neurons: usize,
    scale: usize,
) -> Result<Raster, SpikeIOError> {
    // Reject oversized geometry (and bad scale) before allocating anything.
    raster_geometry(n_steps, n_neurons, scale)?;
    // Substitute zero amplitudes with unit amplitude before accumulating.
    let grid = match events.amp() {
        Some(amp) => {
            let substituted: Vec<f32> = amp
                .iter()
                .map(|value| if *value == 0.0 { 1.0 } else { *value })
                .collect();
            let substituted_events = SpikeEvents::new(
                events.t().to_vec(),
                events.neuron_id().to_vec(),
                Some(substituted),
            )?;
            sparse_to_dense(&substituted_events, n_steps, n_neurons)?
        }
        None => sparse_to_dense(events, n_steps, n_neurons)?,
    };
    render_dense(&grid, scale)
}

/// Render a dense `[T, N]` grid with the same intensity rules as
/// [`render_raster`].
pub fn render_dense(grid: &DenseGrid, scale: usize) -> Result<Raster, SpikeIOError> {
    let (n_steps, n_neurons) = (grid.n_steps(), grid.n_neurons());
    let (len, width, height) = raster_geometry(n_steps, n_neurons, scale)?;

    // Frame is [N, T]: rows = neurons, columns = time steps.
    let mut peak = f32::NEG_INFINITY;
    for n in 0..n_neurons {
        for t in 0..n_steps {
            let value = grid.get(t, n);
            if !value.is_finite() {
                return Err(SpikeIOError::Invalid(
                    "data contains non-finite or out-of-range values".to_string(),
                ));
            }
            peak = peak.max(value);
        }
    }

    let mut pixels = buffer(len)?;
    for n in 0..n_neurons {
        for t in 0..n_steps {
            let value = grid.get(t, n);
            // peak > 0 normalizes; otherwise paint raw values, which the
            // clip below turns into an all-black (or all-clip) frame — the
            // same "honest silence" as the Python renderer.
            let intensity = if peak > 0.0 { value / peak } else { value };
            let clipped = intensity.clamp(0.0, 1.0);
            pixels.push(to_u8(clipped));
        }
    }

    Ok(Raster {
        n_steps,
        n_neurons,
        scale,
        width,
        height,
        peak,
        pixels,
    })
}

/// Map a clipped `[0, 1]` intensity to `u8`.
fn to_u8(intensity: f32) -> u8 {
    // intensity is clipped to [0, 1], so the rounded value is in [0, 255].
    round_half_even(intensity * 255.0) as u8
}

/// Round to nearest with halves to even, exactly like NumPy's `np.round`
/// (the Python renderer relies on it, e.g. `0.5 * 255 = 127.5 -> 128`).
fn round_half_even(value: f32) -> f32 {
    let floor = value.floor();
    if value - floor == 0.5 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

/// Write a raster to a PNG file.
///
/// The parent directory must already exist (matching the Python renderer's
/// behavior of letting `OSError` propagate). Output is 8-bit RGB, identical
/// in mode to the PNGs written by the Python renderer.
pub fn write_png<P: AsRef<Path>>(raster: &Raster, path: P) -> Result<(), SpikeIOError> {
    // Encode the bytes before touching the destination so a failure does
    // not leave a truncated PNG behind. The u32 casts re-check the PNG
    // dimension limit at the encoder boundary (construction already
    // validates it; this guards direct struct misuse).
    let width = u32::try_from(raster.width())
        .map_err(|_| SpikeIOError::Invalid("PNG width exceeds u32".to_string()))?;
    let height = u32::try_from(raster.height())
        .map_err(|_| SpikeIOError::Invalid("PNG height exceeds u32".to_string()))?;
    let rgb = raster.rgb_bytes()?;
    let path = path.as_ref();
    let file = File::create(path)
        .map_err(|e| SpikeIOError::io(format!("failed to write PNG at {}: {e}", path.display())))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| SpikeIOError::io(format!("failed to encode PNG header: {e}")))?;
    writer
        .write_image_data(&rgb)
        .map_err(|e| SpikeIOError::io(format!("failed to encode PNG pixels: {e}")))?;
    writer.finish().map_err(|e| {
        SpikeIOError::io(format!("failed to finish PNG at {}: {e}", path.display()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_matches_numpy() {
        assert_eq!(round_half_even(0.0), 0.0);
        assert_eq!(round_half_even(255.0), 255.0);
        assert_eq!(round_half_even(127.5), 128.0);
        // 24.5 rounds to 24 (half to even), 25.5 to 26 — like np.round.
        assert_eq!(round_half_even(24.5), 24.0);
        assert_eq!(round_half_even(25.5), 26.0);
    }

    #[test]
    fn intensity_halves_render_like_numpy() {
        // amp = 0.5 against peak = 1.0 -> 0.5 * 255 = 127.5 -> 128.
        assert_eq!(to_u8(0.5), 128);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(0.0), 0);
    }

    #[test]
    fn dense_grid_rejects_bad_geometry() {
        assert!(DenseGrid::new(0, 2, vec![]).is_err());
        assert!(DenseGrid::new(2, 0, vec![]).is_err());
        assert!(DenseGrid::new(2, 2, vec![0.0; 3]).is_err());
        assert!(DenseGrid::new(2, 2, vec![0.0; 4]).is_ok());
    }

    #[test]
    fn invalid_raster_sizes_fail_before_png_creation() {
        let path = std::env::temp_dir().join("spike-viz-missing-parent/invalid.png");
        for (width, height) in [
            (usize::MAX, 1),                        // width exceeds the PNG u32 limit
            (1, usize::MAX),                        // height exceeds the PNG u32 limit
            (i32::MAX as usize, i32::MAX as usize), // RGB buffer exceeds addressable size
        ] {
            // Exercise the buffer and PNG boundaries independently of
            // render_dense (which validates geometry at construction).
            let raster = Raster {
                n_steps: width,
                n_neurons: height,
                scale: 1,
                width,
                height,
                peak: 0.0,
                pixels: vec![],
            };
            assert!(matches!(
                raster.scaled_pixels(),
                Err(SpikeIOError::Invalid(_))
            ));
            assert!(matches!(raster.rgb_bytes(), Err(SpikeIOError::Invalid(_))));
            assert!(matches!(
                write_png(&raster, &path),
                Err(SpikeIOError::Invalid(_))
            ));
        }
    }
}
