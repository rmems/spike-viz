use std::io;

/// Missing or invalid spike data — fail loud, never invent spikes.
///
/// Mirrors the Python package's `SpikeIOError`: a missing `meta.json`,
/// `spikes.npz`, or any contract violation is an error, never empty
/// activity that would look like "no spikes". Loading errors must not be
/// weakened to make a pretty PNG.
#[derive(Debug, thiserror::Error)]
pub enum SpikeIOError {
    /// Filesystem / archive level failure (missing file, corrupt archive, ...).
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The data or metadata violated the export contract.
    #[error("{0}")]
    Invalid(String),
}

impl SpikeIOError {
    /// Build an `Io` error with a context message (path, archive, ...).
    pub fn io(message: impl Into<String>) -> Self {
        SpikeIOError::Io(io::Error::new(io::ErrorKind::InvalidData, message.into()))
    }
}
