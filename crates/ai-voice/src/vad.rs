//! Voice activity detection: a real energy-based VAD (RMS threshold with
//! hangover). No external models — deterministic and testable.

use std::time::Duration;

use crate::Audio;

/// The VAD's decision for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    Speech,
    Silence,
    /// Speech just ended (post-speech hangover frame).
    SpeechEnd,
}

/// VAD configuration.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// RMS threshold below which a frame is silent.
    pub rms_threshold: f32,
    /// Frames of silence before declaring `SpeechEnd`.
    pub hangover_frames: usize,
    /// Frame size in samples.
    pub frame_samples: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 500.0,
            hangover_frames: 8,
            frame_samples: 320, // 20 ms @ 16 kHz
        }
    }
}

/// Energy-based voice activity detector.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    config: VadConfig,
    silence_run: usize,
    in_speech: bool,
}

impl VoiceActivityDetector {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            silence_run: 0,
            in_speech: false,
        }
    }

    /// Processes one audio chunk (may span several frames); returns the
    /// decision of the LAST frame.
    pub fn process(&mut self, audio: &Audio) -> VadDecision {
        let mut decision = VadDecision::Silence;
        for frame in audio.samples.chunks(self.config.frame_samples) {
            let rms = rms(frame);
            if rms >= self.config.rms_threshold {
                self.in_speech = true;
                self.silence_run = 0;
                decision = VadDecision::Speech;
            } else if self.in_speech {
                self.silence_run += 1;
                if self.silence_run >= self.config.hangover_frames {
                    self.in_speech = false;
                    self.silence_run = 0;
                    decision = VadDecision::SpeechEnd;
                } else {
                    decision = VadDecision::Speech;
                }
            } else {
                decision = VadDecision::Silence;
            }
        }
        decision
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Resets the detector state.
    pub fn reset(&mut self) {
        self.in_speech = false;
        self.silence_run = 0;
    }

    /// Convenience for callers that need the configured frame duration.
    pub fn frame_duration(&self) -> Duration {
        Duration::from_millis((self.config.frame_samples as u64 * 1000) / 16_000)
    }
}

fn rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_matches_expected_values() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[0i16, 0, 0]), 0.0);
        // Constant 1000 → RMS 1000.
        assert!((rms(&[1000i16; 4]) - 1000.0).abs() < 0.001);
    }

    #[test]
    fn hangover_delays_speech_end() {
        let mut detector = VoiceActivityDetector::new(VadConfig {
            hangover_frames: 3,
            ..Default::default()
        });
        let speech = Audio::from_samples(vec![2000i16; 320], 16_000);
        let silence = Audio::from_samples(vec![0i16; 320], 16_000);

        assert_eq!(detector.process(&speech), VadDecision::Speech);
        // Two silence frames: still "speech" (hangover).
        assert_eq!(detector.process(&silence), VadDecision::Speech);
        assert_eq!(detector.process(&silence), VadDecision::Speech);
        // Third silence frame: end.
        assert_eq!(detector.process(&silence), VadDecision::SpeechEnd);
        assert!(!detector.in_speech());
    }

    #[test]
    fn reset_clears_state() {
        let mut detector = VoiceActivityDetector::new(VadConfig::default());
        detector.process(&Audio::from_samples(vec![2000i16; 320], 16_000));
        assert!(detector.in_speech());
        detector.reset();
        assert!(!detector.in_speech());
    }
}
