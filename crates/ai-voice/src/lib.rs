//! Voice (PRD §3.5): audio types, an energy-based VAD, and STT/TTS traits
//! with a real OpenAI-compatible adapter. Realtime full-duplex streaming
//! requires provider credentials and is documented as a limitation.

mod vad;

pub use vad::{VadConfig, VadDecision, VoiceActivityDetector};

use async_trait::async_trait;

use ai_errors::{AiError, WebError};

/// PCM audio (16-bit little-endian, mono).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio {
    /// Raw PCM samples.
    pub samples: Vec<i16>,
    /// Sample rate in Hz (e.g. 16_000).
    pub sample_rate: u32,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

impl Audio {
    pub fn from_samples(samples: Vec<i16>, sample_rate: u32) -> Self {
        let duration_ms = if sample_rate == 0 {
            0
        } else {
            (samples.len() as u64 * 1000) / sample_rate as u64
        };
        Self {
            samples,
            sample_rate,
            duration_ms,
        }
    }

    /// Resamples to a target rate (linear interpolation) — real, simple
    /// resampling used before provider upload.
    pub fn resample(&self, target_rate: u32) -> Self {
        if target_rate == 0 || target_rate == self.sample_rate || self.samples.is_empty() {
            return self.clone();
        }
        let ratio = self.sample_rate as f64 / target_rate as f64;
        let new_len = (self.samples.len() as f64 / ratio) as usize;
        let mut out = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src = (i as f64 * ratio) as usize;
            let a = self.samples.get(src).copied().unwrap_or(0);
            let b = self.samples.get(src + 1).copied().unwrap_or(a);
            let frac = (i as f64 * ratio) - src as f64;
            out.push((a as f64 + (b as f64 - a as f64) * frac) as i16);
        }
        Self::from_samples(out, target_rate)
    }
}

/// Speech-to-text provider.
#[async_trait]
pub trait SpeechToText: Send + Sync {
    async fn transcribe(&self, audio: &Audio) -> Result<String, AiError>;
}

/// Text-to-speech provider.
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    /// Returns audio bytes (WAV/MP3 per provider) plus mime type.
    async fn synthesize(&self, text: &str) -> Result<(Vec<u8>, String), AiError>;
}

/// Real OpenAI-compatible STT adapter (`POST {base}/audio/transcriptions`,
/// whisper model). Requires an API key.
pub struct OpenAiCompatSpeechToText {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiCompatSpeechToText {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self, AiError> {
        Ok(Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: "whisper-1".to_string(),
            client: reqwest::Client::builder()
                .user_agent("ai-sdk-voice/0.1")
                .build()
                .map_err(|e| AiError::Web(WebError::new("stt client", e.to_string())))?,
        })
    }
}

#[async_trait]
impl SpeechToText for OpenAiCompatSpeechToText {
    async fn transcribe(&self, audio: &Audio) -> Result<String, AiError> {
        let url = format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        );
        let audio = audio.resample(16_000);
        let pcm = audio.samples;
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let wav = wav_from_pcm(&bytes, audio.sample_rate);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .multipart(
                reqwest::multipart::Form::new()
                    .text("model", self.model.clone())
                    .part(
                        "file",
                        reqwest::multipart::Part::bytes(wav)
                            .file_name("audio.wav")
                            .mime_str("audio/wav")
                            .map_err(|e| AiError::Web(WebError::new("stt", e.to_string())))?,
                    ),
            )
            .send()
            .await
            .map_err(|e| AiError::Web(WebError::new("stt", e.to_string())))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AiError::Web(WebError::new(
                "stt",
                format!("HTTP {status}: {body}"),
            )));
        }
        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AiError::Web(WebError::new("stt parse", e.to_string())))?;
        Ok(json
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string())
    }
}

/// Real OpenAI-compatible TTS adapter (`POST {base}/audio/speech`).
pub struct OpenAiCompatTextToSpeech {
    base_url: String,
    api_key: String,
    voice: String,
    client: reqwest::Client,
}

impl OpenAiCompatTextToSpeech {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self, AiError> {
        Ok(Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            voice: "alloy".to_string(),
            client: reqwest::Client::builder()
                .user_agent("ai-sdk-voice/0.1")
                .build()
                .map_err(|e| AiError::Web(WebError::new("tts client", e.to_string())))?,
        })
    }
}

#[async_trait]
impl TextToSpeech for OpenAiCompatTextToSpeech {
    async fn synthesize(&self, text: &str) -> Result<(Vec<u8>, String), AiError> {
        let url = format!("{}/audio/speech", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": "tts-1",
                "input": text,
                "voice": self.voice
            }))
            .send()
            .await
            .map_err(|e| AiError::Web(WebError::new("tts", e.to_string())))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AiError::Web(WebError::new(
                "tts",
                format!("HTTP {status}: {body}"),
            )));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_string();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| AiError::Web(WebError::new("tts body", e.to_string())))?
            .to_vec();
        Ok((bytes, content_type))
    }
}

/// Wraps raw PCM in a minimal WAV container (16-bit mono).
fn wav_from_pcm(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let byte_rate = sample_rate * 2;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_duration_is_computed() {
        let audio = Audio::from_samples(vec![0i16; 16_000], 16_000);
        assert_eq!(audio.duration_ms, 1000);
    }

    #[test]
    fn resample_halves_and_doubles() {
        let audio = Audio::from_samples(vec![0i16; 32_000], 32_000);
        let half = audio.resample(16_000);
        assert_eq!(half.sample_rate, 16_000);
        assert!(half.samples.len() < audio.samples.len());
        let back = half.resample(32_000);
        assert_eq!(back.sample_rate, 32_000);
    }

    #[test]
    fn wav_header_is_valid() {
        let pcm = vec![0u8; 4];
        let wav = wav_from_pcm(&pcm, 16_000);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 4);
        assert_eq!(wav.len(), 48);
    }

    #[test]
    fn vad_detects_speech_and_silence() {
        let mut detector = VoiceActivityDetector::new(VadConfig::default());
        // Pure silence.
        let silence = Audio::from_samples(vec![0i16; 320], 16_000);
        assert_eq!(detector.process(&silence), VadDecision::Silence);
        // Loud signal.
        let speech = Audio::from_samples(vec![3000i16; 320], 16_000);
        assert_eq!(detector.process(&speech), VadDecision::Speech);
    }
}
