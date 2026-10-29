"""Validate English transcription requests without API calls or speech fixtures."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave

from interview_services import InterviewServiceError, transcribe


def pcm_wav(samples: bytes, *, width: int = 2, channels: int = 1) -> bytes:
    """Wrap deterministic PCM samples in an in-memory WAV container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(width)
        recording.setframerate(16000)
        recording.writeframes(samples)
    return output.getvalue()


class InterviewTranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="  How did you validate the model?  "
        )

    def test_english_hint_preserves_model_and_original_audio(self) -> None:
        audio = pcm_wav(b"\x01\x00\xff\xff" * 8)
        result = transcribe(self.client, audio)
        self.assertEqual(result, "How did you validate the model?")
        request = self.client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["model"], "whisper-1")
        self.assertEqual(request["language"], "en")
        self.assertEqual(request["file"].name, "speech.wav")
        self.assertEqual(request["file"].read(), audio)

    def test_empty_invalid_and_oversized_audio_never_reach_api(self) -> None:
        for audio in (None, "audio", bytearray(b"audio"), b"", b"not a WAV", b"x" * 15_000_001):
            with self.subTest(audio_type=type(audio).__name__):
                with self.assertRaises(InterviewServiceError):
                    transcribe(self.client, audio)
        self.client.audio.transcriptions.create.assert_not_called()

    def test_zero_frame_and_truncated_recordings_never_reach_api(self) -> None:
        valid = pcm_wav(b"\x01\x00\xff\xff" * 8)
        truncated_data = bytearray(valid)
        truncated_data[40:44] = (len(valid) - 44 + 2).to_bytes(4, "little")
        partial_frame = pcm_wav(b"\x01\x00\xff")
        for audio in (pcm_wav(b""), valid[:20], valid[:-1], bytes(truncated_data), partial_frame):
            with self.subTest(size=len(audio)):
                with self.assertRaises(InterviewServiceError):
                    transcribe(self.client, audio)
        self.client.audio.transcriptions.create.assert_not_called()

    def test_invalid_format_and_zero_sample_rate_never_reach_api(self) -> None:
        valid = pcm_wav(b"\x01\x00\xff\xff" * 8)
        invalid_fields = ((16, 4, 255), (20, 2, 3), (34, 2, 40), (24, 4, 0))
        for offset, size, value in invalid_fields:
            malformed = bytearray(valid)
            malformed[offset:offset + size] = value.to_bytes(size, "little")
            with self.subTest(offset=offset, value=value):
                with self.assertRaises(InterviewServiceError):
                    transcribe(self.client, bytes(malformed))
        self.client.audio.transcriptions.create.assert_not_called()

    def test_valid_pcm_with_ancillary_chunk_is_accepted(self) -> None:
        standard = pcm_wav(b"\x01\x00\xff\xff" * 8)
        audio = bytearray(standard[:36] + b"JUNK\x04\x00\x00\x00test" + standard[36:])
        audio[4:8] = (len(audio) - 8).to_bytes(4, "little")
        self.assertEqual(transcribe(self.client, bytes(audio)), "How did you validate the model?")

    def test_digital_silence_is_rejected_for_supported_pcm_widths(self) -> None:
        for width in (1, 2, 3, 4):
            for channels in (1, 2):
                silence = b"\x80" if width == 1 else b"\x00" * width
                audio = pcm_wav(silence * channels * 16, width=width, channels=channels)
                with self.subTest(width=width, channels=channels):
                    with self.assertRaisesRegex(InterviewServiceError, "silent"):
                        transcribe(self.client, audio)
        self.client.audio.transcriptions.create.assert_not_called()

    def test_quiet_nonzero_pcm_is_accepted_without_loudness_threshold(self) -> None:
        for width in (1, 2, 3, 4):
            for channels in (1, 2):
                samples = b"\x7f\x81" if width == 1 else (
                    (1).to_bytes(width, "little", signed=True)
                    + (-1).to_bytes(width, "little", signed=True)
                )
                audio = pcm_wav(samples * channels * 16, width=width, channels=channels)
                with self.subTest(width=width, channels=channels):
                    self.assertEqual(transcribe(self.client, audio), "How did you validate the model?")
        self.assertEqual(self.client.audio.transcriptions.create.call_count, 8)

    def test_service_failure_is_safe_and_same_recording_can_be_retried(self) -> None:
        audio = pcm_wav(b"\x01\x00\xff\xff" * 8)
        self.client.audio.transcriptions.create.side_effect = [
            TimeoutError("private raw response"), SimpleNamespace(text="A usable transcript"),
        ]
        with self.assertRaises(InterviewServiceError) as raised:
            transcribe(self.client, audio)
        self.assertIn("retry", str(raised.exception))
        self.assertNotIn("private raw response", str(raised.exception))
        self.assertEqual(transcribe(self.client, audio), "A usable transcript")
        for call in self.client.audio.transcriptions.create.call_args_list:
            self.assertEqual(call.kwargs["file"].read(), audio)

    def test_missing_or_blank_transcript_is_reported_as_failure(self) -> None:
        audio = pcm_wav(b"\x01\x00\xff\xff" * 8)
        for text in (None, 42, "", " \n "):
            self.client.audio.transcriptions.create.return_value = SimpleNamespace(text=text)
            with self.subTest(text=text):
                with self.assertRaises(InterviewServiceError):
                    transcribe(self.client, audio)


if __name__ == "__main__":
    unittest.main()
