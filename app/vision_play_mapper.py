"""Vision-based play detection using Claude Vision to map CFBD plays to video timestamps.

This module replaces broken DTW/OCR timestamp mapping with semantic understanding.
Instead of trying to find plays mathematically, we ask Claude Vision to watch the video
and identify where each play occurs based on visual content.

Architecture:
1. Extract frames from video at regular intervals (every 10 seconds)
2. Batch frames into groups for Claude Vision API calls
3. Send frames + CFBD play context to Claude Vision
4. Claude Vision returns precise timestamps for each play
5. Use these timestamps to cut accurate clips

Key differences from claude_play_detector.py:
- Dense frame sampling (every 10s vs 60 total frames)
- Batched API calls (can't send 1000+ frames at once)
- Returns precise timestamps (not frame indices)
- Designed for high-volume processing (150+ plays per game)
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import asyncio

logger = logging.getLogger(__name__)


class VisionPlayMapper:
    """Map CFBD plays to video timestamps using Claude Vision."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude Vision client.

        Args:
            api_key: Anthropic API key. If None, will try to import from environment.
        """
        import os

        logger.info("[DEBUG] [VISION MAPPER] ===== INITIALIZING VISION PLAY MAPPER =====")
        logger.info(f"[DEBUG] [VISION MAPPER] api_key parameter provided: {api_key is not None}")
        if api_key:
            logger.info(f"[DEBUG] [VISION MAPPER] api_key length: {len(api_key)} chars")
            logger.info(f"[DEBUG] [VISION MAPPER] api_key preview: {api_key[:20]}...")
        else:
            logger.warning(f"[DEBUG] [VISION MAPPER] No API key provided to __init__, will check environment")
            logger.info(f"[DEBUG] [VISION MAPPER] ANTHROPIC_API_KEY in env: {'ANTHROPIC_API_KEY' in os.environ}")

        try:
            import anthropic
            self.anthropic = anthropic
            logger.info(f"[DEBUG] [VISION MAPPER] anthropic library imported successfully")

            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"[DEBUG] [VISION MAPPER] Anthropic client created successfully")

            self.model = "claude-opus-4-20250514"  # Latest Opus model with vision
            logger.info(f"[DEBUG] [VISION MAPPER] Model set to: {self.model}")
            logger.info("[VISION MAPPER] ✓ Initialized Vision Play Mapper successfully")

        except ImportError as e:
            logger.error(f"[VISION MAPPER] ❌ Failed to import anthropic library: {e}")
            logger.error(f"[DEBUG] [VISION MAPPER] ImportError details: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"[VISION MAPPER] ❌ Failed to initialize Claude client: {e}")
            logger.error(f"[DEBUG] [VISION MAPPER] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG] [VISION MAPPER] Exception details: {e}")
            import traceback
            logger.error(f"[DEBUG] [VISION MAPPER] Traceback:\n{traceback.format_exc()}")
            raise

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            duration = float(output.strip())
            logger.info(f"[VISION MAPPER] Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            return duration
        except Exception as e:
            logger.error(f"[VISION MAPPER] Failed to get video duration: {e}")
            raise

    def extract_frames_dense(
        self,
        video_path: str,
        interval_seconds: float = 10.0,
        max_frames: Optional[int] = None
    ) -> List[Tuple[bytes, float]]:
        """Extract frames at regular intervals for dense sampling.

        Args:
            video_path: Path to video file
            interval_seconds: Time between frames (default: 10 seconds)
            max_frames: Maximum number of frames to extract (optional)

        Returns:
            List of (frame_bytes, timestamp) tuples
        """
        logger.info(f"[DEBUG] [VISION MAPPER] ===== STARTING FRAME EXTRACTION =====")
        logger.info(f"[VISION MAPPER] Starting dense frame extraction")
        logger.info(f"  Video path: {video_path}")
        logger.info(f"  Interval: {interval_seconds}s between frames")
        logger.info(f"  Max frames: {max_frames if max_frames else 'unlimited'}")

        duration = self.get_video_duration(video_path)

        # Calculate frame timestamps
        num_frames = int(duration / interval_seconds)
        if max_frames and num_frames > max_frames:
            num_frames = max_frames
            interval_seconds = duration / num_frames
            logger.info(f"  Limiting to {max_frames} frames, adjusted interval to {interval_seconds:.1f}s")

        logger.info(f"  Will extract {num_frames} frames from {duration:.1f}s video")

        frames: List[Tuple[bytes, float]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_frames):
                timestamp = i * interval_seconds

                # Don't go past video duration
                if timestamp >= duration:
                    break

                frame_path = f"{tmpdir}/frame_{i:05d}.jpg"

                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",  # High quality JPEG
                    "-vf", "scale='min(1920,iw)':-2",  # Limit to 1920px width
                    frame_path,
                    "-y"
                ]

                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=10
                    )

                    with open(frame_path, "rb") as f:
                        frame_bytes = f.read()
                        frames.append((frame_bytes, timestamp))

                    if (i + 1) % 100 == 0:
                        logger.info(f"  Extracted {i + 1}/{num_frames} frames...")

                except subprocess.TimeoutExpired:
                    logger.warning(f"  Timeout extracting frame at {timestamp:.1f}s")
                except Exception as e:
                    logger.warning(f"  Failed to extract frame at {timestamp:.1f}s: {e}")

        logger.info(f"[VISION MAPPER] Extracted {len(frames)} frames successfully")
        return frames

    @staticmethod
    def clean_json_response(response_text: str) -> str:
        """Clean Claude's response to extract valid JSON.

        Handles various response formats:
        - Markdown code blocks (```json ... ```)
        - Plain backticks (``` ... ```)
        - Preamble text before JSON
        - Trailing text after JSON

        Args:
            response_text: Raw response text from Claude

        Returns:
            Cleaned JSON string ready for parsing
        """
        if not response_text:
            return response_text

        # Strip leading/trailing whitespace
        cleaned = response_text.strip()

        # Handle markdown code blocks with json language tag
        if "```json" in cleaned:
            parts = cleaned.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0]
                cleaned = json_part.strip()
        # Handle plain markdown code blocks
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        # Remove any remaining leading/trailing whitespace
        cleaned = cleaned.strip()

        # Find JSON start - look for opening bracket or brace
        json_start = -1
        for char in ['{', '[']:
            pos = cleaned.find(char)
            if pos != -1 and (json_start == -1 or pos < json_start):
                json_start = pos

        # Find JSON end - look for closing bracket or brace (from the end)
        json_end = -1
        for char in ['}', ']']:
            pos = cleaned.rfind(char)
            if pos != -1 and pos > json_end:
                json_end = pos

        # Extract JSON if valid boundaries found
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned = cleaned[json_start:json_end + 1]

        return cleaned

    async def map_plays_to_timestamps(
        self,
        video_path: str,
        cfbd_plays: List[Dict],
        game_info: Optional[Dict] = None,
        frame_interval: float = 10.0,
        batch_size: int = 20,
        max_frames: Optional[int] = None
    ) -> Dict[int, Tuple[float, float]]:
        """Map CFBD plays to video timestamps using Claude Vision.

        This is the main entry point for vision-based play detection.

        Args:
            video_path: Path to video file
            cfbd_plays: List of CFBD play dictionaries
            game_info: Optional game context (teams, date, etc.)
            frame_interval: Seconds between extracted frames (default: 10s)
            batch_size: Number of plays to process per API call (default: 20)
            max_frames: Maximum frames to extract (optional, for testing)

        Returns:
            Dictionary mapping play_number -> (start_time, end_time)
        """
        logger.info("="*80)
        logger.info("[VISION PLAY MAPPER] Starting play detection")
        logger.info("="*80)
        logger.info(f"  Video: {video_path}")
        logger.info(f"  CFBD plays to map: {len(cfbd_plays)}")
        logger.info(f"  Frame interval: {frame_interval}s")
        logger.info(f"  Batch size: {batch_size} plays per API call")

        # Extract frames densely
        frames = self.extract_frames_dense(
            video_path,
            interval_seconds=frame_interval,
            max_frames=max_frames
        )

        if not frames:
            logger.error("[VISION MAPPER] No frames extracted!")
            return {}

        # Prepare frame data for Claude
        frame_data = []
        for frame_bytes, timestamp in frames:
            base64_frame = base64.standard_b64encode(frame_bytes).decode("utf-8")
            frame_data.append({
                "timestamp": timestamp,
                "base64": base64_frame,
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_frame
                }
            })

        logger.info(f"[VISION MAPPER] Prepared {len(frame_data)} frames for Claude Vision")

        # Build game context
        game_context = ""
        if game_info:
            away = game_info.get("away_team", "Unknown")
            home = game_info.get("home_team", "Unknown")
            game_context = f"Game: {away} @ {home}\n"

        # Process plays in batches
        all_detections = {}
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Split plays into batches
        play_batches = []
        for i in range(0, len(cfbd_plays), batch_size):
            play_batches.append(cfbd_plays[i:i + batch_size])

        logger.info(f"[VISION MAPPER] Processing {len(play_batches)} batches of plays")

        for batch_idx, play_batch in enumerate(play_batches):
            logger.info(f"\n[BATCH {batch_idx + 1}/{len(play_batches)}] Processing {len(play_batch)} plays...")

            try:
                # Call Claude Vision for this batch
                detections, metrics = await self._detect_plays_batch(
                    frame_data,
                    play_batch,
                    game_context,
                    batch_idx
                )

                # Merge results
                all_detections.update(detections)

                # Track metrics
                total_input_tokens += metrics.get("input_tokens", 0)
                total_output_tokens += metrics.get("output_tokens", 0)
                total_cost += metrics.get("cost", 0.0)

                logger.info(f"[BATCH {batch_idx + 1}] Detected {len(detections)} plays")
                logger.info(f"  Input tokens: {metrics.get('input_tokens', 0)}")
                logger.info(f"  Output tokens: {metrics.get('output_tokens', 0)}")
                logger.info(f"  Estimated cost: ${metrics.get('cost', 0.0):.4f}")

            except Exception as e:
                logger.error(f"[BATCH {batch_idx + 1}] Failed to process batch: {e}")
                continue

        # Log final statistics
        logger.info("\n" + "="*80)
        logger.info("[VISION MAPPER] Detection complete")
        logger.info("="*80)
        logger.info(f"  Total plays requested: {len(cfbd_plays)}")
        logger.info(f"  Total plays detected: {len(all_detections)}")
        logger.info(f"  Detection rate: {len(all_detections)/len(cfbd_plays)*100:.1f}%")
        logger.info(f"  Total input tokens: {total_input_tokens:,}")
        logger.info(f"  Total output tokens: {total_output_tokens:,}")
        logger.info(f"  Total estimated cost: ${total_cost:.2f}")
        logger.info("="*80 + "\n")

        return all_detections

    async def _detect_plays_batch(
        self,
        frame_data: List[Dict],
        plays: List[Dict],
        game_context: str,
        batch_idx: int
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict]:
        """Process a batch of plays with Claude Vision.

        Args:
            frame_data: List of frame data dictionaries
            plays: List of CFBD plays to detect in this batch
            game_context: Game context string
            batch_idx: Index of this batch (for logging)

        Returns:
            Tuple of (detections dict, metrics dict)
        """
        # Build frame timeline for prompt
        frame_timeline = "\n".join([
            f"Frame {i}: {frame['timestamp']:.1f}s ({frame['timestamp']/60:.1f} min)"
            for i, frame in enumerate(frame_data)
        ])

        # Build play list for prompt
        play_list = "\n".join([
            f"Play #{play['play_number']}: Q{play.get('quarter', '?')} "
            f"{play.get('clock_minutes', '?')}:{play.get('clock_seconds', 0):02d} - "
            f"{play.get('play_text', 'Unknown play')}"
            for play in plays
        ])

        # Build prompt
        video_duration = frame_data[-1]["timestamp"] if frame_data else 0
        prompt = f"""You are analyzing a college football game video to identify when specific plays occur.

{game_context}
Video duration: {video_duration:.1f} seconds (~{video_duration/60:.0f} minutes)

I'm providing you with {len(frame_data)} frames sampled every ~{frame_data[1]['timestamp'] - frame_data[0]['timestamp']:.1f}s if len(frame_data) > 1 else 'N/A'} from the video:

{frame_timeline}

TASK: Find the following {len(plays)} plays in the video and report their START and END timestamps:

{play_list}

For each play:
1. Look through the frames to understand the game progression
2. Identify where the play occurs (you may need to infer between frames)
3. Report the START timestamp (when the play begins - typically the snap)
4. Report the END timestamp (when the play ends - whistle, tackle, completion, TD, etc.)
5. Provide a confidence level (high/medium/low)

IMPORTANT:
- Start times should be when the play BEGINS (snap, kickoff, etc.)
- End times should be when the play ENDS (whistle, tackle, etc.)
- If a play occurs between two frames, estimate the timestamp based on the frame context
- Only report plays you can confidently identify (confidence must be at least "medium")

RESPOND WITH ONLY A JSON OBJECT:
{{
  "detected_plays": [
    {{
      "play_number": 1,
      "start_time": 45.2,
      "end_time": 58.5,
      "confidence": "high",
      "notes": "23-yard pass completion visible in frames 4-5"
    }},
    ...
  ]
}}

Only include plays you can identify with at least medium confidence."""

        # Prepare messages for Claude
        frame_messages = [frame["source"] for frame in frame_data]

        try:
            logger.info(f"[DEBUG] [BATCH {batch_idx + 1}] ===== CALLING CLAUDE VISION API =====")
            logger.info(f"[BATCH {batch_idx + 1}] Calling Claude Vision API...")
            logger.info(f"  Frames: {len(frame_messages)}")
            logger.info(f"  Plays: {len(plays)}")
            logger.info(f"  Prompt length: {len(prompt)} chars")
            logger.info(f"  Model: {self.model}")
            logger.info(f"  Max tokens: 8000")
            logger.info(f"[DEBUG] [BATCH {batch_idx + 1}] Client type: {type(self.client)}")
            logger.info(f"[DEBUG] [BATCH {batch_idx + 1}] Client initialized: {self.client is not None}")

            start_time = time.time()

            logger.info(f"[DEBUG] [BATCH {batch_idx + 1}] Making API request now...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *frame_messages,
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            elapsed = time.time() - start_time

            # Extract response
            response_text = response.content[0].text.strip()

            logger.info(f"[DEBUG] [BATCH {batch_idx + 1}] ✓ API call succeeded!")
            logger.info(f"[BATCH {batch_idx + 1}] API call completed in {elapsed:.1f}s")
            logger.info(f"  Response length: {len(response_text)} chars")
            logger.info(f"  Response preview: {response_text[:200]}...")

            # Parse response
            cleaned_response = self.clean_json_response(response_text)
            result = json.loads(cleaned_response)

            detected_plays_list = result.get("detected_plays", [])

            # Convert to dictionary mapping
            detections = {}
            for play in detected_plays_list:
                play_number = play.get("play_number")
                start_time = play.get("start_time")
                end_time = play.get("end_time")
                confidence = play.get("confidence", "low")

                # Only accept medium or high confidence
                if confidence not in ["medium", "high"]:
                    continue

                if play_number and start_time is not None and end_time is not None:
                    detections[play_number] = (float(start_time), float(end_time))
                    logger.info(f"  Play #{play_number}: {start_time:.1f}s - {end_time:.1f}s ({confidence} confidence)")

            # Calculate cost (rough estimate)
            # Claude Opus 4: $15/MTok input, $75/MTok output
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            cost = (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)

            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "elapsed": elapsed
            }

            return detections, metrics

        except json.JSONDecodeError as e:
            logger.error(f"[DEBUG] [BATCH {batch_idx + 1}] ❌ JSON PARSE FAILED")
            logger.error(f"[BATCH {batch_idx + 1}] Failed to parse JSON response: {e}")
            logger.error(f"  Raw response: {response_text[:500]}")
            logger.error(f"[DEBUG] JSONDecodeError type: {type(e).__name__}")
            logger.error(f"[DEBUG] JSONDecodeError: {e}")
            return {}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "elapsed": 0.0}
        except Exception as e:
            logger.error(f"[DEBUG] [BATCH {batch_idx + 1}] ❌ API CALL FAILED")
            logger.error(f"[BATCH {batch_idx + 1}] API call failed: {e}")
            logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG] Exception details: {e}")
            import traceback
            logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return {}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "elapsed": 0.0}

    def validate_timestamps(
        self,
        detections: Dict[int, Tuple[float, float]],
        video_duration: float,
        cfbd_plays: List[Dict]
    ) -> Dict[int, Tuple[float, float]]:
        """Validate detected timestamps and fix common issues.

        Args:
            detections: Dictionary mapping play_number -> (start, end)
            video_duration: Video duration in seconds
            cfbd_plays: Original CFBD plays for context

        Returns:
            Validated dictionary mapping play_number -> (start, end)
        """
        logger.info("[VISION MAPPER] Validating timestamps...")

        validated = {}
        issues = []

        for play_number, (start_time, end_time) in detections.items():
            # Check if start < end
            if start_time >= end_time:
                issues.append(f"Play #{play_number}: start >= end ({start_time:.1f}s >= {end_time:.1f}s)")
                continue

            # Check if within video duration
            if start_time < 0:
                issues.append(f"Play #{play_number}: start < 0 ({start_time:.1f}s)")
                start_time = 0

            if end_time > video_duration:
                issues.append(f"Play #{play_number}: end > duration ({end_time:.1f}s > {video_duration:.1f}s)")
                end_time = video_duration

            # Check duration is reasonable (most plays are 5-60 seconds)
            duration = end_time - start_time
            if duration < 3:
                issues.append(f"Play #{play_number}: duration too short ({duration:.1f}s)")
            elif duration > 120:
                issues.append(f"Play #{play_number}: duration too long ({duration:.1f}s)")

            validated[play_number] = (start_time, end_time)

        if issues:
            logger.warning("[VISION MAPPER] Timestamp validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        logger.info(f"[VISION MAPPER] Validated {len(validated)}/{len(detections)} timestamps")

        return validated
