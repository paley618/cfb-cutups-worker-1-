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

            self.client = anthropic.Anthropic(api_key=api_key, timeout=300.0)
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

        extraction_start_time = time.time()
        duration = self.get_video_duration(video_path)

        # Calculate frame timestamps
        num_frames = int(duration / interval_seconds)
        if max_frames and num_frames > max_frames:
            num_frames = max_frames
            interval_seconds = duration / num_frames
            logger.info(f"  Limiting to {max_frames} frames, adjusted interval to {interval_seconds:.1f}s")

        logger.info(f"[VISION MAPPER] Frame extraction configuration:")
        logger.info(f"  Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        logger.info(f"  Expected frames: {num_frames}")
        logger.info(f"  Frame interval: every {interval_seconds:.1f} seconds")
        logger.info(f"  Frame range: 0.0s to {(num_frames-1)*interval_seconds:.1f}s")
        logger.info(f"[VISION MAPPER] ⏱️  Starting frame extraction at {time.strftime('%H:%M:%S')}")

        frames: List[Tuple[bytes, float]] = []
        total_frame_bytes = 0

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
                        total_frame_bytes += len(frame_bytes)

                    # Progress checkpoint every 50 frames
                    if (i + 1) % 50 == 0:
                        elapsed = time.time() - extraction_start_time
                        pct = (i + 1) / num_frames * 100
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        eta = (num_frames - (i + 1)) / rate if rate > 0 else 0
                        logger.info(
                            f"[VISION MAPPER] [FRAME EXTRACTION] Progress: {i + 1}/{num_frames} ({pct:.1f}%) | "
                            f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} frames/sec | ETA: {eta:.1f}s"
                        )

                    # Heavier checkpoint every 200 frames with memory estimate
                    if (i + 1) % 200 == 0:
                        mem_mb = total_frame_bytes / (1024 * 1024)
                        logger.info(f"[VISION MAPPER] [FRAME EXTRACTION] Checkpoint at {i + 1} frames - Memory used: ~{mem_mb:.1f}MB")

                except subprocess.TimeoutExpired:
                    logger.warning(f"[VISION MAPPER] ⚠️  Timeout extracting frame at {timestamp:.1f}s")
                except Exception as e:
                    logger.warning(f"[VISION MAPPER] ⚠️  Failed to extract frame at {timestamp:.1f}s: {e}")

        extraction_elapsed = time.time() - extraction_start_time
        avg_rate = len(frames) / extraction_elapsed if extraction_elapsed > 0 else 0
        mem_mb = total_frame_bytes / (1024 * 1024)

        logger.info(f"[VISION MAPPER] ✓ Frame extraction COMPLETE!")
        logger.info(f"  Frames extracted: {len(frames)} (expected {num_frames})")
        logger.info(f"  Total time: {extraction_elapsed:.1f}s ({extraction_elapsed/60:.1f} minutes)")
        logger.info(f"  Average rate: {avg_rate:.2f} frames/second")
        logger.info(f"  Estimated memory: ~{mem_mb:.1f}MB")

        # CRITICAL CHECK: If 0 frames extracted, log error immediately
        if len(frames) == 0:
            logger.error(f"[VISION MAPPER] ❌ CRITICAL: No frames were extracted!")
            logger.error(f"  Video path: {video_path}")
            logger.error(f"  Video duration: {duration:.1f}s")
            logger.error(f"  Expected frames: {num_frames}")
            logger.error(f"  This will cause the Vision mapper to fail!")

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
        function_start_time = time.time()

        logger.info("="*80)
        logger.info("[VISION MAPPER] ===== STARTING MAP_PLAYS_TO_TIMESTAMPS =====")
        logger.info("="*80)
        logger.info(f"[VISION MAPPER] Function: map_plays_to_timestamps")
        logger.info(f"[VISION MAPPER] Input parameters:")
        logger.info(f"  Video path: {video_path}")
        logger.info(f"  CFBD plays received: {len(cfbd_plays)}")
        logger.info(f"  Frame interval: {frame_interval}s")
        logger.info(f"  Batch size: {batch_size} plays per API call")
        logger.info(f"  Max frames: {max_frames if max_frames else 'unlimited'}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  API client exists: {self.client is not None}")

        if game_info:
            logger.info(f"[VISION MAPPER] Game context:")
            logger.info(f"  Away team: {game_info.get('away_team', 'Unknown')}")
            logger.info(f"  Home team: {game_info.get('home_team', 'Unknown')}")

        logger.info(f"[VISION MAPPER] Starting execution at {time.strftime('%H:%M:%S')}")
        logger.info("="*80)

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
        logger.info(f"[VISION MAPPER] ===== BATCH PREPARATION =====")
        logger.info(f"[VISION MAPPER] Total plays to process: {len(cfbd_plays)}")
        logger.info(f"[VISION MAPPER] Batch size setting: {batch_size} plays per batch")

        all_detections = {}
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Split plays into batches
        play_batches = []
        for i in range(0, len(cfbd_plays), batch_size):
            play_batches.append(cfbd_plays[i:i + batch_size])

        logger.info(f"[VISION MAPPER] Number of batches created: {len(play_batches)}")
        logger.info(f"[VISION MAPPER] Batch distribution: {[len(batch) for batch in play_batches]}")
        logger.info(f"[VISION MAPPER] ===== STARTING BATCH PROCESSING =====")
        logger.info("")

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
        function_elapsed = time.time() - function_start_time
        detection_rate = (len(all_detections) / len(cfbd_plays) * 100) if len(cfbd_plays) > 0 else 0

        logger.info("\n" + "="*80)
        logger.info("[VISION MAPPER] ===== DETECTION COMPLETE =====")
        logger.info("="*80)
        logger.info(f"[VISION MAPPER] FINAL RESULTS SUMMARY:")
        logger.info(f"  Total execution time: {function_elapsed:.1f}s ({function_elapsed/60:.1f} minutes)")
        logger.info(f"  Total plays input (from CFBD): {len(cfbd_plays)}")
        logger.info(f"  Total plays detected: {len(all_detections)}")
        logger.info(f"  Detection rate: {detection_rate:.1f}%")
        logger.info(f"  Total input tokens: {total_input_tokens:,}")
        logger.info(f"  Total output tokens: {total_output_tokens:,}")
        logger.info(f"  Total estimated cost: ${total_cost:.2f}")

        # WARNING if 0 plays detected
        if len(all_detections) == 0:
            logger.warning("="*80)
            logger.warning("[VISION MAPPER] ⚠️  WARNING: DETECTED 0 PLAYS - WILL USE FALLBACK")
            logger.warning("="*80)
            logger.warning(f"  Input plays: {len(cfbd_plays)}")
            logger.warning(f"  Batches processed: {len(play_batches)}")
            logger.warning(f"  Frames extracted: {len(frames)}")
            logger.warning(f"  This indicates Vision Play Mapper failed completely")
            logger.warning("="*80)
        else:
            # List detected play IDs
            detected_ids = sorted(all_detections.keys())
            logger.info(f"[VISION MAPPER] Detected play IDs: {detected_ids}")

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

        # Build enhanced play list with CFBD data
        video_duration = frame_data[-1]["timestamp"] if frame_data else 0
        play_descriptions = []

        for play in plays:
            quarter = play.get('quarter', 1)
            clock_minutes = play.get('clock_minutes', 0)
            clock_seconds = play.get('clock_seconds', 0)

            # Calculate expected location in video
            # Each quarter is ~25% of the video
            # Convert game clock to seconds into quarter (counting down from 15:00)
            quarter_duration = video_duration / 4
            game_clock_seconds = clock_minutes * 60 + clock_seconds
            seconds_into_quarter = (15 * 60) - game_clock_seconds  # Clock counts down

            expected_video_time = ((quarter - 1) * quarter_duration) + (seconds_into_quarter / (15 * 60) * quarter_duration)
            expected_pct = (expected_video_time / video_duration) * 100 if video_duration > 0 else 0

            # Build rich play description
            play_type = play.get('play_type', 'Unknown')
            description = play.get('play_text', play_type)
            offense_team = play.get('offense', play.get('posteam', 'Unknown'))
            defense_team = play.get('defense', play.get('defteam', 'Unknown'))

            play_desc = f"""
Play #{play.get('play_number', '?')}:
  Type: {play_type}
  Description: {description}
  Offense: {offense_team}
  Defense: {defense_team}
  Quarter: Q{quarter}
  Game Clock: {clock_minutes}:{clock_seconds:02d}
  Expected Location: ~{expected_pct:.0f}% through video (~{expected_video_time:.0f}s)
"""
            play_descriptions.append(play_desc)

        play_list = "\n".join(play_descriptions)

        # Build prompt - CFBD-guided location (not blind detection)
        # Calculate frame interval for prompt
        frame_interval_str = f"{frame_data[1]['timestamp'] - frame_data[0]['timestamp']:.1f}s" if len(frame_data) > 1 else "N/A"

        prompt = f"""You are analyzing a college football game video.
I have official play-by-play data from this game from the College Football Database (CFBD).
Your task is to find the exact timestamp when each play occurs in the video.

{game_context}
Video duration: {video_duration:.1f} seconds (~{video_duration/60:.0f} minutes)

I'm providing you with {len(frame_data)} frames sampled every ~{frame_interval_str} from the video:

{frame_timeline}

IMPORTANT: These are KNOWN plays from official game data. I need you to LOCATE them in the video, not detect unknown plays.

Here are the {len(plays)} plays that occurred in this game:

{play_list}

TASK: For each play above, look at the video frames and find when it occurs.

For each play:
1. Use the "Expected Location" as a starting point - the play should be near that timestamp
2. Look at frames around that location to identify the play visually
3. Find the START timestamp (when the play begins - typically the snap, kickoff, or start of action)
4. Find the END timestamp (when the play ends - whistle, tackle, completion, TD, out of bounds, etc.)
5. Provide a confidence level (high/medium/low)

GUIDELINES:
- Start times should be when the play BEGINS (snap, kickoff, etc.)
- End times should be when the play ENDS (whistle, tackle, etc.)
- If a play occurs between two frames, estimate the timestamp based on frame context
- Use the play description to identify the correct play (e.g., "Pass Touchdown" should show a QB throwing and receiver catching in end zone)
- The expected location is approximate - plays may be slightly before or after this time
- Only report plays you can confidently identify (confidence must be at least "medium")

RESPOND WITH ONLY A JSON OBJECT:
{{
  "detected_plays": [
    {{
      "play_number": 1,
      "start_time": 45.2,
      "end_time": 58.5,
      "confidence": "high",
      "notes": "Pass TD visible in frames 4-5, QB throws to WR in end zone"
    }},
    ...
  ]
}}

Only include plays you can identify with at least medium confidence."""

        # Prepare messages for Claude
        frame_messages = [frame["source"] for frame in frame_data]

        try:
            # Log plays we're looking for (preview first few)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Batch details:")
            logger.info(f"  Frames in batch: {len(frame_messages)}")
            logger.info(f"  Plays to detect: {len(plays)}")
            logger.info(f"  Prompt length: {len(prompt)} chars")

            # Preview play descriptions
            play_previews = []
            for i, play in enumerate(plays[:5]):  # First 5 plays
                play_desc = f"Play #{play.get('play_number', '?')}: {play.get('play_text', 'Unknown')[:60]}"
                play_previews.append(play_desc)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Play descriptions (first 5):")
            for preview in play_previews:
                logger.info(f"    {preview}")
            if len(plays) > 5:
                logger.info(f"    ... and {len(plays) - 5} more plays")

            # Estimate payload size
            frame_bytes_estimate = sum(len(f["data"]) for f in frame_data) if frame_data else 0
            frame_mb = frame_bytes_estimate / (1024 * 1024)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Estimated payload size: ~{frame_mb:.1f}MB")

            # CRITICAL MARKER: About to call API
            logger.info("="*80)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ⏱️  ABOUT TO CALL CLAUDE VISION API...")
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Timestamp: {time.strftime('%H:%M:%S')}")
            logger.info("="*80)

            start_time = time.time()
            api_call_start = time.time()

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

            api_call_elapsed = time.time() - api_call_start
            elapsed = time.time() - start_time

            # CRITICAL MARKER: Response received
            logger.info("="*80)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ✓ API RESPONSE RECEIVED in {api_call_elapsed:.1f}s")
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Timestamp: {time.strftime('%H:%M:%S')}")
            logger.info("="*80)

            # Extract response
            response_text = response.content[0].text.strip()

            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Response details:")
            logger.info(f"  Response length: {len(response_text)} chars")
            logger.info(f"  Response preview (first 300 chars):")
            logger.info(f"    {response_text[:300]}...")
            if len(response_text) > 300:
                logger.info(f"  Response preview (last 200 chars):")
                logger.info(f"    ...{response_text[-200:]}")

            # Check for common error messages
            error_indicators = ["cannot", "unable to", "error", "failed", "unclear", "not visible", "can't see"]
            response_lower = response_text.lower()
            for indicator in error_indicators:
                if indicator in response_lower:
                    logger.warning(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ⚠️  Response contains '{indicator}' - may indicate error")

            # Parse response - log the process
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Attempting to extract JSON from response...")
            cleaned_response = self.clean_json_response(response_text)

            if cleaned_response != response_text:
                logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ✓ JSON extracted ({len(cleaned_response)} chars)")
                logger.info(f"  Cleaned JSON preview: {cleaned_response[:200]}...")
            else:
                logger.warning(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ⚠️  No JSON extraction needed (response is already clean)")

            # Check if it looks like JSON
            if not (cleaned_response.startswith('{') or cleaned_response.startswith('[')):
                logger.error(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ❌ Response doesn't look like JSON!")
                logger.error(f"  Expected to start with '{{' or '[', got: {cleaned_response[:50]}")
                logger.error(f"  Full response:")
                logger.error(f"    {response_text}")
                raise ValueError("Response is not valid JSON")

            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Parsing JSON...")
            result = json.loads(cleaned_response)
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ✓ JSON parsed successfully!")

            detected_plays_list = result.get("detected_plays", [])
            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Found 'detected_plays' array with {len(detected_plays_list)} items")

            # Convert to dictionary mapping
            detections = {}
            skipped_low_confidence = 0

            for play in detected_plays_list:
                play_number = play.get("play_number")
                start_time = play.get("start_time")
                end_time = play.get("end_time")
                confidence = play.get("confidence", "low")

                # Only accept medium or high confidence
                if confidence not in ["medium", "high"]:
                    skipped_low_confidence += 1
                    continue

                if play_number and start_time is not None and end_time is not None:
                    detections[play_number] = (float(start_time), float(end_time))
                    logger.info(f"  ✓ Play #{play_number}: {start_time:.1f}s - {end_time:.1f}s ({confidence} confidence)")

            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ✓ Detected {len(detections)} plays!")
            if skipped_low_confidence > 0:
                logger.info(f"  Skipped {skipped_low_confidence} plays due to low confidence")

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

            logger.info(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Batch complete - returning {len(detections)} detections")

            return detections, metrics

        except json.JSONDecodeError as e:
            logger.error("="*80)
            logger.error(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ❌ JSON PARSING FAILED")
            logger.error("="*80)
            logger.error(f"[VISION MAPPER] [BATCH {batch_idx + 1}] JSONDecodeError: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {str(e)}")
            if hasattr(e, 'lineno') and hasattr(e, 'colno'):
                logger.error(f"  Error at line {e.lineno}, column {e.colno}")
            logger.error(f"  Attempted to parse cleaned response:")
            logger.error(f"    {cleaned_response[:500]}")
            if 'response_text' in locals():
                logger.error(f"  Original response text:")
                logger.error(f"    {response_text}")
            logger.error("="*80)
            return {}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "elapsed": 0.0}

        except Exception as e:
            logger.error("="*80)
            logger.error(f"[VISION MAPPER] [BATCH {batch_idx + 1}] ❌ API CALL OR PROCESSING FAILED")
            logger.error("="*80)
            logger.error(f"[VISION MAPPER] [BATCH {batch_idx + 1}] Exception: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {str(e)}")
            import traceback
            logger.error(f"  Full traceback:")
            logger.error(traceback.format_exc())
            logger.error("="*80)
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
