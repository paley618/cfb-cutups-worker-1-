"""Claude Vision API play detector for college football videos."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaudePlayDetector:
    """Use Claude's vision API to detect plays from video keyframes."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. If None, will try to import from environment.
        """
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-opus-4-20250514"  # Latest Opus model with vision
            logger.info("[CLAUDE] Initialized Claude Play Detector")
        except ImportError as e:
            logger.error(f"[CLAUDE] Failed to import anthropic library: {e}")
            raise
        except Exception as e:
            logger.error(f"[CLAUDE] Failed to initialize Claude client: {e}")
            raise

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds, or 600.0 as default if detection fails
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
            logger.info(f"[CLAUDE] Video duration: {duration:.1f}s")
            return duration
        except Exception as e:
            logger.warning(f"[CLAUDE] Could not get video duration: {e}, using default 600s")
            return 600.0

    def extract_keyframes(self, video_path: str, num_frames: int = 12) -> List[tuple[bytes, float]]:
        """Extract keyframes from video at regular intervals.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 12)

        Returns:
            List of (frame_bytes, timestamp) tuples
        """
        logger.info(f"[CLAUDE] Extracting {num_frames} keyframes from video...")

        duration = self.get_video_duration(video_path)
        keyframes: List[tuple[bytes, float]] = []

        # Extract frames at regular intervals throughout video
        interval = duration / (num_frames + 1)  # +1 to avoid very end
        times = [interval * (i + 1) for i in range(num_frames)]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, time_sec in enumerate(times):
                frame_path = f"{tmpdir}/frame_{i:04d}.jpg"

                cmd = [
                    "ffmpeg",
                    "-ss", str(time_sec),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",  # High quality
                    "-vf", "scale='min(1920,iw)':-2",  # Limit width to 1920px
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
                        keyframes.append((frame_bytes, time_sec))
                    logger.info(f"[CLAUDE] Extracted frame {i+1}/{num_frames} at {time_sec:.1f}s")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[CLAUDE] Timeout extracting frame {i} at {time_sec:.1f}s")
                except Exception as e:
                    logger.warning(f"[CLAUDE] Failed to extract frame {i} at {time_sec:.1f}s: {e}")

        logger.info(f"[CLAUDE] Successfully extracted {len(keyframes)} keyframes")
        return keyframes

    def detect_plays(
        self,
        video_path: str,
        game_info: Optional[Dict] = None,
        num_frames: int = 12
    ) -> List[tuple[float, float]]:
        """Analyze video frames with Claude to detect plays.

        Args:
            video_path: Path to video file
            game_info: Optional dict with game context (away_team, home_team, etc.)
            num_frames: Number of keyframes to extract and analyze

        Returns:
            List of (start_time, end_time) tuples representing detected plays
        """
        logger.info(f"[CLAUDE] Starting play detection for {video_path}")

        # Extract keyframes
        keyframes = self.extract_keyframes(video_path, num_frames=num_frames)

        if not keyframes:
            logger.error("[CLAUDE] No keyframes extracted!")
            return []

        video_duration = self.get_video_duration(video_path)

        # Prepare frames for Claude
        frame_messages = []
        frame_times = []
        for frame_bytes, timestamp in keyframes:
            base64_frame = base64.standard_b64encode(frame_bytes).decode("utf-8")
            frame_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_frame,
                },
            })
            frame_times.append(timestamp)

        # Build prompt with game context
        game_context = ""
        if game_info:
            away = game_info.get("away_team", "Unknown")
            home = game_info.get("home_team", "Unknown")
            game_context = f"""
Game Context:
- Away Team: {away}
- Home Team: {home}
"""

        # Map frame indices to actual timestamps for the prompt
        frame_time_info = "\n".join([
            f"Frame {i}: {t:.1f}s" for i, t in enumerate(frame_times)
        ])

        prompt = f"""Analyze these {len(keyframes)} frames from a college football game video and identify actual plays.

{game_context}
Video duration: {video_duration:.1f} seconds
Frame timestamps:
{frame_time_info}

For each ACTUAL PLAY you can identify (not replays, commercials, timeouts, or celebrations), provide:
1. frame_index: Which frame number (0-{len(keyframes)-1}) shows the play
2. play_type: One of these EXACT types:
   - "Rush" (running play)
   - "Pass Reception" (completed pass)
   - "Pass Incompletion" (incomplete pass)
   - "Sack"
   - "Interception Return"
   - "Fumble Recovery (Opponent)"
   - "Punt"
   - "Kickoff"
   - "Field Goal Good"
   - "Field Goal Missed"
   - "Passing Touchdown"
   - "Rushing Touchdown"
   - "Penalty"
   - "Safety"
3. description: Brief description of what you see
4. confidence: 0.0-1.0 (only include plays with confidence > 0.6)

CRITICAL RULES:
- Only report plays where you can SEE the ball in motion or immediate action
- DO NOT report replays, commercials, halftime shows, or celebration footage
- DO NOT report timeouts, huddles, or between-play footage
- Only use the EXACT play_type names listed above
- Each play should have a clear visual indicator of football action

Respond with ONLY a JSON array, no other text:

[
  {{"frame_index": 2, "play_type": "Pass Reception", "description": "QB completes pass to WR", "confidence": 0.9}},
  {{"frame_index": 5, "play_type": "Rush", "description": "RB runs up middle", "confidence": 0.85}}
]

If you see no clear plays, return: []
"""

        try:
            logger.info("[CLAUDE] Sending frames to Claude for analysis...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.0,  # Deterministic for consistent results
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *frame_messages,
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            logger.info(f"[CLAUDE] Claude response ({len(response_text)} chars): {response_text[:200]}...")

            # Parse JSON response
            detected_plays = json.loads(response_text)
            logger.info(f"[CLAUDE] Parsed {len(detected_plays)} plays from response")

            # Convert to (start, end) tuples with timestamps
            play_windows: List[tuple[float, float]] = []
            for play in detected_plays:
                try:
                    frame_idx = int(play.get("frame_index", 0))
                    confidence = float(play.get("confidence", 0.0))
                    play_type = play.get("play_type", "Unknown")
                    description = play.get("description", "")

                    # Skip low confidence plays
                    if confidence < 0.6:
                        logger.debug(f"[CLAUDE] Skipping low confidence play: {confidence}")
                        continue

                    # Get timestamp for this frame
                    if 0 <= frame_idx < len(frame_times):
                        center_time = frame_times[frame_idx]

                        # Create window around detected play (3s before, 5s after)
                        start_time = max(0.0, center_time - 3.0)
                        end_time = min(video_duration, center_time + 5.0)

                        play_windows.append((start_time, end_time))
                        logger.info(
                            f"[CLAUDE] Play detected: {play_type} at {center_time:.1f}s "
                            f"(confidence: {confidence:.2f}) - {description}"
                        )
                    else:
                        logger.warning(f"[CLAUDE] Invalid frame_index {frame_idx}")

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"[CLAUDE] Error processing play entry: {e}")
                    continue

            logger.info(f"[CLAUDE] Returning {len(play_windows)} play windows")
            return play_windows

        except json.JSONDecodeError as e:
            logger.error(f"[CLAUDE] Failed to parse Claude response as JSON: {e}")
            logger.error(f"[CLAUDE] Response was: {response_text[:500]}")
            return []
        except Exception as e:
            logger.error(f"[CLAUDE] Error calling Claude API: {type(e).__name__}: {e}")
            return []
