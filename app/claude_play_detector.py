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

    def extract_keyframes(self, video_path: str, num_frames: int = 60) -> List[tuple[bytes, float]]:
        """Extract keyframes from video at regular intervals.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 60, ~1 every 3 minutes in a 3-hour game)

        Returns:
            List of (frame_bytes, timestamp) tuples
        """
        logger.info(f"[CLAUDE] Extracting {num_frames} keyframes from video...")

        duration = self.get_video_duration(video_path)
        keyframes: List[tuple[bytes, float]] = []

        # Extract frames at regular intervals throughout video
        interval = duration / (num_frames + 1)  # +1 to avoid very end
        times = [interval * (i + 1) for i in range(num_frames)]
        logger.info(f"[CLAUDE] Extracting frames at {interval:.1f}s intervals (duration: {duration:.1f}s)")

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

        prompt = f"""You are analyzing {len(keyframes)} frames from a college football game video to identify EVERY play that occurred.

{game_context}
Video duration: {video_duration:.1f} seconds (~{video_duration/60:.0f} minutes)

IMPORTANT: Each frame is from a DIFFERENT timestamp in the game:
{frame_time_info}

TASK: Identify ALL football plays across these frames. In a typical college football game, you should find 50-200+ plays across {len(keyframes)} frames.

For EACH play you identify, provide:
1. frame_index: Which frame number (0-{len(keyframes)-1}) shows the play
2. play_type: One of these EXACT types:
   - "Rush" - Running play (RB, QB run, option)
   - "Pass Reception" - Completed pass
   - "Pass Incompletion" - Incomplete pass
   - "Sack" - QB sacked
   - "Interception Return" - INT by defense
   - "Fumble Recovery (Opponent)" - Fumble recovered
   - "Punt" - Punting play
   - "Kickoff" - Kickoff play
   - "Field Goal Good" - Successful FG
   - "Field Goal Missed" - Missed FG
   - "Passing Touchdown" - TD via pass
   - "Rushing Touchdown" - TD via rush
   - "Penalty" - Penalty flag/enforcement
   - "Safety" - Safety scored
3. description: Brief description of what you see (e.g., "QB drops back, throws to WR on right side")
4. confidence: 0.0-1.0 (include plays with confidence > 0.5)

IMPORTANT GUIDELINES:
✓ Report ALL plays you can identify - don't filter or limit yourself
✓ Each frame may show MULTIPLE plays (during action, in scorebug recap, etc.)
✓ Look for: players in formation, ball carriers, passes in flight, tackles, scoring plays
✓ Include plays even if partially visible or in progress
✓ Use game context (score, down & distance indicators) to infer play types
✗ Skip obvious replays (slow-motion, different camera angles of same play)
✗ Skip commercials, halftime, crowd shots with no game action
✗ Skip pre-game/post-game ceremonies

CONFIDENCE LEVELS:
- 0.9-1.0: Clear, unambiguous play action visible
- 0.7-0.8: Strong indicators (formation, players in motion, scorebug update)
- 0.5-0.6: Reasonable inference from context (score change, field position)

RESPOND WITH ONLY A JSON ARRAY:
[
  {{"frame_index": 0, "play_type": "Kickoff", "description": "Opening kickoff, ball in air", "confidence": 0.95}},
  {{"frame_index": 0, "play_type": "Pass Reception", "description": "Scorebug shows 1st down completion", "confidence": 0.75}},
  {{"frame_index": 5, "play_type": "Rush", "description": "RB carrying ball, defenders converging", "confidence": 0.90}}
]

Remember: Report ALL plays you find. A college football game has many plays, and your job is to find as many as possible across these {len(keyframes)} sample frames.
"""

        try:
            logger.info("[CLAUDE] Sending frames to Claude for analysis...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=6000,  # Increased to handle 150+ plays
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
                    if confidence < 0.5:
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

            # Log detailed play distribution analytics
            play_counts_by_frame = {}
            play_types_count = {}
            for play in detected_plays:
                frame_idx = play.get("frame_index", -1)
                play_type = play.get("play_type", "Unknown")
                play_counts_by_frame[frame_idx] = play_counts_by_frame.get(frame_idx, 0) + 1
                play_types_count[play_type] = play_types_count.get(play_type, 0) + 1

            frames_with_plays = len([c for c in play_counts_by_frame.values() if c > 0])
            avg_plays_per_frame = len(detected_plays) / len(keyframes) if keyframes else 0

            logger.info(f"[CLAUDE] Returning {len(play_windows)} play windows")
            logger.info(f"[CLAUDE] Play distribution: {dict(sorted(play_counts_by_frame.items()))}")
            logger.info(f"[CLAUDE] Play types: {play_types_count}")
            logger.info(f"[CLAUDE] Frame coverage: {frames_with_plays}/{len(keyframes)} frames with plays")
            logger.info(f"[CLAUDE] Average: {avg_plays_per_frame:.1f} plays per frame")

            return play_windows

        except json.JSONDecodeError as e:
            logger.error(f"[CLAUDE] Failed to parse Claude response as JSON: {e}")
            logger.error(f"[CLAUDE] Response was: {response_text[:500]}")
            return []
        except Exception as e:
            logger.error(f"[CLAUDE] Error calling Claude API: {type(e).__name__}: {e}")
            return []
