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
            logger.info(f"[CLAUDE] Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            return duration
        except Exception as e:
            logger.warning(f"[CLAUDE] Could not get video duration: {e}, using default 600s")
            return 600.0

    def detect_game_start(self, video_path: str, game_info: Optional[Dict] = None) -> float:
        """Detect when actual game footage begins (vs commercials/pregame).

        Args:
            video_path: Path to video file
            game_info: Optional dict with game context (away_team, home_team, etc.)

        Returns:
            Timestamp in seconds when game footage begins
        """
        logger.info("[GAME START DETECTION] Analyzing first 15 minutes of video...")

        # Extract 6 frames from first 15 minutes (0, 2.5, 5, 7.5, 10, 12.5 min)
        max_search_duration = 900.0  # 15 minutes
        num_sample_frames = 6

        sample_frames: List[tuple[bytes, float]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_sample_frames):
                time_sec = (i / (num_sample_frames - 1)) * max_search_duration if num_sample_frames > 1 else 0
                frame_path = f"{tmpdir}/game_start_frame_{i:02d}.jpg"

                cmd = [
                    "ffmpeg",
                    "-ss", str(time_sec),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    "-vf", "scale='min(1920,iw)':-2",
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
                        sample_frames.append((frame_bytes, time_sec))
                    logger.info(f"[GAME START DETECTION] Sampled frame at {time_sec:.1f}s ({time_sec/60:.1f}m)")
                except Exception as e:
                    logger.warning(f"[GAME START DETECTION] Failed to extract frame at {time_sec:.1f}s: {e}")

        if not sample_frames:
            logger.warning("[GAME START DETECTION] No frames extracted, assuming 10 min offset (600s)")
            return 600.0

        # Prepare frames for Claude
        frame_messages = []
        frame_times = []
        for frame_bytes, timestamp in sample_frames:
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

        # Build game context
        game_context = ""
        if game_info:
            away = game_info.get("away_team", "Unknown")
            home = game_info.get("home_team", "Unknown")
            game_context = f"Game: {away} @ {home}"

        # Create prompt for game start detection
        frame_time_info = "\n".join([
            f"Frame {i}: {t:.1f}s ({t/60:.1f} minutes)" for i, t in enumerate(frame_times)
        ])

        prompt = f"""You are analyzing {len(sample_frames)} frames from a college football broadcast to identify when the ACTUAL GAME FOOTAGE begins.

{game_context}

Frames sampled from first 15 minutes:
{frame_time_info}

TASK: Identify the FIRST frame that shows actual game footage (not commercials, not pregame coverage, not halftime).

Game footage indicators:
✓ Football field with yard line markings clearly visible
✓ Scoreboard showing both teams and a game clock
✓ Players in formation (kickoff, offensive/defensive sets)
✓ Active play in progress
✓ Game graphics/overlays from broadcaster

NOT game footage:
✗ Commercials (products, ads, promos)
✗ Studio coverage (analysts at desk)
✗ Pregame interviews
✗ Crowd shots without field visible
✗ Completely black/transition frames

RESPOND WITH ONLY A JSON OBJECT:
{{
  "game_start_frame_index": <index of first game frame, or -1 if not found>,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Example: {{"game_start_frame_index": 3, "confidence": 0.95, "reasoning": "Frame 3 shows kickoff formation with scoreboard"}}
"""

        try:
            logger.info(f"[GAME START DETECTION] Asking Claude to analyze {len(sample_frames)} frames...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
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
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            logger.info(f"[GAME START DETECTION] Claude response: {response_text}")

            # Parse response
            result = json.loads(response_text)
            game_start_frame = int(result.get("game_start_frame_index", -1))
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            if game_start_frame >= 0 and game_start_frame < len(frame_times):
                game_start_time = frame_times[game_start_frame]
                logger.info(f"[GAME START DETECTION] Game footage detected at {game_start_time:.1f}s ({game_start_time/60:.1f} min)")
                logger.info(f"[GAME START DETECTION] Confidence: {confidence:.2f}, Reasoning: {reasoning}")
                logger.info(f"[GAME START DETECTION] Will add {game_start_time:.1f}s offset to all play timestamps")
                return game_start_time
            else:
                logger.warning(f"[GAME START DETECTION] No game footage found in first 15 min, assuming 10 min offset (600s)")
                return 600.0

        except json.JSONDecodeError as e:
            logger.error(f"[GAME START DETECTION] Failed to parse Claude response: {e}")
            logger.error(f"[GAME START DETECTION] Response was: {response_text[:200]}")
            return 600.0
        except Exception as e:
            logger.error(f"[GAME START DETECTION] Error: {type(e).__name__}: {e}")
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
        logger.info(f"[CLAUDE DETECTION START]")
        logger.info(f"  Video path: {video_path}")

        # STEP 1: Detect game start offset
        game_start_offset = self.detect_game_start(video_path, game_info=game_info)

        # STEP 2: Get video duration
        video_duration = self.get_video_duration(video_path)
        logger.info(f"  Video duration: {video_duration:.1f}s ({video_duration/60:.1f} minutes)")
        logger.info(f"  Game start offset: {game_start_offset:.1f}s ({game_start_offset/60:.1f} minutes)")

        # STEP 3: Extract keyframes
        logger.info(f"  Frames to extract: {num_frames}")
        logger.info(f"  Frame interval: ~{video_duration/num_frames:.1f}s between frames")

        keyframes = self.extract_keyframes(video_path, num_frames=num_frames)

        if not keyframes:
            logger.error("[CLAUDE] No keyframes extracted!")
            return []

        logger.info(f"[FRAME EXTRACTION] Successfully extracted {len(keyframes)} frames")
        for i, (_, timestamp) in enumerate(keyframes):
            logger.info(f"  Frame {i}: {timestamp:.1f}s ({timestamp/60:.1f}m)")

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
            logger.info(f"[CLAUDE API CALL]")
            logger.info(f"  Model: {self.model}")
            logger.info(f"  Max tokens: 6000")
            logger.info(f"  Frames to analyze: {len(keyframes)}")
            logger.info(f"  Prompt length: {len(prompt)} chars")
            if game_info:
                logger.info(f"  Game context: {game_info}")

            # Time the API call
            import time
            start = time.time()

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

            elapsed = time.time() - start

            response_text = response.content[0].text.strip()
            logger.info(f"[CLAUDE API RESPONSE]")
            logger.info(f"  Time elapsed: {elapsed:.1f}s")
            logger.info(f"  Response length: {len(response_text)} chars")
            if hasattr(response, 'usage'):
                logger.info(f"  Input tokens: {response.usage.input_tokens}")
                logger.info(f"  Output tokens: {response.usage.output_tokens}")
            logger.info(f"  Response preview: {response_text[:200]}...")

            # Parse JSON response
            logger.info(f"[CLAUDE RESPONSE PARSING]")
            detected_plays = json.loads(response_text)
            logger.info(f"  JSON parsed successfully")
            logger.info(f"  Total plays detected: {len(detected_plays)}")

            # Analyze detected plays BEFORE converting
            play_types_count = {}
            confidence_stats = {"min": 1.0, "max": 0.0, "sum": 0.0}
            frames_with_plays = set()
            low_confidence_count = 0

            for play in detected_plays:
                play_type = play.get("play_type", "Unknown")
                play_types_count[play_type] = play_types_count.get(play_type, 0) + 1

                confidence = play.get("confidence", 0.0)
                confidence_stats["min"] = min(confidence_stats["min"], confidence)
                confidence_stats["max"] = max(confidence_stats["max"], confidence)
                confidence_stats["sum"] += confidence

                if confidence < 0.5:
                    low_confidence_count += 1

                frame_idx = play.get("frame_index", -1)
                if frame_idx >= 0:
                    frames_with_plays.add(frame_idx)

            avg_confidence = confidence_stats["sum"] / len(detected_plays) if detected_plays else 0

            logger.info(f"[DETECTION ANALYSIS]")
            logger.info(f"  Play type distribution: {play_types_count}")
            logger.info(f"  Confidence stats: min={confidence_stats['min']:.2f}, max={confidence_stats['max']:.2f}, avg={avg_confidence:.2f}")
            logger.info(f"  Frames with plays: {sorted(frames_with_plays)} ({len(frames_with_plays)} of {len(keyframes)})")
            logger.info(f"  Frames without plays: {len(keyframes) - len(frames_with_plays)}")
            logger.info(f"  Low confidence plays (< 0.5): {low_confidence_count}")

            # Convert to (start, end) tuples with timestamps + GAME START OFFSET
            play_windows: List[tuple[float, float]] = []
            skipped_low_conf = 0
            skipped_invalid_frame = 0

            logger.info(f"[TIMESTAMP CONVERSION] Converting {len(detected_plays)} plays to time windows...")
            logger.info(f"  Applying game start offset: {game_start_offset:.1f}s")

            for i, play in enumerate(detected_plays):
                try:
                    frame_idx = int(play.get("frame_index", 0))
                    confidence = float(play.get("confidence", 0.0))
                    play_type = play.get("play_type", "Unknown")
                    description = play.get("description", "")

                    # Skip low confidence plays
                    if confidence < 0.5:
                        skipped_low_conf += 1
                        logger.debug(f"[PLAY {i}] SKIPPED: Low confidence {confidence:.2f}")
                        continue

                    # Get timestamp for this frame
                    if 0 <= frame_idx < len(frame_times):
                        # Get the base timestamp from the frame
                        frame_time = frame_times[frame_idx]

                        # Apply game start offset to center the play window
                        center_time = game_start_offset + frame_time

                        # Create window around detected play (3s before, 5s after)
                        start_time = max(0.0, center_time - 3.0)
                        end_time = min(video_duration, center_time + 5.0)

                        play_windows.append((start_time, end_time))
                        logger.info(
                            f"[PLAY {i}] {play_type} @ frame {frame_idx} -> {center_time:.1f}s "
                            f"(window: {start_time:.1f}-{end_time:.1f}s, confidence: {confidence:.2f})"
                        )
                        logger.debug(f"  Description: {description}")
                    else:
                        skipped_invalid_frame += 1
                        logger.warning(f"[PLAY {i}] SKIPPED: Invalid frame_index {frame_idx} (max: {len(frame_times)-1})")

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"[PLAY {i}] SKIPPED: Error processing play entry: {e}")
                    continue

            logger.info(f"[CONVERSION SUMMARY]")
            logger.info(f"  Total plays from Claude: {len(detected_plays)}")
            logger.info(f"  Skipped (low confidence < 0.5): {skipped_low_conf}")
            logger.info(f"  Skipped (invalid frame index): {skipped_invalid_frame}")
            logger.info(f"  Play windows created: {len(play_windows)}")

            logger.info(f"[DETECTION COMPLETE]")
            logger.info(f"  Detected plays: {len(detected_plays)}")
            logger.info(f"  Play windows ready for clipping: {len(play_windows)}")

            return play_windows

        except json.JSONDecodeError as e:
            logger.error(f"[CLAUDE] Failed to parse Claude response as JSON: {e}")
            logger.error(f"[CLAUDE] Response was: {response_text[:500]}")
            return []
        except Exception as e:
            logger.error(f"[CLAUDE] Error calling Claude API: {type(e).__name__}: {e}")
            return []
