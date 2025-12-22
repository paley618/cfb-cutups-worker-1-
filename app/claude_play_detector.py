"""Claude Vision API play detector for college football videos."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

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
            # Extract content between ```json and ```
            parts = cleaned.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0]
                cleaned = json_part.strip()
        # Handle plain markdown code blocks
        elif "```" in cleaned:
            # Extract content between first ``` and second ```
            parts = cleaned.split("```")
            if len(parts) >= 3:
                # Take the content between first and second ```
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

    async def validate_plays_with_espn(
        self,
        detected_plays: List[Dict],
        frame_times: List[float],
        espn_game_id: Optional[str] = None,
        team_name: Optional[str] = None
    ) -> Tuple[List[Dict], Dict[str, any]]:
        """Validate detected plays against ESPN game clock data.

        Args:
            detected_plays: List of plays detected by Claude
            frame_times: List of video timestamps for each frame
            espn_game_id: ESPN game ID for fetching play-by-play data
            team_name: Team name for filtering ESPN plays

        Returns:
            Tuple of (validated_plays, validation_stats)
        """
        logger.info("\n" + "="*80)
        logger.info("[ESPN GAME CLOCK SYNC] Starting validation")
        logger.info("="*80)

        # Step 2: Collect game clock sightings
        game_clock_sightings = []
        for play in detected_plays:
            game_clock = play.get('game_clock')
            quarter = play.get('quarter')
            frame_idx = play.get('frame_index', -1)

            if game_clock and quarter and 0 <= frame_idx < len(frame_times):
                video_timestamp = frame_times[frame_idx]
                game_seconds = self.clock_to_absolute_seconds(quarter, game_clock)

                if game_seconds is not None:
                    game_clock_sightings.append({
                        'frame_index': frame_idx,
                        'video_timestamp': video_timestamp,
                        'game_clock': game_clock,
                        'quarter': quarter,
                        'game_seconds': game_seconds
                    })
                    logger.info(f"[GAME CLOCK DETECTED] Frame {frame_idx} @ video {video_timestamp:.1f}s shows Q{quarter} {game_clock} (game time: {game_seconds:.1f}s)")

        logger.info(f"\n[GAME CLOCK COLLECTION] Found {len(game_clock_sightings)} clock sightings")

        if len(game_clock_sightings) < 3:
            logger.warning(f"[ESPN SYNC] Insufficient game clock sightings ({len(game_clock_sightings)}), skipping ESPN validation")
            return detected_plays, {
                'validated': 0,
                'rejected': 0,
                'no_clock': len(detected_plays),
                'reason': 'insufficient_clock_sightings'
            }

        # Step 3: Build video→game_clock mapping using linear regression
        import numpy as np

        video_times = np.array([s['video_timestamp'] for s in game_clock_sightings])
        game_times = np.array([s['game_seconds'] for s in game_clock_sightings])

        # Calculate linear regression: video_time = drift * game_time + offset
        # Rearrange: game_time = (video_time - offset) / drift
        A = np.vstack([game_times, np.ones(len(game_times))]).T
        drift, offset = np.linalg.lstsq(A, video_times, rcond=None)[0]

        logger.info(f"\n[MAPPING CALCULATED]")
        logger.info(f"  Drift: {drift:.6f} (video playback rate)")
        logger.info(f"  Offset: {offset:.1f}s (pre-game content)")
        logger.info(f"  Formula: video_time = {drift:.6f} * game_time + {offset:.1f}")

        # Calculate mapping quality (R² score)
        predicted_video = drift * game_times + offset
        ss_res = np.sum((video_times - predicted_video) ** 2)
        ss_tot = np.sum((video_times - np.mean(video_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        logger.info(f"  R² (mapping quality): {r_squared:.4f}")

        # Step 4 & 5: Fetch ESPN data and match plays
        espn_plays = []
        if espn_game_id and team_name:
            try:
                from .espn import fetch_offensive_play_times
                import httpx

                # Fetch full ESPN play-by-play data (not just timestamps)
                url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay?event={espn_game_id}"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url)
                response.raise_for_status()
                payload = response.json()

                # Extract plays with game clock information
                drives_payload = payload.get("drives") or {}
                drives = []
                previous_drives = drives_payload.get("previous") or []
                if isinstance(previous_drives, list):
                    drives.extend(previous_drives)
                current_drive = drives_payload.get("current")
                if isinstance(current_drive, dict):
                    drives.append(current_drive)

                for drive in drives:
                    if not isinstance(drive, dict):
                        continue
                    plays_list = drive.get("plays", [])
                    for play in plays_list:
                        if not isinstance(play, dict):
                            continue

                        # Get play details
                        clock = (play.get("clock") or {}).get("displayValue")
                        period = (play.get("period") or {}).get("number")
                        play_type = (play.get("type") or {}).get("text", "")

                        if clock and isinstance(period, int):
                            game_seconds = self.clock_to_absolute_seconds(period, clock)
                            if game_seconds is not None:
                                espn_plays.append({
                                    'quarter': period,
                                    'clock': clock,
                                    'game_seconds': game_seconds,
                                    'play_type': play_type,
                                    'text': play.get('text', '')
                                })

                logger.info(f"\n[ESPN DATA] Fetched {len(espn_plays)} plays from ESPN")
            except Exception as e:
                logger.warning(f"[ESPN SYNC] Failed to fetch ESPN data: {e}")
                return detected_plays, {
                    'validated': 0,
                    'rejected': 0,
                    'no_clock': len(detected_plays),
                    'reason': f'espn_fetch_failed: {e}'
                }

        if not espn_plays:
            logger.warning("[ESPN SYNC] No ESPN plays available, skipping validation")
            return detected_plays, {
                'validated': 0,
                'rejected': 0,
                'no_clock': len(detected_plays),
                'reason': 'no_espn_data'
            }

        # Step 5: Match Claude detections against ESPN data
        validated_plays = []
        rejected_plays = []
        no_clock_plays = []

        logger.info(f"\n[VALIDATION] Matching {len(detected_plays)} Claude plays against {len(espn_plays)} ESPN plays")

        for play_idx, play in enumerate(detected_plays):
            frame_idx = play.get('frame_index', -1)
            game_clock = play.get('game_clock')
            quarter = play.get('quarter')
            confidence = play.get('confidence', 0.0)
            play_type = play.get('play_type', '')

            # Skip low confidence plays
            if confidence < 0.5:
                continue

            # If no game clock detected, mark as no_clock
            if not game_clock or not quarter or frame_idx < 0 or frame_idx >= len(frame_times):
                no_clock_plays.append(play)
                logger.info(f"[PLAY {play_idx}] NO CLOCK - Cannot validate")
                continue

            # Convert video timestamp to game clock using mapping
            video_timestamp = frame_times[frame_idx]
            predicted_game_seconds = (video_timestamp - offset) / drift if drift != 0 else None

            if predicted_game_seconds is None:
                no_clock_plays.append(play)
                continue

            predicted_quarter, predicted_clock = self.quarter_clock_from_seconds(predicted_game_seconds)

            logger.info(f"\n[PLAY {play_idx}] Validation:")
            logger.info(f"  Video timestamp: {video_timestamp:.1f}s")
            logger.info(f"  Detected clock: Q{quarter} {game_clock}")
            logger.info(f"  Predicted clock: Q{predicted_quarter} {predicted_clock}")
            logger.info(f"  Play type: {play_type}")

            # Find matching ESPN play (same quarter, close clock time ±30s, similar play type)
            match_found = False
            for espn_play in espn_plays:
                espn_quarter = espn_play['quarter']
                espn_game_seconds = espn_play['game_seconds']
                espn_play_type = espn_play['play_type']

                # Check quarter match
                if espn_quarter != quarter:
                    continue

                # Check time proximity (±30 seconds)
                detected_game_seconds = self.clock_to_absolute_seconds(quarter, game_clock)
                if detected_game_seconds is None:
                    continue

                time_diff = abs(detected_game_seconds - espn_game_seconds)
                if time_diff > 30:  # 30 second tolerance
                    continue

                # Match found!
                match_found = True
                validated_plays.append(play)
                logger.info(f"  ✅ VALIDATED - Matched ESPN play at Q{espn_quarter} {espn_play['clock']} (Δ{time_diff:.1f}s)")
                logger.info(f"     ESPN: {espn_play_type}")
                break

            if not match_found:
                rejected_plays.append(play)
                logger.info(f"  ❌ REJECTED - No matching ESPN play found")

        # Summary
        logger.info(f"\n[VALIDATION SUMMARY]")
        logger.info(f"  Total Claude plays: {len(detected_plays)}")
        logger.info(f"  Validated (matched ESPN): {len(validated_plays)}")
        logger.info(f"  Rejected (no ESPN match): {len(rejected_plays)}")
        logger.info(f"  No clock detected: {len(no_clock_plays)}")
        logger.info("="*80 + "\n")

        validation_stats = {
            'validated': len(validated_plays),
            'rejected': len(rejected_plays),
            'no_clock': len(no_clock_plays),
            'mapping_drift': float(drift),
            'mapping_offset': float(offset),
            'mapping_r_squared': float(r_squared),
            'clock_sightings': len(game_clock_sightings),
            'espn_plays': len(espn_plays)
        }

        return validated_plays, validation_stats

    @staticmethod
    def clock_to_absolute_seconds(quarter: int, clock_str: str) -> Optional[float]:
        """Convert game clock (Q1 10:32) to absolute seconds from game start.

        Args:
            quarter: Quarter number (1-4, 5+ for OT)
            clock_str: Clock string like "10:32" or "0:45"

        Returns:
            Absolute seconds from game start, or None if invalid

        Example:
            Q1 10:32 = 15*60 - (10*60 + 32) = 900 - 632 = 268 seconds elapsed
        """
        try:
            # Parse MM:SS format
            match = re.match(r'(\d+):(\d+)', clock_str.strip())
            if not match:
                return None

            minutes = int(match.group(1))
            seconds = int(match.group(2))

            # Clock shows time REMAINING in quarter
            time_remaining = minutes * 60 + seconds

            # Each quarter is 15 minutes
            quarter_length = 15 * 60

            # Calculate elapsed time in current quarter
            elapsed_in_quarter = quarter_length - time_remaining

            # Calculate total elapsed time from game start
            total_elapsed = (quarter - 1) * quarter_length + elapsed_in_quarter

            return float(total_elapsed)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def quarter_clock_from_seconds(seconds: float) -> Tuple[int, str]:
        """Convert absolute game seconds to (quarter, clock_str).

        Args:
            seconds: Absolute seconds from game start

        Returns:
            (quarter, clock_str) tuple like (1, "10:32")

        Example:
            268 seconds = Q1 with 632 seconds remaining = Q1 10:32
        """
        quarter_length = 15 * 60  # 900 seconds per quarter

        # Calculate quarter (1-indexed)
        quarter = int(seconds // quarter_length) + 1

        # Calculate elapsed time in current quarter
        elapsed_in_quarter = seconds % quarter_length

        # Calculate time remaining (clock shows remaining time)
        time_remaining = quarter_length - elapsed_in_quarter

        # Convert to MM:SS format
        minutes = int(time_remaining // 60)
        secs = int(time_remaining % 60)
        clock_str = f"{minutes}:{secs:02d}"

        return quarter, clock_str

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
            logger.info(f"[GAME START DETECTION] Claude response (raw): {response_text[:200]}...")

            # Clean response before parsing
            cleaned_response = self.clean_json_response(response_text)
            logger.info(f"[GAME START DETECTION] Claude response (cleaned): {cleaned_response[:200]}...")

            # Parse response
            result = json.loads(cleaned_response)
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
            logger.error(f"[GAME START DETECTION] Raw response: {response_text[:200]}")
            # Try to show cleaned response if available
            try:
                cleaned = self.clean_json_response(response_text)
                logger.error(f"[GAME START DETECTION] Cleaned response: {cleaned[:200]}")
            except Exception:
                pass
            return 600.0
        except Exception as e:
            logger.error(f"[GAME START DETECTION] Error: {type(e).__name__}: {e}")
            return 600.0

    def extract_keyframes(self, video_path: str, num_frames: int = 60, game_start_offset: float = 0.0) -> List[tuple[bytes, float]]:
        """Extract keyframes from video at regular intervals.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 60, ~1 every 3 minutes in a 3-hour game)
            game_start_offset: Offset from game start detection (for diagnostic logging)

        Returns:
            List of (frame_bytes, timestamp) tuples
        """
        logger.info("[FRAME EXTRACTION DIAGNOSTIC] Starting frame extraction")

        duration = self.get_video_duration(video_path)
        logger.info(f"  Video duration: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"  Game start offset (from detection): {game_start_offset:.1f}s")
        logger.info(f"  Number of frames to extract: {num_frames}")

        keyframes: List[tuple[bytes, float]] = []

        # Calculate playable duration (game footage only, excluding pregame)
        playable_duration = duration - game_start_offset
        logger.info(f"  Playable duration (game only): {playable_duration:.1f}s ({playable_duration/60:.1f} min)")

        # Extract frames at regular intervals throughout GAME FOOTAGE ONLY
        interval = playable_duration / (num_frames + 1)  # +1 to avoid very end
        times = [game_start_offset + interval * (i + 1) for i in range(num_frames)]

        # Show the calculation
        logger.info(f"  Frame interval calculation: {playable_duration:.1f}s / {num_frames + 1} = {interval:.1f}s per frame")
        logger.info(f"  [FIX APPLIED] Frames will be extracted from GAME FOOTAGE ONLY")
        logger.info(f"  This will extract frames from {times[0]:.1f}s to {times[-1]:.1f}s")

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, time_sec in enumerate(times):
                # Log EACH frame being extracted with context
                logger.info(f"\n[FRAME {i}] Will extract at timestamp: {time_sec:.1f}s ({time_sec/60:.1f} min)")

                # Context: Is this before/after game start?
                logger.info(f"  Context: Is this before/after game start ({game_start_offset:.1f}s)?")
                if time_sec < game_start_offset:
                    logger.warning(f"  ⚠️  BEFORE game start! (offset={game_start_offset:.1f}s, frame={time_sec:.1f}s)")
                elif time_sec > (duration - 600):
                    logger.warning(f"  ⚠️  NEAR END OF VIDEO (within 10 min of end)")
                else:
                    logger.info(f"  ✓ In game timeframe")

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
                    logger.info(f"  ✓ Successfully extracted frame {i+1}/{num_frames}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"  ❌ Timeout extracting frame {i} at {time_sec:.1f}s")
                except Exception as e:
                    logger.warning(f"  ❌ Failed to extract frame {i} at {time_sec:.1f}s: {e}")

        logger.info(f"\n[FRAME EXTRACTION DIAGNOSTIC] Complete - Successfully extracted {len(keyframes)} keyframes")
        return keyframes

    async def detect_plays(
        self,
        video_path: str,
        game_info: Optional[Dict] = None,
        num_frames: int = 12,
        espn_game_id: Optional[str] = None,
        enable_espn_validation: bool = True
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

        keyframes = self.extract_keyframes(video_path, num_frames=num_frames, game_start_offset=game_start_offset)

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

TASK 1: DETECT GAME CLOCK ON SCREEN
For EACH frame, look for the game clock display (usually in bottom-right or top corner):
- Look for formats like "10:32", "Q1 10:32", "1st 10:32", or quarter indicator + time
- The clock shows TIME REMAINING in the current quarter (counts down)
- Report the quarter (1-4) and time (MM:SS format)

TASK 2: IDENTIFY FOOTBALL PLAYS
Identify ALL football plays across these frames. In a typical college football game, you should find 50-200+ plays across {len(keyframes)} frames.

For EACH play you identify, provide:
1. frame_index: Which frame number (0-{len(keyframes)-1}) shows the play
2. game_clock: The game clock shown on screen (e.g., "10:32") or null if not visible
3. quarter: The quarter number (1-4) or null if not visible
4. play_type: One of these EXACT types:
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
5. description: Brief description of what you see (e.g., "QB drops back, throws to WR on right side")
6. confidence: 0.0-1.0 (include plays with confidence > 0.5)

IMPORTANT GUIDELINES:
✓ ALWAYS try to detect and report the game clock if visible on screen
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
  {{"frame_index": 0, "game_clock": "14:32", "quarter": 1, "play_type": "Kickoff", "description": "Opening kickoff, ball in air", "confidence": 0.95}},
  {{"frame_index": 0, "game_clock": "14:32", "quarter": 1, "play_type": "Pass Reception", "description": "Scorebug shows 1st down completion", "confidence": 0.75}},
  {{"frame_index": 5, "game_clock": "10:15", "quarter": 1, "play_type": "Rush", "description": "RB carrying ball, defenders converging", "confidence": 0.90}},
  {{"frame_index": 12, "game_clock": null, "quarter": null, "play_type": "Pass Reception", "description": "Play in progress, clock not visible", "confidence": 0.80}}
]

Remember:
1. DETECT GAME CLOCK whenever possible - this is critical for validation
2. Report ALL plays you find. A college football game has many plays, and your job is to find as many as possible across these {len(keyframes)} sample frames.
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
            logger.info(f"  Response preview (raw): {response_text[:200]}...")

            # Clean response before parsing
            logger.info(f"[CLAUDE RESPONSE PARSING]")
            cleaned_response = self.clean_json_response(response_text)
            logger.info(f"  Response after cleaning: {cleaned_response[:200]}...")

            # Parse JSON response
            detected_plays = json.loads(cleaned_response)
            logger.info(f"  JSON parsed successfully")
            logger.info(f"  Total plays detected: {len(detected_plays)}")

            # ==================== PHASE 2: CLAUDE DETECTION DIAGNOSTICS ====================
            logger.info(f"\n{'='*80}")
            logger.info("[CLAUDE DETECTION DIAGNOSTIC] Analyzing each detected play")
            logger.info(f"{'='*80}")

            # Log EVERY detected play with FULL details
            for play_idx, play in enumerate(detected_plays):
                frame_idx = play.get('frame_index', -1)
                frame_time = frame_times[frame_idx] if frame_idx >= 0 and frame_idx < len(frame_times) else None
                play_type = play.get('play_type', 'UNKNOWN')
                confidence = play.get('confidence', 0.0)
                description = play.get('description', 'NONE')

                logger.info(f"\n[PLAY {play_idx}] Raw Claude Detection:")
                logger.info(f"  Frame index: {frame_idx}")
                if frame_time is not None:
                    logger.info(f"  Frame timestamp: {frame_time:.1f}s ({frame_time/60:.1f} min)")
                else:
                    logger.info(f"  Frame timestamp: UNKNOWN")
                logger.info(f"  Play type: {play_type}")
                logger.info(f"  Confidence: {confidence:.2f}")
                logger.info(f"  Description: {description}")

                # CRITICAL: Show what Claude literally saw
                logger.info(f"  ⚠️  Claude saw: {description}")

                # Flag suspicious descriptions
                description_lower = description.lower()
                if 'studio' in description_lower:
                    logger.warning(f"  🚩 STUDIO CONTENT DETECTED")
                if 'commercial' in description_lower:
                    logger.warning(f"  🚩 COMMERCIAL DETECTED")
                if 'replay' in description_lower:
                    logger.warning(f"  🚩 REPLAY DETECTED")
                if 'halftime' in description_lower:
                    logger.warning(f"  🚩 HALFTIME DETECTED")
                if 'pregame' in description_lower or 'pre-game' in description_lower:
                    logger.warning(f"  🚩 PREGAME DETECTED")
                if 'crowd' in description_lower and 'field' not in description_lower:
                    logger.warning(f"  🚩 CROWD SHOT (no field) DETECTED")
                if 'analyst' in description_lower or 'desk' in description_lower:
                    logger.warning(f"  🚩 ANALYST/DESK DETECTED")

            logger.info(f"\n{'='*80}")

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

            # ==================== ESPN GAME CLOCK VALIDATION ====================
            # Validate plays against ESPN data if enabled
            plays_to_process = detected_plays
            if enable_espn_validation and espn_game_id:
                team_name = game_info.get("team", "") if game_info else ""
                validated_plays, validation_stats = await self.validate_plays_with_espn(
                    detected_plays=detected_plays,
                    frame_times=frame_times,
                    espn_game_id=espn_game_id,
                    team_name=team_name
                )
                if validated_plays:
                    logger.info(f"[ESPN VALIDATION] Using {len(validated_plays)} validated plays (rejected {len(detected_plays) - len(validated_plays)})")
                    plays_to_process = validated_plays
                else:
                    logger.warning(f"[ESPN VALIDATION] No plays validated, using all {len(detected_plays)} detected plays")
            else:
                logger.info(f"[ESPN VALIDATION] Skipped (enable_espn_validation={enable_espn_validation}, espn_game_id={espn_game_id})")

            # Convert to (start, end) tuples with absolute timestamps from frame extraction
            play_windows: List[tuple[float, float]] = []
            skipped_low_conf = 0
            skipped_invalid_frame = 0

            # ==================== DIAGNOSTIC LOGGING ====================
            logger.info(f"\n{'='*80}")
            logger.info(f"[TIMESTAMP CALC DIAGNOSTIC] Starting timestamp conversion...")
            logger.info(f"{'='*80}")
            logger.info(f"[INPUTS]")
            logger.info(f"  video_duration: {video_duration}s ({video_duration/60:.1f} minutes)")
            logger.info(f"  game_start_offset: {game_start_offset}s ({game_start_offset/60:.1f} minutes)")
            logger.info(f"  num_frames: {num_frames}")
            logger.info(f"  frame_interval: {video_duration / num_frames:.1f}s")
            logger.info(f"  Total plays to process: {len(detected_plays)}")

            # Show frame_times distribution
            if frame_times:
                logger.info(f"\n[FRAME TIMES DISTRIBUTION]")
                logger.info(f"  First frame time: {frame_times[0]:.1f}s")
                logger.info(f"  Last frame time: {frame_times[-1]:.1f}s")
                logger.info(f"  Frame time range: {frame_times[0]:.1f}s - {frame_times[-1]:.1f}s")
                logger.info(f"  Expected last frame (video_end - interval): ~{video_duration - (video_duration / num_frames):.1f}s")
                if frame_times[-1] > video_duration:
                    logger.warning(f"  ⚠️  Last frame time ({frame_times[-1]:.1f}s) EXCEEDS video duration ({video_duration}s)!")
            logger.info(f"{'='*80}\n")

            # Track statistics for analysis
            timestamps_beyond_video = []
            timestamps_within_video = []

            logger.info(f"[TIMESTAMP CONVERSION] Converting {len(plays_to_process)} plays to time windows...")
            logger.info(f"  Using absolute timestamps from frame extraction (NOT adding game_start_offset={game_start_offset:.1f}s)")

            for i, play in enumerate(plays_to_process):
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

                        # DIAGNOSTIC: Log step-by-step calculation
                        logger.info(f"\n[PLAY {i}] {play_type} (confidence={confidence:.2f})")
                        logger.info(f"  Input frame number: {frame_idx}")
                        logger.info(f"  Frame range: 0-{len(frame_times)-1}")
                        logger.info(f"  Frame time (from extraction): {frame_time:.1f}s ({frame_time/60:.1f}m)")

                        # FIX: Use absolute timestamp directly (frame_time is already absolute)
                        # Previously: center_time = game_start_offset + frame_time (WRONG: double-offset)
                        center_time = frame_time

                        logger.info(f"  Calculation: center_time = frame_time (absolute)")
                        logger.info(f"              center_time = {center_time:.1f}s")
                        logger.info(f"  [FIX APPLIED] Removed game_start_offset (+{game_start_offset:.1f}s) to prevent double-offset")

                        # Check if beyond video duration
                        if center_time > video_duration:
                            logger.warning(f"  ⚠️  BEYOND VIDEO! {center_time:.1f}s > {video_duration}s (overage: {center_time - video_duration:.1f}s)")
                            timestamps_beyond_video.append({
                                "frame_idx": frame_idx,
                                "frame_time": frame_time,
                                "center_time": center_time,
                                "overage": center_time - video_duration,
                                "play_type": play_type
                            })
                        else:
                            logger.info(f"  ✓ Within bounds ({center_time:.1f}s < {video_duration}s)")
                            timestamps_within_video.append(center_time)

                        # Create window around detected play (3s before, 5s after)
                        start_time = max(0.0, center_time - 3.0)
                        end_time = min(video_duration, center_time + 5.0)

                        logger.info(f"  Window: {start_time:.1f}s - {end_time:.1f}s (duration: {end_time-start_time:.1f}s)")

                        play_windows.append((start_time, end_time))
                    else:
                        skipped_invalid_frame += 1
                        logger.warning(f"[PLAY {i}] SKIPPED: Invalid frame_index {frame_idx} (max: {len(frame_times)-1})")

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"[PLAY {i}] SKIPPED: Error processing play entry: {e}")
                    continue

            # ==================== AGGREGATE ANALYSIS ====================
            logger.info(f"\n{'='*80}")
            logger.info(f"[TIMESTAMP ANALYSIS]")
            logger.info(f"{'='*80}")
            logger.info(f"  Total plays from Claude: {len(detected_plays)}")
            logger.info(f"  Skipped (low confidence < 0.5): {skipped_low_conf}")
            logger.info(f"  Skipped (invalid frame index): {skipped_invalid_frame}")
            logger.info(f"  Play windows created: {len(play_windows)}")
            logger.info(f"  Timestamps within video: {len(timestamps_within_video)}")
            logger.info(f"  Timestamps BEYOND video: {len(timestamps_beyond_video)}")

            if timestamps_beyond_video:
                logger.warning(f"\n[BEYOND VIDEO TIMESTAMPS]")
                for item in sorted(timestamps_beyond_video, key=lambda x: x["center_time"]):
                    logger.warning(f"  Frame {item['frame_idx']}: {item['center_time']:.1f}s (overage: +{item['overage']:.1f}s) - {item['play_type']}")

                # Calculate statistics
                overages = [item["overage"] for item in timestamps_beyond_video]
                logger.warning(f"\n  Average overage: {sum(overages)/len(overages):.1f}s")
                logger.warning(f"  Max overage: {max(overages):.1f}s")
                logger.warning(f"  Min overage: {min(overages):.1f}s")

            # ==================== HYPOTHESIS TESTING ====================
            logger.info(f"\n{'='*80}")
            logger.info(f"[HYPOTHESIS TESTING]")
            logger.info(f"{'='*80}")

            # Hypothesis 1: num_frames is wrong
            logger.info(f"\nHypothesis 1: Is num_frames={num_frames} correct?")
            if num_frames != 60:
                logger.warning(f"  ⚠️  num_frames is {num_frames}, expected 60!")
            else:
                logger.info(f"  ✓ num_frames is 60 as expected")

            # Hypothesis 2: frame numbers from Claude are outside expected range
            logger.info(f"\nHypothesis 2: Are Claude frame numbers in 0-{num_frames-1} range?")
            frame_numbers = [play.get("frame_index") for play in detected_plays if play.get("frame_index") is not None]
            if frame_numbers:
                min_frame = min(frame_numbers)
                max_frame = max(frame_numbers)
                logger.info(f"  Frame range from Claude: {min_frame}-{max_frame}")
                if min_frame < 0 or max_frame >= num_frames:
                    logger.warning(f"  ⚠️  Frames outside expected 0-{num_frames-1} range!")
                else:
                    logger.info(f"  ✓ Frames within expected range")

            # Hypothesis 3: Is video_duration being calculated incorrectly?
            logger.info(f"\nHypothesis 3: Is video_duration={video_duration}s correct?")
            if frame_times:
                expected_last_frame_time = video_duration - (video_duration / num_frames)
                logger.info(f"  Expected last frame time: ~{expected_last_frame_time:.1f}s")
                logger.info(f"  Actual last frame time: {frame_times[-1]:.1f}s")

                if timestamps_beyond_video:
                    last_beyond = max(timestamps_beyond_video, key=lambda x: x["center_time"])
                    logger.info(f"  Last timestamp calculated: {last_beyond['center_time']:.1f}s")
                    logger.warning(f"  ⚠️  Last timestamp ({last_beyond['center_time']:.1f}s) exceeds video duration ({video_duration}s) by {last_beyond['overage']:.1f}s")

            # Hypothesis 4: Is game_start_offset being applied incorrectly?
            logger.info(f"\nHypothesis 4: Is game_start_offset being applied correctly?")
            logger.info(f"  Current calculation: center_time = game_start_offset + frame_time")
            logger.info(f"  game_start_offset: {game_start_offset:.1f}s")
            if frame_times:
                logger.info(f"  frame_times are ABSOLUTE timestamps in video: {frame_times[0]:.1f}s to {frame_times[-1]:.1f}s")
                if timestamps_beyond_video:
                    logger.warning(f"  ⚠️  Adding offset to absolute timestamps causes overage!")
                    logger.warning(f"  Example: Frame {timestamps_beyond_video[0]['frame_idx']}: {game_start_offset:.1f}s + {timestamps_beyond_video[0]['frame_time']:.1f}s = {timestamps_beyond_video[0]['center_time']:.1f}s")
                    logger.warning(f"  This suggests game_start_offset should NOT be added to frame_times!")

            # Hypothesis 5: Frame extraction timing issue
            logger.info(f"\nHypothesis 5: Are frame extraction times calculated correctly?")
            if frame_times and num_frames > 0:
                expected_interval = video_duration / (num_frames + 1)
                actual_interval = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1) if len(frame_times) > 1 else 0
                logger.info(f"  Expected interval: {expected_interval:.1f}s")
                logger.info(f"  Actual interval: {actual_interval:.1f}s")
                logger.info(f"  Frame extraction formula: interval * (i + 1) for i in range({num_frames})")
                logger.info(f"  This means frame 0 = {expected_interval:.1f}s, frame {num_frames-1} = {expected_interval * num_frames:.1f}s")

                if expected_interval * num_frames > video_duration:
                    logger.warning(f"  ⚠️  Last frame time ({expected_interval * num_frames:.1f}s) exceeds video duration!")

            logger.info(f"{'='*80}\n")

            # ==================== PHASE 5: OFFSET DETECTION AND ANALYSIS SUMMARY ====================
            logger.info("\n" + "="*80)
            logger.info("[DIAGNOSTIC SUMMARY] Analyzing all plays for patterns")
            logger.info("="*80)

            # Collect all mismatches by category
            commercial_plays = []
            studio_plays = []
            replay_plays = []
            pregame_plays = []
            halftime_plays = []
            analyst_plays = []
            valid_plays = []

            for play_idx, play in enumerate(detected_plays):
                frame_idx = play.get('frame_index', -1)
                frame_time = frame_times[frame_idx] if 0 <= frame_idx < len(frame_times) else None
                description = play.get('description', '').lower()
                confidence = play.get('confidence', 0.0)

                # Skip low confidence
                if confidence < 0.5:
                    continue

                # Categorize the play
                if 'commercial' in description:
                    commercial_plays.append((play_idx, frame_time, play))
                elif 'studio' in description:
                    studio_plays.append((play_idx, frame_time, play))
                elif 'analyst' in description or 'desk' in description:
                    analyst_plays.append((play_idx, frame_time, play))
                elif 'replay' in description:
                    replay_plays.append((play_idx, frame_time, play))
                elif 'halftime' in description:
                    halftime_plays.append((play_idx, frame_time, play))
                elif frame_time and frame_time < game_start_offset:
                    pregame_plays.append((play_idx, frame_time, play))
                else:
                    valid_plays.append((play_idx, frame_time, play))

            # Report findings
            logger.info(f"\n[PATTERN ANALYSIS]")
            logger.info(f"  Total plays detected: {len(detected_plays)}")
            logger.info(f"  Plays after confidence filter (>= 0.5): {len(commercial_plays) + len(studio_plays) + len(analyst_plays) + len(replay_plays) + len(halftime_plays) + len(pregame_plays) + len(valid_plays)}")
            logger.info(f"  Commercials: {len(commercial_plays)}")
            logger.info(f"  Studio content: {len(studio_plays)}")
            logger.info(f"  Analyst/desk: {len(analyst_plays)}")
            logger.info(f"  Replays: {len(replay_plays)}")
            logger.info(f"  Halftime: {len(halftime_plays)}")
            logger.info(f"  Pre-game (before {game_start_offset:.1f}s): {len(pregame_plays)}")
            logger.info(f"  Potentially valid: {len(valid_plays)}")

            # Show specific examples of problems
            if commercial_plays:
                logger.warning(f"\n[COMMERCIALS DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(commercial_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            if studio_plays:
                logger.warning(f"\n[STUDIO CONTENT DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(studio_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            if analyst_plays:
                logger.warning(f"\n[ANALYST/DESK CONTENT DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(analyst_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            if replay_plays:
                logger.warning(f"\n[REPLAYS DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(replay_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            if halftime_plays:
                logger.warning(f"\n[HALFTIME CONTENT DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(halftime_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            if pregame_plays:
                logger.warning(f"\n[PRE-GAME CONTENT DETECTED]")
                for idx, (play_idx, frame_time, play) in enumerate(pregame_plays[:3]):
                    if frame_time:
                        logger.warning(f"  Example {idx+1}: Frame {frame_time:.1f}s (before game start at {game_start_offset:.1f}s) - {play.get('description', 'N/A')}")
                    else:
                        logger.warning(f"  Example {idx+1}: Frame time unknown - {play.get('description', 'N/A')}")

            # Check for systematic offset issues
            problem_plays = commercial_plays + studio_plays + analyst_plays + halftime_plays
            if problem_plays:
                # Calculate average timestamp for problem content
                problem_times = [frame_time for _, frame_time, _ in problem_plays if frame_time is not None]
                if problem_times:
                    avg_problem_time = sum(problem_times) / len(problem_times)
                    logger.warning(f"\n[OFFSET HYPOTHESIS]")
                    logger.warning(f"  Problem content (commercials/studio/analyst/halftime) average at: {avg_problem_time:.1f}s ({avg_problem_time/60:.1f} min)")
                    logger.warning(f"  Game start offset: {game_start_offset:.1f}s ({game_start_offset/60:.1f} min)")
                    potential_offset = avg_problem_time - game_start_offset
                    logger.warning(f"  Potential systematic offset: {potential_offset:.1f}s")

                    # Check if problem content is clustered at beginning or end
                    if problem_times:
                        min_problem_time = min(problem_times)
                        max_problem_time = max(problem_times)
                        logger.warning(f"  Problem content time range: {min_problem_time:.1f}s - {max_problem_time:.1f}s")
                        if max_problem_time < game_start_offset:
                            logger.warning(f"  ⚠️  All problem content is BEFORE game start - may indicate incorrect game start detection")
                        elif min_problem_time > (video_duration - 600):
                            logger.warning(f"  ⚠️  All problem content is NEAR END - may indicate post-game content")

            # Show examples of valid plays
            if valid_plays:
                logger.info(f"\n[EXAMPLES OF POTENTIALLY VALID PLAYS]")
                for idx, (play_idx, frame_time, play) in enumerate(valid_plays[:5]):
                    if frame_time:
                        logger.info(f"  Example {idx+1}: {frame_time:.1f}s ({frame_time/60:.1f} min) - {play.get('play_type')} (conf: {play.get('confidence', 0.0):.2f})")
                        logger.info(f"    Description: {play.get('description', 'N/A')}")
                    else:
                        logger.info(f"  Example {idx+1}: Frame time unknown - {play.get('play_type')} (conf: {play.get('confidence', 0.0):.2f})")
            else:
                logger.warning(f"\n[WARNING] NO VALID PLAYS DETECTED!")
                logger.warning(f"  This is highly unusual for a football game")
                logger.warning(f"  Possible causes:")
                logger.warning(f"    - Incorrect game start detection")
                logger.warning(f"    - Video is not actual game footage")
                logger.warning(f"    - Frame extraction timing issues")

            logger.info("="*80 + "\n")

            logger.info(f"[DETECTION COMPLETE]")
            logger.info(f"  Detected plays: {len(detected_plays)}")
            logger.info(f"  Play windows ready for clipping: {len(play_windows)}")

            return play_windows

        except json.JSONDecodeError as e:
            logger.error(f"[CLAUDE] Failed to parse Claude response as JSON: {e}")
            logger.error(f"[CLAUDE] Raw response (first 500 chars): {response_text[:500]}")
            # Try to show cleaned response if available
            try:
                cleaned = self.clean_json_response(response_text)
                logger.error(f"[CLAUDE] Cleaned response (first 500 chars): {cleaned[:500]}")
            except Exception:
                pass
            return []
        except Exception as e:
            logger.error(f"[CLAUDE] Error calling Claude API: {type(e).__name__}: {e}")
            return []
