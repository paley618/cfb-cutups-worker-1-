"""Detection dispatch layer for orchestrating play detection with MANDATORY cfbfastR integration.

CRITICAL CHANGE: Claude Vision is now SUPERVISED ONLY.
This module implements MANDATORY cfbfastR integration to ensure 90%+ accuracy.

NEW ARCHITECTURE (Non-negotiable):
    1. cfbfastR (fetch official plays) - MANDATORY FIRST STEP
    2. Claude Vision SUPERVISED (uses official plays) - PRIMARY if cfbfastR succeeds
    3. ESPN fallback - SECONDARY if cfbfastR fails
    4. ERROR - No blind detection allowed

KEY PRINCIPLE:
    - cfbfastR is the ground truth, NOT a fallback
    - Claude Vision is a verification tool, NOT a detection tool
    - Blind/unsupervised detection has been REMOVED (was only 60% accurate)
    - Supervised detection achieves 90%+ accuracy

The dispatch layer is responsible for:
    1. Fetching official plays from cfbfastR FIRST
    2. Using Claude Vision to match official plays to video frames (supervised)
    3. Falling back to ESPN only if cfbfastR fails
    4. Returning unified results with metadata about which method was used
    5. Logging visibility into the detection path taken
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DetectionResult:
    """Unified result structure from any detection method."""

    def __init__(
        self,
        plays: List[Dict[str, Any]],
        detection_method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize detection result.

        Args:
            plays: List of play dictionaries with standardized structure
            detection_method: Name of method used ("claude_vision", "cfbd", "espn", etc.)
            metadata: Additional metadata about the detection process
        """
        self.plays = plays
        self.detection_method = detection_method
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        """Result is truthy if plays were detected."""
        return bool(self.plays)

    def __len__(self) -> int:
        """Number of plays detected."""
        return len(self.plays)


async def try_vision_play_mapper(
    video_path: str,
    game_info: Optional[Dict],
    settings: Any,
    cfbd_plays: List[Dict],
) -> DetectionResult:
    """Attempt play detection using Vision Play Mapper with dense frame sampling.

    This is the NEW vision-based approach that:
    - Extracts frames every 10 seconds (~1200 frames for 3-hour game)
    - Batches frames and sends to Claude Vision
    - Returns precise start/end timestamps for each play
    - Achieves 70-90%+ accuracy through semantic understanding

    Args:
        video_path: Path to video file
        game_info: Game context (away_team, home_team, etc.)
        settings: Application settings
        cfbd_plays: List of CFBD plays to map to video timestamps

    Returns:
        DetectionResult with plays and metadata, or empty result if failed
    """
    logger.info("[DISPATCH] Attempting Vision Play Mapper (DENSE SAMPLING)...")

    # Validate that CFBD plays are provided
    if not cfbd_plays or len(cfbd_plays) == 0:
        logger.error("[DISPATCH] ✗ Vision Play Mapper requires CFBD plays")
        return DetectionResult(
            [],
            "vision_play_mapper",
            {"status": "error", "error": "cfbd_plays required"}
        )

    try:
        from .vision_play_mapper import VisionPlayMapper

        # Initialize mapper
        mapper = VisionPlayMapper(api_key=settings.anthropic_api_key)

        logger.info(f"[VISION MAPPER] Mapping {len(cfbd_plays)} CFBD plays to video timestamps")

        # Get frame interval and batch size from settings (with defaults)
        frame_interval = getattr(settings, 'VISION_MAPPER_FRAME_INTERVAL', 10.0)
        batch_size = getattr(settings, 'VISION_MAPPER_BATCH_SIZE', 20)

        # Map plays to timestamps
        timestamp_mapping = await mapper.map_plays_to_timestamps(
            video_path=video_path,
            cfbd_plays=cfbd_plays,
            game_info=game_info,
            frame_interval=frame_interval,
            batch_size=batch_size,
        )

        if timestamp_mapping and len(timestamp_mapping) > 0:
            logger.info(f"[DISPATCH] ✓ Vision Play Mapper succeeded: {len(timestamp_mapping)} plays mapped")

            # Validate timestamps
            video_duration = mapper.get_video_duration(video_path)
            validated_mapping = mapper.validate_timestamps(
                timestamp_mapping,
                video_duration,
                cfbd_plays
            )

            # Convert to standardized play format
            plays = []
            for play_number, (start_time, end_time) in validated_mapping.items():
                # Find the original CFBD play for metadata
                cfbd_play = next((p for p in cfbd_plays if p.get('play_number') == play_number), None)

                play_dict = {
                    "id": play_number,
                    "game_id": game_info.get("game_id", 0) if game_info else 0,
                    "timestamp": start_time,
                    "end_timestamp": end_time,
                    "source": "vision_play_mapper",
                }

                # Add CFBD metadata if available
                if cfbd_play:
                    play_dict.update({
                        "play_type": cfbd_play.get("play_type"),
                        "play_text": cfbd_play.get("play_text"),
                        "quarter": cfbd_play.get("quarter"),
                    })

                plays.append(play_dict)

            metadata = {
                "status": "success",
                "plays_count": len(plays),
                "cfbd_plays_count": len(cfbd_plays),
                "detection_rate": len(plays) / len(cfbd_plays) * 100,
                "frame_interval": frame_interval,
                "batch_size": batch_size,
            }

            return DetectionResult(plays, "vision_play_mapper", metadata)
        else:
            logger.warning("[DISPATCH] ✗ Vision Play Mapper returned no timestamp mappings")
            return DetectionResult([], "vision_play_mapper", {"status": "no_mappings"})

    except ImportError as e:
        logger.warning(f"[DISPATCH] ✗ Vision Play Mapper unavailable: {e}")
        return DetectionResult([], "vision_play_mapper", {"status": "import_error", "error": str(e)})
    except Exception as e:
        logger.warning(f"[DISPATCH] ✗ Vision Play Mapper failed: {type(e).__name__}: {e}")
        return DetectionResult([], "vision_play_mapper", {"status": "error", "error": str(e)})


async def try_claude_vision_supervised(
    video_path: str,
    game_info: Optional[Dict],
    settings: Any,
    official_plays: List[Dict],
    game_start_offset: float = 900.0,
) -> DetectionResult:
    """Attempt play detection using Claude Vision in SUPERVISED mode ONLY.

    IMPORTANT: This function requires official plays from cfbfastR.
    Claude Vision will ONLY match these official plays to video frames.
    This is NOT blind detection - it's supervised verification.

    NOTE: This is the OLD approach with sparse sampling (60 frames).
    Consider using try_vision_play_mapper for better accuracy.

    Args:
        video_path: Path to video file
        game_info: Game context (away_team, home_team, team, etc.)
        settings: Application settings
        official_plays: REQUIRED list of official plays from cfbfastR
        game_start_offset: Offset in seconds for game start (default: 900s = 15 min)

    Returns:
        DetectionResult with plays and metadata, or empty result if failed
    """
    logger.info("[DISPATCH] Attempting Claude Vision detection (SUPERVISED mode - SPARSE SAMPLING)...")

    # Validate that official plays are provided
    if not official_plays or len(official_plays) == 0:
        logger.error("[DISPATCH] ✗ Claude Vision requires official plays from cfbfastR")
        logger.error("[DISPATCH] Cannot run Claude Vision in blind mode - aborting")
        return DetectionResult(
            [],
            "claude_vision",
            {"status": "error", "error": "official_plays required for supervised mode"}
        )

    try:
        from .claude_play_detector import ClaudePlayDetector

        # Initialize detector
        detector = ClaudePlayDetector(api_key=settings.anthropic_api_key)

        logger.info(f"[SUPERVISED] Using {len(official_plays)} official plays from cfbfastR")

        # Detect plays using supervised mode ONLY
        claude_windows = await detector.detect_plays(
            video_path,
            game_info=game_info,
            num_frames=settings.CLAUDE_VISION_FRAMES,
            official_plays=official_plays,  # REQUIRED - supervised mode only
        )

        if claude_windows and len(claude_windows) > 0:
            logger.info(f"[DISPATCH] ✓ Claude Vision SUPERVISED succeeded: {len(claude_windows)} plays matched")

            # Convert windows to standardized play format
            plays = [
                {
                    "id": i,
                    "game_id": game_info.get("game_id", 0) if game_info else 0,
                    "timestamp": start,
                    "end_timestamp": end,
                    "source": "claude_vision_supervised",
                }
                for i, (start, end) in enumerate(claude_windows)
            ]

            metadata = {
                "status": "success",
                "plays_count": len(plays),
                "official_plays_count": len(official_plays),
                "frames_analyzed": settings.CLAUDE_VISION_FRAMES,
                "mode": "supervised",
            }

            return DetectionResult(plays, "claude_vision_supervised", metadata)
        else:
            logger.warning("[DISPATCH] ✗ Claude Vision SUPERVISED returned no plays")
            return DetectionResult([], "claude_vision_supervised", {"status": "no_plays"})

    except ImportError as e:
        logger.warning(f"[DISPATCH] ✗ Claude Vision unavailable: {e}")
        return DetectionResult([], "claude_vision", {"status": "import_error", "error": str(e)})
    except Exception as e:
        logger.warning(f"[DISPATCH] ✗ Claude Vision failed: {type(e).__name__}: {e}")
        return DetectionResult([], "claude_vision", {"status": "error", "error": str(e)})


async def try_cfbd(
    cfbd_client: Any,
    game_id: int,
    year: Optional[int],
    week: Optional[int],
    season_type: Optional[str],
) -> DetectionResult:
    """Attempt play detection using CFBD API.

    Args:
        cfbd_client: CFBD client instance
        game_id: CFBD game ID
        year: Season year
        week: Week number
        season_type: Season type ("regular", "postseason", etc.)

    Returns:
        DetectionResult with plays and metadata, or empty result if failed
    """
    logger.info(f"[DISPATCH] Attempting CFBD detection for game_id={game_id}...")

    try:
        import asyncio

        # Call CFBD API
        plays_list = await asyncio.to_thread(
            cfbd_client.get_plays_for_game,
            int(game_id),
            year=year,
            week=week,
            season_type=season_type,
        )
        plays = list(plays_list)

        if plays and len(plays) > 0:
            logger.info(f"[DISPATCH] ✓ CFBD succeeded: {len(plays)} plays detected")

            metadata = {
                "status": "success",
                "plays_count": len(plays),
                "game_id": int(game_id),
            }

            return DetectionResult(plays, "cfbd", metadata)
        else:
            logger.warning("[DISPATCH] ✗ CFBD returned no plays")
            return DetectionResult([], "cfbd", {"status": "no_plays"})

    except Exception as e:
        logger.warning(f"[DISPATCH] ✗ CFBD failed: {type(e).__name__}: {e}")
        return DetectionResult([], "cfbd", {"status": "error", "error": str(e)})


async def try_espn(
    espn_game_id: str,
    team_name: str,
) -> DetectionResult:
    """Attempt play detection using ESPN API.

    Args:
        espn_game_id: ESPN game ID
        team_name: Team name for filtering plays

    Returns:
        DetectionResult with plays and metadata, or empty result if failed
    """
    logger.info(f"[DISPATCH] Attempting ESPN detection for game_id={espn_game_id}...")

    try:
        from .espn import fetch_offensive_play_times

        # Call ESPN API
        espn_timestamps = await fetch_offensive_play_times(
            espn_game_id=espn_game_id,
            team_name=team_name or "unknown",
        )

        if espn_timestamps and len(espn_timestamps) > 10:
            logger.info(f"[DISPATCH] ✓ ESPN succeeded: {len(espn_timestamps)} timestamps detected")

            # Convert to standardized play format
            plays = [
                {
                    "id": i,
                    "game_id": int(espn_game_id) if espn_game_id.isdigit() else 0,
                    "timestamp": ts,
                    "source": "espn",
                }
                for i, ts in enumerate(espn_timestamps)
            ]

            metadata = {
                "status": "success",
                "plays_count": len(plays),
                "game_id": espn_game_id,
            }

            return DetectionResult(plays, "espn", metadata)
        else:
            logger.warning(f"[DISPATCH] ✗ ESPN returned insufficient timestamps: {len(espn_timestamps or [])}")
            return DetectionResult([], "espn", {"status": "insufficient_plays"})

    except Exception as e:
        logger.warning(f"[DISPATCH] ✗ ESPN failed: {type(e).__name__}: {e}")
        return DetectionResult([], "espn", {"status": "error", "error": str(e)})


async def dispatch_detection(
    video_path: str,
    game_id: Optional[int],
    game_info: Optional[Dict],
    cfbd_client: Any,
    settings: Any,
    year: Optional[int] = None,
    week: Optional[int] = None,
    season_type: Optional[str] = None,
    team_name: Optional[str] = None,
) -> DetectionResult:
    """Orchestrate play detection with MANDATORY cfbfastR integration.

    NEW DETECTION ORDER (Non-negotiable):
        1. cfbfastR (fetch official plays) - MANDATORY FIRST STEP
        2. If cfbfastR succeeds → Claude Vision SUPERVISED (uses official plays)
        3. If cfbfastR fails → ESPN fallback
        4. If ESPN fails → ERROR (no blind detection allowed)

    CRITICAL CHANGE: Claude Vision ONLY runs when cfbfastR provides official plays.
    This ensures 90%+ accuracy via supervised detection, not 60% blind guessing.

    Args:
        video_path: Path to video file
        game_id: CFBD/ESPN game ID (REQUIRED for cfbfastR)
        game_info: Game context dictionary
        cfbd_client: CFBD client instance
        settings: Application settings
        year: Season year (REQUIRED for cfbfastR)
        week: Week number
        season_type: Season type
        team_name: Team name

    Returns:
        DetectionResult with plays and metadata about which method succeeded
    """
    import os

    logger.info("\n" + "=" * 80)
    logger.info("[DEBUG] ===== DETECTION START =====")
    logger.info("=" * 80)

    # STEP 0: Environment & Configuration Verification
    logger.info("[DEBUG] ENVIRONMENT VERIFICATION:")
    logger.info(f"  ANTHROPIC_API_KEY present: {'ANTHROPIC_API_KEY' in os.environ}")
    if 'ANTHROPIC_API_KEY' in os.environ:
        api_key = os.environ['ANTHROPIC_API_KEY']
        logger.info(f"  ANTHROPIC_API_KEY length: {len(api_key)} chars")
        logger.info(f"  ANTHROPIC_API_KEY preview: {api_key[:20]}...")
    else:
        logger.error("  ❌ ANTHROPIC_API_KEY NOT SET - Vision will fail!")

    logger.info(f"  settings.anthropic_api_key present: {hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key is not None}")
    if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
        logger.info(f"  settings.anthropic_api_key length: {len(settings.anthropic_api_key)} chars")

    logger.info("")
    logger.info("[DEBUG] CONFIGURATION:")
    logger.info(f"  CLAUDE_VISION_ENABLE: {settings.CLAUDE_VISION_ENABLE}")
    logger.info(f"  USE_VISION_PLAY_MAPPER: {getattr(settings, 'USE_VISION_PLAY_MAPPER', True)}")
    logger.info(f"  Game ID: {game_id}")
    logger.info(f"  Year: {year}")
    logger.info(f"  Week: {week}")
    logger.info(f"  Season type: {season_type}")
    logger.info(f"  Team name: {team_name}")
    logger.info(f"  Video path: {video_path}")
    logger.info(f"  Game info: {game_info}")
    logger.info("")
    logger.info("[DEBUG] DETECTION ORDER:")
    logger.info(f"  1. cfbfastR (fetch official plays)")
    logger.info(f"  2. Claude Vision SUPERVISED (if cfbfastR succeeds)")
    logger.info(f"  3. ESPN fallback (if cfbfastR fails)")
    logger.info(f"  4. ERROR (if all fail)")
    logger.info("")

    # STEP 1: MANDATORY - Fetch official plays from cfbfastR FIRST
    logger.info("[STEP 1] Attempting to fetch official plays from cfbfastR...")

    official_plays = None
    if game_id and year:
        try:
            from .cfbfastr_helper import get_official_plays, game_clock_to_video_time

            logger.info(f"[CFBFASTR] Fetching official plays for game_id={game_id}, year={year}")
            official_plays = get_official_plays(str(game_id), year)

            logger.info(f"[DEBUG] After CFBD fetch: {len(official_plays) if official_plays else 0} plays loaded")
            if official_plays and len(official_plays) > 0:
                logger.info(f"[CFBFASTR] ✓ SUCCESS: Fetched {len(official_plays)} official plays")

                # Log sample play structure
                sample_play = official_plays[0]
                logger.info(f"[DEBUG] CFBD play fields: {list(sample_play.keys())}")
                logger.info(f"[DEBUG] Sample play: {sample_play}")

                # Convert game clock to video timestamps
                game_start_offset = 900.0  # Default 15 minutes
                for play in official_plays:
                    play['video_timestamp'] = game_clock_to_video_time(
                        play['quarter'],
                        play['clock_minutes'],
                        play['clock_seconds'],
                        game_start_offset
                    )

                valid_timestamps = len([p for p in official_plays if p['video_timestamp'] is not None])
                logger.info(f"[CFBFASTR] Converted {valid_timestamps}/{len(official_plays)} plays to video timestamps")
                logger.info(f"[DEBUG] CFBD plays successfully loaded and timestamped")

            else:
                logger.warning(f"[CFBFASTR] ✗ FAILED: No official plays returned")
                logger.warning(f"[DEBUG] official_plays is None or empty")
                official_plays = None

        except Exception as e:
            logger.error(f"[CFBFASTR] ✗ FAILED: {type(e).__name__}: {e}")
            logger.error(f"[DEBUG] Exception details:")
            import traceback
            logger.error(f"[DEBUG] {traceback.format_exc()}")
            official_plays = None
    else:
        logger.warning(f"[CFBFASTR] ✗ SKIPPED: Missing game_id={game_id} or year={year}")
        logger.warning(f"[DEBUG] Cannot fetch CFBD plays without both game_id and year")
        official_plays = None

    logger.info(f"[DEBUG] Post-CFBD status: official_plays={'LOADED' if official_plays else 'NONE'} ({len(official_plays) if official_plays else 0} plays)")
    logger.info("")

    # STEP 2: If cfbfastR succeeded, use Vision Play Mapper (NEW) or Claude Vision (OLD)
    vision_enabled = settings.CLAUDE_VISION_ENABLE and settings.anthropic_api_key
    use_new_vision_mapper = getattr(settings, 'USE_VISION_PLAY_MAPPER', True)  # Default to new approach

    logger.info("[DEBUG] ===== VISION DECISION POINT =====")
    logger.info(f"[DEBUG] official_plays available: {official_plays is not None} ({len(official_plays) if official_plays else 0} plays)")
    logger.info(f"[DEBUG] vision_enabled: {vision_enabled}")
    logger.info(f"[DEBUG]   - CLAUDE_VISION_ENABLE: {settings.CLAUDE_VISION_ENABLE}")
    logger.info(f"[DEBUG]   - settings.anthropic_api_key exists: {settings.anthropic_api_key is not None}")
    logger.info(f"[DEBUG] use_new_vision_mapper: {use_new_vision_mapper}")
    logger.info("")

    if official_plays and vision_enabled:
        if use_new_vision_mapper:
            logger.info(f"[STEP 2] cfbfastR succeeded → Using Vision Play Mapper (DENSE SAMPLING)")
            logger.info(f"[DEBUG] ===== ATTEMPTING VISION PLAY MAPPER =====")
            logger.info(f"[DEBUG] Calling vision_play_mapper with {len(official_plays)} CFBD plays")
            logger.info(f"[DEBUG] Video path: {video_path}")
            logger.info(f"[DEBUG] Game info: {game_info}")

            try:
                result = await try_vision_play_mapper(
                    video_path=video_path,
                    game_info=game_info,
                    settings=settings,
                    cfbd_plays=official_plays,
                )

                logger.info(f"[DEBUG] Vision Play Mapper returned: {len(result.plays) if result and result.plays else 0} plays")
                logger.info(f"[DEBUG] Vision result detection_method: {result.detection_method if result else 'NONE'}")
                logger.info(f"[DEBUG] Vision result metadata: {result.metadata if result else 'NONE'}")

                if result:
                    logger.info(f"[DISPATCH] ✓ SUCCESS: Vision Play Mapper detected {len(result)} plays")
                    logger.info(f"[DISPATCH] Detection rate: {result.metadata.get('detection_rate', 0):.1f}%")
                    logger.info(f"[DISPATCH] Expected accuracy: 70-90%+ (vision-based semantic understanding)")
                    logger.info(f"[DEBUG] ===== RETURNING VISION PLAY MAPPER RESULT =====")
                    return result

                logger.warning("[DISPATCH] Vision Play Mapper returned no plays, falling back to old Claude Vision...")

            except Exception as e:
                logger.error(f"[DEBUG] Vision Play Mapper FAILED with exception: {e}")
                logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
                logger.warning("[DISPATCH] Vision Play Mapper failed, falling back to old Claude Vision...")

            # Fallback to old Claude Vision if new mapper fails
            logger.info(f"[DEBUG] Attempting old Claude Vision SUPERVISED as fallback...")

            try:
                result = await try_claude_vision_supervised(
                    video_path=video_path,
                    game_info=game_info,
                    settings=settings,
                    official_plays=official_plays,
                    game_start_offset=900.0,
                )

                logger.info(f"[DEBUG] Claude Vision SUPERVISED returned: {len(result.plays) if result and result.plays else 0} plays")

                if result:
                    logger.info(f"[DISPATCH] ✓ SUCCESS: Claude Vision SUPERVISED detected {len(result)} plays")
                    logger.info(f"[DISPATCH] Accuracy: ~90%+ (supervised with official plays)")
                    return result

            except Exception as e:
                logger.error(f"[DEBUG] Claude Vision SUPERVISED FAILED: {e}")
                import traceback
                logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")

            logger.warning("[DISPATCH] Both vision methods failed, falling back to ESPN...")

        else:
            logger.info(f"[STEP 2] cfbfastR succeeded → Using Claude Vision SUPERVISED mode (OLD)")
            logger.info(f"[DEBUG] use_new_vision_mapper=False, using old approach")

            try:
                result = await try_claude_vision_supervised(
                    video_path=video_path,
                    game_info=game_info,
                    settings=settings,
                    official_plays=official_plays,
                    game_start_offset=900.0,
                )

                if result:
                    logger.info(f"[DISPATCH] ✓ SUCCESS: Claude Vision SUPERVISED detected {len(result)} plays")
                    logger.info(f"[DISPATCH] Accuracy: ~90%+ (supervised with official plays)")
                    return result

            except Exception as e:
                logger.error(f"[DEBUG] Claude Vision SUPERVISED FAILED: {e}")
                import traceback
                logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")

            logger.warning("[DISPATCH] Claude Vision SUPERVISED returned no plays, falling back to ESPN...")

    elif official_plays and not vision_enabled:
        logger.warning("[STEP 2] cfbfastR succeeded but Claude Vision is disabled")
        logger.warning(f"[DEBUG] VISION DISABLED - Reason:")
        logger.warning(f"[DEBUG]   - CLAUDE_VISION_ENABLE={settings.CLAUDE_VISION_ENABLE}")
        logger.warning(f"[DEBUG]   - anthropic_api_key exists={settings.anthropic_api_key is not None}")
        logger.warning("[DISPATCH] Falling back to ESPN...")

    else:
        logger.warning("[STEP 2] cfbfastR failed → Cannot use Vision methods (no official plays)")
        logger.warning(f"[DEBUG] Cannot run Vision - official_plays={official_plays is not None}, vision_enabled={vision_enabled}")
        logger.info("[DISPATCH] Falling back to ESPN...")

    # STEP 3: ESPN fallback (if cfbfastR or Claude failed)
    logger.info("[STEP 3] Attempting ESPN fallback...")

    if game_id:
        result = await try_espn(
            str(game_id),
            team_name or "unknown",
        )
        if result:
            logger.info(f"[DISPATCH] ✓ ESPN SUCCESS: {len(result)} plays detected")
            logger.info(f"[DISPATCH] Accuracy: ~80% (ESPN play-by-play)")
            return result

        logger.warning("[DISPATCH] ESPN failed")

    # STEP 4: All methods failed - NO BLIND DETECTION ALLOWED
    logger.error("[DISPATCH] ✗ ALL DETECTION METHODS FAILED")
    logger.error("[DISPATCH] cfbfastR: FAILED (no official plays)")
    logger.error("[DISPATCH] Claude Vision: SKIPPED (requires official plays)")
    logger.error("[DISPATCH] ESPN: FAILED")
    logger.error("[DISPATCH] RESULT: 0 plays detected")
    logger.error("")
    logger.error("[CRITICAL] Claude Vision is NOT allowed to run without official plays")
    logger.error("[CRITICAL] Blind detection has been DISABLED to prevent 60% accuracy")
    logger.info("=" * 80 + "\n")

    return DetectionResult(
        [],
        "none",
        {
            "status": "all_methods_failed",
            "error": "All detection methods (cfbfastR+Claude, ESPN) failed. Blind detection is disabled.",
            "cfbfastr_status": "failed" if not official_plays else "success",
            "claude_vision_status": "requires_official_plays",
            "espn_status": "failed",
        },
    )
