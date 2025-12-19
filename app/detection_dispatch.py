"""Detection dispatch layer for orchestrating play detection methods with graceful fallback.

This module implements a clean separation between detection strategy selection and
the individual detection methods (Claude Vision, CFBD, ESPN, OpenCV, FFprobe, Timegrid).

Architecture:
    - Primary Strategy: Claude Vision (if enabled and configured as primary)
    - Fallback Strategy: CFBD → ESPN chain
    - Legacy Fallback: OpenCV → FFprobe → Timegrid (when no game context)

The dispatch layer is responsible for:
    1. Choosing which detection method to try
    2. Handling graceful fallback if a method fails
    3. Returning unified results with metadata about which method was used
    4. Logging visibility into the detection path taken
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


async def try_claude_vision(
    video_path: str,
    game_info: Optional[Dict],
    settings: Any,
    espn_game_id: Optional[str] = None,
) -> DetectionResult:
    """Attempt play detection using Claude Vision.

    Args:
        video_path: Path to video file
        game_info: Game context (away_team, home_team, team, etc.)
        settings: Application settings
        espn_game_id: Optional ESPN game ID for validation

    Returns:
        DetectionResult with plays and metadata, or empty result if failed
    """
    logger.info("[DISPATCH] Attempting Claude Vision detection...")

    try:
        from .claude_play_detector import ClaudePlayDetector

        # Initialize detector
        detector = ClaudePlayDetector(api_key=settings.anthropic_api_key)

        # Detect plays
        claude_windows = await detector.detect_plays(
            video_path,
            game_info=game_info,
            num_frames=settings.CLAUDE_VISION_FRAMES,
            espn_game_id=espn_game_id,
            enable_espn_validation=bool(espn_game_id),
        )

        if claude_windows and len(claude_windows) > 0:
            logger.info(f"[DISPATCH] ✓ Claude Vision succeeded: {len(claude_windows)} plays detected")

            # Convert windows to standardized play format
            plays = [
                {
                    "id": i,
                    "game_id": game_info.get("game_id", 0) if game_info else 0,
                    "timestamp": start,
                    "end_timestamp": end,
                    "source": "claude_vision",
                }
                for i, (start, end) in enumerate(claude_windows)
            ]

            metadata = {
                "status": "success",
                "plays_count": len(plays),
                "frames_analyzed": settings.CLAUDE_VISION_FRAMES,
                "espn_validation": bool(espn_game_id),
            }

            return DetectionResult(plays, "claude_vision", metadata)
        else:
            logger.warning("[DISPATCH] ✗ Claude Vision returned no plays")
            return DetectionResult([], "claude_vision", {"status": "no_plays"})

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
    """Orchestrate play detection with configurable strategy and graceful fallback.

    Detection order depends on CLAUDE_VISION_PRIMARY setting:
        - If True: Claude Vision → CFBD → ESPN
        - If False: CFBD → ESPN → Claude Vision (original behavior)

    Args:
        video_path: Path to video file
        game_id: CFBD/ESPN game ID
        game_info: Game context dictionary
        cfbd_client: CFBD client instance
        settings: Application settings
        year: Season year
        week: Week number
        season_type: Season type
        team_name: Team name

    Returns:
        DetectionResult with plays and metadata about which method succeeded
    """
    logger.info("\n" + "=" * 80)
    logger.info("[DETECTION DISPATCH] Starting detection with graceful fallback")
    logger.info("=" * 80)
    logger.info(f"  Configuration:")
    logger.info(f"    CLAUDE_VISION_ENABLE: {settings.CLAUDE_VISION_ENABLE}")
    logger.info(f"    CLAUDE_VISION_PRIMARY: {getattr(settings, 'CLAUDE_VISION_PRIMARY', False)}")
    logger.info(f"    Game ID: {game_id}")
    logger.info(f"    Video path: {video_path}")
    logger.info("")

    # Determine detection order based on configuration
    claude_vision_primary = getattr(settings, "CLAUDE_VISION_PRIMARY", False)
    claude_vision_enabled = settings.CLAUDE_VISION_ENABLE and settings.anthropic_api_key

    if claude_vision_primary and claude_vision_enabled:
        logger.info("[DISPATCH] Detection order: Claude Vision (PRIMARY) → CFBD → ESPN")

        # Try Claude Vision first
        result = await try_claude_vision(
            video_path,
            game_info,
            settings,
            espn_game_id=str(game_id) if game_id else None,
        )
        if result:
            logger.info(f"[DISPATCH] ✓ PRIMARY SUCCESS: Claude Vision detected {len(result)} plays")
            return result

        logger.info("[DISPATCH] Claude Vision returned no plays, falling back to CFBD/ESPN...")

    else:
        logger.info("[DISPATCH] Detection order: CFBD → ESPN → Claude Vision (FALLBACK)")

    # Try CFBD if game_id is available
    if game_id and cfbd_client:
        result = await try_cfbd(
            cfbd_client,
            game_id,
            year,
            week,
            season_type,
        )
        if result:
            logger.info(f"[DISPATCH] ✓ CFBD SUCCESS: {len(result)} plays detected")
            return result

        logger.info("[DISPATCH] CFBD failed, attempting ESPN fallback...")

        # Try ESPN fallback
        if game_id:
            result = await try_espn(
                str(game_id),
                team_name or "unknown",
            )
            if result:
                logger.info(f"[DISPATCH] ✓ ESPN SUCCESS: {len(result)} plays detected")
                return result

            logger.info("[DISPATCH] ESPN failed, attempting Claude Vision fallback...")

    # Try Claude Vision as fallback (if not already tried as primary)
    if not claude_vision_primary and claude_vision_enabled:
        result = await try_claude_vision(
            video_path,
            game_info,
            settings,
            espn_game_id=str(game_id) if game_id else None,
        )
        if result:
            logger.info(f"[DISPATCH] ✓ FALLBACK SUCCESS: Claude Vision detected {len(result)} plays")
            return result

        logger.info("[DISPATCH] Claude Vision fallback returned no plays")

    # All methods failed
    logger.error("[DISPATCH] ✗ ALL DETECTION METHODS FAILED")
    logger.info("=" * 80 + "\n")

    return DetectionResult(
        [],
        "none",
        {
            "status": "all_methods_failed",
            "error": "All detection methods (Claude Vision, CFBD, ESPN) failed to detect plays",
        },
    )
