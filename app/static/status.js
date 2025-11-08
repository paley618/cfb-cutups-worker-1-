document.addEventListener('DOMContentLoaded', () => {
  const stageChips = ['Queued', 'Downloading', 'Detecting', 'Bucketing', 'Segmenting', 'Packaging', 'Uploading', 'Completed', 'Failed', 'Canceled'];

  const form = document.getElementById('job-form');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const errorEl = document.getElementById('error_box');
  const submitBtn = document.getElementById('submit_btn');
  const cookieEl = document.getElementById('cookie_status');
  const selftestBtn = document.getElementById('selftest');
  const selftestOut = document.getElementById('selftest_out');

  const cfbdUseCheckbox = document.getElementById('cfbd_use');
  const cfbdLinkInput = document.getElementById('cfbd_espn_link');
  const cfbdAutofillBtn = document.getElementById('cfbd_autofill_btn');
  const cfbdAutofillStatus = document.getElementById('cfbd_autofill_status');
  const cfbdExtraFields = document.getElementById('cfbd-extra-fields');
  const cfbdYearInput = document.getElementById('cfbdYear');
  const cfbdWeekInput = document.getElementById('cfbdWeek');
  window.__cfbOrchestrated = null;

  // NEW: Team/Conference/Game selection dropdowns
  const teamSelect = document.getElementById('team_select');
  const conferenceSelect = document.getElementById('conference_select');
  const yearSelect = document.getElementById('year_select');
  const gameSelect = document.getElementById('game_select');
  window.__selectedGameData = null;

  // Load teams and conferences on page load
  const loadDropdownOptions = async () => {
    try {
      const response = await fetch('/api/options');
      const data = await response.json();

      // Populate conference dropdown
      if (conferenceSelect && data.conferences) {
        data.conferences.forEach(conf => {
          const option = document.createElement('option');
          option.value = conf;
          option.textContent = conf;
          conferenceSelect.appendChild(option);
        });
      }

      // Populate year dropdown
      if (yearSelect && data.years) {
        data.years.forEach(year => {
          const option = document.createElement('option');
          option.value = year;
          option.textContent = year;
          yearSelect.appendChild(option);
        });
        // Set default year to 2024
        yearSelect.value = '2024';
      }
    } catch (error) {
      console.error('Failed to load dropdown options:', error);
    }
  };

  // When conference is selected, load teams for that conference
  const loadTeamsForConference = async () => {
    if (!conferenceSelect || !teamSelect) return;

    const conference = conferenceSelect.value;

    // Clear team and game dropdowns
    teamSelect.innerHTML = '<option value="">-- Loading teams... --</option>';
    gameSelect.innerHTML = '<option value="">-- Select Team and Year First --</option>';
    gameSelect.disabled = true;
    window.__selectedGameData = null;

    if (!conference) {
      teamSelect.innerHTML = '<option value="">-- Pick a Conference First --</option>';
      teamSelect.disabled = true;
      return;
    }

    try {
      const response = await fetch(`/api/teams-by-conference?conference=${encodeURIComponent(conference)}`);
      const data = await response.json();

      teamSelect.innerHTML = '<option value="">-- Select a Team --</option>';

      if (data.error || !data.teams || data.teams.length === 0) {
        teamSelect.innerHTML += '<option disabled>No teams found</option>';
        teamSelect.disabled = true;
      } else {
        data.teams.forEach(team => {
          const option = document.createElement('option');
          option.value = team;
          option.textContent = team;
          teamSelect.appendChild(option);
        });
        teamSelect.disabled = false;
      }
    } catch (error) {
      console.error('Failed to load teams:', error);
      teamSelect.innerHTML = '<option disabled>Error loading teams</option>';
      teamSelect.disabled = true;
    }
  };

  // When team is selected, fetch available games
  const loadGamesForTeam = async () => {
    if (!teamSelect || !yearSelect || !gameSelect) return;

    const team = teamSelect.value;
    const year = yearSelect.value;

    if (!team) {
      gameSelect.disabled = true;
      gameSelect.innerHTML = '<option value="">-- Pick a team first --</option>';
      window.__selectedGameData = null;
      return;
    }

    try {
      gameSelect.disabled = true;
      gameSelect.innerHTML = '<option value="">Loading games...</option>';

      const response = await fetch(`/api/games?team=${encodeURIComponent(team)}&year=${year}`);
      const data = await response.json();

      gameSelect.innerHTML = '<option value="">-- Select Game --</option>';

      if (data.error || !data.games || data.games.length === 0) {
        gameSelect.innerHTML += '<option disabled>No games available</option>';
        gameSelect.disabled = true;
        window.__selectedGameData = null;
      } else {
        data.games.forEach(game => {
          const option = document.createElement('option');
          option.value = game.id;
          option.textContent = game.display;
          option.dataset.gameData = JSON.stringify(game);
          gameSelect.appendChild(option);
        });
        gameSelect.disabled = false;
      }
    } catch (error) {
      console.error('Failed to load games:', error);
      gameSelect.innerHTML = '<option value="">Error loading games</option>';
      gameSelect.disabled = true;
      window.__selectedGameData = null;
    }
  };

  // When a game is selected, store the game data and create orchestrated payload
  const handleGameSelection = async () => {
    if (!gameSelect) return;

    const selectedOption = gameSelect.options[gameSelect.selectedIndex];
    if (!selectedOption || !selectedOption.value) {
      window.__selectedGameData = null;
      window.__cfbOrchestrated = null;
      return;
    }

    try {
      const gameData = JSON.parse(selectedOption.dataset.gameData || '{}');
      window.__selectedGameData = gameData;

      // Create a simple orchestrated payload for the dropdown selection
      // This mimics what the ESPN autofill does
      const team = teamSelect.value;
      const year = yearSelect.value;

      window.__cfbOrchestrated = {
        status: 'READY',
        decision: {
          chosen_source: 'cfbd',
          anomalies: []
        },
        cfbd_match: {
          status: 'OK',
          cfbdGameId: gameData.id,
          gameId: gameData.id,
          year: parseInt(year),
          week: gameData.week,
          seasonType: 'regular',
          cfbdHome: gameData.home_team,
          cfbdAway: gameData.away_team,
          homeTeam: gameData.home_team,
          awayTeam: gameData.away_team,
          playsCount: 0  // Will be fetched when job is submitted
        }
      };

      // Update the status display
      if (cfbdAutofillStatus) {
        renderCfbdAutofillStatus([
          `${gameData.away_team} @ ${gameData.home_team}`,
          `Week ${gameData.week} - ${gameData.start_date}`,
          'Game selected — ready to submit.'
        ], 'ok');
      }

      // Enable CFBD checkbox
      if (cfbdUseCheckbox) {
        cfbdUseCheckbox.checked = true;
      }
    } catch (error) {
      console.error('Failed to handle game selection:', error);
      window.__selectedGameData = null;
      window.__cfbOrchestrated = null;
    }
  };

  // Attach event listeners for cascading dropdowns
  if (conferenceSelect) {
    conferenceSelect.addEventListener('change', loadTeamsForConference);
  }
  if (teamSelect) {
    teamSelect.addEventListener('change', loadGamesForTeam);
  }
  if (yearSelect) {
    yearSelect.addEventListener('change', loadGamesForTeam);
  }
  if (gameSelect) {
    gameSelect.addEventListener('change', handleGameSelection);
  }

  // Load dropdown options on page load
  loadDropdownOptions();

  const clearCfbdStatus = () => {
    window.__cfbOrchestrated = null;
    if (!cfbdAutofillStatus) return;
    cfbdAutofillStatus.innerHTML = '';
    cfbdAutofillStatus.textContent = '';
    cfbdAutofillStatus.style.color = '';
    cfbdAutofillStatus.classList.remove('error');
    cfbdAutofillStatus.classList.remove('needs-year');
    cfbdAutofillStatus.classList.remove('warn');
    delete cfbdAutofillStatus.dataset.status;
    if (cfbdExtraFields) {
      cfbdExtraFields.style.display = 'none';
    }
  };

  const renderCfbdAutofillStatus = (lines, mode = '') => {
    if (!cfbdAutofillStatus) return;
    cfbdAutofillStatus.innerHTML = '';
    cfbdAutofillStatus.classList.remove('error');
    cfbdAutofillStatus.classList.remove('needs-year');
    cfbdAutofillStatus.classList.remove('warn');
    if (mode) {
      cfbdAutofillStatus.dataset.status = mode;
    } else {
      delete cfbdAutofillStatus.dataset.status;
    }
    if (mode === 'error') {
      cfbdAutofillStatus.classList.add('error');
    } else if (mode === 'needs-year') {
      cfbdAutofillStatus.classList.add('needs-year');
    } else if (mode === 'warn') {
      cfbdAutofillStatus.classList.add('warn');
    }
    const colorMap = {
      ok: '#6f6',
      warn: '#f9b234',
      error: '#f66',
      'needs-year': '#f66',
      info: '#ccc',
    };
    if (mode && colorMap[mode]) {
      cfbdAutofillStatus.style.color = colorMap[mode];
    } else {
      cfbdAutofillStatus.style.color = '';
    }
    if (!Array.isArray(lines) || !lines.length) {
      return;
    }
    lines.forEach((text, idx) => {
      const div = document.createElement('div');
      div.className = 'cfbd-line';
      if (mode === 'ok' && idx === 0) {
        div.classList.add('cfbd-line--primary');
      }
      if ((mode === 'ok' || mode === 'warn') && idx === lines.length - 1) {
        div.classList.add('cfbd-line--status');
      }
      div.textContent = text;
      cfbdAutofillStatus.appendChild(div);
    });
  };

  if (cfbdLinkInput) {
    cfbdLinkInput.addEventListener('input', () => {
      clearCfbdStatus();
      if (cfbdYearInput) cfbdYearInput.value = '';
      if (cfbdWeekInput) cfbdWeekInput.value = '';
    });
  }

  if (cfbdAutofillBtn) {
    cfbdAutofillBtn.addEventListener('click', async (event) => {
      event.preventDefault();
      if (!cfbdLinkInput) return;
      const espnUrl = cfbdLinkInput.value.trim();
      if (!espnUrl) {
        window.__cfbOrchestrated = null;
        renderCfbdAutofillStatus(
          ['Paste an ESPN game link or event id.'],
          'error',
        );
        if (cfbdExtraFields) {
          cfbdExtraFields.style.display = 'none';
        }
        return;
      }

      cfbdAutofillBtn.disabled = true;
      window.__cfbOrchestrated = null;
      renderCfbdAutofillStatus(['Resolving ESPN summary…'], 'info');

      try {
        const espnResp = await fetch(
          `/api/util/espn-resolve?espnUrl=${encodeURIComponent(espnUrl)}`,
          { cache: 'no-store' },
        );
        if (!espnResp.ok) {
          throw new Error(`HTTP ${espnResp.status}`);
        }
        const espnData = await espnResp.json();
        if (espnData.status !== 'ESPN_OK') {
          const lines = ['ESPN failed or needs more info.'];
          const resolver = espnData.resolver || {};
          const nextAction = resolver.next_action || '';
          if (nextAction === 'ask_user_for_year_week' && cfbdExtraFields) {
            cfbdExtraFields.style.display = 'flex';
          }
          if (resolver.message) {
            lines.push(resolver.message);
          }
          if (Array.isArray(resolver.suggestions) && resolver.suggestions.length) {
            lines.push(`Suggestions: ${resolver.suggestions.join('; ')}`);
          }
          renderCfbdAutofillStatus(lines, 'error');
          return;
        }

        const summary = espnData.espn_summary || {};
        const competition = summary?.header?.competitions?.[0] || {};
        const competitors = Array.isArray(competition?.competitors)
          ? competition.competitors
          : [];
        const pickTeam = (side) => {
          const entry = competitors.find(
            (comp) => (comp?.homeAway || comp?.homeaway) === side,
          );
          if (!entry) return null;
          const team = entry.team || {};
          return (
            team.displayName ||
            team.shortDisplayName ||
            team.name ||
            team.abbreviation ||
            null
          );
        };
        const home = pickTeam('home') || 'Home team';
        const away = pickTeam('away') || 'Away team';
        const seasonYear = summary?.header?.season?.year;
        const weekText = summary?.header?.week?.text;
        const statusText =
          competition?.status?.type?.detail ||
          competition?.status?.type?.shortDetail ||
          competition?.status?.type?.description ||
          null;

        const baseLines = [`${home} vs ${away}`];
        if (seasonYear) {
          let seasonLine = `Season ${seasonYear}`;
          if (weekText) {
            seasonLine += ` — ${weekText}`;
          }
          baseLines.push(seasonLine);
        }
        if (statusText) {
          baseLines.push(statusText);
        }

        if (cfbdExtraFields) {
          cfbdExtraFields.style.display = 'none';
        }
        if (cfbdYearInput) cfbdYearInput.value = '';
        if (cfbdWeekInput) cfbdWeekInput.value = '';
        renderCfbdAutofillStatus(
          [...baseLines, 'ESPN OK — matching to CFBD...'],
          'ok',
        );

        const cfbdResp = await fetch('/api/util/cfbd-match-from-espn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ espn_summary: summary }),
        });
        const cfbdData = await cfbdResp.json();

        let espnPbpData = null;
        let statusLines = [...baseLines];
        let statusMode = 'ok';

        if (
          cfbdData.status === 'CFBD_SUSPECT' &&
          cfbdData.fallback?.action === 'use_espn_pbp'
        ) {
          try {
            const pbpResp = await fetch(
              `/api/util/espn-playbyplay?espnUrl=${encodeURIComponent(espnUrl)}`,
            );
            if (!pbpResp.ok) {
              throw new Error(`HTTP ${pbpResp.status}`);
            }
            espnPbpData = await pbpResp.json();
          } catch (err) {
            const message = err && err.message ? err.message : 'Unknown error';
            espnPbpData = { status: 'ESPN_PBP_ERROR', message };
          }

          if (espnPbpData?.status === 'ESPN_PBP_OK') {
            statusMode = 'warn';
            statusLines = [
              ...baseLines,
              'CFBD looked wrong, but ESPN play-by-play is available (using ESPN data).',
            ];
          } else {
            statusMode = 'error';
            statusLines = [...baseLines, 'CFBD looked wrong and ESPN play-by-play also failed.'];
            if (espnPbpData?.message) {
              statusLines.push(`ESPN error: ${espnPbpData.message}`);
            }
          }
        } else if (cfbdData.status === 'OK') {
          const weekLabel = cfbdData.week != null ? cfbdData.week : 'N/A';
          statusMode = 'ok';
          statusLines = [
            ...baseLines,
            `CFBD OK — ${cfbdData.cfbdHome || home} vs ${
              cfbdData.cfbdAway || away
            }, ${cfbdData.year} week ${weekLabel} — ${cfbdData.playsCount} plays`,
          ];
        } else {
          try {
            const pbpResp = await fetch(
              `/api/util/espn-playbyplay?espnUrl=${encodeURIComponent(espnUrl)}`,
            );
            if (!pbpResp.ok) {
              throw new Error(`HTTP ${pbpResp.status}`);
            }
            espnPbpData = await pbpResp.json();
          } catch (err) {
            const message = err && err.message ? err.message : 'Unknown error';
            espnPbpData = { status: 'ESPN_PBP_ERROR', message };
          }

          if (espnPbpData?.status === 'ESPN_PBP_OK') {
            statusMode = 'warn';
            statusLines = [...baseLines, 'CFBD failed, using ESPN play-by-play.'];
          } else {
            statusMode = 'error';
            const message =
              cfbdData.message || cfbdData.status || 'CFBD match failed';
            statusLines = [...baseLines, `CFBD failed: ${message}`];
            if (espnPbpData?.message) {
              statusLines.push(`ESPN error: ${espnPbpData.message}`);
            }
          }
        }

        renderCfbdAutofillStatus(statusLines, statusMode);

        const videoInput = document.getElementById('video_url');
        const videoUrl = videoInput?.value?.trim() || '';
        let videoMeta = null;
        if (videoUrl) {
          try {
            const videoResp = await fetch(
              `/api/util/video-meta?videoUrl=${encodeURIComponent(videoUrl)}`,
            );
            if (videoResp.ok) {
              videoMeta = await videoResp.json();
            }
          } catch (_) {
            videoMeta = null;
          }
        }

        const orchestrateResp = await fetch('/api/util/orchestrate-game', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            espn_summary: summary,
            espn_pbp: espnPbpData?.raw || null,
            cfbd_match: cfbdData,
            video_meta: videoMeta,
          }),
        });
        if (!orchestrateResp.ok) {
          throw new Error(`Orchestrator HTTP ${orchestrateResp.status}`);
        }
        const orchData = await orchestrateResp.json();
        window.__cfbOrchestrated = orchData;

        const anomalies = Array.isArray(orchData.decision?.anomalies)
          ? orchData.decision.anomalies
          : [];
        const finalLines = [...statusLines];
        let finalMode = orchData.status === 'READY' ? 'ok' : 'warn';
        if (statusMode === 'error' && orchData.status !== 'READY') {
          finalMode = 'error';
        }
        if (orchData.status === 'READY') {
          finalLines.push('Data looks good — ready to submit.');
        } else {
          const anomalyText = anomalies.length
            ? `Data is usable but has issues: ${JSON.stringify(anomalies)}`
            : 'Data is usable but has issues.';
          finalLines.push(anomalyText);
        }
        renderCfbdAutofillStatus(finalLines, finalMode);

        if (cfbdUseCheckbox) {
          const chosen = orchData?.decision?.chosen_source;
          if (chosen === 'cfbd') {
            cfbdUseCheckbox.checked = true;
          } else if (chosen === 'espn') {
            cfbdUseCheckbox.checked = false;
          }
        }
        const requireCheckbox = document.getElementById('cfbd_require');
        if (requireCheckbox && window.__cfbOrchestrated?.decision?.chosen_source !== 'cfbd') {
          requireCheckbox.checked = false;
        }
      } catch (err) {
        const message = err && err.message ? err.message : 'Unknown error';
        window.__cfbOrchestrated = null;
        renderCfbdAutofillStatus([`Autofill failed: ${message}`], 'error');
        if (cfbdExtraFields) {
          cfbdExtraFields.style.display = 'none';
        }
      } finally {
        cfbdAutofillBtn.disabled = false;
      }
    });
  }

  const addLine = (text) => {
    if (!statusEl) return;
    const line = document.createElement('div');
    line.className = 'status-extra muted';
    line.textContent = text;
    statusEl.appendChild(line);
  };

  function renderCFBD(j) {
    if (!statusEl) return;
    const meta = (j.manifest && j.manifest.detector_meta) || {};
    const s = meta.cfbd_state || 'off';
    const r = meta.cfbd_reason || ''; // reason now includes filtered play counts when available
    const cached = meta.cfbd_cached ? ` • cached=${meta.cfbd_cached_count}` : '';
    const el =
      document.getElementById('cfbd-state') ||
      (() => {
        const d = document.createElement('div');
        d.id = 'cfbd-state';
        statusEl.appendChild(d);
        return d;
      })();
    const classes = {
      ready: 'badge-success',
      pending: 'badge-info',
      error: 'badge-warn',
      unavailable: 'badge-warn',
      off: 'badge',
    };
    el.innerHTML = `<span class="badge ${classes[s] || 'badge'}">CFBD: ${s.toUpperCase()}</span>${cached} ${
      r ? `<span class="muted">(${r})</span>` : ''
    }`;
  }

  if (selftestBtn && selftestOut) {
    selftestBtn.onclick = async () => {
      selftestOut.style.display = 'block';
      selftestOut.textContent = 'Running…';
      try {
        const r = await fetch('/__selftest', { cache: 'no-store' });
        const j = await r.json();
        selftestOut.textContent = JSON.stringify(j, null, 2);
      } catch (e) {
        selftestOut.textContent = 'Self-test failed to run: ' + e.message;
      }
    };
  }

  const stageKey = (job) => (job.stage || job.status || 'queued').toLowerCase();

  const attachSummary = (manifest) => {
    if (!manifest || typeof manifest !== 'object') return;
    const line = statusEl.querySelector('.status-line');
    if (!line) return;
    statusEl.querySelectorAll('.status-extra').forEach((el) => el.remove());
    line.querySelectorAll('.status-summary').forEach((el) => el.remove());
    const src = manifest.source || {};
    const meta = manifest.detector_meta || {};
    const minutes = Math.round((Number(src.duration_sec) || 0) / 60);
    const clipCount = manifest.metrics?.num_clips ?? meta.clips_found ?? 0;
    const summary = document.createElement('span');
    summary.className = 'status-summary muted';
    let text = `Source ${minutes} min • ${clipCount} clips`;
    if (meta.low_confidence) {
      text += ' • Low confidence (relaxed thresholds)';
    }
    const detectors = [];
    if (meta.audio_spikes_used != null) {
      detectors.push(meta.audio_spikes_used ? 'Audio spikes' : 'Audio off');
    }
    if (meta.scorebug_used != null) {
      detectors.push(meta.scorebug_used ? 'Scorebug ROI' : 'Scorebug off');
    }
    if (detectors.length) {
      text += ` • Detectors: ${detectors.join(' + ')}`;
    }
    summary.textContent = text;
    line.append(' — ', summary);

    const metrics = manifest.metrics || {};
    const debug = manifest.debug || {};
    addLine(
      `Audio spikes: ${metrics.audio_spikes ?? 0} • OCR: ${metrics.ocr_samples ?? 0} • Candidates: ${metrics.vision_candidates ?? 0} • Windows: ${metrics.post_merge_windows ?? 0}`,
    );
    if (Array.isArray(debug.timeline) && debug.timeline.length) {
      addLine(`Debug timeline: ${debug.timeline.length} imgs`);
    }
    if (Array.isArray(debug.candidates) && debug.candidates.length) {
      addLine(`Debug candidates: ${debug.candidates.length} imgs`);
    }

    const confSummary = manifest.quality?.confidence || {};
    const hideThreshold = manifest.settings?.CONF_HIDE_THRESHOLD ?? 40;
    addLine(
      `Confidence — median ${confSummary.median ?? 0} (p25 ${confSummary.p25 ?? 0}, p75 ${confSummary.p75 ?? 0}) • low (<${hideThreshold}): ${confSummary.low_count ?? 0}/${confSummary.total ?? 0}`,
    );

    if (statusEl) {
      let toggleLabel = document.getElementById('hideLowConfLabel');
      if (!toggleLabel) {
        const chk = document.createElement('input');
        chk.type = 'checkbox';
        chk.id = 'hideLowConf';
        chk.checked = true;
        chk.dataset.threshold = String(hideThreshold);
        toggleLabel = document.createElement('label');
        toggleLabel.id = 'hideLowConfLabel';
        toggleLabel.appendChild(chk);
        toggleLabel.appendChild(document.createTextNode(` Hide clips < ${hideThreshold} confidence`));
        statusEl.appendChild(document.createElement('br'));
        statusEl.appendChild(toggleLabel);
        const renderClips = () => {
          const thresh = Number(chk.dataset.threshold ?? hideThreshold);
          const minConf = chk.checked ? thresh : 0;
          document.dispatchEvent(
            new CustomEvent('clips-filter-change', { detail: { minConfidence: minConf } }),
          );
        };
        chk.addEventListener('change', renderClips);
        renderClips();
      } else {
        const chk = toggleLabel.querySelector('#hideLowConf');
        if (chk) {
          chk.dataset.threshold = String(hideThreshold);
          chk.dispatchEvent(new Event('change'));
        }
        const textNode = toggleLabel.childNodes[toggleLabel.childNodes.length - 1];
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          textNode.textContent = ` Hide clips < ${hideThreshold} confidence`;
        }
      }
    }
  };

  const attachCfbdSummary = (manifest) => {
    if (!manifest || typeof manifest !== 'object') return;
    const cfbd = manifest.cfbd || {};
    if (!(cfbd.requested || cfbd.used || cfbd.error)) return;
    const line = statusEl.querySelector('.status-line');
    if (!line) return;
    const parts = [
      `requested=${cfbd.requested ? 'true' : 'false'}`,
      `used=${cfbd.used ? 'true' : 'false'}`,
      `plays=${cfbd.plays ?? 0}`,
      `cfbd clips=${cfbd.clips ?? 0}`,
      `fallback clips=${cfbd.fallback_clips ?? 0}`,
    ];
    if (cfbd.mapping) parts.push(`mapping=${cfbd.mapping}`);
    if (cfbd.error) parts.push(`error=${cfbd.error}`);
    if (cfbd.finder?.match) parts.push(`finder=${cfbd.finder.match}`);
    if (cfbd.finder?.usedWeek != null) parts.push(`usedWeek=${cfbd.finder.usedWeek}`);
    if (cfbd.finder?.seasonType) parts.push(`seasonType=${cfbd.finder.seasonType}`);
    const meta = document.createElement('span');
    meta.className = 'status-summary muted';
    meta.textContent = `CFBD: ${parts.join(' • ')}`;
    line.append(' — ', meta);
  };

  const renderOutputs = (data) => {
    if (!statusEl) return;
    const manifest = data && data.manifest ? data.manifest : data;
    if (!manifest || typeof manifest !== 'object') return;
    const outputs = manifest.outputs || {};
    const reels = outputs.reels_by_bucket || {};
    const counts = manifest.bucket_counts || {};
    const teamNameRaw =
      (manifest.cfbd &&
        ((manifest.cfbd.request && manifest.cfbd.request.team) || manifest.cfbd.team)) ||
      '';
    const teamName = (teamNameRaw || '').toString().trim() || 'Team';
    const offenseLabel = `${teamName} Offense`;
    const oppLabel = teamName ? `Opponent Offense` : 'Opponent Offense';
    const specialLabel = 'Special Teams';
    const ensureBox = () => {
      const existing = document.getElementById('outputs');
      if (existing) return existing;
      const el = document.createElement('div');
      el.id = 'outputs';
      statusEl.appendChild(el);
      return el;
    };
    const box = ensureBox();

    const mk = (label, url, count) =>
      url
        ? `<a href="${url}" target="_blank">${label} (${count ?? 0})</a>`
        : `<span class="muted">${label} (${count ?? 0})</span>`;

    box.innerHTML = `
    <div class="reels">
      ${mk(offenseLabel, reels.team_offense, counts.team_offense)}
      &nbsp;|&nbsp;
      ${mk(oppLabel, reels.opp_offense, counts.opp_offense)}
      &nbsp;|&nbsp;
      ${mk(specialLabel, reels.special_teams, counts.special_teams)}
    </div>`;
  };

  const renderTimeline = (job) => {
    const wrap = document.createElement('div');
    wrap.className = 'timeline';
    const currentKey = stageKey(job);
    const currentIdx = stageChips.findIndex((name) => name.toLowerCase() === currentKey);
    stageChips.forEach((name, idx) => {
      const span = document.createElement('span');
      span.className = 'timeline-step';
      span.textContent = name;
      const lower = name.toLowerCase();
      if (currentIdx !== -1 && idx < currentIdx) {
        span.classList.add('done');
      }
      if (lower === currentKey) {
        span.classList.add('active');
      }
      if (['completed', 'failed', 'canceled'].includes(lower) && (job.status || '').toLowerCase() === lower) {
        span.classList.add('active');
      }
      wrap.appendChild(span);
    });
    return wrap;
  };

  const humanEta = (seconds) => {
    if (seconds == null) return '';
    const m = Math.floor(seconds / 60);
    const s = Math.max(0, Math.floor(seconds % 60));
    return ` • ETA ${m}m ${s}s`;
  };

  const renderStatus = (job, cancelBtn) => {
    const progress = job.progress || {};
    const pct = typeof job.pct === 'number' ? job.pct : 0;
    const detail = job.detail ? ` • ${job.detail}` : '';
    const etaSource = job.eta_sec != null ? job.eta_sec : progress.eta_seconds;
    const etaTxt = humanEta(etaSource);
    const key = stageKey(job);
    const label = stageChips.find((name) => name.toLowerCase() === key) || (job.stage || job.status || 'queued');

    statusEl.innerHTML = '';
    const line = document.createElement('div');
    line.className = 'status-line';
    line.textContent = `${label} — ${pct.toFixed(1)}%${etaTxt}${detail}`;
    statusEl.appendChild(line);
    statusEl.appendChild(renderTimeline(job));
    renderCFBD(job);

    const metaParts = [];
    if (job.elapsed_seconds != null) metaParts.push(`Elapsed: ${job.elapsed_seconds}s`);
    if (job.idle_seconds != null) metaParts.push(`Idle: ${job.idle_seconds}s`);
    if (progress.eta_seconds != null) metaParts.push(`ETA: ${progress.eta_seconds}s`);
    if (progress.clips_done != null && progress.clips_total != null) {
      metaParts.push(`Clips: ${progress.clips_done}/${progress.clips_total}`);
    }
    if (progress.downloaded_mb != null) {
      if (progress.total_mb != null) {
        metaParts.push(`Downloaded: ${progress.downloaded_mb}/${progress.total_mb} MB`);
      } else {
        metaParts.push(`Downloaded: ${progress.downloaded_mb} MB`);
      }
    }
    if (progress.last_uploaded) metaParts.push(`Last upload: ${progress.last_uploaded}`);
    if (metaParts.length) {
      const metaLine = document.createElement('div');
      metaLine.className = 'status-extra muted';
      metaLine.textContent = metaParts.join(' • ');
      statusEl.appendChild(metaLine);
    }

    const isTerminal = ['completed', 'failed', 'canceled'].includes(key);
    if (cancelBtn) {
      if (isTerminal) {
        cancelBtn.disabled = true;
        if (cancelBtn.parentElement) cancelBtn.parentElement.removeChild(cancelBtn);
      } else {
        if (job.cancel) {
          cancelBtn.disabled = true;
          cancelBtn.dataset.locked = '1';
          cancelBtn.textContent = 'Cancel Requested';
        } else {
          cancelBtn.disabled = cancelBtn.dataset.locked === '1';
          cancelBtn.textContent = 'Cancel';
        }
        cancelBtn.style.marginTop = '8px';
        statusEl.appendChild(cancelBtn);
      }
    }
  };

  const resetOutputs = () => {
    resultEl.style.display = 'none';
    resultEl.textContent = '';
    errorEl.textContent = '';
    const outBox = document.getElementById('outputs');
    if (outBox) outBox.innerHTML = '';
  };

  const fetchCookieStatus = async () => {
    try {
      const resp = await fetch('/has_cookies');
      if (resp.ok) {
        const payload = await resp.json();
        cookieEl.textContent = payload.has_cookies ? 'Server cookies: loaded' : 'Server cookies: not set';
      }
    } catch (_) {
      /* noop */
    }
  };

  fetchCookieStatus();

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const orchestrated = window.__cfbOrchestrated;
    if (!orchestrated) {
      alert('Please select a game using the team dropdown OR run the ESPN/CFBD autofill.');
      return;
    }

    submitBtn.disabled = true;
    statusEl.textContent = 'Submitting job…';
    resetOutputs();

    const payload = {
      video_url: document.getElementById('video_url').value.trim() || null,
      webhook_url: document.getElementById('webhook_url').value.trim() || null,
      options: {
        play_padding_pre: parseFloat(document.getElementById('play_padding_pre').value || '3'),
        play_padding_post: parseFloat(document.getElementById('play_padding_post').value || '5'),
        scene_thresh: parseFloat(document.getElementById('scene_thresh').value || '0.30'),
        min_duration: parseFloat(document.getElementById('min_duration').value || '4'),
        max_duration: parseFloat(document.getElementById('max_duration').value || '20'),
      },
    };

    const useCfbdToggle = document.getElementById('cfbd_use')?.checked ?? false;
    const requireToggle = document.getElementById('cfbd_require')?.checked ?? false;
    const chosenSource = orchestrated?.decision?.chosen_source;
    const usingCfbd = useCfbdToggle && chosenSource === 'cfbd';
    const requiringCfbd = requireToggle && usingCfbd;
    const cfbd = {
      use_cfbd: usingCfbd,
      require_cfbd: requiringCfbd,
      game_id: null,
      season: null,
      week: null,
      team: null,
      season_type: null,
      home_team: null,
      away_team: null,
    };

    const cfbdMatch = orchestrated?.cfbd_match;
    if (usingCfbd && cfbdMatch) {
      const rawGameId =
        cfbdMatch.cfbdGameId ?? cfbdMatch.gameId ?? cfbdMatch.game_id;
      const parsedGameId = Number(rawGameId);
      cfbd.game_id = Number.isFinite(parsedGameId) ? parsedGameId : rawGameId;
      cfbd.season = cfbdMatch.year ?? null;
      cfbd.week = cfbdMatch.week ?? null;
      const seasonType =
        cfbdMatch.seasonType || cfbdMatch.season_type || 'regular';
      if (seasonType) {
        cfbd.season_type = seasonType;
      }
      const homeTeam = cfbdMatch.cfbdHome || cfbdMatch.homeTeam || null;
      const awayTeam = cfbdMatch.cfbdAway || cfbdMatch.awayTeam || null;
      const teamName = homeTeam || awayTeam || null;
      cfbd.team = teamName;
      cfbd.home_team = homeTeam;
      cfbd.away_team = awayTeam;
    }

    payload.cfbd = cfbd;
    payload.orchestrator = orchestrated;

    if (cfbd.use_cfbd && !cfbd.game_id) {
      submitBtn.disabled = false;
      alert('CFBD autofill did not produce a valid game id.');
      return;
    }

    let response;
    try {
      response = await fetch('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      statusEl.textContent = 'Network error creating job.';
      submitBtn.disabled = false;
      return;
    }

    if (!response.ok) {
      statusEl.textContent = `Failed to create job (HTTP ${response.status}).`;
      submitBtn.disabled = false;
      return;
    }

    const data = await response.json();
    const jobId = data.job_id;

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.className = 'btn-secondary';
    cancelBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      cancelBtn.disabled = true;
      cancelBtn.dataset.locked = '1';
      try {
        await fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
      } catch (_) {
        /* noop */
      }
    });

    statusEl.textContent = `Job queued: ${jobId}`;

    const poll = async () => {
      let job;
      try {
        const sr = await fetch(`/jobs/${jobId}`, { cache: 'no-store' });
        if (!sr.ok) throw new Error('status_not_ok');
        job = await sr.json();
      } catch (err) {
        console.error('poll_failed', err);
        setTimeout(poll, 2000);
        return;
      }

      renderStatus(job, cancelBtn);
      renderCFBD(job);

      if (job.status === 'completed') {
        errorEl.textContent = '';
        try {
          const res = await fetch(`/jobs/${jobId}/result`, { cache: 'no-store' });
          if (!res.ok) throw new Error('result_not_ready');
          const payload = await res.json();
          const manifestUrl = payload.manifest_url;
          const zipUrl = payload.archive_url;
          const reelUrl = payload.reel_url;

          let manifestLink;
          if (manifestUrl) {
            manifestLink = document.createElement('a');
            manifestLink.href = manifestUrl;
            manifestLink.textContent = 'Manifest JSON';
            manifestLink.className = 'link';
            manifestLink.target = '_blank';
          }

          let zipLink;
          if (zipUrl) {
            zipLink = document.createElement('a');
            zipLink.href = zipUrl;
            zipLink.textContent = 'Download ZIP';
            zipLink.className = 'link';
            zipLink.target = '_blank';
          }

          if (manifestLink || zipLink) {
            const parts = [' '];
            if (manifestLink) parts.push(manifestLink);
            if (manifestLink && zipLink) parts.push(' ');
            if (zipLink) parts.push(zipLink);
            statusEl.append(...parts);
          }

          if (reelUrl) {
            const reelLink = document.createElement('a');
            reelLink.href = reelUrl;
            reelLink.textContent = 'Download Combined Video';
            reelLink.className = 'link';
            reelLink.target = '_blank';
            statusEl.append(' • ', reelLink);
          }

          const showManifest = (manifest) => {
            resultEl.style.display = 'block';
            if (typeof manifest === 'string') {
              resultEl.textContent = manifest;
              errorEl.textContent = '';
              return;
            }
            resultEl.textContent = JSON.stringify(manifest, null, 2);
            errorEl.textContent = '';
            attachSummary(manifest);
            attachCfbdSummary(manifest);
            renderCFBD({ manifest });
            renderOutputs({ manifest });
          };

          const parseManifestResponse = async (resp) => {
            try {
              return await resp.json();
            } catch (_) {
              return await resp.text();
            }
          };

          if (manifestUrl) {
            try {
              const r2 = await fetch(manifestUrl, { mode: 'cors', cache: 'no-store' });
              if (!r2.ok) throw new Error('HTTP ' + r2.status);
              const manifest = await parseManifestResponse(r2);
              showManifest(manifest);
            } catch (e) {
              try {
                const pr = await fetch(`/manifest-proxy?url=${encodeURIComponent(manifestUrl)}`, { cache: 'no-store' });
                if (!pr.ok) throw new Error('proxy ' + pr.status);
                const manifest = await parseManifestResponse(pr);
                showManifest(manifest);
              } catch (e2) {
                console.error('manifest_fetch_error', e, e2);
                errorEl.textContent = 'Completed, but manifest fetch failed (CORS/URL). Use the Manifest JSON link above.';
              }
            }
          }

          submitBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        } catch (err) {
          console.error('result_fetch_error', err);
          errorEl.textContent = 'Completed, but result is unavailable.';
          submitBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }
      }

      if (job.status === 'failed') {
        try {
          const er = await fetch(`/jobs/${jobId}/error`, { cache: 'no-store' });
          if (er.ok) {
            const ej = await er.json();
            errorEl.textContent = ej.error || 'Unknown error';
          } else {
            errorEl.textContent = 'Unknown error.';
          }
        } catch (_) {
          errorEl.textContent = 'Unknown error.';
        }
        submitBtn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      if (job.status === 'canceled') {
        errorEl.textContent = '';
        submitBtn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      setTimeout(poll, 1500);
    };

    poll();
  });
});
