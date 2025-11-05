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

  const gidInput = document.querySelector('input[name="cfbd_game_id"]');
  if (gidInput && !gidInput.dataset.espnReady) {
    gidInput.dataset.espnReady = '1';
    gidInput.addEventListener('input', () => {
      const v = gidInput.value.trim();
      const m = v.match(/\/gameId\/(\d+)/i);
      if (m) gidInput.value = m[1];
    });
  }

  let btn = document.getElementById('cfbd-selftest');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'cfbd-selftest';
    btn.className = 'btn';
    btn.textContent = 'Run CFBD self-test';
    const targetForm = form || document.querySelector('form');
    if (targetForm) {
      targetForm.insertBefore(btn, targetForm.firstChild);
    }
  }
  if (btn) {
    btn.onclick = async (e) => {
      e.preventDefault();
      let gid = (gidInput?.value || '').trim();
      if (!gid) return alert('Enter a CFBD game_id or ESPN URL first.');
      const m = gid.match(/\/gameId\/(\d+)/i);
      if (m) gid = m[1];
      const r = await fetch(`/diag/cfbd?gameId=${encodeURIComponent(gid)}`).then((x) => x.json());
      alert(r.ok ? `CFBD OK\n${r.url}\nstatus ${r.status}` : `CFBD ERROR\n${r.url}\n${r.error || r.body_prefix}`);
    };
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
    const r = meta.cfbd_reason || '';
    const el =
      document.getElementById('cfbd-state') ||
      (() => {
        const d = document.createElement('div');
        d.id = 'cfbd-state';
        statusEl.appendChild(d);
        return d;
      })();
    const map = { ready: 'badge-success', error: 'badge-warn', unavailable: 'badge-warn', off: 'badge' };
    el.innerHTML = `<span class="badge ${map[s] || 'badge'}">CFBD: ${s.toUpperCase()}</span> ${
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

    const gameIdRaw = (document.getElementById('cfbd_game_id').value || '').trim();
    const seasonRaw = (document.getElementById('cfbd_season').value || '').trim();
    const weekRaw = (document.getElementById('cfbd_week').value || '').trim();
    const seasonVal = seasonRaw ? parseInt(seasonRaw, 10) : NaN;
    const weekVal = weekRaw ? parseInt(weekRaw, 10) : NaN;
    const cfbd = {
      use_cfbd: document.getElementById('cfbd_use').checked,
      require_cfbd: document.getElementById('cfbd_require')?.checked ?? false,
      game_id: gameIdRaw || null,
      season: Number.isNaN(seasonVal) ? null : seasonVal,
      week: Number.isNaN(weekVal) ? null : weekVal,
      team: (document.getElementById('cfbd_team').value || '').trim() || null,
    };
    payload.cfbd = cfbd;

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
