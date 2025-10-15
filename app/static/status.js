document.addEventListener('DOMContentLoaded', () => {
  const stageChips = ['Queued', 'Downloading', 'Detecting', 'Bucketing', 'Segmenting', 'Packaging', 'Completed', 'Failed', 'Canceled'];

  const form = document.getElementById('job-form');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const errorEl = document.getElementById('error_box');
  const submitBtn = document.getElementById('submit_btn');
  const cookieEl = document.getElementById('cookie_status');
  const selftestBtn = document.getElementById('selftest');
  const selftestOut = document.getElementById('selftest_out');

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
  };

  const attachCfbdSummary = (manifest) => {
    if (!manifest || typeof manifest !== 'object') return;
    const cfbd = manifest.cfbd || {};
    if (!(cfbd.cfbd_plays || cfbd.ocr_samples || cfbd.aligned_clips)) return;
    const line = statusEl.querySelector('.status-line');
    if (!line) return;
    const meta = document.createElement('span');
    meta.className = 'status-summary muted';
    meta.textContent = `CFBD plays: ${cfbd.cfbd_plays ?? 0} • OCR samples: ${cfbd.ocr_samples ?? 0} • Aligned: ${cfbd.aligned_clips ?? 0}`;
    line.append(' — ', meta);
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
    const pct = typeof job.pct === 'number' ? job.pct : 0;
    const detail = job.detail ? ` • ${job.detail}` : '';
    const etaTxt = humanEta(job.eta_sec);
    const key = stageKey(job);
    const label = stageChips.find((name) => name.toLowerCase() === key) || (job.stage || job.status || 'queued');

    statusEl.innerHTML = '';
    const line = document.createElement('div');
    line.className = 'status-line';
    line.textContent = `${label} — ${pct.toFixed(1)}%${etaTxt}${detail}`;
    statusEl.appendChild(line);
    statusEl.appendChild(renderTimeline(job));

    const isTerminal = ['completed', 'failed', 'canceled'].includes(key);
    if (cancelBtn) {
      if (isTerminal) {
        cancelBtn.disabled = true;
        if (cancelBtn.parentElement) cancelBtn.parentElement.removeChild(cancelBtn);
      } else {
        cancelBtn.disabled = cancelBtn.dataset.locked === '1';
        cancelBtn.style.marginTop = '8px';
        statusEl.appendChild(cancelBtn);
      }
    }
  };

  const resetOutputs = () => {
    resultEl.style.display = 'none';
    resultEl.textContent = '';
    errorEl.textContent = '';
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

    const cfbd = {
      use_cfbd: document.getElementById('cfbd_use').checked,
      game_id: parseInt(document.getElementById('cfbd_game_id').value || '0', 10) || null,
      season: parseInt(document.getElementById('cfbd_season').value || '0', 10) || null,
      week: parseInt(document.getElementById('cfbd_week').value || '0', 10) || null,
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

      if (job.status === 'completed') {
        errorEl.textContent = '';
        try {
          const res = await fetch(`/jobs/${jobId}/result`, { cache: 'no-store' });
          if (!res.ok) throw new Error('result_not_ready');
          const payload = await res.json();
          const manifestUrl = payload.manifest_url;
          const zipUrl = payload.archive_url;

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

          const showManifest = (manifest) => {
            resultEl.style.display = 'block';
            if (typeof manifest === 'string') {
              resultEl.textContent = manifest;
            } else {
              resultEl.textContent = JSON.stringify(manifest, null, 2);
            }
            errorEl.textContent = '';
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
              attachSummary(manifest);
              attachCfbdSummary(manifest);
            } catch (e) {
              try {
                const pr = await fetch(`/manifest-proxy?url=${encodeURIComponent(manifestUrl)}`, { cache: 'no-store' });
                if (!pr.ok) throw new Error('proxy ' + pr.status);
                const manifest = await parseManifestResponse(pr);
                showManifest(manifest);
                attachSummary(manifest);
                attachCfbdSummary(manifest);
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
