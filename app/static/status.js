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

          if (manifestUrl) {
            try {
              const manifestResp = await fetch(manifestUrl, { cache: 'no-store', mode: 'cors' });
              const manifest = await manifestResp.json();
              resultEl.style.display = 'block';
              resultEl.textContent = JSON.stringify(manifest, null, 2);
            } catch (err) {
              console.error('manifest_fetch_error', err);
              errorEl.textContent = 'Completed, but failed to fetch manifest (CORS/URL).';
            }
          }

          if (zipUrl) {
            const link = document.createElement('a');
            link.href = zipUrl;
            link.textContent = 'Download ZIP';
            link.className = 'link';
            link.target = '_blank';
            statusEl.append(' ', link);
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
