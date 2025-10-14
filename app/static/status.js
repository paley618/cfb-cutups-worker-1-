// app/static/status.js
document.addEventListener('DOMContentLoaded', async () => {
  const form = document.getElementById('job-form');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const btn = document.getElementById('submit_btn');
  const cookieEl = document.getElementById('cookie_status');

  try {
    const resp = await fetch('/has_cookies');
    if (resp.ok) {
      const payload = await resp.json();
      cookieEl.textContent = payload.has_cookies ? 'Server cookies: loaded' : 'Server cookies: not set';
    }
  } catch (_) {}

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    btn.disabled = true;
    statusEl.textContent = 'Submitting job…';
    resultEl.style.display = 'none';
    resultEl.textContent = '';

    const payload = {
      video_url: document.getElementById('video_url').value.trim() || null,
      webhook_url: document.getElementById('webhook_url').value.trim() || null,
      options: {
        play_padding_pre: parseFloat(document.getElementById('play_padding_pre').value || '3'),
        play_padding_post: parseFloat(document.getElementById('play_padding_post').value || '5'),
        scene_thresh: parseFloat(document.getElementById('scene_thresh').value || '0.30'),
        min_duration: parseFloat(document.getElementById('min_duration').value || '4'),
        max_duration: parseFloat(document.getElementById('max_duration').value || '20'),
      }
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
      btn.disabled = false;
      return;
    }

    if (!response.ok) {
      statusEl.textContent = `Failed to create job (HTTP ${response.status}).`;
      btn.disabled = false;
      return;
    }

    const data = await response.json();
    const jobId = data.job_id;

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.marginLeft = '12px';
    cancelBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      cancelBtn.disabled = true;
      try {
        await fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
      } catch (_) {}
    });

    statusEl.textContent = `Job queued: ${jobId}`;
    statusEl.appendChild(cancelBtn);

    const poll = async () => {
      let js;
      try {
        const sr = await fetch(`/jobs/${jobId}`, { cache: 'no-store' });
        if (!sr.ok) {
          throw new Error('status_not_ok');
        }
        js = await sr.json();
      } catch (err) {
        console.error('poll_failed', err);
        setTimeout(poll, 2000);
        return;
      }

      const stageLabel = js.detail ? `${js.stage || js.status} (${js.detail})` : (js.stage || js.status);
      const pctValue = typeof js.pct === 'number' ? js.pct : 0;
      let etaTxt = '';
      if (js.eta_sec != null) {
        const m = Math.floor(js.eta_sec / 60);
        const s = Math.max(0, Math.floor(js.eta_sec % 60));
        etaTxt = ` • ETA ${m}m ${s}s`;
      }
      statusEl.textContent = `${stageLabel} — ${pctValue.toFixed(1)}%${etaTxt}`;
      statusEl.appendChild(cancelBtn);

      if (js.status === 'completed') {
        let manifest;
        try {
          const mr = await fetch(`/jobs/${jobId}/manifest`, { cache: 'no-store' });
          if (!mr.ok) {
            throw new Error('manifest_not_ready');
          }
          const meta = await mr.json();
          const manifestUrl = meta.redirect || meta.url || null;
          if (manifestUrl) {
            const r2 = await fetch(manifestUrl, { cache: 'no-store' });
            if (!r2.ok) {
              throw new Error('manifest_fetch_failed');
            }
            manifest = await r2.json();
          } else {
            manifest = meta;
          }
        } catch (err) {
          console.error('manifest_fetch_error', err);
          statusEl.textContent = 'Completed, but failed to fetch manifest.';
          btn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }

        resultEl.style.display = 'block';
        resultEl.textContent = JSON.stringify(manifest, null, 2);
        const link = document.createElement('a');
        link.href = `/jobs/${jobId}/download`;
        link.textContent = 'Download ZIP';
        link.className = 'link';
        statusEl.appendChild(document.createTextNode(' '));
        statusEl.appendChild(link);
        btn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      if (js.status === 'failed') {
        try {
          const er = await fetch(`/jobs/${jobId}/error`, { cache: 'no-store' });
          if (er.ok) {
            const ej = await er.json();
            statusEl.textContent = 'Failed: ' + (ej.error || 'Unknown');
          }
        } catch (_) {}
        btn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      if (js.status === 'canceled') {
        statusEl.textContent = 'Canceled';
        btn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      setTimeout(poll, 2000);
    };

    poll();
  });
});
