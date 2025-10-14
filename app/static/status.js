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
    statusEl.textContent = `Job queued: ${jobId}. Processing…`;

    const humanStage = (job) => {
      if (job.status === 'completed') return 'Completed';
      if (job.status === 'failed') return 'Failed';
      switch (job.stage) {
        case 'downloading':
          return 'Downloading video…';
        case 'detecting':
          return 'Detecting plays…';
        case 'segmenting':
          return 'Cutting clips…';
        default:
          return 'Queued…';
      }
    };

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

      let line = humanStage(js);
      if (js.progress !== null && js.progress !== undefined) {
        line += ` ${js.progress}%`;
      }
      if (js.download && js.download.bytes) {
        const mb = (js.download.bytes / 1048576).toFixed(1);
        const tot = js.download.total_bytes ? (js.download.total_bytes / 1048576).toFixed(1) : '?';
        line += ` — ${mb} / ${tot} MB`;
      }
      statusEl.textContent = line;

      if (js.status === 'failed') {
        try {
          const er = await fetch(`/jobs/${jobId}/error`, { cache: 'no-store' });
          if (er.ok) {
            const ej = await er.json();
            statusEl.textContent = 'Failed: ' + (ej.error || 'Unknown');
          }
        } catch (_) {}
        btn.disabled = false;
        return;
      }

      if (js.status === 'completed') {
        let manifest;
        try {
          const mr = await fetch(`/jobs/${jobId}/manifest`, { cache: 'no-store' });
          if (!mr.ok) {
            throw new Error('manifest_not_ready');
          }
          const meta = await mr.json();
          if (meta.redirect) {
            const r2 = await fetch(meta.redirect, { cache: 'no-store' });
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
          return;
        }

        resultEl.style.display = 'block';
        resultEl.textContent = JSON.stringify(manifest, null, 2);
        const link = document.createElement('a');
        link.href = `/jobs/${jobId}/download`;
        link.textContent = 'Download ZIP';
        link.className = 'link';
        link.addEventListener('click', async (event) => {
          event.preventDefault();
          try {
            const dr = await fetch(`/jobs/${jobId}/download`, { cache: 'no-store' });
            if (!dr.ok) return;
            const dj = await dr.json();
            if (dj.redirect) {
              window.location = dj.redirect;
            }
          } catch (_) {}
        });
        statusEl.textContent = 'Completed';
        statusEl.append(' ', link);
        btn.disabled = false;
        return;
      }

      setTimeout(poll, 2000);
    };

    poll();
  });
});
