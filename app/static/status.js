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

    const poll = async (tries = 0) => {
      const manifestResp = await fetch(`/jobs/${jobId}/manifest`, { cache: 'no-store' });
      if (manifestResp.ok) {
        let manifest;
        try {
          const meta = await manifestResp.json();
          if (meta.redirect) {
            const follow = await fetch(meta.redirect, { cache: 'no-store' });
            if (!follow.ok) {
              throw new Error('follow_failed');
            }
            manifest = await follow.json();
          } else {
            manifest = meta;
          }
        } catch (err) {
          statusEl.textContent = 'Completed, but failed to fetch manifest.';
          btn.disabled = false;
          return;
        }

        statusEl.textContent = 'Completed.';
        resultEl.style.display = 'block';
        resultEl.textContent = JSON.stringify(manifest, null, 2);
        const link = document.createElement('a');
        link.href = '#';
        link.textContent = 'Download ZIP';
        link.className = 'link';
        link.addEventListener('click', async (event) => {
          event.preventDefault();
          try {
            const redirect = (await (await fetch(`/jobs/${jobId}/download`, { cache: 'no-store' })).json()).redirect;
            if (redirect) {
              window.location = redirect;
            }
          } catch (_) {}
        });
        statusEl.append(' ', link);
        btn.disabled = false;
        return;
      }

      if (tries % 6 === 0) {
        const errResp = await fetch(`/jobs/${jobId}/error`);
        if (errResp.ok) {
          const errJson = await errResp.json();
          statusEl.textContent = 'Failed: ' + (errJson.error || 'Unknown');
          btn.disabled = false;
          return;
        }
      }

      setTimeout(() => poll(tries + 1), 5000);
    };

    poll();
  });
});
