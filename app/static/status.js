// app/static/status.js
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('job-form');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    statusEl.textContent = 'Submitting job...';
    resultEl.innerHTML = '';

    const video_url = document.getElementById('video_url').value.trim();
    const webhook_url = document.getElementById('webhook_url').value.trim();
    const play_padding_pre = parseFloat(document.getElementById('play_padding_pre').value || '3');
    const play_padding_post = parseFloat(document.getElementById('play_padding_post').value || '5');

    const payload = {
      video_url,
      webhook_url: webhook_url || null,
      options: { play_padding_pre, play_padding_post }
    };

    let resp;
    try {
      resp = await fetch('/jobs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
    } catch (err) {
      statusEl.textContent = 'Network error creating job.';
      return;
    }

    if (!resp.ok) {
      statusEl.textContent = `Failed to create job (HTTP ${resp.status}).`;
      return;
    }

    const data = await resp.json();
    const jobId = data.job_id;
    statusEl.textContent = `Job queued: ${jobId}. Processing...`;

    const poll = async () => {
      try {
        const mr = await fetch(`/jobs/${jobId}/manifest`, { cache: 'no-store' });
        if (mr.ok) {
          const manifest = await mr.json();
          statusEl.textContent = 'Completed.';
          resultEl.innerHTML = `
            <pre>${JSON.stringify(manifest.metrics, null, 2)}</pre>
            <a href="/jobs/${jobId}/download">Download ZIP</a>
          `;
          return;
        }
      } catch (_) {}
      setTimeout(poll, 5000);
    };
    poll();
  });
});
