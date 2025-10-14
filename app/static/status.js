// app/static/status.js
document.addEventListener('DOMContentLoaded', async () => {
  const STAGES = ['queued', 'downloading', 'detecting', 'bucketing', 'segmenting', 'packaging', 'completed', 'failed', 'canceled'];
  const form = document.getElementById('job-form');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const btn = document.getElementById('submit_btn');
  const cookieEl = document.getElementById('cookie_status');

  const timelineEl = document.createElement('div');
  timelineEl.id = 'stage_timeline';
  timelineEl.className = 'timeline';
  timelineEl.style.display = 'none';
  statusEl.insertAdjacentElement('afterend', timelineEl);

  const renderTimeline = (stage) => {
    if (!stage) {
      timelineEl.style.display = 'none';
      timelineEl.innerHTML = '';
      return;
    }
    const foundIdx = STAGES.indexOf(stage);
    if (foundIdx === -1) {
      timelineEl.style.display = 'none';
      timelineEl.innerHTML = '';
      return;
    }
    const currentIdx = foundIdx;
    timelineEl.innerHTML = '';
    STAGES.forEach((name, idx) => {
      const step = document.createElement('div');
      step.className = 'timeline-step';
      if (idx < currentIdx) step.classList.add('done');
      if (idx === currentIdx) step.classList.add('active');
      step.textContent = name;
      timelineEl.appendChild(step);
    });
    timelineEl.style.display = 'flex';
  };

  const renderStatus = (job, cancelBtn) => {
    const stage = job.stage || job.status || 'queued';
    const pct = typeof job.pct === 'number' ? job.pct : 0;
    const detail = job.detail ? ` • ${job.detail}` : '';
    let etaTxt = '';
    if (job.eta_sec != null) {
      const m = Math.floor(job.eta_sec / 60);
      const s = Math.max(0, Math.floor(job.eta_sec % 60));
      etaTxt = ` • ETA ${m}m ${s}s`;
    }
    statusEl.textContent = `${stage} — ${pct.toFixed(1)}%${detail}${etaTxt}`;
    const isTerminal = ['completed', 'failed', 'canceled'].includes(stage);
    if (cancelBtn) {
      if (isTerminal) {
        cancelBtn.remove();
      } else {
        statusEl.appendChild(document.createTextNode(' '));
        statusEl.appendChild(cancelBtn);
      }
    }
    renderTimeline(stage);
  };

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
    timelineEl.style.display = 'none';
    timelineEl.innerHTML = '';
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
    renderTimeline('queued');

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
        try {
          const res = await fetch(`/jobs/${jobId}/result`, { cache: 'no-store' });
          if (!res.ok) throw new Error('result_not_ready');
          const payload = await res.json();
          const manifestUrl = payload.manifest_url;
          const archiveUrl = payload.archive_url;

          if (manifestUrl) {
            try {
              const manifestResp = await fetch(manifestUrl, { cache: 'no-store', mode: 'cors' });
              const manifest = await manifestResp.json();
              resultEl.style.display = 'block';
              resultEl.textContent = JSON.stringify(manifest, null, 2);
            } catch (err) {
              console.error('manifest_fetch_error', err);
              resultEl.style.display = 'block';
              resultEl.textContent = 'Completed, but failed to fetch manifest (CORS/URL).';
            }
          }

          statusEl.textContent = 'completed — 100.0% • Ready';
          renderTimeline('completed');
          if (manifestUrl) {
            const manifestLink = document.createElement('a');
            manifestLink.href = manifestUrl;
            manifestLink.textContent = 'Manifest JSON';
            manifestLink.className = 'link';
            manifestLink.target = '_blank';
            statusEl.appendChild(document.createTextNode(' '));
            statusEl.appendChild(manifestLink);
          }
          if (archiveUrl) {
            const zipLink = document.createElement('a');
            zipLink.href = archiveUrl;
            zipLink.textContent = 'Download ZIP';
            zipLink.className = 'link';
            zipLink.target = '_blank';
            statusEl.appendChild(document.createTextNode(' '));
            statusEl.appendChild(zipLink);
          }
          btn.disabled = false;
          cancelBtn.disabled = true;
          return;
        } catch (err) {
          console.error('result_fetch_error', err);
          statusEl.textContent = 'Completed, but result is unavailable.';
          renderTimeline('completed');
          btn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }
      }

      if (job.status === 'failed') {
        try {
          const er = await fetch(`/jobs/${jobId}/error`, { cache: 'no-store' });
          if (er.ok) {
            const ej = await er.json();
            statusEl.textContent = 'failed — ' + (ej.error || 'Unknown');
          } else {
            statusEl.textContent = 'failed — Unknown error';
          }
        } catch (_) {
          statusEl.textContent = 'failed — Unknown error';
        }
        renderTimeline('failed');
        btn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      if (job.status === 'canceled') {
        statusEl.textContent = 'canceled';
        renderTimeline('canceled');
        btn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      setTimeout(poll, 1500);
    };

    poll();
  });
});
