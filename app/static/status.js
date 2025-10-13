// app/static/status.js
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('job-form');
  const uploadForm = document.getElementById('upload-form');
  const uploadFileInput = document.getElementById('upload_file');
  const uploadStatusEl = document.getElementById('upload-status');
  const uploadIdInput = document.getElementById('upload_id');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  if (!form || !statusEl || !resultEl) {
    return;
  }

  const setUploadStatus = (message, src) => {
    if (!uploadStatusEl) return;
    uploadStatusEl.textContent = message;
    if (src) {
      uploadStatusEl.dataset.src = src;
    } else if (uploadStatusEl.dataset) {
      delete uploadStatusEl.dataset.src;
    }
  };

  const performUpload = async () => {
    const file = uploadFileInput && uploadFileInput.files ? uploadFileInput.files[0] : null;
    if (!file) {
      setUploadStatus('Select a file before uploading.');
      return null;
    }

    const formData = new FormData();
    formData.append('file', file);

    let uploadResp;
    try {
      uploadResp = await fetch('/upload', {
        method: 'POST',
        body: formData
      });
    } catch (err) {
      setUploadStatus('Network error uploading file.');
      return null;
    }

    if (!uploadResp.ok) {
      setUploadStatus(`Upload failed (HTTP ${uploadResp.status}).`);
      return null;
    }

    const uploadData = await uploadResp.json();
    if (uploadIdInput) {
      uploadIdInput.value = uploadData.upload_id || '';
    }
    setUploadStatus(`Uploaded ${file.name} -> ${uploadData.src}`, uploadData.src || '');
    return uploadData;
  };

  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      setUploadStatus('Uploading...');
      await performUpload();
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    statusEl.textContent = 'Submitting job...';
    resultEl.innerHTML = '';

    const video_url = document.getElementById('video_url').value.trim();
    const webhook_url = document.getElementById('webhook_url').value.trim();
    const play_padding_pre = parseFloat(document.getElementById('play_padding_pre').value || '3');
    const play_padding_post = parseFloat(document.getElementById('play_padding_post').value || '5');

    const payload = {
      options: { play_padding_pre, play_padding_post }
    };

    if (webhook_url) {
      payload.webhook_url = webhook_url;
    }

    let uploadId = uploadIdInput ? uploadIdInput.value.trim() : '';
    const file = uploadFileInput && uploadFileInput.files ? uploadFileInput.files[0] : null;

    if (!video_url && !uploadId && file) {
      setUploadStatus('Uploading file...');
      const uploadData = await performUpload();
      if (!uploadData) {
        statusEl.textContent = 'Upload failed. Please try again.';
        return;
      }
      uploadId = uploadData.upload_id;
    }

    if (!uploadId) {
      uploadId = uploadIdInput ? uploadIdInput.value.trim() : '';
    }

    let sourceLabel = video_url;

    if (uploadId) {
      payload.upload_id = uploadId;
      const uploadSrc = uploadStatusEl && uploadStatusEl.dataset ? uploadStatusEl.dataset.src : '';
      sourceLabel = uploadSrc || `upload ${uploadId}`;
    } else if (video_url) {
      payload.video_url = video_url;
    } else {
      statusEl.textContent = 'Provide either a video URL or upload a file.';
      return;
    }

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
    statusEl.textContent = `Job queued: ${jobId}. Processing ${sourceLabel}...`;

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
