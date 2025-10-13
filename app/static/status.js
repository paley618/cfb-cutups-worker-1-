const form = document.getElementById("job-form");
const submitButton = document.getElementById("submit-button");
const statusPanel = document.getElementById("status-panel");
const statusMessage = document.getElementById("status-message");
const statusExtra = document.getElementById("status-extra");
const downloadLink = document.getElementById("download-link");

let pollingTimeoutId = null;
let activeJobId = null;

const POLL_INTERVAL_MS = 5000;

function setPanelState({
  message,
  extra = "",
  tone = "info",
  showDownload = false,
}) {
  statusPanel.classList.add("visible");
  statusPanel.classList.remove("success", "error");
  if (tone === "success") {
    statusPanel.classList.add("success");
  } else if (tone === "error") {
    statusPanel.classList.add("error");
  }

  statusMessage.textContent = message;
  if (extra) {
    statusExtra.hidden = false;
    statusExtra.textContent = extra;
  } else {
    statusExtra.hidden = true;
    statusExtra.textContent = "";
  }

  if (showDownload && activeJobId) {
    downloadLink.hidden = false;
    downloadLink.href = `/jobs/${encodeURIComponent(activeJobId)}/download`;
  } else {
    downloadLink.hidden = true;
    downloadLink.removeAttribute("href");
  }
}

function clearPolling() {
  if (pollingTimeoutId !== null) {
    window.clearTimeout(pollingTimeoutId);
    pollingTimeoutId = null;
  }
  activeJobId = null;
}

async function pollForManifest(jobId) {
  activeJobId = jobId;

  const poll = async () => {
    try {
      const response = await fetch(`/jobs/${encodeURIComponent(jobId)}/manifest`, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });

      if (response.ok) {
        setPanelState({
          message: "Job complete! Download your cutups archive.",
          extra: `Job ID: ${jobId}`,
          tone: "success",
          showDownload: true,
        });
        submitButton.disabled = false;
        pollingTimeoutId = null;
        return;
      }

      if (response.status === 404) {
        setPanelState({
          message: "Processing… We’ll keep checking for the manifest.",
          extra: `Job ID: ${jobId}`,
        });
      } else {
        const body = await response.text();
        throw new Error(`Unexpected status ${response.status}: ${body}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setPanelState({
        message: "We hit a snag while checking the job.",
        extra: message,
        tone: "error",
      });
      submitButton.disabled = false;
      pollingTimeoutId = null;
      return;
    }

    pollingTimeoutId = window.setTimeout(poll, POLL_INTERVAL_MS);
  };

  await poll();
}

function buildPayload(formData) {
  const payload = {};

  const videoUrl = (formData.get("video_url") || "").toString().trim();
  if (!videoUrl) {
    throw new Error("A video URL is required.");
  }
  payload.video_url = videoUrl;

  const webhook = (formData.get("webhook_url") || "").toString().trim();
  if (webhook) {
    payload.webhook_url = webhook;
  }

  const options = {};
  const pre = formData.get("play_padding_pre");
  const post = formData.get("play_padding_post");

  if (pre !== null && pre !== "") {
    const preNumber = Number(pre);
    if (Number.isNaN(preNumber) || preNumber < 0) {
      throw new Error("Padding before each play must be a non-negative number.");
    }
    options.play_padding_pre = preNumber;
  }

  if (post !== null && post !== "") {
    const postNumber = Number(post);
    if (Number.isNaN(postNumber) || postNumber < 0) {
      throw new Error("Padding after each play must be a non-negative number.");
    }
    options.play_padding_post = postNumber;
  }

  if (Object.keys(options).length > 0) {
    payload.options = options;
  }

  return payload;
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearPolling();

  try {
    const formData = new FormData(form);
    const payload = buildPayload(formData);

    submitButton.disabled = true;
    setPanelState({
      message: "Submitting job…",
    });

    const response = await fetch("/jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Submission failed with status ${response.status}: ${body}`);
    }

    const result = await response.json();
    const jobId = result?.job_id;

    if (!jobId) {
      throw new Error("The server response did not include a job ID.");
    }

    setPanelState({
      message: "Job accepted! Waiting for processing to finish…",
      extra: `Job ID: ${jobId}`,
    });

    pollForManifest(jobId).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      setPanelState({
        message: "We hit a snag while checking the job.",
        extra: message,
        tone: "error",
      });
      submitButton.disabled = false;
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setPanelState({
      message: "Please fix the issue below and try again.",
      extra: message,
      tone: "error",
    });
    submitButton.disabled = false;
  }
});

setPanelState({
  message: "Paste a URL and submit to begin.",
});
