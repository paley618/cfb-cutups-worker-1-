# syntax=docker/dockerfile:1
FROM python:3.10-slim as base

ARG FFMPEG_VERSION=6.1.1
# The yt-dlp FFmpeg builds are released under tags like `ffmpeg-n6.1.1-latest`
# and the assets follow the same naming scheme. The previous URL attempted to
# fetch from a `v${FFMPEG_VERSION}` tag which does not exist and resulted in a
# 404 at deploy time. Point directly at the `ffmpeg-n${FFMPEG_VERSION}-latest`
# tag to ensure the asset is resolvable.
ARG FFMPEG_DOWNLOAD_URL=https://github.com/yt-dlp/FFmpeg-Builds/releases/download/ffmpeg-n${FFMPEG_VERSION}-latest/ffmpeg-n${FFMPEG_VERSION}-latest-linux64-gpl.tar.xz
ARG FFMPEG_SHA256=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -H "User-Agent: Mozilla/5.0" "${FFMPEG_DOWNLOAD_URL}" -o /tmp/ffmpeg.tar.xz \
    && if [ -n "${FFMPEG_SHA256}" ]; then echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c -; fi \
    && mkdir -p /opt/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && mv /opt/ffmpeg/ffmpeg /usr/local/bin/ffmpeg \
    && mv /opt/ffmpeg/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg.tar.xz /opt/ffmpeg \
    && ffmpeg -version

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && ffmpeg -version \
    && yt-dlp --version

COPY ./app /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
