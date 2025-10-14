import json
import socket
import time

import httpx


TEST_URL = "https://httpbin.org/bytes/1048576"  # ~1MB
OKRU = "ok.ru"


async def test_http_egress():
    t0 = time.time()
    timeout = httpx.Timeout(60, connect=10, read=30)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(TEST_URL)
        size = len(r.content)
        return {
            "ok": r.status_code == 200 and size >= 1_000_000,
            "ms": int((time.time() - t0) * 1000),
            "status": r.status_code,
            "bytes": size,
        }


def test_dns(host=OKRU):
    t0 = time.time()
    try:
        socket.gethostbyname(host)
        return {"ok": True, "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def test_s3(storage):
    # skips if storage backend isn't S3
    try:
        if storage.__class__.__name__ != "S3Storage":
            return {"ok": True, "skipped": True, "why": "not using S3"}
        key = f"selftest/{int(time.time())}.json"
        local = "/tmp/selftest.json"
        with open(local, "w") as f:
            json.dump({"ok": True, "ts": time.time()}, f)
        t0 = time.time()
        await storage.write_file(local, key)
        url = storage.url_for(key)
        timeout = httpx.Timeout(30, connect=10, read=20)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
        return {
            "ok": r.status_code == 200,
            "ms": int((time.time() - t0) * 1000),
            "status": r.status_code,
            "url": url,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def run_all(storage):
    http = await test_http_egress()
    dns = test_dns()
    s3 = await test_s3(storage)
    overall = http.get("ok") and dns.get("ok") and s3.get("ok")
    return {"overall_ok": bool(overall), "http": http, "dns_okru": dns, "s3": s3}
