# app/selftest.py
import asyncio, socket, os, json, time, inspect
import httpx

TEST_URL = "https://httpbin.org/bytes/102400"  # ~100KB, fast & consistent
OKRU = "ok.ru"

async def test_http_egress():
    t0 = time.time()
    try:
        timeout = httpx.Timeout(60, connect=10, read=30)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(TEST_URL)
        size = len(r.content or b"")
        ok = (r.status_code == 200) and (size >= 100_000)
        return {"ok": ok, "ms": int((time.time()-t0)*1000), "status": r.status_code, "bytes": size}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "ms": int((time.time()-t0)*1000)}

def test_dns(host=OKRU):
    t0 = time.time()
    try:
        socket.gethostbyname(host)
        return {"ok": True, "ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def _maybe_async_call(fn, *args, **kwargs):
    # Supports sync and async implementations transparently
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    # If it's a bound method but sync, run in a thread
    return await asyncio.to_thread(fn, *args, **kwargs)

async def test_s3(storage):
    try:
        if getattr(storage, "__class__", type("X",(object,),{})).__name__ != "S3Storage":
            return {"ok": True, "skipped": True, "why": "not using S3"}
        key = f"selftest/{int(time.time())}.json"
        local = "/tmp/selftest.json"
        with open(local, "w") as f:
            json.dump({"ok": True, "ts": time.time()}, f)
        t0 = time.time()
        # write_file may be sync or async; handle both
        await _maybe_async_call(storage.write_file, local, key)
        url = storage.url_for(key)
        timeout = httpx.Timeout(30, connect=10, read=20)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
        return {"ok": r.status_code == 200, "ms": int((time.time()-t0)*1000), "status": r.status_code, "url": url}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

async def run_all(storage):
    http = await test_http_egress()
    dns = test_dns()
    s3 = await test_s3(storage)
    overall = (http.get("ok") is True) and (dns.get("ok") is True) and (s3.get("ok") is True)
    return {"overall_ok": bool(overall), "http": http, "dns_okru": dns, "s3": s3}
