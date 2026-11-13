#!/usr/bin/env python3
"""Controllable end-to-end demo of the strict last-waiter cancellation.

Run against a live service (``docker compose up`` then this script, or a local
``uvicorn`` instance).  Only the Python standard library is required::

    python -I -m scripts.demo_cancel_teardown            # default http://localhost:8000
    python -I -m scripts.demo_cancel_teardown --base-url http://localhost:8000

The script reproduces, step by step, the scenario from the defect report:

1. submit two jobs that attach to one shared baseline run;
2. cancel the first waiter -- the shared computation keeps running;
3. cancel the last waiter while the shared baseline worker is still alive;
4. hammer cancel again -- the job stays ``cancelling`` and keeps its slot;
5. submit a new job with the remaining slot: it starts a *fresh* baseline run
   and the old teardown never deletes or overwrites its files;
6. wait for the old run to be terminated/reaped/cleaned: only then does the
   last waiter become ``cancelled`` (directory gone, slot released);
7. confirm the new job runs untouched and finishes with a downloadable result;
8. confirm a cancelled job never serves a download (HTTP 409).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(method: str, url: str, payload: bytes | None = None,
             content_type: str | None = None) -> tuple[int, dict | str]:
    headers = {}
    data = None
    if payload is not None:
        data = payload
        headers["Content-Type"] = content_type or "application/octet-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    text = body.decode("utf-8", "replace")
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def _post_multipart(url: str, filename: str, content: bytes) -> tuple[int, dict]:
    boundary = "----tas-demo-boundary-0123456789abcdef"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/x-netcdf\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    status, body = _request(
        "POST", url, head + content + tail,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    assert isinstance(body, dict), body
    return status, body


def _step(num: int, title: str) -> None:
    print(f"\n=== 步骤 {num}：{title} ===", flush=True)


def _wait(predicate, timeout: float = 30.0, interval: float = 0.2):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not reached before timeout; last={last!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="service base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="path to a target NetCDF; defaults to downloading target_sample_a.nc",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = _request("GET", f"{base}/healthz")
    if status != 200:
        print(f"service not healthy: {status} {health}", file=sys.stderr)
        return 2
    print(f"服务健康检查通过：{base}")

    if args.sample:
        payload = Path(args.sample).read_bytes()
        sample_name = Path(args.sample).name
    else:
        status, listing = _request("GET", f"{base}/api/samples")
        assert status == 200 and listing["samples"], (status, listing)
        sample_name = "target_sample_a.nc"
        with urllib.request.urlopen(
            f"{base}/samples/{sample_name}", timeout=30
        ) as resp:
            payload = resp.read()
    print(f"使用样例文件：{sample_name}（{len(payload)} 字节）")

    def job_state(job_id: str) -> str:
        status, body = _request("GET", f"{base}/api/jobs/{job_id}")
        assert status == 200, body
        return body["state"]

    def baseline() -> dict:
        status, body = _request("GET", f"{base}/api/baseline")
        assert status == 200, body
        return body

    _step(1, "连续提交两份任务，挂接同一个共享基准运行")
    status, job_a = _post_multipart(f"{base}/api/jobs", "a.nc", payload)
    assert status == 201, job_a
    status, job_b = _post_multipart(f"{base}/api/jobs", "b.nc", payload)
    assert status == 201, job_b
    print(f"job A = {job_a['id'][:8]}  job B = {job_b['id'][:8]}")
    _wait(lambda: baseline().get("active_run", {}).get("status") == "running")
    run_info = baseline()["active_run"]
    print(
        f"共享基准运行 run={run_info['run_id'][:8]} status={run_info['status']} "
        f"等待者={[w[:8] for w in run_info['waiters']]}"
    )

    _step(2, "取消 A（仍有 B 在等待）：共享计算不得终止")
    status, body = _request("POST", f"{base}/api/jobs/{job_a['id']}/cancel")
    assert status == 200, body
    _wait(lambda: job_state(job_a["id"]) == "cancelled")
    run_after = baseline()["active_run"]
    print(
        f"A={job_state(job_a['id'])}；共享运行仍为 "
        f"status={run_after['status']} run={run_after['run_id'][:8]}"
    )
    assert run_after["run_id"] == run_info["run_id"]
    assert run_after["status"] == "running"

    _step(3, "取消最后等待者 B：B 必须保持 cancelling 并占住名额")
    status, body = _request("POST", f"{base}/api/jobs/{job_b['id']}/cancel")
    assert status == 200, body
    assert body["state"] == "cancelling", body
    print(f"首次取消 B -> state={body['state']}，消息：{body['message']}")
    draining = _wait(
        lambda: (
            info if (info := baseline().get("active_run"))
            and info.get("status") == "draining" else None
        )
    )
    print(f"共享运行进入收尾：status={draining['status']}（进程退出/回收/清理进行中）")

    _step(4, "重复取消 B：不得提前结束收尾，状态仍是 cancelling")
    for i in range(3):
        status, body = _request("POST", f"{base}/api/jobs/{job_b['id']}/cancel")
        assert status == 200, body
        print(f"  第 {i + 1} 次重复取消 -> state={body['state']}")
        assert body["state"] == "cancelling", body
    status, jobs = _request("GET", f"{base}/api/jobs")
    print(f"当前活跃任务计数 active_count={jobs['active_count']}（B 仍占名额）")
    assert jobs["active_count"] == 1

    _step(5, "用剩余名额提交新任务 C：另起基准计算")
    status, job_c = _post_multipart(f"{base}/api/jobs", "c.nc", payload)
    assert status == 201, job_c
    print(f"job C = {job_c['id'][:8]} 已受理，开始全新的基准运行")
    status, jobs = _request("GET", f"{base}/api/jobs")
    assert jobs["active_count"] == 2, jobs

    def _fresh_run():
        info = baseline().get("active_run")
        return info if info and info["run_id"] != run_info["run_id"] else None

    fresh = _wait(_fresh_run, timeout=15)
    print(
        f"新基准运行 run={fresh['run_id'][:8]} status={fresh['status']}，"
        f"与旧 run {run_info['run_id'][:8]} 不同；旧收尾不能触碰 C 的目录"
    )
    assert fresh["run_id"] != run_info["run_id"]

    _step(6, "等待旧运行回收清理完成，B 此时才允许变为 cancelled")
    # Give the old run a moment to still be observable in the closing window,
    # then wait for B to finalize.
    _wait(lambda: job_state(job_b["id"]) == "cancelled", timeout=30)
    print(f"B 最终状态 = {job_state(job_b['id'])}（进程已退出/回收、目录已清理）")
    status, jobs = _request("GET", f"{base}/api/jobs")
    print(f"名额释放后 active_count={jobs['active_count']}（仅 C 仍未结束）")
    assert jobs["active_count"] == 1

    _step(7, "新任务 C 不受旧收尾影响，正常完成并可下载")
    _wait(lambda: job_state(job_c["id"]) == "done", timeout=60)
    status, _ = _request("GET", f"{base}/api/jobs/{job_c['id']}/download")
    print(f"C 下载 HTTP {status}（200 表示距平 NetCDF 可下载）")
    assert status == 200

    _step(8, "已取消任务 A/B 一律不得提供结果")
    for job in (job_a, job_b):
        status, body = _request("GET", f"{base}/api/jobs/{job['id']}/download")
        print(f"  job {job['id'][:8]} 下载 HTTP {status}：{body}")
        assert status == 409, body

    print("\n全部演示断言通过：最后等待者严格在基准进程回收清理后才 cancelled，"
          "重复取消不会提前收尾，新任务另起计算且不被旧收尾影响。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
