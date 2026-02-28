"""
DoS simulation for docs-agent rate limiting

Prerequisites:
    1. Redis running
    2. Server running
    3. Install test deps

Usage:
    python tests/dos_simulation.py [--url http://localhost:8000] [--limit 10]
"""

import argparse
import asyncio
import json
import time
import httpx
import websockets

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

CHAT_PAYLOAD = {"message": "What is Kubeflow?", "stream": True}

# Helpers

def header(text: str) -> None:
    print(f"\n{BOLD}{BLUE}  {text}{RESET}")


def result_line(n: int, status: int, elapsed_ms: float) -> None:
    if status == 200:
        tag = f"{GREEN}[ALLOWED] {status}{RESET}"
    elif status == 429:
        tag = f"{RED}[BLOCKED]  {status}{RESET}"
    else:
        tag = f"{YELLOW}[OTHER]    {status}{RESET}"
    print(f"  Request {n:>3}: {tag}  ({elapsed_ms:.0f} ms)")


async def fire(client: httpx.AsyncClient, url: str, n: int) -> tuple[int, float]:
    """Send a single chat request and return (status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{url}/chat", json=CHAT_PAYLOAD, timeout=10)
        elapsed = (time.perf_counter() - t0) * 1000
        return resp.status_code, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Request {n:>3}: {YELLOW}[ERROR] {exc}{RESET}")
        return 0, elapsed



# Test: burst above the limit from a single IP

async def test_burst(url: str, limit: int) -> None:
    header(f"Burst Attack — {limit * 2} requests, limit = {limit} RPM")
    print(f"  Sending {limit * 2} rapid requests from the same IP.\n"
          f"  Expect: first {limit} allowed, remainder blocked.\n")

    allowed = blocked = 0
    async with httpx.AsyncClient() as client:
        for i in range(1, limit * 2 + 1):
            status, ms = await fire(client, url, i)
            result_line(i, status, ms)
            if status == 200:
                allowed += 1
            elif status == 429:
                blocked += 1

    print()
    print(f"  {BOLD}Summary:{RESET} {GREEN}Allowed: {allowed}{RESET}  |  {RED}Blocked: {blocked}{RESET}")
    passed = allowed == limit and blocked == limit
    outcome = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Result: {outcome}{RESET}")



# Test: two IPs don't share the counter

async def test_ip_isolation(url: str, limit: int) -> None:
    header("IP Isolation — two IPs, same limit")
    print(f"  Two clients with different spoofed IPs each send {limit} requests.\n"
          f"  Expect: ALL {limit * 2} requests allowed (separate counters).\n")

    async def fire_as(client: httpx.AsyncClient, ip: str, count: int) -> list[int]:
        statuses = []
        for i in range(count):
            resp = await client.post(
                f"{url}/chat",
                json=CHAT_PAYLOAD,
                headers={"X-Forwarded-For": ip},
                timeout=10,
            )
            statuses.append(resp.status_code)
        return statuses

    # Flush the window for these IPs by using fresh addresses
    ip_a = "10.0.0.1"
    ip_b = "10.0.0.2"

    async with httpx.AsyncClient() as client:
        results_a, results_b = await asyncio.gather(
            fire_as(client, ip_a, limit),
            fire_as(client, ip_b, limit),
        )

    for i, s in enumerate(results_a, 1):
        result_line(i, s, 0)
    for i, s in enumerate(results_b, 1):
        result_line(i + limit, s, 0)

    all_allowed = all(s == 200 for s in results_a + results_b)
    print()
    print(f"  {BOLD}Summary:{RESET} IP A: {results_a.count(200)} allowed  | "
          f"IP B: {results_b.count(200)} allowed")
    outcome = f"{GREEN}PASS{RESET}" if all_allowed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Result: {outcome}{RESET}")



# Test: window resets after 60 seconds

async def test_window_reset(url: str, limit: int) -> None:
    header("Window Reset — traffic resumes after 60s")
    print(f"  Filling the window for IP 10.99.99.99 ({limit} requests),")
    print(f"  wait for 62 seconds and retry\n")

    ip = "10.99.99.99"
    async with httpx.AsyncClient() as client:
        # Fill the window
        for i in range(1, limit + 1):
            resp = await client.post(
                f"{url}/chat", json=CHAT_PAYLOAD,
                headers={"X-Forwarded-For": ip}, timeout=10,
            )
            result_line(i, resp.status_code, 0)

        # Confirm blocked
        resp = await client.post(
            f"{url}/chat", json=CHAT_PAYLOAD,
            headers={"X-Forwarded-For": ip}, timeout=10,
        )
        print(f"\n  Next request (should be blocked): ", end="")
        result_line(limit + 1, resp.status_code, 0)
        blocked_ok = resp.status_code == 429

        print(f"\n  {YELLOW}Waiting 62 seconds for the window to expire...{RESET}")
        await asyncio.sleep(62)

        resp = await client.post(
            f"{url}/chat", json=CHAT_PAYLOAD,
            headers={"X-Forwarded-For": ip}, timeout=10,
        )
        print(f"  First request after reset: ", end="")
        result_line(limit + 2, resp.status_code, 0)
        reset_ok = resp.status_code == 200

    passed = blocked_ok and reset_ok
    outcome = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n  {BOLD}Result: {outcome}{RESET}")



# WebSocket helpers

def ws_result_line(n: int, allowed: bool) -> None:
    tag = f"{GREEN}[ALLOWED]{RESET}" if allowed else f"{RED}[BLOCKED]{RESET}"
    print(f"  Message {n:>3}: {tag}")


async def ws_send_recv(ws) -> bool:
    """Send one message, return True if allowed (not rate-limited)."""
    await ws.send("What is Kubeflow?")
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("type") == "error" and "Rate limited" in msg.get("content", ""):
            return False
        if msg.get("type") in ("done", "error"):
            return True
    return True


# Test: WebSocket message burst

async def test_ws_burst(ws_url: str, limit: int) -> None:
    header(f"WebSocket Burst — {limit * 2} messages, limit = {limit} RPM")
    print(f"  Sending {limit * 2} rapid messages on one WS connection.\n"
          f"  Expect: first {limit} allowed, remainder blocked.\n")

    allowed = blocked = 0
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.recv()  # consume welcome message
            for i in range(1, limit * 2 + 1):
                result = await ws_send_recv(ws)
                ws_result_line(i, result)
                if result:
                    allowed += 1
                else:
                    blocked += 1
    except Exception as exc:
        print(f"  {YELLOW}[ERROR] {exc}{RESET}")

    print()
    print(f"  {BOLD}Summary:{RESET} {GREEN}Allowed: {allowed}{RESET}  |  {RED}Blocked: {blocked}{RESET}")
    passed = allowed == limit and blocked == limit
    outcome = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Result: {outcome}{RESET}")


# Test: WebSocket connection limit

async def test_ws_conn_limit(ws_url: str, conn_limit: int) -> None:
    header(f"WebSocket Connection Limit — {conn_limit + 1} connections, limit = {conn_limit}")
    print(f"  Opening {conn_limit + 1} connections from the same IP.\n"
          f"  Expect: first {conn_limit} accepted, last one rejected.\n")

    conns = []
    accepted = rejected = 0
    try:
        for i in range(conn_limit + 1):
            try:
                ws = await websockets.connect(ws_url)
                await ws.recv()  # welcome message
                conns.append(ws)
                accepted += 1
                print(f"  Connection {i + 1:>3}: {GREEN}[ACCEPTED]{RESET}")
            except websockets.exceptions.ConnectionClosedError as e:
                rejected += 1
                print(f"  Connection {i + 1:>3}: {RED}[REJECTED] code={e.code}{RESET}")
            except Exception as exc:
                rejected += 1
                print(f"  Connection {i + 1:>3}: {RED}[REJECTED] {exc}{RESET}")
    finally:
        for ws in conns:
            await ws.close()

    print()
    print(f"  {BOLD}Summary:{RESET} {GREEN}Accepted: {accepted}{RESET}  |  {RED}Rejected: {rejected}{RESET}")
    passed = accepted == conn_limit and rejected == 1
    outcome = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Result: {outcome}{RESET}")


# Test: MCP tool burst

async def test_mcp_burst(mcp_url: str, limit: int) -> None:
    header(f"MCP Tool Burst — {limit * 2} calls, limit = {limit} RPM")
    print(f"  Firing {limit * 2} search_kubeflow_docs calls.\n"
          f"  Expect: first {limit} allowed, remainder rate-limited.\n")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search_kubeflow_docs", "arguments": {"query": "kubeflow", "top_k": 1}},
    }

    allowed = blocked = 0
    async with httpx.AsyncClient() as client:
        for i in range(1, limit * 2 + 1):
            try:
                resp = await client.post(mcp_url, json=payload, timeout=10)
                body = resp.json()
                result_text = body.get("result", {}).get("content", [{"text": ""}])[0].get("text", "")
                is_blocked = "Rate limited" in result_text
                tag = f"{RED}[BLOCKED]{RESET}" if is_blocked else f"{GREEN}[ALLOWED]{RESET}"
                print(f"  Call {i:>3}: {tag}")
                if is_blocked:
                    blocked += 1
                else:
                    allowed += 1
            except Exception as exc:
                print(f"  Call {i:>3}: {YELLOW}[ERROR] {exc}{RESET}")

    print()
    print(f"  {BOLD}Summary:{RESET} {GREEN}Allowed: {allowed}{RESET}  |  {RED}Blocked: {blocked}{RESET}")
    passed = allowed == limit and blocked == limit
    outcome = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Result: {outcome}{RESET}")


# Health check
async def check_server(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/health", timeout=5)
            return resp.status_code == 200
    except Exception:
        return False



# Entry point

async def main() -> None:
    parser = argparse.ArgumentParser(description="docs-agent rate limit DoS simulation")
    parser.add_argument("--url",        default="http://localhost:8000",    help="HTTPS server base URL")
    parser.add_argument("--ws-url",     default="ws://localhost:8001",      help="WebSocket server URL")
    parser.add_argument("--mcp-url",    default="http://localhost:8002/mcp",help="MCP server tool endpoint")
    parser.add_argument("--limit",      type=int, default=10,               help="RATE_LIMIT_RPM the servers run with")
    parser.add_argument("--conn-limit", type=int, default=5,                help="RATE_LIMIT_CONNECTIONS for WS server")
    parser.add_argument("--skip-reset", action="store_true",                help="Skip the 62-second window reset test")
    parser.add_argument("--skip-ws",    action="store_true",                help="Skip WebSocket tests")
    parser.add_argument("--skip-mcp",   action="store_true",                help="Skip MCP tests")
    args = parser.parse_args()

    print(f"\n{BOLD}docs-agent Rate Limit — DoS Simulation{RESET}")
    print(f"  HTTPS server : {args.url}")
    print(f"  WS server    : {args.ws_url}")
    print(f"  MCP server   : {args.mcp_url}")
    print(f"  Rate limit   : {args.limit} req / 60 s  |  conn limit: {args.conn_limit}")

    print(f"\n  Checking HTTPS server health...", end=" ")
    if not await check_server(args.url):
        print(f"{RED}UNREACHABLE{RESET}")
        print(f"\n  Start: cd server-https && "
              f"RATE_LIMIT_RPM={args.limit} REDIS_URL=redis://localhost:6379 "
              f"uvicorn app:app --port 8000\n")
        return
    print(f"{GREEN}OK{RESET}")

    # HTTPS tests
    await test_burst(args.url, args.limit)
    await test_ip_isolation(args.url, args.limit)
    if not args.skip_reset:
        await test_window_reset(args.url, args.limit)
    else:
        print(f"\n{YELLOW}  Window reset test skipped (--skip-reset){RESET}")

    # WebSocket tests
    if not args.skip_ws:
        await test_ws_burst(args.ws_url, args.limit)
        await test_ws_conn_limit(args.ws_url, args.conn_limit)
    else:
        print(f"\n{YELLOW}  WebSocket tests skipped (--skip-ws){RESET}")

    # MCP tests
    if not args.skip_mcp:
        await test_mcp_burst(args.mcp_url, args.limit)
    else:
        print(f"\n{YELLOW}  MCP tests skipped (--skip-mcp){RESET}")

    print(f"\n{BOLD}  Simulation complete{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
