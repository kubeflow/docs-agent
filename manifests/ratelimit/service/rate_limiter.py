# Custom rate limit service using the sliding window log algorithm


import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response

# Configuration (environment variables) 

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis-ratelimit:6379")
REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD")
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "20"))
WINDOW_SECONDS: int = int(os.getenv("WINDOW_SECONDS", "60"))
TTL_BUFFER: int = 10  # extra seconds so Redis auto-expires keys

RL_PREFIX = "rl"  # prefix for rl:{ip}

# Logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ratelimit")

#  Redis connection 

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            REDIS_URL,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
    return _redis


# Sliding window log algorithm 


async def check_rate_limit(ip: str) -> tuple[bool, int, int]:
    
    try:
        r = await get_redis()
        redis_key = f"{RL_PREFIX}:{ip}"
        now = time.time()
        window_start = now - WINDOW_SECONDS
        member = str(uuid.uuid4())

        # Atomic pipeline — single round-trip to Redis
        pipe = r.pipeline()
        pipe.zremrangebyscore(redis_key, "-inf", window_start)  
        pipe.zadd(redis_key, {member: now})                     
        pipe.zcard(redis_key)                                   
        pipe.expire(redis_key, WINDOW_SECONDS + TTL_BUFFER)     
        results = await pipe.execute()

        count = results[2]

        if count > RATE_LIMIT_RPM:
            # Over limit — undo the ZADD so rejected requests leave no trace
            await r.zrem(redis_key, member)
            logger.warning(
                "DENIED ip=%s count=%d limit=%d", ip, count - 1, RATE_LIMIT_RPM
            )
            return False, count - 1, RATE_LIMIT_RPM

        logger.debug(
            "ALLOWED ip=%s count=%d limit=%d", ip, count, RATE_LIMIT_RPM
        )
        return True, count, RATE_LIMIT_RPM

    except Exception as exc:
        # Fail open — Redis down means we allow traffic
        logger.error("Redis error, failing open: %s", exc)
        global _redis
        _redis = None  # reset connection for next attempt
        return True, 0, RATE_LIMIT_RPM


# FastAPI application

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: nothing to do, Redis connection is lazy
    yield
    # shutdown: close Redis connection pool
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


app = FastAPI(title="Sliding Window Rate Limiter", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/healthcheck")
async def healthcheck():
    """Health check for Kubernetes probes."""
    return {"status": "healthy"}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def check_auth(request: Request, path: str = ""):
    
    # Envoy ext_authz check endpoint

    # Extract client IP from X-Forwarded-For (set by load balancer)
    xff = request.headers.get("x-forwarded-for", "")
    client_ip = xff.split(",")[0].strip() if xff else ""

    if not client_ip:
        # No IP available — allow (can't rate limit without identifier)
        return Response(status_code=200)

    allowed, count, limit = await check_rate_limit(client_ip)
    remaining = max(0, limit - count)

    headers = {
        "x-ratelimit-limit": str(limit),
        "x-ratelimit-remaining": str(remaining),
        "x-ratelimit-reset": str(WINDOW_SECONDS),
    }

    if allowed:
        return Response(status_code=200, headers=headers)
    else:
        return Response(
            status_code=429,
            headers={**headers, "x-ratelimit-remaining": "0"},
            content="rate limit exceeded",
        )


# Entry point 
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    logger.info(
        "Starting sliding window rate limiter on :%d  "
        "(limit=%d req/%ds, redis=%s)",
        port, RATE_LIMIT_RPM, WINDOW_SECONDS, REDIS_URL,
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
