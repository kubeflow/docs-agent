# Redis-backed sliding window rate limiter for docs-agent

import logging
import os
import time
import uuid
from typing import Optional, Tuple

import redis.asyncio as aioredis
import redis as syncredis

logger = logging.getLogger(__name__)

# Configuration

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "20"))
RATE_LIMIT_CONNECTIONS: int = int(os.getenv("RATE_LIMIT_CONNECTIONS", "5"))

_RL_PREFIX = "rl"        # sorted-set keys: rl:{ip}
_CONN_PREFIX = "rl:conn"  # connection counter keys: rl:conn:{ip}
_WINDOW_SECONDS = 60
_TTL_BUFFER = 10  # extra seconds so Redis cleans up keys itself



# Async rate limiter  (WebSocket server + FastAPI server)
class RedisRateLimiter:

    def __init__(
        self,
        requests_per_window: int = RATE_LIMIT_RPM,
        window_seconds: int = _WINDOW_SECONDS,
        enabled: bool = RATE_LIMIT_ENABLED,
        _client: Optional[aioredis.Redis] = None,  
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._redis: Optional[aioredis.Redis] = _client
        self._injected = _client is not None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                REDIS_URL,
                password=REDIS_PASSWORD,
                decode_responses=True,
            )
        return self._redis

    async def check(self, key: str) -> Tuple[bool, int, int]:
        """
        Check the rate limit for client IP.

        Returns:
            (allowed, current_count, limit)
        Fails open on any Redis service error
        """
        if not self.enabled:
            return True, 0, self.requests_per_window

        try:
            r = await self._get_redis()
            redis_key = f"{_RL_PREFIX}:{key}"
            now = time.time()
            window_start = now - self.window_seconds
            member = str(uuid.uuid4())

            pipe = r.pipeline()
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            pipe.zadd(redis_key, {member: now})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, self.window_seconds + _TTL_BUFFER)
            results = await pipe.execute()

            count = results[2]  

            if count > self.requests_per_window:
                # Over limit: undo the add and reject
                await r.zrem(redis_key, member)
                logger.warning("[RedisRateLimiter] %s exceeded %d RPM (count=%d)", key, self.requests_per_window, count - 1)
                return False, count - 1, self.requests_per_window

            return True, count, self.requests_per_window

        except Exception as exc:
            logger.warning("[RedisRateLimiter] Redis unreachable, failing open: %s", exc)
            if not self._injected:
                self._redis = None  # reset so the next call retries the connection
            return True, 0, self.requests_per_window

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis and not self._injected:
            await self._redis.aclose()
            self._redis = None


# Async connection limiter  (WebSocket server only)

class RedisConnectionLimiter:

# Tracks concurrent WebSocket connections per IP using Redis INCR/DECR
# Multi-pod support: the count is shared across all replicas

    def __init__(
        self,
        max_per_ip: int = RATE_LIMIT_CONNECTIONS,
        enabled: bool = RATE_LIMIT_ENABLED,
        _client: Optional[aioredis.Redis] = None,
    ):
        self.max_per_ip = max_per_ip
        self.enabled = enabled
        self._redis: Optional[aioredis.Redis] = _client
        self._injected = _client is not None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                REDIS_URL,
                password=REDIS_PASSWORD,
                decode_responses=True,
            )
        return self._redis

    async def acquire(self, ip: str) -> bool:
        # Reserve a connection slot, returns True if allowed
        if not self.enabled:
            return True
        try:
            r = await self._get_redis()
            redis_key = f"{_CONN_PREFIX}:{ip}"
            count = await r.incr(redis_key)
            await r.expire(redis_key, 3600)  # safety TTL: 1 hour
            if count > self.max_per_ip:
                await r.decr(redis_key)
                logger.warning("[RedisConnectionLimiter] %s hit max connections (%d)", ip, self.max_per_ip)
                return False
            logger.debug("[RedisConnectionLimiter] %s active connections: %d", ip, count)
            return True
        except Exception as exc:
            logger.warning("[RedisConnectionLimiter] Redis unreachable, failing open: %s", exc)
            if not self._injected:
                self._redis = None
            return True

    async def release(self, ip: str) -> None:
        """Release a connection slot."""
        try:
            r = await self._get_redis()
            redis_key = f"{_CONN_PREFIX}:{ip}"
            new_count = await r.decr(redis_key)
            if new_count < 0:
                await r.set(redis_key, 0)
        except Exception as exc:
            logger.warning("[RedisConnectionLimiter] Redis error on release: %s", exc)
            if not self._injected:
                self._redis = None

    async def close(self) -> None:
        if self._redis and not self._injected:
            await self._redis.aclose()
            self._redis = None



# Sync rate limiter  (MCP tool server — no event loop)

class RedisSyncRateLimiter:
# Same algorithm as RedisRateLimiter, using the synchronous redis client

    def __init__(
        self,
        requests_per_window: int = RATE_LIMIT_RPM,
        window_seconds: int = _WINDOW_SECONDS,
        enabled: bool = RATE_LIMIT_ENABLED,
        _client: Optional[syncredis.Redis] = None,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._redis: Optional[syncredis.Redis] = _client
        self._injected = _client is not None

    def _get_redis(self) -> syncredis.Redis:
        if self._redis is None:
            self._redis = syncredis.from_url(
                REDIS_URL,
                password=REDIS_PASSWORD,
                decode_responses=True,
            )
        return self._redis

    def check_sync(self, key: str) -> Tuple[bool, int, int]:

        if not self.enabled:
            return True, 0, self.requests_per_window

        try:
            r = self._get_redis()
            redis_key = f"{_RL_PREFIX}:{key}"
            now = time.time()
            window_start = now - self.window_seconds
            member = str(uuid.uuid4())

            pipe = r.pipeline()
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            pipe.zadd(redis_key, {member: now})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, self.window_seconds + _TTL_BUFFER)
            results = pipe.execute()

            count = results[2]

            if count > self.requests_per_window:
                r.zrem(redis_key, member)
                logger.warning("[RedisSyncRateLimiter] %s exceeded %d RPM (count=%d)", key, self.requests_per_window, count - 1)
                return False, count - 1, self.requests_per_window

            return True, count, self.requests_per_window

        except Exception as exc:
            logger.warning("[RedisSyncRateLimiter] Redis unreachable, failing open: %s", exc)
            if not self._injected:
                self._redis = None
            return True, 0, self.requests_per_window

    def close(self) -> None:
        if self._redis and not self._injected:
            self._redis.close()
            self._redis = None
