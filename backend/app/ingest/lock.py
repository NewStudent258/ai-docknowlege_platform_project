"""Celery 幂等锁 — 基于 Redis SET NX 实现，防止同一文档重复入队

幂等键格式: doc_lock:{doc_id}（ingest/delete 共享互斥锁）
锁 TTL: 600s（settings.IDEMPOTENCY_LOCK_TTL）

⚠️ TTL 与任务超时的对齐要求:
    锁 TTL 必须 >= 持锁任务的最长执行时间，否则任务被 soft_time_limit
    中断后锁仍被持有，重试将一直拿不到锁。当前各任务的 soft_time_limit:
      - ingest_document  : 600s  → 与锁 TTL 相等（临界，可接受）
      - delete_kb        : 600s  → 与锁 TTL 相等（临界，可接受）
      - delete_document  : 300s  → 短于锁 TTL，见下方说明
    因此所有任务在抢锁失败时统一抛 ResourceLockedError（可重试），
    由 Celery autoretry_for 兜底重试，避免「拿到 locked 就直接放弃」的静默失败。

触发规则（对齐 ARCHITECTURE.md §4.5）:
- 无锁 → 正常创建任务
- 有锁 + 运行中 → 拒绝，返回 E2011「文档正在处理中」
- 有锁 + Worker crash → 等待锁过期后自动允许重新触发
- 终态 + 无锁 + reprocess → 允许重新触发（清理旧数据）
"""

from app.core.redis_client import get_redis

from app.config import settings


class ResourceLockedError(Exception):
    """幂等锁被占用 —— 可重试异常。

    为什么需要它：任务被 soft_time_limit 中断后，Redis 锁不会随之释放，
    Celery 重试时会发现锁仍被自己（上一次执行）持有。若此时直接 return
    一个 {"status": "locked"} 的普通返回值，Celery 不会触发 autoretry，
    任务就被**静默放弃**了（用户以为删除完成，实际未执行）。

    抛出本异常可让 autoretry_for=(Exception,) 捕获并按 retry_backoff
    重试，直到锁自然过期。因此本异常必须在「抢锁失败」路径抛出的位置
    使用，而不是作为普通业务返回。
    """

    def __init__(self, resource: str, resource_id: int, task_type: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        self.task_type = task_type
        super().__init__(
            f"{resource} {resource_id} 的 {task_type} 任务因幂等锁被占用而重试"
        )


# 幂等键前缀
IDEMPOTENCY_KEY_PREFIX = "doc_lock"


def _build_lock_key(doc_id: int, task_type: str) -> str:
    """构建幂等锁 Redis key（ingest/delete 共享同一锁，确保互斥）"""
    return f"{IDEMPOTENCY_KEY_PREFIX}:{doc_id}"


def acquire_idempotency_lock(
    doc_id: int, task_type: str, ttl: int = settings.IDEMPOTENCY_LOCK_TTL
) -> bool:
    """尝试获取幂等锁（原子操作 SET key value EX ttl NX）。

    Args:
        doc_id: 文档 ID
        task_type: 任务类型（如 ingest、delete）
        ttl: 锁过期时间（秒），默认 600s

    Returns:
        True: 获取成功，可继续执行
        False: 锁已被占用，应拒绝重复入队（→ E2011）
    """
    key = _build_lock_key(doc_id, task_type)
    return bool(get_redis().set(key, "locked", ex=ttl, nx=True))


def release_idempotency_lock(doc_id: int, task_type: str) -> None:
    """释放幂等锁（任务完成/失败后调用，幂等操作）"""
    key = _build_lock_key(doc_id, task_type)
    get_redis().delete(key)


def check_idempotency_lock(doc_id: int, task_type: str) -> bool:
    """检查是否存在幂等锁。

    Returns:
        True: 已锁定（任务正在处理中）
        False: 未锁定
    """
    key = _build_lock_key(doc_id, task_type)
    return get_redis().exists(key) > 0


async def acquire_idempotency_lock_async(
    doc_id: int, task_type: str, ttl: int = settings.IDEMPOTENCY_LOCK_TTL
) -> bool:
    """异步版幂等锁获取，供 async 上下文使用（避免阻塞事件循环）。"""
    from app.core.redis_client import get_async_redis

    key = _build_lock_key(doc_id, task_type)
    redis_client = await get_async_redis()
    return bool(await redis_client.set(key, "locked", ex=ttl, nx=True))


async def release_idempotency_lock_async(doc_id: int, task_type: str) -> None:
    """异步版幂等锁释放，供 async 上下文使用（避免阻塞事件循环）。"""
    from app.core.redis_client import get_async_redis

    key = _build_lock_key(doc_id, task_type)
    redis_client = await get_async_redis()
    await redis_client.delete(key)
