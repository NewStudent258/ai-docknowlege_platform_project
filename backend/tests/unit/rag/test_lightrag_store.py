"""LightRAG 实例管理单元测试（ADR-025）

覆盖：
- 工作目录按 KB 隔离
- 构建标记文件（.building）的读写与读写分离语义
- 实例缓存 LRU 淘汰与 invalidate
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag import lightrag_store as ls

pytestmark = pytest.mark.unit


class TestKbDir:
    """按 KB 隔离的工作目录"""

    def test_目录按_kb_id_分子目录(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            assert ls.kb_dir(1) == tmp_path / "1"
            assert ls.kb_dir(2) == tmp_path / "2"

    def test_不同_KB_目录互不相同(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            assert ls.kb_dir(1) != ls.kb_dir(2)


class TestBuildingMarker:
    """构建中标记 — 检索侧据此跳过该 KB，避免读到中间态"""

    def test_无目录时_has_graph_data_为假(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            assert ls.has_graph_data(1) is False
            assert ls.is_building(1) is False

    def test_目录存在且无标记时_视为已建图谱(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            ls.kb_dir(1).mkdir(parents=True)
            assert ls.has_graph_data(1) is True
            assert ls.is_building(1) is False

    def test_存在构建标记时_不算已建图谱(self, tmp_path):
        """构建中：is_building 为真，has_graph_data 为假"""
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            d = ls.kb_dir(1)
            d.mkdir(parents=True)
            (d / ls._BUILDING_MARKER).touch()

            assert ls.is_building(1) is True
            assert ls.has_graph_data(1) is False


class TestInstanceCache:
    """实例缓存：LRU 上限 + invalidate"""

    def _store_with_mock_build(self, max_instances=2, ttl=300):
        store = ls.LightRAGStore()
        built: list[int] = []

        def _fake_build(kb_id):
            built.append(kb_id)
            handle = MagicMock()
            handle.kb_id = kb_id
            handle.loop = MagicMock()
            handle.loop.is_closed.return_value = True
            # 必须是真实时间戳，否则 LRU 淘汰里的时间比较会拿到 MagicMock
            handle.last_access = time.monotonic()
            return handle

        store._build = _fake_build
        store._built = built
        return store

    @pytest.mark.asyncio
    async def test_同一_kb_重复获取复用实例(self):
        store = self._store_with_mock_build()
        h1 = await store.get_handle(1)
        h2 = await store.get_handle(1)
        assert h1 is h2
        assert store._built == [1]

    @pytest.mark.asyncio
    async def test_超过上限时淘汰最久未用(self):
        store = self._store_with_mock_build(max_instances=2)
        with patch.object(ls.settings, "LIGHTRAG_MAX_INSTANCES", 2), patch.object(
            ls.settings, "LIGHTRAG_INSTANCE_TTL", 99999
        ):
            await store.get_handle(1)
            await store.get_handle(2)
            # 访问 1 使其成为最近使用，再插入 3 应淘汰 2
            await store.get_handle(1)
            await store.get_handle(3)

        assert 1 in store._handles
        assert 3 in store._handles
        assert 2 not in store._handles

    @pytest.mark.asyncio
    async def test_invalidate_移除实例(self):
        store = self._store_with_mock_build()
        await store.get_handle(1)
        store.invalidate(1)
        assert 1 not in store._handles

    def test_invalidate_remove_data_删除工作目录(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            d = ls.kb_dir(1)
            d.mkdir(parents=True)
            (d / "graph.bin").write_text("x", encoding="utf-8")

            store = self._store_with_mock_build()
            store.invalidate(1, remove_data=True)

            assert not d.exists()

    def test_invalidate_不删数据时保留目录(self, tmp_path):
        with patch.object(ls.settings, "LIGHTRAG_DATA_DIR", str(tmp_path)):
            d = ls.kb_dir(1)
            d.mkdir(parents=True)

            store = self._store_with_mock_build()
            store.invalidate(1, remove_data=False)

            assert d.exists()


class TestSingleton:
    def test_全局单例(self):
        assert ls.LightRAGStore.get() is ls.LightRAGStore.get()
