"""临时脚本：把 resources/policy_docs 下的制度文档批量灌入知识库

用法：python upload_policy_docs.py
完成后可删除。
"""
import sys
import time
from pathlib import Path

import httpx

BASE = "http://localhost:8000"
DOC_DIR = Path("F:/pycharm_t/docmind-main/resources/policy_docs")
KB_NAME = "公司制度汇编（100 份）"
BATCH = 50  # 对齐 BATCH_UPLOAD_MAX_COUNT


def main() -> None:
    files = sorted(DOC_DIR.glob("*.txt"))
    if not files:
        print(f"未找到文档：{DOC_DIR}")
        sys.exit(1)
    print(f"待上传 {len(files)} 份文档")

    c = httpx.Client(base_url=BASE, timeout=180)

    # 1. 登录
    r = c.post("/api/auth/login", json={"username": "testuser", "password": "Test123456"})
    r.raise_for_status()
    c.headers["Authorization"] = f"Bearer {r.json()['data']['access_token']}"
    print("1. 登录 OK")

    # 2. 建知识库（public，便于同时展示在「公开知识库」）
    r = c.post("/api/knowledge-bases", json={
        "name": KB_NAME,
        "description": "公司制度汇编，涵盖薪酬福利、考勤休假、奖惩管理、人力资源、信息与数据安全、行政后勤、财务与采购七大类共 100 份制度文件",
        "visibility": "public",
    })
    if r.status_code not in (200, 201):
        print("建库失败:", r.status_code, r.text[:400])
        sys.exit(1)
    kb = r.json()["data"]
    kb_uuid = kb.get("uuid") or kb.get("kb_uuid") or kb.get("id")
    print(f"2. 建库 OK  kb_uuid={kb_uuid}")

    # 3. 分批上传
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        payload = [
            ("files", (f.name, f.read_bytes(), "text/plain; charset=utf-8"))
            for f in chunk
        ]
        r = c.post(f"/api/knowledge-bases/{kb_uuid}/documents/batch-upload", files=payload)
        if r.status_code not in (200, 201, 202):
            print(f"   批次 {i // BATCH + 1} 失败: {r.status_code} {r.text[:400]}")
            continue
        d = r.json()["data"]
        ok, bad = len(d.get("success", [])), len(d.get("failed", []))
        print(f"3. 批次 {i // BATCH + 1}: 成功 {ok}，失败 {bad}")
        if bad:
            for f in d["failed"][:5]:
                print("     失败详情:", f)

    # 4. 轮询入库进度
    print("4. 等待 Celery 入库…")
    last_done = -1
    for i in range(120):
        time.sleep(10)
        r = c.get(f"/api/knowledge-bases/{kb_uuid}/documents",
                  params={"page": 1, "page_size": 1})
        if r.status_code != 200:
            continue
        # 用列表接口的 total + 状态统计
        # 注意：page_size 上限为 100（接口 le=100 校验）
        rows = c.get(f"/api/knowledge-bases/{kb_uuid}/documents",
                     params={"page": 1, "page_size": 100}).json()["data"]
        items = rows.get("items", [])
        stat = {}
        for it in items:
            stat[it["status"]] = stat.get(it["status"], 0) + 1
        done = stat.get("completed", 0)
        if done != last_done:
            print(f"   [{i * 10}s] 共 {len(items)} 份  状态分布: {stat}")
            last_done = done
        if stat.get("completed", 0) + stat.get("failed", 0) >= len(files):
            print("   全部处理完毕")
            break

    # 5. 汇总
    rows = c.get(f"/api/knowledge-bases/{kb_uuid}/documents",
                 params={"page": 1, "page_size": 200}).json()["data"]
    items = rows.get("items", [])
    stat = {}
    for it in items:
        stat[it["status"]] = stat.get(it["status"], 0) + 1
    total_chunks = sum(it.get("chunk_count") or 0 for it in items)
    print(f"\n最终：{len(items)} 份文档，状态 {stat}，分块合计 {total_chunks}")
    print(f"知识库名称：{KB_NAME}")
    print(f"kb_uuid  ：{kb_uuid}")


if __name__ == "__main__":
    main()
