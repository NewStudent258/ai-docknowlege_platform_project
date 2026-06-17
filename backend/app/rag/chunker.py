"""智能分块 — 先识别章节，再在章节内执行固定大小分块

对齐 ARCHITECTURE.md §4.2:
- 算法: RecursiveCharacterTextSplitter
- 分隔符优先级: 段落 → 换行 → 中文标点 → 英文标点 → 空格 → 字符
- chunk_size: 1000 chars（800-1200 范围）
- chunk_overlap: 150 chars（≈50 tokens）
- keep_separator: True（中文场景保留语义完整性）
- Token 估算: int(len(content) / 1.5)，不引入 tiktoken

对齐 Phase 6 PR2:
- detect_sections(): Markdown 标题检测
- SectionResult: 章节结构化结果
- chunk_document(): 先按 section 切分，再在 section 内分 chunk
"""

import logging
import re
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.rag.parser import ParsedPage

from app.config import settings

logger = logging.getLogger(__name__)

# 分隔符优先级（对齐 ARCHITECTURE.md §4.2）
# RecursiveCharacterTextSplitter.separators 是精确字符串匹配，非正则，
# 因此中文/英文标点展开为独立字符，才能在每个标点处正确断句。
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# Markdown ATX 标题正则（行首 # 开头，支持 # 至 ######）
# 使用 [^\S\n] 替代 \s：排除换行符，避免将 "# \n内容" 误判为标题
_MD_HEADING_PATTERN = re.compile(r'^(#{1,6})[^\S\n]+(.+)$', re.MULTILINE)
_SYNTHETIC_SECTION_TITLE = "全文"


@dataclass
class SectionResult:
    """单个章节结果"""
    title: str
    path: str
    level: int
    start_offset: int
    end_offset: int
    start_chunk_index: int
    end_chunk_index: int
    synthetic: bool = False


@dataclass
class ChunkResult:
    """单个分块结果"""
    content: str
    chunk_index: int
    page_number: int | None
    estimated_tokens: int
    section_index: int | None = None
    section_title: str | None = None   # 当前所属章节标题（如 "§6.1 SSE 事件格式"）
    section_path: str | None = None    # 章节路径（如 "RAG Pipeline > §6. SSE 事件流"）


@dataclass
class ChunkingResult:
    """分块聚合结果"""
    sections: list[SectionResult] = field(default_factory=list)
    chunks: list[ChunkResult] = field(default_factory=list)
    total_chunks: int = 0


def chunk_document(
    text: str,
    pages: list[ParsedPage] | None = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> ChunkingResult:
    """对文档全文执行智能分块，可选附带页码元数据。

    Args:
        text: 文档全文（ParseResult.full_text）
        pages: 解析结果中的页面列表，用于回溯每块的来源页码
        chunk_size: 每块最大字符数
        chunk_overlap: 块间重叠字符数

    Returns:
        ChunkingResult: 包含所有分块及总数
    """
    if not text or not text.strip():
        logger.warning("文档全文为空，跳过智能分块")
        return ChunkingResult()

    splitter = RecursiveCharacterTextSplitter(
        separators=CHUNK_SEPARATORS,
        keep_separator=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # 构建页码偏移映射，用于回溯每块的来源页码
    page_offset_map = build_page_offset_map(pages) if pages else []

    raw_sections = _build_section_ranges(text)
    logger.info("文档章节切分完成: %d 个 section", len(raw_sections))

    # 在 section 内分块，再将 chunk 映射回全文偏移量
    sections: list[SectionResult] = []
    chunks: list[ChunkResult] = []
    for raw_section in raw_sections:
        section_text = text[raw_section.start_offset:raw_section.end_offset].strip()
        if not section_text:
            continue

        section_chunk_texts = splitter.split_text(section_text)
        if not section_chunk_texts:
            continue

        section_index = len(sections)
        section_search_start = raw_section.start_offset
        start_chunk_index = len(chunks)

        for chunk_text in section_chunk_texts:
            start_offset = text.find(chunk_text, section_search_start)
            if start_offset == -1:
                # 极端情况下退化为 section 起点，至少保证页码和 section 归属稳定
                logger.warning("section 内 chunk 偏移定位失败，回退到 section 起点")
                start_offset = section_search_start
            else:
                section_search_start = start_offset + len(chunk_text)

            page_number = resolve_page_number(start_offset, page_offset_map)
            estimated_tokens = estimate_tokens(chunk_text)
            chunks.append(ChunkResult(
                content=chunk_text,
                chunk_index=len(chunks),
                page_number=page_number,
                estimated_tokens=estimated_tokens,
                section_index=section_index,
                section_title=None if raw_section.synthetic else raw_section.title,
                section_path=None if raw_section.synthetic else raw_section.path,
            ))

        sections.append(SectionResult(
            title=raw_section.title,
            path=raw_section.path,
            level=raw_section.level,
            start_offset=raw_section.start_offset,
            end_offset=raw_section.end_offset,
            start_chunk_index=start_chunk_index,
            end_chunk_index=len(chunks) - 1,
            synthetic=raw_section.synthetic,
        ))

    logger.info("文档分块完成: %d 个 section, %d 块", len(sections), len(chunks))
    return ChunkingResult(sections=sections, chunks=chunks, total_chunks=len(chunks))


# 页分隔符：必须与 parser.py 中 ParsedResult.full_text 的 join 分隔符一致
# parser.py 使用 "\n\n".join(p.content for p in self.pages ...) 拼接全文
_PAGE_SEPARATOR = "\n\n"
_PAGE_SEPARATOR_LEN = len(_PAGE_SEPARATOR)  # = 2


def build_page_offset_map(pages: list[ParsedPage]) -> list[tuple[int, int]]:
    """构建 (字符偏移量, 页码) 映射列表，按偏移量升序。

    纯函数。重建 full_text 拼接逻辑：每页 content 后跟 _PAGE_SEPARATOR 分隔符。
    偏移量计算必须与 parser.py 中 ParsedResult.full_text 的拼接方式一致。

    验证要点：ParsedPage → 偏移元组的正确转换，特别是 "\\n\\n" 分隔符长度计算。
    """
    offset_map: list[tuple[int, int]] = []
    pos = 0
    for page in pages:
        if page.success and page.content:
            offset_map.append((pos, page.page_number))
            pos += len(page.content) + _PAGE_SEPARATOR_LEN
    return offset_map


def _build_section_ranges(text: str) -> list[SectionResult]:
    """构建覆盖全文的 section 范围。

    规则：
    - 有标题时，每个标题形成一个显式 section
    - 若首个标题前有前言内容，则补一个 synthetic section
    - 无标题文档则生成一个 synthetic 根 section，保证新文档 chunk 全部可关联 section
    """
    headings = detect_sections(text)
    if not headings:
        return [SectionResult(
            title=_SYNTHETIC_SECTION_TITLE,
            path=_SYNTHETIC_SECTION_TITLE,
            level=1,
            start_offset=0,
            end_offset=len(text),
            start_chunk_index=0,
            end_chunk_index=0,
            synthetic=True,
        )]

    sections: list[SectionResult] = []
    first_heading_offset = headings[0][0]
    if text[:first_heading_offset].strip():
        sections.append(SectionResult(
            title=_SYNTHETIC_SECTION_TITLE,
            path=_SYNTHETIC_SECTION_TITLE,
            level=1,
            start_offset=0,
            end_offset=first_heading_offset,
            start_chunk_index=0,
            end_chunk_index=0,
            synthetic=True,
        ))

    level_stack: list[tuple[int, str]] = []
    for i, (offset, level, title) in enumerate(headings):
        while level_stack and level_stack[-1][0] >= level:
            level_stack.pop()
        level_stack.append((level, title))
        next_offset = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        sections.append(SectionResult(
            title=title,
            path=" > ".join(item_title for _, item_title in level_stack),
            level=level,
            start_offset=offset,
            end_offset=next_offset,
            start_chunk_index=0,
            end_chunk_index=0,
            synthetic=False,
        ))

    return sections


def resolve_page_number(
    start_offset: int,
    offset_map: list[tuple[int, int]],
) -> int | None:
    """通过字符偏移量在偏移映射中反查来源页码。

    纯函数。边界验证：首块、末块、跨页块、空映射。
    """
    if not offset_map or start_offset == -1:
        return None

    page_number: int | None = None
    for offset, pg in offset_map:
        if offset <= start_offset:
            page_number = pg
        else:
            break

    return page_number


def detect_sections(text: str) -> list[tuple[int, int, str]]:
    """从文本中提取 Markdown 标题层级，返回 (偏移量, 层级, 标题文本) 列表。

    纯函数。对齐 ROADMAP.md §8.7：Markdown 通过 #/##/### 正则提取标题，
    DOCX 标题样式已由 parser.py 转换为 # 标记，因此可跨格式统一检测。

    验证要点：ATX 标题层级、非标题行过滤、空标题过滤。

    层级规则：
    - # → level 1
    - ## → level 2
    - 以此类推，最多六级

    Args:
        text: 文档全文

    Returns:
        按字符偏移量升序排列的 section 列表
    """
    if not text:
        return []

    sections: list[tuple[int, int, str]] = []
    for m in _MD_HEADING_PATTERN.finditer(text):
        level = len(m.group(1))  # 1-6
        title = m.group(2).strip()
        if title:
            sections.append((m.start(), level, title))

    return sections


def resolve_section(
    start_offset: int,
    sections: list[tuple[int, int, str]],
) -> tuple[str | None, str | None]:
    """根据字符偏移量反查当前所属章节。

    纯函数。对齐 ROADMAP.md §8.7：类似 resolve_page_number() 的回溯逻辑，
    找到当前偏移量之前最近的一个 heading → 即为当前 section_title，
    结合 heading 层级栈构建 section_path。

    验证要点：同级替换、子嵌套、空输入。

    Args:
        start_offset: chunk 在全文中的起始偏移量
        sections: detect_sections() 返回的 section 列表

    Returns:
        (section_title, section_path) — section_title 为当前节标题，
        section_path 为从顶层到当前节的完整路径
    """
    if not sections or start_offset == -1:
        return None, None

    # 回溯偏移量之前经过的所有 heading，维护层级栈
    level_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    section_title: str | None = None

    for offset, level, title in sections:
        if offset > start_offset:
            break
        # 弹出 >= 当前层级的旧标题（同级或更高层级替换，低层级为子节）
        while level_stack and level_stack[-1][0] >= level:
            level_stack.pop()
        level_stack.append((level, title))
        section_title = title

    if section_title is None:
        return None, None

    # 构建路径：父标题 > 子标题
    path_parts = [t for _, t in level_stack]
    section_path = " > ".join(path_parts) if len(path_parts) > 1 else path_parts[0] if path_parts else None

    return section_title, section_path


def estimate_tokens(text: str) -> int:
    """用字符数估算 token 数。

    中文字符占比 > 30% → 1 token ≈ 1.5 字符
    否则（纯英文/英文为主）→ 1 token ≈ 4.0 字符
    """
    if not text:
        return 1

    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    ratio = (
        settings.TOKEN_CHINESE_RATIO
        if chinese_chars / len(text) > settings.TOKEN_CHINESE_THRESHOLD
        else settings.TOKEN_ENGLISH_RATIO
    )
    return max(1, int(len(text) / ratio))
