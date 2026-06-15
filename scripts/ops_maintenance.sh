#!/bin/bash
# ============================================================
# DocMind 运维脚本 — 每日自动备份 + 日志清理 + 健康检查 + 告警
# ============================================================
# 功能：
#   1. MySQL 数据备份（mysqldump → gzip，保留最近 N 天）
#   2. 过期日志文件清理（按天数 + 目录白名单）
#   3. 服务健康检查（端口连通性 + HTTP 探活）
#   4. 异常自动重启（docker compose restart + 重试上限）
#   5. 告警通知（钉钉 / 飞书 / Slack / 自定义 Webhook）
#
# 用法：
#   bash scripts/ops_maintenance.sh              # 执行全部任务
#   bash scripts/ops_maintenance.sh backup       # 仅备份
#   bash scripts/ops_maintenance.sh health       # 仅健康检查
#   bash scripts/ops_maintenance.sh cleanup      # 仅日志清理
#   bash scripts/ops_maintenance.sh --dry-run    # 干运行（不实际执行）
#
# Cron 配置（每日凌晨 2:00 执行）：
#   0 2 * * * /bin/bash /opt/docmind/scripts/ops_maintenance.sh >> /var/log/docmind/ops.log 2>&1
# ============================================================

set -euo pipefail

# ============================================================
# 配置区（可按需修改）
# ============================================================

# ── 项目路径 ──
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_DIR}/backups}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
OPS_LOG="${OPS_LOG:-${LOG_DIR}/ops_maintenance.log}"

# ── MySQL 备份 ──
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"          # 备份保留天数
# 容器名使用模糊匹配（Docker Compose v2 实际名如 docmind-mysql-1）
MYSQL_CONTAINER="${MYSQL_CONTAINER:-mysql}"                   # MySQL 容器名（模糊匹配）
MYSQL_DATABASE="${MYSQL_DATABASE:-docmind}"
MYSQL_USER="${MYSQL_USER:-root}"
# MYSQL_PASSWORD 从 .env 自动读取，也可手动设置
MYSQL_PASSWORD="${MYSQL_PASSWORD:-}"

# ── 日志清理 ──
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"               # 普通日志保留天数
BACKUP_LOG_RETENTION_DAYS="${BACKUP_LOG_RETENTION_DAYS:-90}" # 运维日志保留天数
# 白名单目录（只清理这些目录下的日志，留空则跳过文件日志清理）
LOG_CLEANUP_DIRS="${LOG_CLEANUP_DIRS:-${PROJECT_DIR}/logs}"

# ── 健康检查 ──
# 格式: "标签|主机|端口|类型(tcp/http)|HTTP路径(仅http类型)"
# 每行一个服务，冒号分隔字段
HEALTH_CHECKS=(
    "MySQL|127.0.0.1|3306|tcp|"
    "Redis|127.0.0.1|6379|tcp|"
    "Backend|127.0.0.1|8000|http|/docs"
)

# 后端健康检查 URL（HTTP 200 探测）
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/docs}"

# ── 自动重启 ──
AUTO_RESTART="${AUTO_RESTART:-true}"                         # 是否自动重启故障服务
MAX_RESTART_RETRIES="${MAX_RESTART_RETRIES:-3}"              # 最大重启次数（单次运行内）
RESTART_COOLDOWN="${RESTART_COOLDOWN:-10}"                   # 重启冷却时间（秒）

# Docker Compose 服务名映射（健康检查标签 → docker compose 服务名）
declare -A SERVICE_MAP=(
    ["MySQL"]="mysql"
    ["Redis"]="redis"
    ["Backend"]="backend"
    ["Celery"]="celery"
)

# ── 告警通知 ──
# Webhook 类型: dingtalk | feishu | slack | custom
ALERT_WEBHOOK_TYPE="${ALERT_WEBHOOK_TYPE:-}"                 # 留空则禁用告警
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"                   # Webhook 地址
ALERT_MENTION_ALL="${ALERT_MENTION_ALL:-false}"              # 是否 @所有人

# ── 锁文件（防止并发执行）──
LOCK_FILE="${LOCK_FILE:-/tmp/docmind_ops_maintenance.lock}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-3600}"                         # 锁超时（秒），防止死锁

# ── 干运行模式 ──
DRY_RUN=false

# ============================================================
# 工具函数
# ============================================================

# 日志输出（同时输出到终端和日志文件）
_log() {
    local level="$1"; shift
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
    echo "$msg"
    if [ -d "$(dirname "$OPS_LOG")" ]; then
        echo "$msg" >> "$OPS_LOG"
    fi
}

log_info()  { _log "INFO" "$@"; }
log_warn()  { _log "WARN" "$@"; }
log_error() { _log "ERROR" "$@"; }

# 获取锁
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_age
        lock_age=$(($(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || stat -f %m "$LOCK_FILE" 2>/dev/null || echo 0)))
        if [ "$lock_age" -lt "$LOCK_TIMEOUT" ]; then
            log_warn "检测到锁文件 $LOCK_FILE（已存在 ${lock_age}s），跳过本次执行"
            exit 0
        fi
        log_warn "锁文件已超时（${lock_age}s > ${LOCK_TIMEOUT}s），强制获取"
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
    # 退出时自动释放锁
    trap 'rm -f "$LOCK_FILE"' EXIT
}

# 读取 .env 中的变量
load_env() {
    local env_file="${PROJECT_DIR}/backend/.env"
    if [ -f "$env_file" ]; then
        # 只读取非注释行，导出为环境变量
        while IFS='=' read -r key value; do
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # 跳过空行和注释
            [ -z "$key" ] && continue
            [ "${key:0:1}" = "#" ] && continue
            # 不覆盖已设置的环境变量（但 MYSQL_PASSWORD 特殊处理）
            if [ "$key" = "MYSQL_PASSWORD" ] && [ -z "$MYSQL_PASSWORD" ]; then
                MYSQL_PASSWORD="$value"
            fi
            if [ "$key" = "MYSQL_DATABASE" ] && [ "$MYSQL_DATABASE" = "docmind" ]; then
                MYSQL_DATABASE="$value"
            fi
        done < "$env_file"
    else
        log_warn ".env 文件不存在: $env_file，使用默认/环境变量配置"
    fi
}

# 发送 Webhook 告警
send_alert() {
    local title="$1"
    local content="$2"
    local level="${3:-error}"  # error | warning | info

    if [ -z "$ALERT_WEBHOOK_URL" ]; then
        return 0
    fi

    local hostname
    hostname=$(hostname 2>/dev/null || echo "unknown")
    local full_title="[DocMind][${hostname}] ${title}"
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')

    case "$ALERT_WEBHOOK_TYPE" in
        dingtalk)
            _send_dingtalk "$full_title" "$content" "$now"
            ;;
        feishu)
            _send_feishu "$full_title" "$content" "$now" "$level"
            ;;
        slack)
            _send_slack "$full_title" "$content" "$now"
            ;;
        custom)
            _send_custom "$full_title" "$content" "$now" "$level"
            ;;
        *)
            log_warn "未知的告警渠道: $ALERT_WEBHOOK_TYPE，跳过通知"
            ;;
    esac
}

_send_dingtalk() {
    local title="$1" content="$2" now="$3"
    local mention=""
    if [ "$ALERT_MENTION_ALL" = "true" ]; then
        mention=', "at": {"isAtAll": true}'
    fi
    local payload
    payload=$(cat <<EOF
{
    "msgtype": "markdown",
    "markdown": {
        "title": "${title}",
        "text": "## ${title}\n\n**时间**: ${now}\n\n${content}"
    }${mention}
}
EOF
)
    curl -s -X POST "$ALERT_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" > /dev/null 2>&1 || true
}

_send_feishu() {
    local title="$1" content="$2" now="$3" level="$4"
    local color="red"
    [ "$level" = "warning" ] && color="yellow"
    [ "$level" = "info" ] && color="green"
    local payload
    payload=$(cat <<EOF
{
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": "${title}"},
            "template": "${color}"
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": "**时间**: ${now}\n${content}"}}
        ]
    }
}
EOF
)
    curl -s -X POST "$ALERT_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" > /dev/null 2>&1 || true
}

_send_slack() {
    local title="$1" content="$2" now="$3"
    local payload
    payload=$(cat <<EOF
{
    "attachments": [{
        "fallback": "${title}",
        "title": "${title}",
        "text": "${content}",
        "footer": "DocMind Ops | ${now}",
        "color": "danger"
    }]
}
EOF
)
    curl -s -X POST "$ALERT_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" > /dev/null 2>&1 || true
}

_send_custom() {
    local title="$1" content="$2" now="$3" level="$4"
    local payload
    payload=$(cat <<EOF
{
    "source": "DocMind",
    "title": "${title}",
    "content": "${content}",
    "time": "${now}",
    "hostname": "$(hostname 2>/dev/null || echo 'unknown')",
    "level": "${level}"
}
EOF
)
    curl -s -X POST "$ALERT_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" > /dev/null 2>&1 || true
}

# ============================================================
# 模块 1：MySQL 数据备份
# ============================================================

do_backup() {
    log_info "========== 开始 MySQL 备份 =========="

    mkdir -p "$BACKUP_DIR"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/docmind_${timestamp}.sql.gz"
    local checksum_file="${BACKUP_DIR}/docmind_${timestamp}.sql.gz.sha256"

    # 检查 MySQL 密码
    if [ -z "$MYSQL_PASSWORD" ]; then
        log_error "MYSQL_PASSWORD 未设置，无法执行备份"
        send_alert "MySQL 备份失败" "原因: MYSQL_PASSWORD 未配置" "error"
        return 1
    fi

    # 检查 MySQL 容器是否运行
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "${MYSQL_CONTAINER}"; then
        log_warn "MySQL 容器未运行，尝试使用宿主机 mysqldump"

        if command -v mysqldump &> /dev/null; then
            log_info "使用宿主机 mysqldump 备份..."
            if [ "$DRY_RUN" = false ]; then
                set +e  # 临时关闭 pipefail，避免管道失败导致脚本退出
                mysqldump -h 127.0.0.1 -P 3306 \
                    -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
                    --single-transaction --routines --triggers --events \
                    --databases "$MYSQL_DATABASE" 2>/dev/null \
                    | gzip > "$backup_file"
                local _exit_code=$?
                set -e
                _verify_backup "$backup_file" "$checksum_file" "$_exit_code"
            else
                log_info "[DRY-RUN] mysqldump -h 127.0.0.1 ... | gzip > $backup_file"
            fi
        else
            log_error "mysqldump 不可用且容器未运行，备份失败"
            send_alert "MySQL 备份失败" "原因: 容器未运行且宿主机无 mysqldump" "error"
            return 1
        fi
    else
        log_info "通过 Docker 容器执行 mysqldump..."
        if [ "$DRY_RUN" = false ]; then
            set +e  # 临时关闭 pipefail，避免管道失败导致脚本退出
            docker exec "$MYSQL_CONTAINER" \
                mysqldump -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
                --single-transaction --routines --triggers --events \
                --databases "$MYSQL_DATABASE" 2>/dev/null \
                | gzip > "$backup_file"
            local _exit_code=$?
            set -e
            _verify_backup "$backup_file" "$checksum_file" "$_exit_code"
        else
            log_info "[DRY-RUN] docker exec $MYSQL_CONTAINER mysqldump ... | gzip > $backup_file"
        fi
    fi

    # 清理过期备份
    _cleanup_old_backups

    log_info "========== MySQL 备份完成 =========="
}

_verify_backup() {
    local backup_file="$1"
    local checksum_file="$2"
    local exit_code="$3"

    if [ "$exit_code" -ne 0 ] || [ ! -s "$backup_file" ]; then
        log_error "mysqldump 失败（退出码: $exit_code），备份文件可能不完整"
        rm -f "$backup_file"
        send_alert "MySQL 备份失败" "原因: mysqldump 退出码 $exit_code" "error"
        return 1
    fi

    local size
    size=$(stat -c %s "$backup_file" 2>/dev/null || stat -f %z "$backup_file" 2>/dev/null || echo 0)
    shasum -a 256 "$backup_file" | cut -d' ' -f1 > "$checksum_file"
    log_info "备份成功: $backup_file (${size} bytes)"

    # 磁盘空间检查
    local avail_kb
    avail_kb=$(df -k "$BACKUP_DIR" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    if [ "$avail_kb" -lt 1048576 ]; then  # < 1GB
        log_warn "备份目录磁盘空间不足: ${avail_kb}KB 可用"
        send_alert "磁盘空间告警" "备份目录 $BACKUP_DIR 剩余空间: $((avail_kb / 1024))MB" "warning"
    fi
}

_cleanup_old_backups() {
    log_info "清理 ${BACKUP_RETENTION_DAYS} 天前的备份文件..."

    local deleted_count=0
    while IFS= read -r old_file; do
        [ -z "$old_file" ] && continue
        if [ "$DRY_RUN" = false ]; then
            rm -f "$old_file"
            deleted_count=$((deleted_count + 1))
            log_info "  删除: $old_file"
        else
            log_info "  [DRY-RUN] 将删除: $old_file"
            deleted_count=$((deleted_count + 1))
        fi
    done < <(find "$BACKUP_DIR" -name "docmind_*.sql.gz" -mtime +"$BACKUP_RETENTION_DAYS" 2>/dev/null || true)

    # 同时清理过期校验和文件
    find "$BACKUP_DIR" -name "*.sha256" -mtime +"$BACKUP_RETENTION_DAYS" -delete 2>/dev/null || true

    log_info "清理完成，删除 ${deleted_count} 个过期备份"
}

# ============================================================
# 模块 2：日志文件清理
# ============================================================

do_cleanup() {
    log_info "========== 开始日志清理 =========="

    local total_deleted=0

    # 2.1 清理项目日志目录
    if [ -n "$LOG_CLEANUP_DIRS" ]; then
        for dir in $(echo "$LOG_CLEANUP_DIRS" | tr ',' ' '); do
            if [ ! -d "$dir" ]; then
                log_info "  日志目录不存在，跳过: $dir"
                continue
            fi
            log_info "清理目录: $dir（>${LOG_RETENTION_DAYS}天的 *.log 文件）"
            if [ "$DRY_RUN" = false ]; then
                local count
                count=$(find "$dir" -name "*.log" -mtime +"$LOG_RETENTION_DAYS" -type f 2>/dev/null | wc -l)
                find "$dir" -name "*.log" -mtime +"$LOG_RETENTION_DAYS" -type f -delete 2>/dev/null || true
                log_info "  删除 ${count} 个日志文件"
                total_deleted=$((total_deleted + count))
            else
                local count
                count=$(find "$dir" -name "*.log" -mtime +"$LOG_RETENTION_DAYS" -type f 2>/dev/null | wc -l)
                log_info "  [DRY-RUN] 将删除 ${count} 个日志文件"
                total_deleted=$((total_deleted + count))
            fi
        done
    fi

    # 2.2 清理 Docker 容器日志（截断而非删除，避免容器 I/O 异常）
    log_info "截断 Docker 容器日志..."
    if [ "$DRY_RUN" = false ]; then
        for container in mysql redis backend celery; do
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "${container}"; then
                local log_path
                log_path=$(docker inspect --format='{{.LogPath}}' "$container" 2>/dev/null || echo "")
                if [ -n "$log_path" ] && [ -f "$log_path" ]; then
                    local before_size after_size
                    before_size=$(stat -c %s "$log_path" 2>/dev/null || stat -f %z "$log_path" 2>/dev/null || echo 0)
                    # 截断保留最后 50MB
                    tail -c 50M "$log_path" > "${log_path}.tmp" 2>/dev/null && mv "${log_path}.tmp" "$log_path" || true
                    after_size=$(stat -c %s "$log_path" 2>/dev/null || stat -f %z "$log_path" 2>/dev/null || echo 0)
                    log_info "  ${container}: ${before_size} → ${after_size} bytes"
                fi
            fi
        done
    else
        log_info "  [DRY-RUN] 将截断运行中容器的日志"
    fi

    # 2.3 清理过期运维日志
    if [ -d "$LOG_DIR" ]; then
        if [ "$DRY_RUN" = false ]; then
            local ops_count
            ops_count=$(find "$LOG_DIR" -name "ops_*.log" -mtime +"$BACKUP_LOG_RETENTION_DAYS" -type f 2>/dev/null | wc -l)
            find "$LOG_DIR" -name "ops_*.log" -mtime +"$BACKUP_LOG_RETENTION_DAYS" -type f -delete 2>/dev/null || true
            log_info "清理 ${ops_count} 个过期运维日志"
        else
            local ops_count
            ops_count=$(find "$LOG_DIR" -name "ops_*.log" -mtime +"$BACKUP_LOG_RETENTION_DAYS" -type f 2>/dev/null | wc -l)
            log_info "  [DRY-RUN] 将删除 ${ops_count} 个过期运维日志"
        fi
    fi

    log_info "========== 日志清理完成（共清理 ${total_deleted} 个文件）=========="
}

# ============================================================
# 模块 3：服务健康检查 + 自动重启
# ============================================================

do_health() {
    log_info "========== 开始健康检查 =========="

    local failed_services=()
    local all_ok=true

    for check in "${HEALTH_CHECKS[@]}"; do
        IFS='|' read -r label host port type http_path <<< "$check"

        local ok=false
        case "$type" in
            tcp)
                if _check_tcp "$host" "$port"; then
                    ok=true
                fi
                ;;
            http)
                if _check_http "$host" "$port" "$http_path"; then
                    ok=true
                fi
                ;;
            *)
                log_warn "未知检查类型: $type（服务: $label）"
                continue
                ;;
        esac

        if [ "$ok" = true ]; then
            log_info "  [OK] $label ($host:$port)"
        else
            log_error "  [FAIL] $label ($host:$port) — 不可达"
            all_ok=false
            failed_services+=("$label")
        fi
    done

    # 处理故障服务
    if [ "$all_ok" = false ]; then
        local failed_list
        failed_list=$(printf '%s, ' "${failed_services[@]}" | sed 's/, $//')

        if [ "$AUTO_RESTART" = true ]; then
            log_warn "以下服务异常，尝试重启: $failed_list"
            _restart_services "${failed_services[@]}"
        else
            log_warn "以下服务异常（未启用自动重启）: $failed_list"
            send_alert "服务异常告警" "异常服务: ${failed_list}\n已禁用自动重启，请手动处理" "error"
        fi
    else
        log_info "所有服务健康检查通过"
    fi

    log_info "========== 健康检查完成 =========="
}

_check_tcp() {
    local host="$1" port="$2"
    # 使用 bash 内置 /dev/tcp（超时 5s）
    if timeout 5 bash -c "echo > /dev/tcp/${host}/${port}" 2>/dev/null; then
        return 0
    fi
    # 回退方案：nc
    if command -v nc &> /dev/null; then
        if nc -z -w 5 "$host" "$port" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

_check_http() {
    local host="$1" port="$2" path="$3"
    local url="http://${host}:${port}${path}"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 "$url" 2>/dev/null || echo "000")

    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 500 ]; then
        return 0
    fi
    log_warn "    HTTP 状态码: $http_code (URL: $url)"
    return 1
}

_restart_services() {
    local services=("$@")
    local restart_failed=()

    for svc_label in "${services[@]}"; do
        local compose_svc="${SERVICE_MAP[$svc_label]:-}"

        if [ -z "$compose_svc" ]; then
            log_warn "  未找到 $svc_label 对应的 docker compose 服务名，跳过重启"
            restart_failed+=("$svc_label")
            continue
        fi

        local success=false
        for attempt in $(seq 1 "$MAX_RESTART_RETRIES"); do
            log_info "  重启 $svc_label ($compose_svc) — 第 $attempt/$MAX_RESTART_RETRIES 次"

            if [ "$DRY_RUN" = false ]; then
                cd "$PROJECT_DIR"
                if docker compose restart "$compose_svc" 2>/dev/null; then
                    log_info "    重启命令执行成功，等待 ${RESTART_COOLDOWN}s 后验证..."
                    sleep "$RESTART_COOLDOWN"

                    # 重启后验证
                    if _verify_service "$svc_label"; then
                        log_info "    [OK] $svc_label 重启后恢复正常"
                        send_alert "服务已恢复" "服务: ${svc_label}\n重启次数: ${attempt}\n状态: 已恢复正常" "info"
                        success=true
                        break
                    else
                        log_warn "    重启后验证未通过，继续重试..."
                    fi
                else
                    log_error "    docker compose restart 命令失败"
                fi
            else
                log_info "    [DRY-RUN] docker compose restart $compose_svc"
                success=true
                break
            fi
        done

        if [ "$success" = false ]; then
            log_error "  [FAIL] $svc_label 重启 ${MAX_RESTART_RETRIES} 次后仍未恢复"
            restart_failed+=("$svc_label")
        fi

        # 服务间冷却
        if [ "$svc_label" != "${services[-1]}" ]; then
            sleep 3
        fi
    done

    # 最终告警
    if [ ${#restart_failed[@]} -gt 0 ]; then
        local failed_list
        failed_list=$(printf '%s, ' "${restart_failed[@]}" | sed 's/, $//')
        log_error "以下服务重启失败，需要人工介入: $failed_list"
        send_alert "⚠️ 服务重启失败（需人工介入）" \
            "异常服务: ${failed_list}\n最大重试次数: ${MAX_RESTART_RETRIES}\n主机: $(hostname 2>/dev/null || echo 'unknown')\n请立即检查！" \
            "error"
    fi
}

_verify_service() {
    local label="$1"

    for check in "${HEALTH_CHECKS[@]}"; do
        IFS='|' read -r check_label host port type http_path <<< "$check"
        if [ "$check_label" != "$label" ]; then
            continue
        fi

        case "$type" in
            tcp) _check_tcp "$host" "$port" && return 0 ;;
            http) _check_http "$host" "$port" "$http_path" && return 0 ;;
        esac
    done
    return 1
}

# ============================================================
# 主流程
# ============================================================

main() {
    # 解析参数
    local mode="all"

    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            backup|health|cleanup)
                mode="$1"
                shift
                ;;
            --help|-h)
                _print_help
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                _print_help
                exit 1
                ;;
        esac
    done

    # 干运行提示
    if [ "$DRY_RUN" = true ]; then
        echo "============================================="
        echo "  ⚠ DRY-RUN 模式 — 不会执行实际操作"
        echo "============================================="
    fi

    # 初始化
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    load_env
    acquire_lock

    log_info "DocMind 运维脚本启动 (PID=$$, 模式=$mode)"

    case "$mode" in
        all)
            do_backup
            do_cleanup
            do_health
            ;;
        backup)
            do_backup
            ;;
        cleanup)
            do_cleanup
            ;;
        health)
            do_health
            ;;
    esac

    log_info "DocMind 运维脚本执行完毕"
}

_print_help() {
    cat << 'EOF'
DocMind 运维脚本 — 每日自动备份 + 日志清理 + 健康检查 + 告警

用法:
  bash scripts/ops_maintenance.sh [MODE] [OPTIONS]

MODE（可选，默认执行全部）:
  backup       仅执行 MySQL 备份
  cleanup      仅执行日志清理
  health       仅执行健康检查 + 自动重启

OPTIONS:
  --dry-run    干运行模式，只输出将要执行的操作，不实际执行
  --help, -h   显示此帮助信息

环境变量（按需覆盖）:
  BACKUP_RETENTION_DAYS    备份保留天数（默认 7）
  LOG_RETENTION_DAYS       日志保留天数（默认 30）
  AUTO_RESTART             是否自动重启故障服务（默认 true）
  ALERT_WEBHOOK_TYPE       告警渠道: dingtalk|feishu|slack|custom
  ALERT_WEBHOOK_URL        Webhook 地址
  MYSQL_PASSWORD           MySQL 密码（默认从 backend/.env 读取）

Cron 配置示例（每日凌晨 2:00）:
  0 2 * * * /bin/bash /opt/docmind/scripts/ops_maintenance.sh >> /var/log/docmind/ops.log 2>&1

日志文件:
  运维日志: $OPS_LOG
  备份目录: $BACKUP_DIR
EOF
}

# 入口
main "$@"
