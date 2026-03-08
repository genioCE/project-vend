#!/usr/bin/env bash
# dev.sh — corpus intelligence dev engine
#
# Usage:
#   ./dev.sh              Start engines
#   ./dev.sh stop         Stop engines
#   ./dev.sh status       Check engine health
#   ./dev.sh restart      Cycle engines
#   ./dev.sh logs         Tail docker logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Palette (monochrome) ─────────────────────────────────────
RST=$'\033[0m'
WHITE=$'\033[38;2;255;255;255m'       # #ffffff — peak, success
BRIGHT=$'\033[38;2;224;224;224m'      # #e0e0e0 — important text, info
SILVER=$'\033[38;2;176;176;176m'     # #b0b0b0 — standard text, warnings
MID=$'\033[38;2;128;128;128m'         # #808080 — secondary, spinner
DIM=$'\033[38;2;88;88;88m'            # #585858 — subtle, borders
MUTED=$'\033[38;2;56;56;56m'          # #383838 — barely visible
BOLD=$'\033[1m'
CLR=$'\033[K'
HIDE_CUR=$'\033[?25l'
SHOW_CUR=$'\033[?25h'

# ── Services ──────────────────────────────────────────────────
SVC_NAMES=(  "neo4j"  "embeddings"  "graph"    "analysis"  "mcp-server" )
SVC_PORTS=(  "7474"   "8000"        "8001"     "8002"      "3001"       )
SVC_PATHS=(  "/"      "/health"     "/health"  "/health"   "/health"    )
TOTAL=${#SVC_NAMES[@]}

DOCKER_SVCS="neo4j embeddings-service graph-service analysis-service mcp-server"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.neo4j-ports.yml"

SPIN="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ── Banner Art ────────────────────────────────────────────────
BANNER_LINES=(
  " ██████╗ ██████╗ ██████╗ ██████╗ ██╗   ██╗███████╗"
  "██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║   ██║██╔════╝"
  "██║     ██║   ██║██████╔╝██████╔╝██║   ██║███████╗"
  "██║     ██║   ██║██╔══██╗██╔═══╝ ██║   ██║╚════██║"
  "╚██████╗╚██████╔╝██║  ██║██║     ╚██████╔╝███████║"
  " ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚══════╝"
)
BANNER_ROWS=${#BANNER_LINES[@]}

GREY_STEPS=( 32 56 88 128 176 224 255 )
STEP_COUNT=${#GREY_STEPS[@]}

# ── Helpers ───────────────────────────────────────────────────
cleanup() { printf '%s' "$SHOW_CUR"; }
trap cleanup EXIT

check_health() {
    curl -sf --max-time 1 "http://127.0.0.1:${1}${2}" &>/dev/null 2>&1
}

preflight() {
    if ! command -v docker &>/dev/null; then
        echo ""; echo "  ${WHITE}${BOLD}✗${RST} docker not found"; echo ""; exit 1
    fi
    if ! docker info &>/dev/null 2>&1; then
        echo ""; echo "  ${WHITE}${BOLD}✗${RST} docker daemon not running"; echo ""; exit 1
    fi
    if [[ ! -f .env ]]; then
        echo ""; echo "  ${WHITE}${BOLD}✗${RST} .env not found"; echo ""; exit 1
    fi
    source .env
    if [[ -z "${CORPUS_PATH:-}" ]] || [[ ! -d "${CORPUS_PATH}" ]]; then
        echo ""; echo "  ${WHITE}${BOLD}✗${RST} CORPUS_PATH not set or directory missing"; echo ""; exit 1
    fi
}

# ── Animated Banner ───────────────────────────────────────────
print_banner() {
    local tagline="dev engine"
    local display_lines=$(( BANNER_ROWS + 1 ))  # art + tagline
    local total_art_frames=$(( BANNER_ROWS - 1 + STEP_COUNT ))  # 12
    local tagline_steps=( 56 128 224 )
    local tagline_count=${#tagline_steps[@]}
    local total_frames=$(( total_art_frames + tagline_count ))  # 15

    printf '%s' "$HIDE_CUR"
    echo ""

    # Reserve vertical space
    for (( i=0; i<display_lines; i++ )); do echo ""; done

    # Animate fade-in
    for (( frame=0; frame<total_frames; frame++ )); do
        printf '\033[%dA' "$display_lines"

        for (( row=0; row<BANNER_ROWS; row++ )); do
            printf '\r\033[2K'
            local step_idx=$(( frame - row ))
            if (( step_idx >= 0 && step_idx < STEP_COUNT )); then
                local g=${GREY_STEPS[$step_idx]}
                printf '  \033[38;2;%d;%d;%dm%s%s' "$g" "$g" "$g" "${BANNER_LINES[$row]}" "$RST"
            elif (( step_idx >= STEP_COUNT )); then
                printf '  %s%s%s' "$WHITE" "${BANNER_LINES[$row]}" "$RST"
            fi
            printf '\n'
        done

        # Tagline
        printf '\r\033[2K'
        local tag_frame=$(( frame - total_art_frames ))
        if (( tag_frame >= 0 && tag_frame < tagline_count )); then
            local tg=${tagline_steps[$tag_frame]}
            printf '  \033[38;2;%d;%d;%dm%s%s' "$tg" "$tg" "$tg" "$tagline" "$RST"
        elif (( tag_frame >= tagline_count )); then
            printf '  %s%s%s' "$DIM" "$tagline" "$RST"
        fi
        printf '\n'

        sleep 0.05
    done

    printf '%s' "$SHOW_CUR"
    echo ""
}

# Static banner (no animation, for status/stop)
print_banner_static() {
    echo ""
    for (( i=0; i<BANNER_ROWS; i++ )); do
        echo "  ${WHITE}${BANNER_LINES[$i]}${RST}"
    done
    echo "  ${DIM}dev engine${RST}"
    echo ""
}

# Cursor offset from park position to service line i
svc_up() { echo $(( TOTAL - $1 + 2 )); }

# ── Start ─────────────────────────────────────────────────────
cmd_start() {
    preflight
    clear
    print_banner
    printf '%s' "$HIDE_CUR"

    for (( i=0; i<TOTAL; i++ )); do
        printf "  ${DIM}○${RST}  ${DIM}%-16s${RST} ${MUTED}:%s${RST}${CLR}\n" "${SVC_NAMES[$i]}" "${SVC_PORTS[$i]}"
    done
    echo ""
    printf "  ${DIM}starting...${RST}${CLR}\n"

    # Launch containers
    if ! docker compose $COMPOSE_FILES up -d $DOCKER_SVCS &>/dev/null 2>&1; then
        printf '\033[1A\r'
        printf "  ${WHITE}${BOLD}✗${RST} compose failed${CLR}\n"
        printf '%s' "$SHOW_CUR"
        echo ""
        echo "  ${DIM}run manually to see errors:${RST}"
        echo "  docker compose $COMPOSE_FILES up $DOCKER_SVCS"
        echo ""
        exit 1
    fi

    # Poll for health
    local done_arr=()
    for (( i=0; i<TOTAL; i++ )); do done_arr+=( 0 ); done
    local ready=0
    local spin_i=0
    local frame=0
    local start_ts
    start_ts=$(date +%s)
    local max_wait=120

    while [[ $ready -lt $TOTAL ]]; do
        local now
        now=$(date +%s)
        if (( now - start_ts >= max_wait )); then break; fi

        local sc="${SPIN:$spin_i:1}"
        spin_i=$(( (spin_i + 1) % ${#SPIN} ))

        for (( i=0; i<TOTAL; i++ )); do
            [[ ${done_arr[$i]} -eq 1 ]] && continue

            local up
            up=$(svc_up $i)

            # Check health every 4 frames (~1.2s)
            if (( frame % 4 == 0 )); then
                if check_health "${SVC_PORTS[$i]}" "${SVC_PATHS[$i]}"; then
                    done_arr[$i]=1
                    ready=$((ready + 1))
                    printf '\033[%dA\r' "$up"
                    printf "  ${WHITE}✓${RST}  ${SILVER}%-16s${RST} ${MUTED}:%s${RST}${CLR}" "${SVC_NAMES[$i]}" "${SVC_PORTS[$i]}"
                    printf '\033[%dB\r' "$up"
                    continue
                fi
            fi

            # Spinner for pending services
            printf '\033[%dA\r' "$up"
            printf "  ${MID}%s${RST}  ${DIM}%-16s${RST} ${MUTED}:%s${RST}${CLR}" "$sc" "${SVC_NAMES[$i]}" "${SVC_PORTS[$i]}"
            printf '\033[%dB\r' "$up"
        done

        # Update status line
        printf '\033[1A\r'
        printf "  ${DIM}engines %d/%d${RST}${CLR}" "$ready" "$TOTAL"
        printf '\033[1B\r'

        frame=$((frame + 1))
        sleep 0.3
    done

    # Final status
    local elapsed=$(( $(date +%s) - start_ts ))
    printf '\033[1A\r'
    if [[ $ready -eq $TOTAL ]]; then
        printf "  ${WHITE}✓${RST} ${SILVER}all engines online${RST} ${DIM}(%ds)${RST}${CLR}\n" "$elapsed"
    else
        printf "  ${SILVER}!${RST} ${SILVER}%d/%d engines${RST} ${DIM}(timed out)${RST}${CLR}\n" "$ready" "$TOTAL"
    fi

    printf '%s' "$SHOW_CUR"
    echo ""

    if [[ $ready -eq $TOTAL ]]; then
        echo "  ${BRIGHT}→${RST} ${DIM}mcp${RST}    ${BRIGHT}http://127.0.0.1:3001/sse${RST}"
        echo "  ${BRIGHT}→${RST} ${DIM}neo4j${RST}  ${BRIGHT}http://127.0.0.1:7474${RST}"
    else
        for (( i=0; i<TOTAL; i++ )); do
            if [[ ${done_arr[$i]} -eq 0 ]]; then
                echo "  ${WHITE}${BOLD}✗${RST} ${SVC_NAMES[$i]} ${DIM}didn't start${RST}"
            fi
        done
        echo ""
        echo "  ${DIM}docker compose logs <service>${RST}"
    fi
    echo ""
}

# ── Status ────────────────────────────────────────────────────
cmd_status() {
    clear
    print_banner_static

    local all_up=1
    for (( i=0; i<TOTAL; i++ )); do
        if check_health "${SVC_PORTS[$i]}" "${SVC_PATHS[$i]}"; then
            printf "  ${WHITE}✓${RST}  ${SILVER}%-16s${RST} ${MUTED}:%s${RST}\n" "${SVC_NAMES[$i]}" "${SVC_PORTS[$i]}"
        else
            printf "  ${WHITE}${BOLD}✗${RST}  ${DIM}%-16s${RST} ${MUTED}:%s${RST}\n" "${SVC_NAMES[$i]}" "${SVC_PORTS[$i]}"
            all_up=0
        fi
    done

    echo ""
    if [[ $all_up -eq 1 ]]; then
        echo "  ${BRIGHT}→${RST} ${DIM}mcp${RST}    ${BRIGHT}http://127.0.0.1:3001/sse${RST}"
        echo "  ${BRIGHT}→${RST} ${DIM}neo4j${RST}  ${BRIGHT}http://127.0.0.1:7474${RST}"
    else
        echo "  ${DIM}some engines offline${RST}"
    fi
    echo ""
}

# ── Stop ──────────────────────────────────────────────────────
cmd_stop() {
    clear
    print_banner_static
    printf '%s' "$HIDE_CUR"
    echo "  ${DIM}shutting down...${RST}"
    docker compose down &>/dev/null 2>&1 || true
    printf '\033[1A\r'
    echo "  ${WHITE}✓${RST} ${SILVER}engines stopped${RST}${CLR}"
    printf '%s' "$SHOW_CUR"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────
CMD="${1:-start}"

case "$CMD" in
    start|up)       cmd_start ;;
    stop|down)      cmd_stop ;;
    status|ps)      cmd_status ;;
    restart)
        docker compose down &>/dev/null 2>&1 || true
        cmd_start
        ;;
    logs)
        docker compose logs -f --tail=50
        ;;
    *)
        clear
        print_banner_static
        echo "  ${BRIGHT}usage${RST}  ./dev.sh [command]"
        echo ""
        echo "  ${SILVER}start${RST}     ${DIM}spin up engines (default)${RST}"
        echo "  ${SILVER}stop${RST}      ${DIM}shut down engines${RST}"
        echo "  ${SILVER}restart${RST}   ${DIM}cycle engines${RST}"
        echo "  ${SILVER}status${RST}    ${DIM}check engine health${RST}"
        echo "  ${SILVER}logs${RST}      ${DIM}tail docker logs${RST}"
        echo ""
        ;;
esac
