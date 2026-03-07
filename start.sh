#!/usr/bin/env bash
#
# start.sh — Start the corpus-mcp stack
#
# Usage:
#   ./start.sh          # Start core services (neo4j, embeddings, graph, analysis, mcp-server, web-ui)
#   ./start.sh ingest   # Run vector ingest then start core
#   ./start.sh graph    # Run graph ingest then start core
#   ./start.sh full     # Run all ingests, then start core
#   ./start.sh down     # Tear down everything
#   ./start.sh status   # Show service status
#   ./start.sh logs     # Tail logs from all services
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[corpus-mcp]${NC} $*"; }
warn() { echo -e "${YELLOW}[corpus-mcp]${NC} $*"; }
err()  { echo -e "${RED}[corpus-mcp]${NC} $*" >&2; }

# ── Preflight checks ──────────────────────────────────────────────
preflight() {
    if ! command -v docker &>/dev/null; then
        err "Docker not found. Install Docker Desktop first."
        exit 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        err "Docker daemon not running. Start Docker Desktop first."
        exit 1
    fi

    if [[ ! -f .env ]]; then
        err ".env file not found. Copy .env.example and fill in values."
        exit 1
    fi

    # Verify critical env vars
    source .env
    if [[ -z "${CORPUS_PATH:-}" ]]; then
        err "CORPUS_PATH not set in .env"
        exit 1
    fi
    if [[ ! -d "$CORPUS_PATH" ]]; then
        err "CORPUS_PATH directory does not exist: $CORPUS_PATH"
        exit 1
    fi
}

# ── Commands ──────────────────────────────────────────────────────
start_core() {
    log "Starting core services..."
    docker compose -f docker-compose.yml -f docker-compose.neo4j-ports.yml up -d neo4j embeddings-service graph-service analysis-service mcp-server web-ui
    log "Waiting for services to become healthy..."
    
    # Wait for MCP server (the last in the chain)
    local attempts=0
    local max_attempts=60
    while ! curl -sf http://127.0.0.1:3001/health &>/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ $attempts -ge $max_attempts ]]; then
            warn "MCP server didn't respond after ${max_attempts}s — check logs with: ./start.sh logs"
            return 1
        fi
        sleep 1
    done

    echo ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  ${CYAN}corpus-mcp is ready${NC}"
    log ""
    log "  MCP endpoint:  ${CYAN}http://127.0.0.1:3001/sse${NC}"
    log "  Web UI:        ${CYAN}http://127.0.0.1:3000${NC}"
    log "  Neo4j browser: ${CYAN}http://127.0.0.1:7474${NC}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    log "Add to Claude Desktop config:"
    echo '  {'
    echo '    "mcpServers": {'
    echo '      "corpus-intelligence": {'
    echo '        "url": "http://127.0.0.1:3001/sse"'
    echo '      }'
    echo '    }'
    echo '  }'
    echo ""
}

run_ingest() {
    log "Running vector ingest..."
    docker compose --profile ingest run --rm ingest
    log "Vector ingest complete."
}

run_graph_ingest() {
    log "Running graph ingest..."
    docker compose --profile graph-ingest run --rm graph-ingest
    log "Graph ingest complete."
}

run_batch_analysis() {
    log "Running batch analysis..."
    docker compose --profile batch-analysis run --rm batch-analysis
    log "Batch analysis complete."
}

show_status() {
    log "Service status:"
    docker compose ps
    echo ""
    
    # Quick health checks
    for svc_port in "mcp-server:3001" "web-ui:3000"; do
        svc="${svc_port%%:*}"
        port="${svc_port##*:}"
        if curl -sf "http://127.0.0.1:${port}/health" &>/dev/null 2>&1; then
            echo -e "  ${GREEN}●${NC} ${svc} (port ${port})"
        else
            echo -e "  ${RED}●${NC} ${svc} (port ${port})"
        fi
    done
    
    if curl -sf "http://127.0.0.1:7474" &>/dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} neo4j (port 7474)"
    else
        echo -e "  ${RED}●${NC} neo4j (port 7474)"
    fi
}

# ── Main ──────────────────────────────────────────────────────────
CMD="${1:-start}"

case "$CMD" in
    start|up)
        preflight
        start_core
        ;;
    ingest)
        preflight
        start_core
        run_ingest
        ;;
    graph)
        preflight
        start_core
        run_graph_ingest
        ;;
    full)
        preflight
        start_core
        run_ingest
        run_graph_ingest
        run_batch_analysis
        log "Full pipeline complete."
        ;;
    down|stop)
        log "Stopping all services..."
        docker compose down
        log "All services stopped."
        ;;
    restart)
        log "Restarting all services..."
        docker compose down
        preflight
        start_core
        ;;
    status|ps)
        show_status
        ;;
    logs)
        docker compose logs -f --tail=50
        ;;
    build)
        log "Rebuilding images..."
        docker compose build
        log "Build complete. Run ./start.sh to start."
        ;;
    *)
        echo "Usage: ./start.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start    Start core services (default)"
        echo "  ingest   Start core + run vector ingest"
        echo "  graph    Start core + run graph ingest"
        echo "  full     Start core + all ingests + batch analysis"
        echo "  down     Stop all services"
        echo "  restart  Stop then start all services"
        echo "  status   Show service health"
        echo "  logs     Tail service logs"
        echo "  build    Rebuild Docker images"
        exit 1
        ;;
esac
