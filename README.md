# Project Vend

**AI-powered agentic vending machine system for UCO**

Built on the [Gravity-Based Orbital Orchestration System](https://github.com/jpn-hnai/corpus-intelligence) — a data-agnostic, geometric approach to multi-agent dispatch where system events decompose into semantic fragments that exert gravitational pull on agent identity vectors.

---

## architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SIMULATION ENGINE                     │
│  (Virtual vending machine state, clock, user sim)       │
└────────────┬────────────────────────────────────────────┘
             │ stimuli (system events)
             ▼
┌─────────────────────────────────────────────────────────┐
│              GRAVITY ORCHESTRATION ENGINE                │
│                                                         │
│  decompose → embed → gravity field → gap detect →       │
│  parallel dispatch → assemble by gravity weight         │
│                                                         │
│  Fragment Taxonomy:                                     │
│  Product · Signal · Temporal · Context · Customer ·     │
│  Machine                                                │
└──┬──────────────┬──────────────────┬────────────────────┘
   │              │                  │
   ▼              ▼                  ▼
┌────────┐  ┌──────────┐  ┌──────────────┐
│INVENTORY│  │ CUSTOMER │  │   PRICING    │
│ AGENT   │  │  AGENT   │  │    AGENT     │
└────┬────┘  └────┬─────┘  └──────┬───────┘
     │            │               │
     ▼            ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                   SHARED STATE STORE                    │
└─────────────────────────────────────────────────────────┘
```

## how gravity orchestration works

Instead of hardcoded pub/sub subscriptions or LLM-in-the-loop agent selection, the gravity model lets **vector-space physics** decide which agents activate.

```
1. decompose    event → typed semantic fragments
                (product, signal, temporal, context, customer, machine)

2. embed        each fragment + full event via sentence-transformer

3. gravity      cosine similarity between fragment vectors
                and agent identity vectors → pull matrix

4. composite    L2 norm across all pulls per agent
                (multiple moderate pulls compound meaningfully)

5. activate     adaptive gap detection finds the natural elbow
                in sorted scores — self-calibrates per event

6. dispatch     activated agents run in parallel

7. assemble     results ranked by composite gravity score
```

A simple stock update on a quiet Tuesday might activate 1 agent. The same event during homecoming week with low stock and an expiring product might activate all 3. No manual routing rules needed.

## agents

| Agent | Domain | Handles |
|-------|--------|---------|
| **Inventory** | Stock management | Monitoring, demand prediction, restocking, waste reduction |
| **Customer** | User experience | Recommendations, purchase flow, accessibility, personalization |
| **Pricing** | Revenue optimization | Dynamic pricing, promotions, A/B tests, expiration discounts |
| **Analytics** | Reporting (meta) | Dashboards, KPIs, decision logging, anomaly detection |

## project structure

```
src/
├── gravity/          ← orchestration engine (adapted from corpus-intelligence)
│   ├── types.ts          fragment taxonomy, agent types, ledger types
│   ├── identities.ts     agent identity vectors (descriptions → embeddings)
│   ├── decompose.ts      rule-based event → fragment decomposition
│   ├── field.ts          pull matrix, L2 norm, gap detection, reliability bias
│   ├── dispatch.ts       parallel agent dispatch with timeouts
│   ├── assemble.ts       result assembly by gravity weight
│   ├── orchestrate.ts    end-to-end pipeline
│   ├── ledger.ts         learning loop (agent reliability tracking)
│   └── index.ts          public API
├── agents/           ← agent implementations (TODO)
├── simulation/       ← vending machine simulator (TODO)
├── state/            ← shared state store (TODO)
embeddings-service/   ← sentence-transformer embedding API (from corpus-intelligence)
gravity/              ← original Python prototype (reference)
data/                 ← runtime data (ledger, state)
```

## tech stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Gravity engine (TypeScript) |
| Agent runtime | TypeScript (Node.js) |
| LLM | Claude Haiku (decomposition), Claude Sonnet (agent reasoning) |
| Embeddings | all-mpnet-base-v2 (768-dim) via FastAPI service |
| State store | SQLite (prototype) → PostgreSQL (production) |
| Monitoring | Gravity Ledger (built-in self-tuning) |

## getting started

```bash
# install
npm install

# build
npm run build

# type check
npm run check
```

## status

**Early stage — architecture and gravity engine adapted, agent logic TBD.**

The gravity orchestration engine is ported and adapted from corpus-intelligence with the Vend-specific fragment taxonomy (Product/Signal/Temporal/Context/Customer/Machine) and agent identity vectors. Next steps: implement agent logic, build simulation engine, validate gravity activation patterns.

---

*Gravity orchestration system designed by Jer Nguyen — Hewes Nguyen AI Infrastructure*
*Adapted for Project Vend for University of Central Oklahoma*
