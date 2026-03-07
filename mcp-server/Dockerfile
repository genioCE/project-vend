FROM node:20-slim AS builder

RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build

FROM node:20-slim

RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*
RUN groupadd -r mcpuser && useradd -r -g mcpuser -m mcpuser

WORKDIR /app

COPY --from=builder /app/package.json /app/package-lock.json* ./
RUN npm install --omit=dev && npm cache clean --force

COPY --from=builder /app/dist/ ./dist/

RUN mkdir -p /app/data && chown mcpuser:mcpuser /app/data

USER mcpuser

EXPOSE 3001

# Default: run the HTTP API server for the web UI.
# Claude Desktop overrides this with: node dist/index.js
CMD ["node", "dist/api-server.js"]
