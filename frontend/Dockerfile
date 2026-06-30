# ============================================================
# Frontend — Vite dev server (HMR) for Docker Compose
# ============================================================
# Dev-mode image: deps are baked at build time, but compose bind-mounts
# ./frontend over /frontend for live reload. An anonymous volume on
# /frontend/node_modules (declared in docker-compose.yml) keeps THESE
# image-built deps from being shadowed by the host directory.
#
# The Vite dev server proxies /v1 -> the API service (VITE_PROXY_TARGET,
# default http://api:8000), so the browser only talks to this container.
FROM node:22-alpine

WORKDIR /frontend

COPY package.json package-lock.json* ./
# Use 'ci' if a lock file is present, otherwise fall back to 'install'.
RUN if [ -f package-lock.json ]; then npm ci --prefer-offline; else npm install; fi

COPY . ./

EXPOSE 3000

# --host 0.0.0.0 so the dev server is reachable from outside the container.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
