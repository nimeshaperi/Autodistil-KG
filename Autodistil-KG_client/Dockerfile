# Autodistil-KG Client (React / Vite)
# Multi-stage: build with Node, serve with nginx.

# ── Stage 1: Build ──
FROM node:20-alpine AS build

RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .

# API URL is relative — nginx proxies /api and /ws to the API container
ENV VITE_API_URL=/api
RUN pnpm run build

# ── Stage 2: Serve ──
FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
