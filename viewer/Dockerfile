FROM node:22-bookworm-slim AS packages

ENV PNPM_HOME=/pnpm
ENV PATH=/pnpm:$PATH

RUN corepack enable \
    && corepack prepare pnpm@10.22.0 --activate

WORKDIR /app

COPY viewer/package.json viewer/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM packages AS development

COPY viewer/ ./

EXPOSE 4173

CMD ["pnpm", "dev", "--host", "0.0.0.0"]

FROM packages AS build

COPY viewer/ ./
RUN pnpm build

FROM nginx:1.27-alpine AS production

COPY viewer/nginx.conf /etc/nginx/nginx.conf
COPY --from=build --chown=nginx:nginx /app/dist /usr/share/nginx/html

USER nginx

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["wget", "--quiet", "--output-document=/dev/null", "http://127.0.0.1:8080/healthz"]

CMD ["nginx", "-g", "daemon off;"]
