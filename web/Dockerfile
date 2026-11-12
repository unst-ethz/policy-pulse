# The Vue frontend as static files behind nginx. Build context: web/.
#   docker build --build-arg WEB_BASE=/experimental/ -t policy-pulse-web web
# WEB_BASE is the path the site is served under; the server in front strips it before proxying
# here, so this nginx always serves from /.
FROM node:24-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG WEB_BASE=/
RUN WEB_BASE=$WEB_BASE npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
