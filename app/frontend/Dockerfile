# Production Dockerfile for React Web Frontend (Vite + React + NGINX)
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# NGINX stage for serving production React static bundle
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 3000

# Custom NGINX config for React client-side routing & port 3000
RUN sed -i 's/listen       80;/listen       3000;/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]
