FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
RUN apk add --no-cache python3
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/conf.d/nginx.conf.template
COPY generate_nginx.sh /generate_nginx.sh
RUN chmod +x /generate_nginx.sh
EXPOSE 80
CMD ["sh", "/generate_nginx.sh"]