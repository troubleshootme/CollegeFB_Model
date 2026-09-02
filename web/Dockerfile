FROM node:20-alpine
WORKDIR /app
COPY package.json /app/package.json
RUN npm install
COPY . /app
ENV API_PROXY=http://api:8000
EXPOSE 5173
CMD ["npm", "run", "dev"]
