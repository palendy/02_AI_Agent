사내에서 ghcr.io에 직접 접근이 안 되는 경우, 로컬로 이미지를 다운로드해서 사용하는 방법을 안내드리겠습니다.

## 1. 외부에서 Docker 이미지 다운로드

**외부 네트워크가 가능한 환경에서:**

```bash
# 1. GitHub MCP Server 이미지 pull
docker pull ghcr.io/github/github-mcp-server:latest

# 2. 이미지를 tar 파일로 저장
docker save ghcr.io/github/github-mcp-server:latest > github-mcp-server.tar

# 3. 이미지 정보 확인
docker images | grep github-mcp-server
```

## 2. 사내 환경으로 이미지 전송 및 로드

**사내 환경에서:**

```bash
# 1. tar 파일을 사내로 복사 (USB, 내부 파일 서버 등 활용)
# 2. Docker에 이미지 로드
docker load < github-mcp-server.tar

# 3. 로드된 이미지 확인
docker images | grep github-mcp-server

# 4. 태그 변경 (선택사항)
docker tag ghcr.io/github/github-mcp-server:latest local/github-mcp-server:latest
```

## 3. 사내용 Docker Compose 설정

```yaml
version: '3.8'
services:
  github-mcp-server:
    image: local/github-mcp-server:latest  # 로컬 이미지 사용
    environment:
      - HTTP_PROXY=http://your-corporate-proxy:8080
      - HTTPS_PROXY=http://your-corporate-proxy:8080
      - NO_PROXY=localhost,127.0.0.1,*.internal.company.com
      - GITHUB_PERSONAL_ACCESS_TOKEN=${GITHUB_TOKEN}
      - GITHUB_TOOLSETS=repos,issues,pull_requests,actions,code_security
    ports:
      - "8080:8080"
    volumes:
      - ./config:/config
    restart: unless-stopped
    networks:
      - internal

  mcp-proxy:
    build:
      context: ./mcp-proxy  # 소스코드로 빌드
      dockerfile: Dockerfile
    ports:
      - "9090:9090"
    environment:
      - HTTP_PROXY=http://your-corporate-proxy:8080
      - HTTPS_PROXY=http://your-corporate-proxy:8080
    volumes:
      - ./proxy-config.json:/config/config.json
    networks:
      - internal
    depends_on:
      - github-mcp-server

networks:
  internal:
    driver: bridge
```

## 4. 소스코드 기반 빌드 방법

**GitHub MCP Server를 소스코드로 빌드:**

```bash
# 1. 외부에서 소스코드 다운로드
git clone https://github.com/github/github-mcp-server.git
cd github-mcp-server

# 2. Dockerfile 확인 및 수정 (프록시 설정 추가)
cat > Dockerfile.corporate << 'EOF'
FROM golang:1.21-alpine AS builder

# 프록시 설정
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

WORKDIR /app
COPY . .
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o github-mcp-server ./cmd/server

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/github-mcp-server .
EXPOSE 8080
CMD ["./github-mcp-server"]
EOF

# 3. 이미지 빌드
docker build -f Dockerfile.corporate \
  --build-arg HTTP_PROXY=http://your-proxy:8080 \
  --build-arg HTTPS_PROXY=http://your-proxy:8080 \
  -t local/github-mcp-server:latest .

# 4. 빌드된 이미지를 tar로 저장
docker save local/github-mcp-server:latest > github-mcp-server-local.tar
```

## 5. 사내 Private Registry 활용

**사내에 Docker Registry가 있다면:**

```bash
# 1. 로컬 이미지를 사내 레지스트리로 push
docker tag local/github-mcp-server:latest your-registry.company.com/github-mcp-server:latest
docker push your-registry.company.com/github-mcp-server:latest

# 2. docker-compose.yml에서 사내 레지스트리 사용
```

```yaml
version: '3.8'
services:
  github-mcp-server:
    image: your-registry.company.com/github-mcp-server:latest
    environment:
      - GITHUB_PERSONAL_ACCESS_TOKEN=${GITHUB_TOKEN}
    # ... 나머지 설정
```

## 6. 실행 및 테스트

```bash
# 1. 환경변수 설정
export GITHUB_TOKEN="your-personal-access-token"

# 2. Docker Compose 실행
docker-compose up -d

# 3. 로그 확인
docker-compose logs -f github-mcp-server

# 4. 헬스 체크
curl http://localhost:8080/health

# 5. MCP 서버 테스트 (stdio 모드)
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}' | docker exec -i $(docker-compose ps -q github-mcp-server) ./github-mcp-server
```

## 7. 보안 및 네트워크 설정

```bash
# Dockerfile에 프록시 설정 추가
ENV HTTP_PROXY=http://your-proxy:8080
ENV HTTPS_PROXY=http://your-proxy:8080
ENV NO_PROXY=localhost,127.0.0.1,*.company.internal

# 인증서가 필요한 경우
COPY corporate-ca-certificates.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
```

이 방법으로 사내 환경에서 GitHub MCP Server를 안전하게 사용할 수 있습니다. 중요한 것은 보안 정책에 맞게 프록시 설정과 인증서 관리를 적절히 해주는 것입니다.