# Multi-Agent Chatbot System

3개의 전문 Agent가 협력하여 사용자의 질문을 처리하는 지능형 챗봇 시스템입니다.

## 시스템 아키텍처

### Agents

1. **ChatAgent** - 질문 분류 및 라우팅
   - 사용자 질문을 분석하여 정보 문의인지 VOC인지 분류
   - 적절한 Agent로 라우팅
   - 대화 컨텍스트 관리

2. **RAG Agent** - 정보 검색 및 안내
   - GitHub MCP Server를 통해 문서 검색
   - 관련 정보를 찾아 자세한 안내 제공
   - 기존 RAG 기능 활용

3. **Issue Agent** - 이슈 검색, 생성, 관리
   - GitHub MCP Server를 통해 이슈 검색
   - 유사한 이슈가 있는지 확인
   - 새로운 이슈 생성 지원
   - 이슈 상태 관리

### MCP Servers

- **GitHub MCP Server**: GitHub API와의 통신을 담당

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd multi_agent_chatbot

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env_example .env
# .env 파일을 편집하여 필요한 API 키들을 설정
```

### 2. 실행

```bash
# 메인 애플리케이션 실행
python main.py

# Streamlit 웹 인터페이스 실행
python -m streamlit run view/streamlit_app.py
```

## 사용법

### 명령행 인터페이스

```bash
# 대화 모드
python main.py chat

# 시스템 정보 확인
python main.py info

# 도움말
python main.py help
```

### 웹 인터페이스

Streamlit을 실행한 후 브라우저에서 `http://localhost:8501`에 접속

## 설정

`.env` 파일에서 다음 설정들을 조정할 수 있습니다:

- `OPENAI_API_KEY`: OpenAI API 키
- `GITHUB_TOKEN`: GitHub Personal Access Token
- `GITHUB_REPOSITORIES`: 검색할 GitHub 저장소 목록
- `MCP_SERVER_URL`: MCP 서버 URL
- 각 Agent별 모델 설정

## 개발

### 프로젝트 구조

```
multi_agent_chatbot/
├── agents/                    # Agent 구현체들
│   ├── base_agent.py         # 기본 Agent 클래스
│   ├── chat_agent.py         # Chat Agent 
│   ├── rag_agent.py          # RAG Agent 
│   └── issue_agent.py        # Issue Agent 
├── core/                     # 핵심 워크플로우
│   └── multi_agent_workflow.py  # Multi-Agent 워크플로우
├── mcp_clients/              # MCP 클라이언트들
│   └── github_mcp_client.py  # GitHub MCP Client
├── utils/                    # 유틸리티
│   └── config.py             # 설정 관리
├── view/                     # 웹 인터페이스
│   └── streamlit_app.py      # Streamlit 앱
├── main.py                   # 메인 애플리케이션
├── run_streamlit.py          # Streamlit 실행 스크립트
├── test_system.py            # 시스템 테스트
└── requirements.txt          # 의존성 목록
```

### 테스트 실행

```bash
pytest tests/
```

## 라이선스

MIT License
