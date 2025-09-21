"""
AI Agent Chatbot Streamlit Web Interface
GitHub 문서 기반 지능형 챗봇 웹 인터페이스
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model import AIChatbot
from config import get_config
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.repository_manager import render_repository_manager

# 페이지 설정
st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .system-info {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """세션 상태 초기화"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"
    
    if 'repository_urls' not in st.session_state:
        st.session_state.repository_urls = []


def initialize_chatbot():
    """챗봇 초기화"""
    try:
        if st.session_state.chatbot is None:
            with st.spinner("🤖 AI Chatbot을 초기화하는 중..."):
                st.session_state.chatbot = AIChatbot()
                st.session_state.system_initialized = True
                st.success("✅ AI Chatbot 초기화 완료!")
        return True
    except Exception as e:
        st.error(f"❌ 챗봇 초기화 실패: {str(e)}")
        return False


def render_header():
    """헤더 렌더링"""
    st.markdown('<h1 class="main-header">🤖 AI Agent Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### GitHub 문서 기반 지능형 챗봇 - Corrective RAG + LangGraph")
    
    # 상태 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.system_initialized:
            st.success("✅ 시스템 준비됨")
        else:
            st.warning("⚠️ 시스템 초기화 필요")
    
    with col2:
        if st.session_state.chatbot:
            info = st.session_state.chatbot.get_system_info()
            doc_count = info.get('vector_store', {}).get('document_count', 0)
            st.metric("📚 문서 수", doc_count)
        else:
            st.metric("📚 문서 수", 0)
    
    with col3:
        st.metric("💬 대화 수", len(st.session_state.chat_history))
    
    with col4:
        if st.session_state.repository_urls:
            st.metric("📁 Repository", len(st.session_state.repository_urls))
        else:
            st.metric("📁 Repository", 0)


def render_navigation():
    """네비게이션 렌더링"""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💬 채팅", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    with col2:
        if st.button("📁 Repository 관리", use_container_width=True):
            st.session_state.current_page = "repository"
            st.rerun()
    
    with col3:
        if st.button("⚙️ 설정", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
    
    with col4:
        if st.button("📊 시스템 정보", use_container_width=True):
            st.session_state.current_page = "info"
            st.rerun()


def render_main_content():
    """메인 콘텐츠 렌더링"""
    if st.session_state.current_page == "chat":
        render_chat_interface()
    elif st.session_state.current_page == "repository":
        render_repository_manager()
    elif st.session_state.current_page == "settings":
        render_sidebar()
    elif st.session_state.current_page == "info":
        render_system_info()


def render_system_info():
    """시스템 정보 렌더링"""
    st.header("📊 시스템 정보")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        info = st.session_state.chatbot.get_system_info()
        
        # 벡터 스토어 정보
        st.subheader("🗄️ 벡터 스토어")
        vector_info = info.get('vector_store', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("문서 수", vector_info.get('document_count', 0))
        with col2:
            st.metric("컬렉션", vector_info.get('collection_name', 'Unknown'))
        with col3:
            st.metric("저장 경로", vector_info.get('persist_directory', 'Unknown'))
        
        # 워크플로우 정보
        st.subheader("🔄 워크플로우")
        workflow_info = info.get('workflow', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("모델", workflow_info.get('model_name', 'Unknown'))
        with col2:
            st.metric("최대 재시도", workflow_info.get('max_retries', 0))
        with col3:
            st.metric("관련성 임계값", f"{workflow_info.get('relevance_threshold', 0):.2f}")
        
        # 설정 정보
        st.subheader("⚙️ 설정")
        config_info = info.get('config', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**임베딩 모델**: {config_info.get('embedding_model', 'Unknown')}")
            st.write(f"**청크 크기**: {config_info.get('chunk_size', 0)}")
            st.write(f"**청크 오버랩**: {config_info.get('chunk_overlap', 0)}")
        
        with col2:
            st.write(f"**최대 검색 결과**: {config_info.get('max_search_results', 0)}")
            st.write(f"**대화 수**: {info.get('conversation_count', 0)}")
            st.write(f"**초기화 상태**: {'✅' if info.get('initialized') else '❌'}")
        
        # 대화 기록
        if st.session_state.chat_history:
            st.subheader("💬 최근 대화 기록")
            
            for i, entry in enumerate(st.session_state.chat_history[-5:], 1):
                with st.expander(f"대화 {i}: {entry['question'][:50]}..."):
                    st.write(f"**질문**: {entry['question']}")
                    st.write(f"**답변**: {entry['answer']}")
                    st.write(f"**검색 소스**: {entry['search_source']}")
                    st.write(f"**관련성 점수**: {entry['relevance_score']:.3f}")
                    st.write(f"**시간**: {entry['timestamp']}")
    
    except Exception as e:
        st.error(f"❌ 시스템 정보 조회 실패: {str(e)}")


def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더 렌더링
    render_header()
    
    # 네비게이션 렌더링
    render_navigation()
    
    # 챗봇 초기화
    if not st.session_state.system_initialized:
        if st.button("🚀 시스템 초기화", use_container_width=True):
            initialize_chatbot()
            st.rerun()
    else:
        # 메인 콘텐츠 렌더링
        render_main_content()


if __name__ == "__main__":
    main()
