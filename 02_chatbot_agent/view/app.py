"""
AI Agent Chatbot Streamlit Web Interface
GitHub 문서 기반 지능형 챗봇 웹 인터페이스
"""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import get_config

# 환경 변수에서 로깅 레벨 읽기
config = get_config()
log_level = getattr(logging, config.log_level.upper(), logging.INFO)

# 로깅 설정 (환경 변수에서 읽은 레벨 사용)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 콘솔에 출력
    ]
)

from model import AIChatbot
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface

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
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = "default"
    
    if 'available_sessions' not in st.session_state:
        st.session_state.available_sessions = []
    
    if 'selected_repository' not in st.session_state:
        st.session_state.selected_repository = None


def initialize_chatbot():
    """챗봇 초기화"""
    try:
        if st.session_state.chatbot is None:
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
    st.markdown("#### GitHub 문서 기반 지능형 챗봇 - Corrective RAG + LangGraph")
    
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
        if st.session_state.chatbot and st.session_state.chatbot.vector_stores:
            st.metric("🎯 서비스", len(st.session_state.chatbot.vector_stores))
        else:
            st.metric("🎯 서비스", 0)
    
    # Repository 선택
    if st.session_state.system_initialized and st.session_state.chatbot:
        render_repository_selector()


def render_repository_selector():
    """서비스 선택기 렌더링"""
    st.markdown("---")
    
    # 서비스 선택 헤더
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎯 문의할 서비스 선택</h3>
        <p style="color: #666; font-size: 0.9rem;">어떤 서비스에 대해 문의하시겠습니까?</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # 사용 가능한 repository 목록 조회
        repositories = st.session_state.chatbot.get_available_repositories()
        
        if not repositories:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; margin: 0.5rem 0;">
                <h4 style="color: #856404; margin-bottom: 0.5rem;">⚠️ 사용 가능한 서비스가 없습니다</h4>
                <p style="color: #856404; font-size: 0.9rem;">시스템을 먼저 초기화해주세요.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # 서비스 선택 드롭다운 (더 간단하고 작게)
        st.markdown("#### 📋 사용 가능한 서비스")
        
        # 현재 선택된 repository 찾기
        current_repo = st.session_state.chatbot.get_current_repository()
        repo_options = [f"{repo['name']} ({repo['document_count']}개)" for repo in repositories]
        repo_urls = [repo['url'] for repo in repositories]
        
        current_index = 0
        if current_repo and current_repo in repo_urls:
            current_index = repo_urls.index(current_repo)
        
        # 서비스 선택 드롭다운
        selected_index = st.selectbox(
            "서비스를 선택하세요:",
            range(len(repo_options)),
            index=current_index,
            format_func=lambda x: repo_options[x],
            help="문의할 서비스를 선택하세요.",
            key="service_selector"
        )
        
        # 서비스 변경 시 처리
        selected_url = repo_urls[selected_index]
        if selected_url != current_repo:
            if st.session_state.chatbot.set_current_repository(selected_url):
                st.session_state.selected_repository = selected_url
                st.success(f"✅ 서비스 변경됨: {repositories[selected_index]['name']}")
                st.rerun()
            else:
                st.error("❌ 서비스 변경에 실패했습니다.")
        
        # 현재 선택된 서비스 정보를 작은 카드로 표시
        if current_repo:
            current_repo_info = next((repo for repo in repositories if repo['url'] == current_repo), None)
            if current_repo_info:
                st.markdown(f"""
                <div style="
                    text-align: center; 
                    padding: 0.5rem; 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    border-radius: 6px; 
                    margin: 0.3rem 0;
                ">
                    <h5 style="margin: 0; color: white; font-size: 0.9rem;">🎯 현재 선택: {current_repo_info['name']}</h5>
                    <p style="margin: 0; color: white; opacity: 0.9; font-size: 0.8rem;">📚 {current_repo_info['document_count']}개 문서</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 새로고침 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            if st.button("🔄 새로고침", use_container_width=True, help="서비스 목록을 새로고침합니다."):
                st.rerun()
        
    
    except Exception as e:
        st.error(f"❌ 서비스 선택기 오류: {str(e)}")


def render_navigation():
    """네비게이션 렌더링"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 채팅", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    with col2:
        if st.button("📚 채팅 히스토리", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    
    with col3:
        if st.button("📊 시스템 정보", use_container_width=True):
            st.session_state.current_page = "info"
            st.rerun()


def render_main_content():
    """메인 콘텐츠 렌더링"""
    if st.session_state.current_page == "chat":
        render_chat_interface()
    elif st.session_state.current_page == "history":
        render_chat_history()
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


def render_chat_history():
    """채팅 히스토리 렌더링"""
    st.header("📚 채팅 히스토리")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        # 세션 관리
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 사용 가능한 세션 목록 조회
            available_sessions = st.session_state.chatbot.get_all_sessions()
            if not available_sessions:
                available_sessions = ["default"]
            
            selected_session = st.selectbox(
                "세션 선택",
                available_sessions,
                index=available_sessions.index(st.session_state.current_session_id) if st.session_state.current_session_id in available_sessions else 0
            )
            
            if selected_session != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_session
                st.rerun()
        
        with col2:
            if st.button("🔄 새로고침"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ 세션 삭제", type="secondary"):
                if st.session_state.chatbot.delete_session(st.session_state.current_session_id):
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
                else:
                    st.error("세션 삭제에 실패했습니다.")
        
        # 채팅 히스토리 조회
        chat_history = st.session_state.chatbot.get_chat_history(
            st.session_state.current_session_id, 
            limit=100
        )
        
        if not chat_history:
            st.info("📝 선택한 세션에 채팅 기록이 없습니다.")
            return
        
        # 히스토리 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 메시지", len(chat_history))
        
        with col2:
            db_messages = len([msg for msg in chat_history if msg.get('search_source') == 'db'])
            st.metric("DB 검색", db_messages)
        
        with col3:
            history_messages = len([msg for msg in chat_history if msg.get('search_source') == 'history'])
            st.metric("히스토리 검색", history_messages)
        
        with col4:
            avg_relevance = sum([msg.get('relevance_score', 0) for msg in chat_history]) / len(chat_history)
            st.metric("평균 관련성", f"{avg_relevance:.3f}")
        
        st.markdown("---")
        
        # 채팅 히스토리 표시
        st.subheader(f"💬 세션: {st.session_state.current_session_id}")
        
        for i, entry in enumerate(reversed(chat_history), 1):
            with st.expander(f"메시지 {len(chat_history) - i + 1}: {entry['question'][:50]}...", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**질문**: {entry['question']}")
                    st.write(f"**답변**: {entry['answer']}")
                
                with col2:
                    st.write(f"**검색 소스**: {entry.get('search_source', 'unknown')}")
                    st.write(f"**관련성 점수**: {entry.get('relevance_score', 0):.3f}")
                    st.write(f"**문서 수**: {entry.get('documents_used', 0)}")
                    st.write(f"**시간**: {entry.get('timestamp', 'Unknown')}")
                
                # 유사한 질문이 있는 경우 표시
                if 'similar_questions' in entry and entry['similar_questions']:
                    st.write("**유사한 질문들**:")
                    for similar in entry['similar_questions'][:3]:
                        st.write(f"- {similar['question']} (유사도: {similar['similarity_score']:.3f})")
        
        # 유사한 질문 검색 기능
        st.markdown("---")
        st.subheader("🔍 유사한 질문 검색")
        
        search_query = st.text_input("검색할 질문을 입력하세요:")
        if search_query:
            similar_questions = st.session_state.chatbot.get_similar_questions(
                search_query, 
                st.session_state.current_session_id, 
                k=5
            )
            
            if similar_questions:
                st.write(f"**'{search_query}'와 유사한 질문들:**")
                for similar in similar_questions:
                    with st.expander(f"유사도: {similar['similarity_score']:.3f} - {similar['question'][:50]}..."):
                        st.write(f"**질문**: {similar['question']}")
                        st.write(f"**답변**: {similar['answer']}")
                        st.write(f"**시간**: {similar['timestamp']}")
            else:
                st.info("유사한 질문을 찾을 수 없습니다.")
    
    except Exception as e:
        st.error(f"❌ 채팅 히스토리 조회 실패: {str(e)}")


def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더 렌더링
    render_header()
    
    # 네비게이션 렌더링
    render_navigation()
    
    # 챗봇 자동 초기화
    if not st.session_state.system_initialized:
        with st.spinner("🤖 시스템을 초기화하는 중..."):
            if initialize_chatbot():
                st.rerun()
            else:
                st.error("❌ 시스템 초기화에 실패했습니다. 페이지를 새로고침해주세요.")
    else:
        # 메인 콘텐츠 렌더링
        render_main_content()


if __name__ == "__main__":
    main()
