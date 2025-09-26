"""
Sidebar Component
사이드바 컴포넌트 (설정, 시스템 정보 등)
"""

import streamlit as st
from config import get_config


def render_sidebar():
    """사이드바 렌더링"""
    st.header("⚙️ 설정 및 시스템 정보")
    
    # 설정 섹션
    render_settings_section()
    
    st.markdown("---")
    
    # 시스템 정보 섹션
    render_system_info_section()
    
    st.markdown("---")
    
    # 도움말 섹션
    render_help_section()


def render_settings_section():
    """설정 섹션 렌더링"""
    st.subheader("🔧 설정")
    
    try:
        config = get_config()
        
        # 기본 설정 표시
        with st.expander("📋 현재 설정", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**모델**: {config.default_model_name}")
                st.write(f"**임베딩**: {config.embedding_model}")
                st.write(f"**최대 재시도**: {config.max_retries}")
                st.write(f"**관련성 임계값**: {config.relevance_threshold}")
            
            with col2:
                st.write(f"**최대 검색 결과**: {config.max_search_results}")
                st.write(f"**청크 크기**: {config.chunk_size}")
                st.write(f"**청크 오버랩**: {config.chunk_overlap}")
                st.write(f"**최대 파일 크기**: {config.max_file_size // (1024*1024)}MB")
        
        # GitHub 설정
        with st.expander("🐙 GitHub 설정"):
            github_repos = config.github_repositories
            if github_repos:
                st.write("**설정된 Repository:**")
                for i, repo in enumerate(github_repos, 1):
                    st.write(f"{i}. {repo}")
            else:
                st.warning("설정된 Repository가 없습니다.")
            
            if config.github_token:
                st.success("✅ GitHub 토큰이 설정되어 있습니다.")
            else:
                st.warning("⚠️ GitHub 토큰이 설정되지 않았습니다.")
        
        # API 키 상태
        with st.expander("🔑 API 키 상태"):
            api_status = check_api_keys(config)
            
            for key, status in api_status.items():
                if status:
                    st.success(f"✅ {key}")
                else:
                    st.error(f"❌ {key}")
    
    except Exception as e:
        st.error(f"❌ 설정 조회 실패: {str(e)}")


def check_api_keys(config):
    """API 키 상태 확인"""
    api_status = {}
    
    try:
        _ = config.openai_api_key
        api_status["OpenAI API Key"] = True
    except:
        api_status["OpenAI API Key"] = False
    
    
    api_status["GitHub Token"] = bool(config.github_token)
    
    return api_status


def render_system_info_section():
    """시스템 정보 섹션 렌더링"""
    st.subheader("📊 시스템 정보")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ 시스템이 초기화되지 않았습니다.")
        return
    
    try:
        info = st.session_state.chatbot.get_system_info()
        
        # 벡터 스토어 정보
        vector_info = info.get('vector_store', {})
        st.metric("📚 문서 수", vector_info.get('document_count', 0))
        st.metric("🗄️ 컬렉션", vector_info.get('collection_name', 'Unknown'))
        
        # 워크플로우 정보
        workflow_info = info.get('workflow', {})
        st.metric("🤖 모델", workflow_info.get('model_name', 'Unknown'))
        st.metric("🔄 최대 재시도", workflow_info.get('max_retries', 0))
        
        # 대화 통계
        st.metric("💬 대화 수", len(st.session_state.chat_history))
        st.metric("📁 Repository", len(st.session_state.repository_urls))
        
        # 초기화 상태
        if info.get('initialized'):
            st.success("✅ 시스템 정상 작동")
        else:
            st.error("❌ 시스템 오류")
    
    except Exception as e:
        st.error(f"❌ 시스템 정보 조회 실패: {str(e)}")


def render_help_section():
    """도움말 섹션 렌더링"""
    st.subheader("❓ 도움말")
    
    with st.expander("🚀 시작하기"):
        st.markdown("""
        1. **시스템 초기화**: 메인 페이지에서 '시스템 초기화' 버튼 클릭
        2. **Repository 추가**: Repository 관리 페이지에서 GitHub 저장소 추가
        3. **대화 시작**: 채팅 페이지에서 질문 입력
        """)
    
    with st.expander("💡 사용 팁"):
        st.markdown("""
        - **구체적인 질문**: 더 정확한 답변을 위해 구체적으로 질문하세요
        - **키워드 사용**: 관련 키워드를 포함하여 질문하세요
        - **다양한 질문**: 시스템의 다양한 기능을 테스트해보세요
        """)
    
    with st.expander("🔧 문제 해결"):
        st.markdown("""
        - **API 키 오류**: .env 파일에 올바른 API 키가 설정되어 있는지 확인
        - **문서 없음**: Repository 관리에서 문서를 추가했는지 확인
        - **느린 응답**: 네트워크 상태와 API 사용량을 확인
        """)
    
    with st.expander("📚 기능 설명"):
        st.markdown("""
        - **Corrective RAG**: 문서 관련성을 자동 평가하고 쿼리를 재작성
        - **LangGraph**: 복잡한 AI 워크플로우를 관리
        - **벡터 검색**: ChromaDB를 사용한 효율적인 문서 검색
        - **채팅 히스토리**: 이전 대화를 벡터화하여 재사용
        """)


def render_quick_actions():
    """빠른 작업 버튼들"""
    st.subheader("⚡ 빠른 작업")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 시스템 새로고침", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
            st.session_state.chat_history = []
            st.success("✅ 대화 기록이 삭제되었습니다.")
            st.rerun()
    
    with col2:
        if st.button("📊 통계 보기", use_container_width=True):
            show_detailed_statistics()
        
        if st.button("📥 로그 내보내기", use_container_width=True):
            export_system_logs()


def show_detailed_statistics():
    """상세 통계 표시"""
    if not st.session_state.chat_history:
        st.warning("⚠️ 통계를 표시할 대화 기록이 없습니다.")
        return
    
    # 통계 계산
    total_chats = len(st.session_state.chat_history)
    
    # 검색 소스별 통계
    source_counts = {}
    total_relevance = 0
    total_retries = 0
    total_documents = 0
    
    for entry in st.session_state.chat_history:
        source = entry['search_source']
        source_counts[source] = source_counts.get(source, 0) + 1
        
        total_relevance += entry['relevance_score']
        total_retries += entry['retry_count']
        total_documents += entry['documents_used']
    
    # 평균 계산
    avg_relevance = total_relevance / total_chats if total_chats > 0 else 0
    avg_retries = total_retries / total_chats if total_chats > 0 else 0
    avg_documents = total_documents / total_chats if total_chats > 0 else 0
    
    # 통계 표시
    st.subheader("📊 상세 통계")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 대화 수", total_chats)
    
    with col2:
        st.metric("평균 관련성", f"{avg_relevance:.3f}")
    
    with col3:
        st.metric("평균 재시도", f"{avg_retries:.1f}")
    
    with col4:
        st.metric("평균 문서 수", f"{avg_documents:.1f}")
    
    # 검색 소스별 차트
    if source_counts:
        st.subheader("🔍 검색 소스별 분포")
        
        import pandas as pd
        df = pd.DataFrame(list(source_counts.items()), columns=['소스', '횟수'])
        st.bar_chart(df.set_index('소스'))


def export_system_logs():
    """시스템 로그 내보내기"""
    try:
        # 시스템 정보 수집
        logs = []
        logs.append("AI Agent Chatbot 시스템 로그")
        logs.append("=" * 50)
        logs.append("")
        
        # 설정 정보
        config = get_config()
        logs.append("설정 정보:")
        logs.append(f"- 모델: {config.default_model_name}")
        logs.append(f"- 임베딩: {config.embedding_model}")
        logs.append(f"- 최대 재시도: {config.max_retries}")
        logs.append(f"- 관련성 임계값: {config.relevance_threshold}")
        logs.append("")
        
        # 대화 기록
        if st.session_state.chat_history:
            logs.append("대화 기록:")
            for i, entry in enumerate(st.session_state.chat_history, 1):
                logs.append(f"[{i}] {entry['timestamp']}")
                logs.append(f"질문: {entry['question']}")
                logs.append(f"답변: {entry['answer'][:100]}...")
                logs.append(f"검색 소스: {entry['search_source']}")
                logs.append(f"관련성: {entry['relevance_score']:.3f}")
                logs.append("-" * 30)
        
        # 로그 파일 생성
        log_text = "\n".join(logs)
        
        # 다운로드 버튼
        st.download_button(
            label="📥 시스템 로그 다운로드",
            data=log_text,
            file_name=f"system_log_{st.session_state.get('session_id', 'unknown')}.txt",
            mime="text/plain"
        )
    
    except Exception as e:
        st.error(f"❌ 로그 내보내기 실패: {str(e)}")
