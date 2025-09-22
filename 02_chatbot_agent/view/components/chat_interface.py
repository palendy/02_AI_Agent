"""
Chat Interface Component
챗봇 대화 인터페이스 컴포넌트
"""

import streamlit as st
import time
from datetime import datetime


def render_chat_interface():
    """챗봇 대화 인터페이스 렌더링"""
    st.header("💬 AI Chatbot과 대화하기")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ 시스템을 먼저 초기화해주세요.")
        return
    
    # 서비스 선택 확인
    if not st.session_state.chatbot.get_current_repository():
        st.warning("⚠️ 문의할 서비스를 먼저 선택해주세요.")
        return
    
    # 세션 선택
    render_session_selector()
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 대화 기록 표시
        display_chat_history()
        
        # 입력 폼
        render_input_form()
        
        # 대화 제어 버튼들
        render_chat_controls()


def render_session_selector():
    """세션 선택기 렌더링"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # 사용 가능한 세션 목록 조회
        available_sessions = st.session_state.chatbot.get_all_sessions()
        if not available_sessions:
            available_sessions = ["default"]
        
        selected_session = st.selectbox(
            "세션 선택",
            available_sessions,
            index=available_sessions.index(st.session_state.current_session_id) if st.session_state.current_session_id in available_sessions else 0,
            help="다른 세션을 선택하면 해당 세션의 채팅 히스토리를 볼 수 있습니다."
        )
        
        if selected_session != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session
            st.rerun()
    
    with col2:
        if st.button("🆕 새 세션", help="새로운 세션을 시작합니다."):
            import uuid
            new_session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.current_session_id = new_session_id
            st.rerun()
    
    with col3:
        if st.button("🔄 새로고침", help="세션 목록을 새로고침합니다."):
            st.rerun()


def display_chat_history():
    """대화 기록 표시"""
    # 현재 세션의 채팅 히스토리 조회
    current_session_history = st.session_state.chatbot.get_chat_history(
        st.session_state.current_session_id, 
        limit=50
    )
    
    if not current_session_history:
        st.info("👋 안녕하세요! AI Chatbot입니다. 무엇을 도와드릴까요?")
        return
    
    # 대화 기록을 역순으로 표시 (최신이 아래)
    for i, entry in enumerate(current_session_history):
        render_chat_message(entry, i)


def render_chat_message(entry, index):
    """개별 채팅 메시지 렌더링"""
    # 사용자 메시지
    with st.chat_message("user"):
        st.write(entry.get('question', ''))
        st.caption(f"⏰ {entry.get('timestamp', 'Unknown')}")
    
    # 챗봇 메시지
    with st.chat_message("assistant"):
        st.write(entry.get('answer', ''))
        
        # 추가 정보 (접을 수 있는 섹션)
        with st.expander("🔍 상세 정보"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("검색 소스", entry.get('search_source', 'unknown'))
            
            with col2:
                st.metric("관련성 점수", f"{entry.get('relevance_score', 0):.3f}")
            
            with col3:
                st.metric("재시도 횟수", entry.get('retry_count', 0))
            
            with col4:
                st.metric("사용된 문서", f"{entry.get('documents_used', 0)}개")
            
            if entry.get('error_message'):
                st.error(f"⚠️ 오류: {entry.get('error_message', '')}")
            
            # GitHub Issue 제안이 있는 경우 표시
            if entry.get('github_issue_suggestion'):
                render_github_issue_suggestion(entry['github_issue_suggestion'])
            
            # 유사한 질문이 있는 경우 표시
            if entry.get('similar_questions'):
                st.write("**🔍 유사한 질문들:**")
                for similar in entry.get('similar_questions', [])[:3]:
                    st.write(f"- {similar.get('question', '')} (유사도: {similar.get('similarity_score', 0):.3f})")


def render_input_form():
    """입력 폼 렌더링"""
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "질문을 입력하세요:",
                placeholder="예: GitHub에서 문서를 추출하는 방법은?",
                key="user_input"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "전송",
                use_container_width=True,
                type="primary"
            )
    
    # 전송 버튼 클릭 시 처리
    if submit_button and user_input:
        process_user_input(user_input)


def process_user_input(user_input):
    """사용자 입력 처리"""
    if not user_input.strip():
        st.warning("⚠️ 질문을 입력해주세요.")
        return
    
    try:
        # 로딩 표시
        with st.spinner("🤔 생각 중..."):
            # 챗봇에 질문 전달 (세션 ID 포함)
            result = st.session_state.chatbot.chat(
                user_input, 
                st.session_state.current_session_id
            )
            
            # 결과를 대화 기록에 추가
            chat_entry = {
                'question': user_input,
                'answer': result['answer'],
                'search_source': result['search_source'],
                'relevance_score': result['relevance_score'],
                'retry_count': result['retry_count'],
                'documents_used': result['documents_used'],
                'timestamp': result['timestamp'],
                'error_message': result.get('error_message', ''),
                'similar_questions': result.get('similar_questions', []),
                'github_issue_suggestion': result.get('github_issue_suggestion', None)
            }
            
            st.session_state.chat_history.append(chat_entry)
            
            # 페이지 새로고침
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ 질문 처리 중 오류가 발생했습니다: {str(e)}")


def render_chat_controls():
    """채팅 제어 버튼들"""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ 현재 세션 삭제", use_container_width=True):
            if st.session_state.chatbot.delete_session(st.session_state.current_session_id):
                st.success("✅ 현재 세션이 삭제되었습니다.")
                st.rerun()
            else:
                st.error("❌ 세션 삭제에 실패했습니다.")
    
    with col2:
        if st.button("📥 대화 내보내기", use_container_width=True):
            export_chat_history()
    
    with col3:
        if st.button("📊 대화 통계", use_container_width=True):
            show_chat_statistics()
    
    with col4:
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()


def export_chat_history():
    """대화 기록 내보내기"""
    if not st.session_state.chat_history:
        st.warning("⚠️ 내보낼 대화 기록이 없습니다.")
        return
    
    try:
        # 대화 기록을 텍스트로 변환
        chat_text = "AI Agent Chatbot 대화 기록\n"
        chat_text += "=" * 50 + "\n\n"
        
        for i, entry in enumerate(st.session_state.chat_history, 1):
            chat_text += f"[{i}] {entry['timestamp']}\n"
            chat_text += f"사용자: {entry['question']}\n"
            chat_text += f"챗봇: {entry['answer']}\n"
            chat_text += f"검색 소스: {entry['search_source']}\n"
            chat_text += f"관련성 점수: {entry['relevance_score']:.3f}\n"
            chat_text += "-" * 30 + "\n\n"
        
        # 파일 다운로드 버튼
        st.download_button(
            label="📥 대화 기록 다운로드",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    except Exception as e:
        st.error(f"❌ 대화 기록 내보내기 실패: {str(e)}")


def show_chat_statistics():
    """대화 통계 표시"""
    if not st.session_state.chat_history:
        st.warning("⚠️ 통계를 표시할 대화 기록이 없습니다.")
        return
    
    try:
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
        st.subheader("📊 대화 통계")
        
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
        
        # 최근 대화 요약
        st.subheader("📝 최근 대화 요약")
        recent_chats = st.session_state.chat_history[-5:]
        
        for i, entry in enumerate(recent_chats, 1):
            with st.expander(f"대화 {i}: {entry['question'][:30]}..."):
                st.write(f"**질문**: {entry['question']}")
                st.write(f"**답변**: {entry['answer'][:200]}...")
                st.write(f"**검색 소스**: {entry['search_source']}")
                st.write(f"**관련성**: {entry['relevance_score']:.3f}")
    
    except Exception as e:
        st.error(f"❌ 통계 계산 실패: {str(e)}")


def render_github_issue_suggestion(suggestion):
    """GitHub Issue 제안 렌더링"""
    if not suggestion or not suggestion.get('suggested'):
        return
    
    st.markdown("---")
    st.markdown("### 🐛 GitHub Issue 제안")
    
    # 제안 메시지
    st.info(suggestion.get('message', '질문에 답변하지 못했습니다.'))
    
    # Issue 정보 표시
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Repository**: {suggestion.get('repository', 'Unknown')}")
        st.write(f"**제목**: {suggestion.get('title', 'Unknown')}")
        
        # Issue 내용 미리보기
        with st.expander("📝 Issue 내용 미리보기"):
            st.text(suggestion.get('body', ''))
    
    with col2:
        # GitHub Issue 생성 버튼
        issue_url = suggestion.get('url', '')
        if issue_url:
            st.markdown(f"[🔗 GitHub Issue 생성하기]({issue_url})")
        
        # Issue 정보 복사 버튼
        if st.button("📋 정보 복사", help="Issue 정보를 클립보드에 복사합니다."):
            issue_info = f"""
**제목**: {suggestion.get('title', '')}
**Repository**: {suggestion.get('repository', '')}
**URL**: {issue_url}

**내용**:
{suggestion.get('body', '')}
            """
            st.code(issue_info, language="text")
            st.success("✅ Issue 정보가 표시되었습니다. 위 내용을 복사하여 사용하세요.")


def render_quick_questions():
    """빠른 질문 버튼들"""
    st.subheader("🚀 빠른 질문")
    
    quick_questions = [
        "안녕하세요!",
        "이 시스템은 어떻게 작동하나요?",
        "GitHub에서 문서를 추출하는 방법은?",
        "Corrective RAG란 무엇인가요?",
        "LangGraph 워크플로우에 대해 설명해주세요."
    ]
    
    cols = st.columns(len(quick_questions))
    
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                process_user_input(question)
