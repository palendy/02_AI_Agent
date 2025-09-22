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
    
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 대화 기록 표시
        display_chat_history()
        
        # 입력 폼
        render_input_form()
        
        # 대화 제어 버튼들
        render_chat_controls()




def display_chat_history():
    """대화 기록 표시"""
    # 현재 세션의 채팅 히스토리 조회
    current_session_history = st.session_state.chatbot.get_chat_history(
        "default", 
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
        
        # 답변 품질 피드백 UI (답변이 있고 GitHub Issue 제안이 없는 경우만)
        if (entry.get('answer') and 
            not entry.get('github_issue_suggestion') and 
            not entry.get('error_message')):
            render_feedback_buttons(entry, index)
        
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
            
            # 답변 품질 점수 표시
            if 'answer_quality_score' in entry:
                quality_score = entry.get('answer_quality_score', 0.0)
                quality_color = "🟢" if quality_score >= 0.7 else "🟡" if quality_score >= 0.4 else "🔴"
                st.metric("답변 품질", f"{quality_color} {quality_score:.2f}")
            
            # 사용자 피드백 표시
            if 'user_feedback' in entry:
                feedback = entry.get('user_feedback')
                if feedback == 'satisfied':
                    st.success("✅ 사용자가 이 답변에 만족했습니다")
                elif feedback == 'dissatisfied':
                    st.warning("❌ 사용자가 이 답변에 불만족했습니다")
            
            if entry.get('error_message'):
                st.error(f"⚠️ 오류: {entry.get('error_message', '')}")
            
            # 유사한 질문이 있는 경우 표시
            if entry.get('similar_questions'):
                st.write("**🔍 유사한 질문들:**")
                for similar in entry.get('similar_questions', [])[:3]:
                    st.write(f"- {similar.get('question', '')} (유사도: {similar.get('similarity_score', 0):.3f})")
            
            # 이슈 검색 결과가 있는 경우 표시
            if entry.get('issue_search_performed') and entry.get('similar_issues'):
                render_issue_search_results(entry.get('similar_issues', []))
    
    # GitHub Issue 제안이 있는 경우 별도의 채팅 메시지로 표시
    if entry.get('github_issue_suggestion'):
        render_github_issue_chat_message(entry['github_issue_suggestion'], index)


def render_feedback_buttons(entry, index):
    """답변 품질 피드백 버튼 렌더링"""
    # 이미 피드백이 있는 경우 표시만
    if 'user_feedback' in entry:
        feedback = entry.get('user_feedback')
        if feedback == 'satisfied':
            st.success("✅ 이 답변에 만족합니다")
        elif feedback == 'dissatisfied':
            st.warning("❌ 이 답변에 불만족합니다")
        return
    
    # 피드백 버튼들
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("👍 만족", key=f"satisfied_{index}", help="이 답변이 도움이 되었습니다"):
            handle_feedback(entry, index, 'satisfied')
    
    with col2:
        if st.button("👎 불만족", key=f"dissatisfied_{index}", help="이 답변이 도움이 되지 않았습니다"):
            handle_feedback(entry, index, 'dissatisfied')
    
    with col3:
        st.caption("💡 만족스러운 답변은 향후 유사한 질문에 재사용됩니다")


def handle_feedback(entry, index, feedback_type):
    """피드백 처리"""
    try:
        # st.session_state.chat_history에서 해당 메시지 찾기 및 업데이트
        if index < len(st.session_state.chat_history):
            st.session_state.chat_history[index]['user_feedback'] = feedback_type
            
            # conversation_history에서도 업데이트
            for msg in st.session_state.chatbot.conversation_history:
                if (msg.get('question') == entry.get('question') and 
                    msg.get('answer') == entry.get('answer') and
                    msg.get('timestamp') == entry.get('timestamp')):
                    msg['user_feedback'] = feedback_type
                    break
            
            # 만족스러운 답변인 경우 채팅 히스토리에 저장
            if feedback_type == 'satisfied':
                save_to_chat_history(entry)
                st.success("✅ 답변이 채팅 히스토리에 저장되었습니다. 향후 유사한 질문에 재사용됩니다.")
            else:
                st.warning("❌ 답변이 채팅 히스토리에 저장되지 않았습니다.")
            
            # 페이지 새로고침
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ 피드백 처리 중 오류가 발생했습니다: {str(e)}")


def save_to_chat_history(entry):
    """만족스러운 답변을 채팅 히스토리에 저장"""
    try:
        if st.session_state.chatbot.chat_history_manager:
            # 답변 품질 점수가 높은 경우에만 저장 (0.5 이상)
            quality_score = entry.get('answer_quality_score', 0.0)
            if quality_score >= 0.5:
                st.session_state.chatbot.chat_history_manager.add_chat_message(
                    question=entry.get('question', ''),
                    answer=entry.get('answer', ''),
                    session_id=entry.get('session_id', "default"),
                    relevance_score=entry.get('relevance_score', 0.0),
                    search_source=entry.get('search_source', 'db'),
                    documents_used=entry.get('documents_used', 0)
                )
                st.info(f"💡 답변 품질 점수: {quality_score:.2f} - 채팅 히스토리에 저장되었습니다.")
            else:
                st.warning(f"⚠️ 답변 품질 점수가 낮습니다 ({quality_score:.2f}). 채팅 히스토리에 저장되지 않았습니다.")
    except Exception as e:
        st.error(f"❌ 채팅 히스토리 저장 실패: {str(e)}")


def render_issue_search_results(similar_issues):
    """이슈 검색 결과 렌더링"""
    if not similar_issues:
        return
    
    st.markdown("---")
    st.markdown("### 🔍 GitHub Issue 검색 결과")
    
    # 이슈 상태별로 분류
    closed_issues = [issue for issue in similar_issues if issue.get('state') == 'closed']
    open_issues = [issue for issue in similar_issues if issue.get('state') == 'open']
    
    # Closed 이슈 표시
    if closed_issues:
        st.markdown("#### ✅ 해결된 이슈들")
        for issue in closed_issues[:3]:  # 최대 3개
            with st.expander(f"#{issue.get('number')} - {issue.get('title')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**상태:** {issue.get('state', 'unknown')}")
                    st.markdown(f"**최종 스코어:** {issue.get('final_score', issue.get('similarity_score', 0)):.3f}")
                    
                    # Hybrid Search + Cross-Encoder 스코어 상세 정보 표시
                    if 'bm25_score' in issue and 'dense_score' in issue and 'cross_encoder_score' in issue:
                        with st.expander("📊 상세 스코어", expanded=False):
                            col_bm25, col_dense, col_cross = st.columns(3)
                            with col_bm25:
                                st.metric("BM25", f"{issue.get('bm25_score', 0):.3f}")
                            with col_dense:
                                st.metric("Dense Embedding", f"{issue.get('dense_score', 0):.3f}")
                            with col_cross:
                                st.metric("Cross-Encoder", f"{issue.get('cross_encoder_score', 0):.3f}")
                    
                    if issue.get('labels'):
                        st.markdown(f"**라벨:** {', '.join(issue.get('labels', []))}")
                
                with col2:
                    st.markdown(f"[🔗 이슈 보기]({issue.get('url', '#')})")
                
                # 답변이 있는 경우 표시
                if issue.get('answer'):
                    st.markdown("**해결 방법:**")
                    st.markdown(issue.get('answer', ''))
    
    # Open 이슈 표시
    if open_issues:
        st.markdown("#### 🔄 진행 중인 이슈들")
        for issue in open_issues[:3]:  # 최대 3개
            with st.expander(f"#{issue.get('number')} - {issue.get('title')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**상태:** {issue.get('state', 'unknown')}")
                    st.markdown(f"**최종 스코어:** {issue.get('final_score', issue.get('similarity_score', 0)):.3f}")
                    
                    # Hybrid Search + Cross-Encoder 스코어 상세 정보 표시
                    if 'bm25_score' in issue and 'dense_score' in issue and 'cross_encoder_score' in issue:
                        with st.expander("📊 상세 스코어", expanded=False):
                            col_bm25, col_dense, col_cross = st.columns(3)
                            with col_bm25:
                                st.metric("BM25", f"{issue.get('bm25_score', 0):.3f}")
                            with col_dense:
                                st.metric("Dense Embedding", f"{issue.get('dense_score', 0):.3f}")
                            with col_cross:
                                st.metric("Cross-Encoder", f"{issue.get('cross_encoder_score', 0):.3f}")
                    
                    if issue.get('labels'):
                        st.markdown(f"**라벨:** {', '.join(issue.get('labels', []))}")
                
                with col2:
                    st.markdown(f"[🔗 이슈 보기]({issue.get('url', '#')})")
                
                # 이슈 본문 일부 표시
                if issue.get('body'):
                    st.markdown("**내용:**")
                    st.markdown(issue.get('body', '')[:200] + "..." if len(issue.get('body', '')) > 200 else issue.get('body', ''))
    
    # 전체 이슈 목록 링크
    if similar_issues:
        st.markdown("---")
        st.markdown(f"**전체 {len(similar_issues)}개의 유사한 이슈가 발견되었습니다.**")
        st.markdown("더 많은 이슈를 확인하려면 [GitHub Issues 페이지](https://github.com/palendy/02_AI_Agent/issues)를 방문해보세요.")


def render_github_issue_suggestion(issue_suggestion):
    """GitHub Issue 제안 렌더링"""
    if not issue_suggestion.get('suggested', False):
        st.warning(f"⚠️ {issue_suggestion.get('message', 'Issue 제안을 생성할 수 없습니다.')}")
        return
    
    st.markdown("---")
    st.markdown("### 🐛 GitHub Issue 제안")
    
    # Issue 정보 표시
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Repository:** `{issue_suggestion.get('repository', 'Unknown')}`")
        st.markdown(f"**제목:** {issue_suggestion.get('title', 'N/A')}")
    
    with col2:
        # Issue 생성 버튼
        issue_url = issue_suggestion.get('url', '')
        if issue_url:
            st.markdown(f"[🔗 Issue 생성하기]({issue_url})")
    
    # Issue 내용 미리보기
    with st.expander("📝 Issue 내용 미리보기"):
        st.markdown(issue_suggestion.get('body', '내용이 없습니다.'))
    
    # 추가 정보
    st.info(f"💡 {issue_suggestion.get('message', '')}")


def render_github_issue_chat_message(issue_suggestion, message_index):
    """GitHub Issue 제안을 채팅 메시지로 렌더링"""
    if not issue_suggestion.get('suggested', False):
        with st.chat_message("assistant"):
            st.warning(f"⚠️ {issue_suggestion.get('message', 'Issue 제안을 생성할 수 없습니다.')}")
        return
    
    with st.chat_message("assistant"):
        st.markdown("### 🐛 GitHub Issue 제안")
        st.markdown("답변을 찾지 못해 GitHub Issue를 생성하는 것을 제안드립니다.")
        
        # Issue 정보 표시
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Repository:** `{issue_suggestion.get('repository', 'Unknown')}`")
            st.markdown(f"**제목:** {issue_suggestion.get('title', 'N/A')}")
        
        with col2:
            issue_url = issue_suggestion.get('url', '')
            if issue_url:
                st.markdown(f"[🔗 Issue 생성하기]({issue_url})")
        
        # Issue 내용 편집 가능한 폼
        with st.form(key=f"github_issue_form_{message_index}"):
            st.markdown("**Issue 내용을 확인하고 수정하세요:**")
            
            # 제목 편집
            edited_title = st.text_input(
                "제목:",
                value=issue_suggestion.get('title', ''),
                key=f"issue_title_{message_index}"
            )
            
            # 본문 편집
            edited_body = st.text_area(
                "본문:",
                value=issue_suggestion.get('body', ''),
                height=300,
                key=f"issue_body_{message_index}"
            )
            
            # 라벨 선택
            labels = st.multiselect(
                "라벨:",
                options=["bug", "question-answer-failure", "auto-generated", "enhancement", "documentation"],
                default=["bug", "question-answer-failure", "auto-generated"],
                key=f"issue_labels_{message_index}"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.form_submit_button("✅ Issue 생성", type="primary"):
                    create_github_issue(issue_suggestion, edited_title, edited_body, labels)
            
            with col2:
                if st.form_submit_button("📝 미리보기"):
                    preview_github_issue(edited_title, edited_body, labels)
            
            with col3:
                if st.form_submit_button("❌ 취소"):
                    st.rerun()


def create_github_issue(issue_suggestion, title, body, labels):
    """GitHub Issue 생성"""
    try:
        # GitHub Issue URL 생성
        repository = issue_suggestion.get('repository', '')
        if not repository:
            st.error("Repository 정보가 없습니다.")
            return
        
        # 라벨을 URL 인코딩
        labels_str = ",".join(labels) if labels else ""
        
        # 제목과 본문을 URL 인코딩
        import urllib.parse
        encoded_title = urllib.parse.quote(title)
        encoded_body = urllib.parse.quote(body)
        encoded_labels = urllib.parse.quote(labels_str)
        
        # GitHub Issue URL 생성
        issue_url = f"https://github.com/{repository}/issues/new?title={encoded_title}&body={encoded_body}&labels={encoded_labels}"
        
        # 성공 메시지와 함께 URL 표시
        st.success("✅ GitHub Issue가 생성되었습니다!")
        st.markdown(f"[🔗 생성된 Issue 보기]({issue_url})")
        
        # 브라우저에서 열기
        st.markdown(f"""
        <script>
        window.open('{issue_url}', '_blank');
        </script>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ GitHub Issue 생성 중 오류가 발생했습니다: {str(e)}")


def preview_github_issue(title, body, labels):
    """GitHub Issue 미리보기"""
    st.markdown("### 📝 Issue 미리보기")
    st.markdown("---")
    st.markdown(f"**제목:** {title}")
    st.markdown(f"**라벨:** {', '.join(labels) if labels else '없음'}")
    st.markdown("**본문:**")
    st.markdown(body)


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
            # 챗봇에 질문 전달
            result = st.session_state.chatbot.chat(
                user_input, 
                "default"
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
                'github_issue_suggestion': result.get('github_issue_suggestion', None),
                'answer_quality_score': result.get('answer_quality_score', 0.0),
                'user_feedback': None,  # 사용자 피드백 (초기값: None)
                'session_id': "default",
                'similar_issues': result.get('similar_issues', []),
                'issue_search_performed': result.get('issue_search_performed', False)
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
        pass  # 빈 컬럼
    
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
