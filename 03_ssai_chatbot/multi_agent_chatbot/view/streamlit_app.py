"""
Streamlit Web Interface for Multi-Agent Chatbot
Langchain 기반 3-Agent 시스템 웹 인터페이스
"""

import streamlit as st
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from core.multi_agent_workflow import MultiAgentWorkflow

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="Multi-Agent Chatbot",
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
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .message-user {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    .message-agent {
        background: #f1f8e9;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    .message-rag {
        border-left: 4px solid #4caf50;
    }
    .message-issue {
        border-left: 4px solid #ff9800;
    }
    .message-chat {
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False


async def initialize_workflow():
    """워크플로우 초기화"""
    try:
        config = get_config()
        if not config.validate():
            st.error("❌ 설정 검증에 실패했습니다. .env 파일을 확인하세요.")
            return False
        
        workflow = MultiAgentWorkflow(config.get_agent_config())
        
        if await workflow.initialize():
            st.session_state.workflow = workflow
            st.session_state.initialized = True
            return True
        else:
            st.error("❌ 워크플로우 초기화에 실패했습니다.")
            return False
            
    except Exception as e:
        st.error(f"❌ 초기화 중 오류가 발생했습니다: {e}")
        return False


def display_agent_status():
    """Agent 상태 표시"""
    if not st.session_state.initialized or not st.session_state.workflow:
        st.warning("⚠️ 시스템이 초기화되지 않았습니다.")
        return
    
    status = st.session_state.workflow.get_workflow_status()
    
    st.subheader("🤖 Agent 상태")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chat_status = status['agents']['chat_agent']
        st.markdown(f"""
        <div class="agent-card">
            <h4>💬 Chat Agent</h4>
            <p>상태: <span class="{'status-success' if chat_status['is_initialized'] else 'status-error'}">
                {'✅ 정상' if chat_status['is_initialized'] else '❌ 오류'}
            </span></p>
            <p>대화 수: {chat_status['conversation_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rag_status = status['agents']['rag_agent']
        st.markdown(f"""
        <div class="agent-card">
            <h4>📚 RAG Agent</h4>
            <p>상태: <span class="{'status-success' if rag_status['is_initialized'] else 'status-error'}">
                {'✅ 정상' if rag_status['is_initialized'] else '❌ 오류'}
            </span></p>
            <p>검색 수: {rag_status['conversation_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        issue_status = status['agents']['issue_agent']
        st.markdown(f"""
        <div class="agent-card">
            <h4>🔧 Issue Agent</h4>
            <p>상태: <span class="{'status-success' if issue_status['is_initialized'] else 'status-error'}">
                {'✅ 정상' if issue_status['is_initialized'] else '❌ 오류'}
            </span></p>
            <p>이슈 수: {issue_status['conversation_count']}</p>
        </div>
        """, unsafe_allow_html=True)


def display_message(message: Dict[str, Any], is_user: bool = False):
    """메시지 표시"""
    if is_user:
        st.markdown(f"""
        <div class="message-user">
            <strong>👤 사용자:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        agent_type = message.get('agent_type', 'unknown')
        agent_emoji = {
            'rag': '📚',
            'issue': '🔧',
            'chat': '💬'
        }.get(agent_type, '🤖')
        
        agent_name = {
            'rag': 'RAG Agent',
            'issue': 'Issue Agent',
            'chat': 'Chat Agent'
        }.get(agent_type, 'Unknown Agent')
        
        css_class = f"message-agent message-{agent_type}"
        
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{agent_emoji} {agent_name}:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # 추가 정보 표시
        if 'metadata' in message:
            metadata = message['metadata']
            with st.expander("📊 상세 정보"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("처리 시간", f"{metadata.get('processing_time', 0):.2f}초")
                
                with col2:
                    if 'confidence' in metadata:
                        st.metric("신뢰도", f"{metadata['confidence']:.3f}")
                
                with col3:
                    if 'documents_used' in metadata:
                        st.metric("사용된 문서", metadata['documents_used'])


async def process_user_message(message: str) -> Dict[str, Any]:
    """사용자 메시지 처리"""
    if not st.session_state.initialized or not st.session_state.workflow:
        return {
            "success": False,
            "error": "시스템이 초기화되지 않았습니다."
        }
    
    try:
        result = await st.session_state.workflow.process_message(message)
        return result
    except Exception as e:
        logger.error(f"메시지 처리 중 오류: {e}")
        return {
            "success": False,
            "error": f"처리 중 오류가 발생했습니다: {e}"
        }


def main():
    """메인 함수"""
    # 헤더
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Chatbot System</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 시스템 제어")
        
        # 초기화 버튼
        if st.button("🚀 시스템 초기화", type="primary"):
            with st.spinner("시스템을 초기화하는 중..."):
                success = asyncio.run(initialize_workflow())
                if success:
                    st.success("✅ 시스템 초기화 완료!")
                else:
                    st.error("❌ 시스템 초기화 실패")
        
        # 상태 표시
        if st.session_state.initialized:
            display_agent_status()
        
        # 대화 기록 초기화
        if st.button("🗑️ 대화 기록 초기화"):
            if st.session_state.workflow:
                asyncio.run(st.session_state.workflow.reset())
                st.session_state.messages = []
                st.success("✅ 대화 기록이 초기화되었습니다.")
        
        # 통계 정보
        if st.session_state.initialized and st.session_state.workflow:
            st.subheader("📊 통계 정보")
            
            chat_stats = st.session_state.workflow.chat_agent.get_classification_stats()
            rag_stats = st.session_state.workflow.rag_agent.get_search_stats()
            issue_stats = st.session_state.workflow.issue_agent.get_issue_stats()
            
            st.metric("총 메시지", len(st.session_state.messages))
            st.metric("Chat Agent 분류", chat_stats['total_messages'])
            st.metric("RAG Agent 검색", rag_stats['total_searches'])
            st.metric("Issue Agent 처리", issue_stats['total_issues'])
    
    # 메인 콘텐츠
    if not st.session_state.initialized:
        st.warning("⚠️ 시스템을 먼저 초기화해주세요.")
        st.info("💡 사이드바의 '시스템 초기화' 버튼을 클릭하세요.")
        return
    
    # 채팅 인터페이스
    st.subheader("💬 대화하기")
    
    # 메시지 입력
    user_input = st.text_input(
        "메시지를 입력하세요:",
        placeholder="예: GitHub에서 문서를 추출하는 방법을 알려주세요",
        key="user_input"
    )
    
    # 전송 버튼
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("📤 전송", type="primary"):
            if user_input.strip():
                # 사용자 메시지 추가
                st.session_state.messages.append({
                    "content": user_input,
                    "is_user": True,
                    "timestamp": datetime.now()
                })
                
                # 메시지 처리
                with st.spinner("처리 중..."):
                    result = asyncio.run(process_user_message(user_input))
                
                if result['success']:
                    # Agent 응답 추가
                    agent_type = result['processing_agent']
                    st.session_state.messages.append({
                        "content": result['answer'],
                        "is_user": False,
                        "agent_type": agent_type,
                        "timestamp": datetime.now(),
                        "metadata": {
                            "processing_time": result['processing_time'],
                            "confidence": result.get('confidence', 0.0),
                            "message_type": result.get('message_type', 'UNKNOWN'),
                            "documents_used": result.get('rag_info', {}).get('documents_used', 0),
                            "similar_issues": result.get('issue_info', {}).get('similar_issues', 0)
                        }
                    })
                else:
                    # 오류 메시지 추가
                    st.session_state.messages.append({
                        "content": f"❌ 오류: {result.get('error', '알 수 없는 오류')}",
                        "is_user": False,
                        "agent_type": "error",
                        "timestamp": datetime.now()
                    })
                
                # 입력 필드 초기화
                st.session_state.user_input = ""
                st.rerun()
    
    with col2:
        if st.button("🗑️ 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    # 메시지 표시
    st.subheader("📝 대화 기록")
    
    if not st.session_state.messages:
        st.info("💬 대화를 시작해보세요!")
    else:
        for message in st.session_state.messages:
            display_message(message, message.get('is_user', False))
    
    # 하단 정보
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🤖 Multi-Agent Chatbot System | 
        💬 Chat Agent | 📚 RAG Agent | 🔧 Issue Agent |
        🚀 Powered by Langchain + LangGraph
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
