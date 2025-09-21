"""
Repository Manager Component
Repository 관리 컴포넌트
"""

import streamlit as st
from datetime import datetime


def render_repository_manager():
    """Repository 관리 인터페이스 렌더링"""
    st.header("📁 Repository 관리")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ 시스템을 먼저 초기화해주세요.")
        return
    
    # Repository 추가 섹션
    render_add_repository_section()
    
    st.markdown("---")
    
    # Repository 목록 섹션
    render_repository_list_section()
    
    st.markdown("---")
    
    # Repository 통계 섹션
    render_repository_statistics_section()


def render_add_repository_section():
    """Repository 추가 섹션 렌더링"""
    st.subheader("➕ Repository 추가")
    
    with st.form(key="add_repository_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            repo_url = st.text_input(
                "GitHub Repository URL:",
                placeholder="https://github.com/owner/repository",
                help="추가할 GitHub repository의 URL을 입력하세요."
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "추가",
                use_container_width=True,
                type="primary"
            )
    
    # Repository 추가 처리
    if submit_button and repo_url:
        add_repository(repo_url)
    
    # 빠른 추가 버튼들
    st.markdown("**🚀 빠른 추가:**")
    quick_repos = [
        "https://github.com/microsoft/vscode",
        "https://github.com/facebook/react",
        "https://github.com/torvalds/linux"
    ]
    
    cols = st.columns(len(quick_repos))
    for i, repo in enumerate(quick_repos):
        with cols[i]:
            if st.button(f"추가 {i+1}", key=f"quick_add_{i}", use_container_width=True):
                add_repository(repo)


def add_repository(repo_url):
    """Repository 추가"""
    try:
        with st.spinner(f"📁 Repository 추가 중: {repo_url}"):
            result = st.session_state.chatbot.add_github_repository(repo_url)
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                st.info(f"📄 {result['documents_count']}개 문서가 추가되었습니다.")
                
                # Repository URL을 세션 상태에 추가
                if repo_url not in st.session_state.repository_urls:
                    st.session_state.repository_urls.append(repo_url)
                
                # 페이지 새로고침
                st.rerun()
            else:
                st.error(f"❌ {result['message']}")
    
    except Exception as e:
        st.error(f"❌ Repository 추가 실패: {str(e)}")


def render_repository_list_section():
    """Repository 목록 섹션 렌더링"""
    st.subheader("📋 Repository 목록")
    
    if not st.session_state.repository_urls:
        st.info("📝 추가된 Repository가 없습니다.")
        return
    
    # Repository 목록 표시
    for i, repo_url in enumerate(st.session_state.repository_urls):
        with st.expander(f"📁 Repository {i+1}: {repo_url}", expanded=False):
            render_repository_details(repo_url, i)


def render_repository_details(repo_url, index):
    """Repository 상세 정보 렌더링"""
    try:
        # Repository 정보 조회
        repo_info = st.session_state.chatbot.github_extractor.get_repository_info(repo_url)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**이름**: {repo_info.get('full_name', 'Unknown')}")
            st.write(f"**설명**: {repo_info.get('description', 'N/A')}")
            st.write(f"**언어**: {repo_info.get('language', 'N/A')}")
            st.write(f"**기본 브랜치**: {repo_info.get('default_branch', 'main')}")
        
        with col2:
            st.write(f"**별표**: {repo_info.get('stars', 0):,}")
            st.write(f"**포크**: {repo_info.get('forks', 0):,}")
            st.write(f"**생성일**: {repo_info.get('created_at', 'N/A')}")
            st.write(f"**업데이트**: {repo_info.get('updated_at', 'N/A')}")
        
        # 문서 수 조회
        try:
            documents = st.session_state.chatbot.vector_store.get_repository_documents(repo_url)
            st.write(f"**문서 수**: {len(documents)}개")
        except:
            st.write("**문서 수**: 조회 실패")
        
        # 작업 버튼들
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 새로고침", key=f"refresh_{index}", use_container_width=True):
                refresh_repository(repo_url)
        
        with col2:
            if st.button("🗑️ 삭제", key=f"delete_{index}", use_container_width=True):
                delete_repository(repo_url, index)
        
        with col3:
            if st.button("📊 통계", key=f"stats_{index}", use_container_width=True):
                show_repository_statistics(repo_url)
    
    except Exception as e:
        st.error(f"❌ Repository 정보 조회 실패: {str(e)}")


def refresh_repository(repo_url):
    """Repository 새로고침"""
    try:
        with st.spinner("🔄 Repository 새로고침 중..."):
            # 기존 문서 삭제
            st.session_state.chatbot.vector_store.delete_repository_documents(repo_url)
            
            # 새로 추가
            result = st.session_state.chatbot.add_github_repository(repo_url)
            
            if result['success']:
                st.success(f"✅ Repository가 새로고침되었습니다. {result['documents_count']}개 문서")
            else:
                st.error(f"❌ 새로고침 실패: {result['message']}")
    
    except Exception as e:
        st.error(f"❌ Repository 새로고침 실패: {str(e)}")


def delete_repository(repo_url, index):
    """Repository 삭제"""
    try:
        # 확인 대화상자
        if st.button("정말 삭제하시겠습니까?", key=f"confirm_delete_{index}"):
            with st.spinner("🗑️ Repository 삭제 중..."):
                # 벡터 스토어에서 문서 삭제
                success = st.session_state.chatbot.vector_store.delete_repository_documents(repo_url)
                
                if success:
                    # 세션 상태에서 제거
                    st.session_state.repository_urls.pop(index)
                    st.success("✅ Repository가 삭제되었습니다.")
                    st.rerun()
                else:
                    st.error("❌ Repository 삭제 실패")
    
    except Exception as e:
        st.error(f"❌ Repository 삭제 실패: {str(e)}")


def show_repository_statistics(repo_url):
    """Repository 통계 표시"""
    try:
        # Repository 정보 조회
        repo_info = st.session_state.chatbot.github_extractor.get_repository_info(repo_url)
        
        # 문서 조회
        documents = st.session_state.chatbot.vector_store.get_repository_documents(repo_url)
        
        # 통계 계산
        total_docs = len(documents)
        
        # 파일 타입별 통계
        file_types = {}
        total_size = 0
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            file_size = doc.metadata.get('file_size', 0)
            total_size += file_size
        
        # 통계 표시
        st.subheader(f"📊 {repo_info.get('full_name', 'Unknown')} 통계")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 문서 수", total_docs)
        
        with col2:
            st.metric("총 크기", f"{total_size / (1024*1024):.1f} MB")
        
        with col3:
            st.metric("별표", f"{repo_info.get('stars', 0):,}")
        
        with col4:
            st.metric("포크", f"{repo_info.get('forks', 0):,}")
        
        # 파일 타입별 차트
        if file_types:
            st.subheader("📁 파일 타입별 분포")
            
            import pandas as pd
            df = pd.DataFrame(list(file_types.items()), columns=['타입', '개수'])
            st.bar_chart(df.set_index('타입'))
        
        # 최근 문서들
        if documents:
            st.subheader("📄 최근 문서들")
            
            for i, doc in enumerate(documents[:5], 1):
                with st.expander(f"문서 {i}: {doc.metadata.get('file_name', 'Unknown')}"):
                    st.write(f"**파일명**: {doc.metadata.get('file_name', 'Unknown')}")
                    st.write(f"**타입**: {doc.metadata.get('file_type', 'Unknown')}")
                    st.write(f"**크기**: {doc.metadata.get('file_size', 0)} bytes")
                    st.write(f"**내용 미리보기**: {doc.page_content[:200]}...")
    
    except Exception as e:
        st.error(f"❌ Repository 통계 조회 실패: {str(e)}")


def render_repository_statistics_section():
    """Repository 통계 섹션 렌더링"""
    st.subheader("📊 전체 Repository 통계")
    
    if not st.session_state.repository_urls:
        st.info("📝 통계를 표시할 Repository가 없습니다.")
        return
    
    try:
        # 전체 통계 계산
        total_repos = len(st.session_state.repository_urls)
        total_docs = 0
        total_size = 0
        
        for repo_url in st.session_state.repository_urls:
            try:
                documents = st.session_state.chatbot.vector_store.get_repository_documents(repo_url)
                total_docs += len(documents)
                
                for doc in documents:
                    total_size += doc.metadata.get('file_size', 0)
            except:
                continue
        
        # 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 Repository", total_repos)
        
        with col2:
            st.metric("총 문서", total_docs)
        
        with col3:
            st.metric("총 크기", f"{total_size / (1024*1024):.1f} MB")
        
        with col4:
            st.metric("평균 문서/Repo", f"{total_docs / total_repos:.1f}" if total_repos > 0 else "0")
        
        # Repository별 문서 수 차트
        if total_repos > 0:
            st.subheader("📈 Repository별 문서 수")
            
            repo_doc_counts = []
            for repo_url in st.session_state.repository_urls:
                try:
                    documents = st.session_state.chatbot.vector_store.get_repository_documents(repo_url)
                    repo_name = repo_url.split('/')[-1]
                    repo_doc_counts.append((repo_name, len(documents)))
                except:
                    repo_name = repo_url.split('/')[-1]
                    repo_doc_counts.append((repo_name, 0))
            
            if repo_doc_counts:
                import pandas as pd
                df = pd.DataFrame(repo_doc_counts, columns=['Repository', '문서 수'])
                st.bar_chart(df.set_index('Repository'))
    
    except Exception as e:
        st.error(f"❌ 전체 통계 계산 실패: {str(e)}")


def render_bulk_operations():
    """대량 작업 섹션 렌더링"""
    st.subheader("⚡ 대량 작업")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 모든 Repository 새로고침", use_container_width=True):
            refresh_all_repositories()
    
    with col2:
        if st.button("🗑️ 모든 Repository 삭제", use_container_width=True):
            delete_all_repositories()


def refresh_all_repositories():
    """모든 Repository 새로고침"""
    if not st.session_state.repository_urls:
        st.warning("⚠️ 새로고침할 Repository가 없습니다.")
        return
    
    try:
        with st.spinner("🔄 모든 Repository 새로고침 중..."):
            success_count = 0
            
            for repo_url in st.session_state.repository_urls:
                try:
                    # 기존 문서 삭제
                    st.session_state.chatbot.vector_store.delete_repository_documents(repo_url)
                    
                    # 새로 추가
                    result = st.session_state.chatbot.add_github_repository(repo_url)
                    
                    if result['success']:
                        success_count += 1
                
                except Exception as e:
                    st.error(f"❌ {repo_url} 새로고침 실패: {str(e)}")
            
            st.success(f"✅ {success_count}/{len(st.session_state.repository_urls)}개 Repository 새로고침 완료")
    
    except Exception as e:
        st.error(f"❌ 대량 새로고침 실패: {str(e)}")


def delete_all_repositories():
    """모든 Repository 삭제"""
    if not st.session_state.repository_urls:
        st.warning("⚠️ 삭제할 Repository가 없습니다.")
        return
    
    try:
        with st.spinner("🗑️ 모든 Repository 삭제 중..."):
            success_count = 0
            
            for repo_url in st.session_state.repository_urls:
                try:
                    success = st.session_state.chatbot.vector_store.delete_repository_documents(repo_url)
                    if success:
                        success_count += 1
                
                except Exception as e:
                    st.error(f"❌ {repo_url} 삭제 실패: {str(e)}")
            
            # 세션 상태 초기화
            st.session_state.repository_urls = []
            
            st.success(f"✅ {success_count}개 Repository 삭제 완료")
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ 대량 삭제 실패: {str(e)}")
