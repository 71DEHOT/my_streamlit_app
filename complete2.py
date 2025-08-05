import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ====== 환경변수 불러오기 (선택사항) ======
load_dotenv()

# ====== 기본 설정 ======
st.set_page_config(page_title="외국어 기사 학습 도우미", layout="wide")

# ====== 상단 헤더 (제목 + 커피 후원 버튼) ======
col1, col2 = st.columns([3, 1])

with col1:
    st.title("NEWStudy")

with col2:
    st.markdown("### ")  # 공간 조정
    if st.button("☕ Give me a coffee!", type="secondary"):
        st.balloons()
        st.success("감사합니다! 후원 링크를 준비 중입니다 ☕")
        # 실제 후원 링크로 연결하려면:
        # st.markdown("[☕ Buy me a coffee](https://buymeacoffee.com/your-username)")
        # 또는 토스, 카카오페이 등의 링크 사용 가능

# ====== 서비스 설명 ======
st.markdown("---")
st.markdown("""
### 🌍 전 세계 뉴스·기사로 외국어를 쉽고 재미있게 학습하세요!

이 서비스는 외국어 학습자를 위한 웹 기반 학습 도구입니다.  
원하는 뉴스·기사 URL만 입력하면 아래 기능을 제공합니다:

**📖 문장별 번역**  
기사 원문을 문장 단위로 분석하고,  
선택한 언어로 자연스러운 번역을 제공합니다.

**📚 핵심 단어장 생성**  
숫자, 관사, 접속사 등 불필요한 단어를 제외하고  
주요 어휘만 뽑아 단어·뜻 형태의 단어장으로 제공합니다.
""")
st.markdown("---")

# ====== API 키 입력 섹션 ======
st.markdown("### 🔑 API 키 설정")
col1, col2 = st.columns([3, 1])

with col1:
    api_key = st.text_input(
        "Gemini API Key를 입력하세요:",
        type="password",
        placeholder="여기에 API 키를 입력하세요...",
        help="API 키는 안전하게 처리되며 저장되지 않습니다."
    )

with col2:
    st.markdown("#### ")  # 공간 조정
    st.markdown("[🔗 API 키 발급받기](https://aistudio.google.com/app/apikey?hl=ko)")

if not api_key:
    st.warning("⚠️ 서비스를 이용하려면 Gemini API 키가 필요합니다.")
    st.info("💡 위 링크를 클릭하여 무료로 API 키를 발급받으세요!")
    st.stop()

# ====== LLM 연결 (사용자 입력 API 키 사용) ======
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    st.success("✅ API 키가 성공적으로 설정되었습니다!")
except Exception as e:
    st.error(f"❌ API 키 설정 중 오류가 발생했습니다: {str(e)}")
    st.info("💡 API 키를 다시 확인해주세요.")
    st.stop()

# ====== 난이도 계산 함수 ======
def calculate_difficulty(sentences):
    """문장 리스트를 기반으로 난이도를 A1~C2로 계산"""
    if not sentences:
        return "A1"
    
    total_words = 0
    complex_words = 0
    
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        complex_words += len([w for w in words if len(w) > 7])
    
    if total_words == 0:
        return "A1"
    
    avg_sentence_length = total_words / len(sentences)
    complex_ratio = complex_words / total_words
    
    # 난이도 계산 로직
    if avg_sentence_length < 10 and complex_ratio < 0.1:
        return "A1"
    elif avg_sentence_length < 15 and complex_ratio < 0.15:
        return "A2"
    elif avg_sentence_length < 20 and complex_ratio < 0.2:
        return "B1"
    elif avg_sentence_length < 25 and complex_ratio < 0.25:
        return "B2"
    elif avg_sentence_length < 30 and complex_ratio < 0.3:
        return "C1"
    else:
        return "C2"

# ====== PDF 생성 함수 ======
def create_pdf(sentences, translations, topic):
    """문장과 번역을 PDF로 생성 (한글 폰트 지원)"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # 한글 폰트 등록 시도
    try:
        # Windows 시스템 폰트 경로들
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",  # 바탕
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        ]
        
        font_registered = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('Korean', font_path))
                    font_registered = True
                    break
                except:
                    continue
        
        # 스타일 설정
        styles = getSampleStyleSheet()
        
        if font_registered:
            # 한글 폰트가 등록된 경우
            from reportlab.lib.styles import ParagraphStyle
            
            korean_title = ParagraphStyle(
                'KoreanTitle',
                parent=styles['Title'],
                fontName='Korean',
                fontSize=16,
                spaceAfter=20
            )
            
            korean_heading = ParagraphStyle(
                'KoreanHeading',
                parent=styles['Heading2'],
                fontName='Korean',
                fontSize=12,
                spaceAfter=10
            )
            
            korean_normal = ParagraphStyle(
                'KoreanNormal',
                parent=styles['Normal'],
                fontName='Korean',
                fontSize=10,
                spaceAfter=8
            )
        else:
            # 폰트 등록 실패 시 기본 스타일 사용
            korean_title = styles['Title']
            korean_heading = styles['Heading2']
            korean_normal = styles['Normal']
            
    except Exception as e:
        # 오류 발생 시 기본 스타일 사용
        styles = getSampleStyleSheet()
        korean_title = styles['Title']
        korean_heading = styles['Heading2']
        korean_normal = styles['Normal']
    
    story = []
    
    # 제목 추가
    title = Paragraph(f"<b>기사 주제: {topic}</b>", korean_title)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # 문장과 번역 추가
    for i, (sentence, translation) in enumerate(zip(sentences, translations), 1):
        # 문장 번호
        story.append(Paragraph(f"<b>문장 {i}:</b>", korean_heading))
        # 원문
        story.append(Paragraph(f"<b>원문:</b> {sentence}", korean_normal))
        # 번역
        story.append(Paragraph(f"<b>번역:</b> {translation}", korean_normal))
        story.append(Spacer(1, 15))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ====== Session State 초기화 ======
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'current_url' not in st.session_state:
    st.session_state.current_url = ""

# ====== UI 구성 ======
url_input = st.text_input("기사 URL을 입력하세요:", value=st.session_state.current_url)

if st.button("기사 분석 시작"):
    if not url_input:
        st.warning("URL을 입력하세요!")
    else:
        # 1️⃣ 기사 크롤링
        loader = WebBaseLoader(url_input)
        docs = loader.load()
        text = docs[0].page_content

        # 2️⃣ 한 번의 API 호출로 모든 데이터 추출
        analysis_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "다음 텍스트에서 뉴스 기사 본문만 찾아서 분석해줘. "
                "광고, 메뉴, 댓글, 관련기사 목록, 사이트 네비게이션 등은 무시하고 실제 뉴스 내용만 처리해줘.\n\n"
                
                "결과를 반드시 다음 JSON 형식으로만 출력해줘:\n"
                "{{\n"
                '  "topic": "기사 주제 (한국어 20자 이내)",\n'
                '  "sentences": ["외국어 문장1", "외국어 문장2", "외국어 문장3", ...],\n'
                '  "translations": ["한국어 번역1", "한국어 번역2", "한국어 번역3", ...],\n'
                '  "words": ["외국어단어1", "외국어단어2", "외국어단어3", ...],\n'
                '  "word_meanings": ["한국어뜻1", "한국어뜻2", "한국어뜻3", ...],\n'
                '  "word_sentences": ["해당 단어가 나온 문장1", "해당 단어가 나온 문장2", "해당 단어가 나온 문장3", ...]\n'
                "}}\n\n"
                
                "중요한 규칙:\n"
                "1. sentences와 translations는 반드시 같은 개수여야 함\n"
                "2. words, word_meanings, word_sentences는 반드시 같은 개수여야 함 (한 문장 당 2단어)\n"
                "3. sentences는 외국어 원문, translations는 한국어 번역\n"
                "4. words는 외국어 단어, word_meanings는 한국어 뜻\n"
                "5. word_sentences는 해당 단어가 실제로 나온 문장 (sentences 배열에서 가져온 것)\n"
                "6. 해당 언어의 관사, 접속사, 대명사, 고유명사(사람 이름, 나라명 등)는 제외(중요)\n"
                "7. 배열 개수를 반드시 확인하고 정확히 맞춰서 출력\n\n"
                
                "텍스트:\n{text}"
            )
        )
        
        chain = LLMChain(llm=llm, prompt=analysis_prompt)
        
        with st.spinner("기사를 분석 중... (잠시만 기다려주세요)"):
            result = chain.run(text)
        
        try:
            # JSON 파싱
            # JSON 부분만 추출 (```json 태그 제거)
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
            elif "{" in result and "}" in result:
                json_start = result.find("{")
                json_end = result.rfind("}") + 1
                json_str = result[json_start:json_end]
            else:
                json_str = result
            
            data = json.loads(json_str)
            
            # 데이터 추출
            topic = data.get("topic", "주제 없음")
            sentences = data.get("sentences", [])
            translations = data.get("translations", [])
            words = data.get("words", [])
            word_meanings = data.get("word_meanings", [])
            word_sentences = data.get("word_sentences", [])
            
            # 배열 길이 검증 및 보정
            if len(words) != len(word_meanings) or len(words) != len(word_sentences):
                st.warning(f"⚠️ 단어 배열 길이 불일치 감지: 단어 {len(words)}개, 뜻 {len(word_meanings)}개, 문장 {len(word_sentences)}개")
                
                # 가장 짧은 배열에 맞춰서 자르기
                min_length = min(len(words), len(word_meanings), len(word_sentences))
                words = words[:min_length]
                word_meanings = word_meanings[:min_length]
                word_sentences = word_sentences[:min_length]
                
                st.info(f"✅ 배열 길이를 {min_length}개로 맞췄습니다.")
            
            if len(sentences) != len(translations):
                st.warning(f"⚠️ 문장 배열 길이 불일치 감지: 문장 {len(sentences)}개, 번역 {len(translations)}개")
                
                # 더 짧은 배열에 맞춰서 자르기
                min_length = min(len(sentences), len(translations))
                sentences = sentences[:min_length]
                translations = translations[:min_length]
                
                st.info(f"✅ 배열 길이를 {min_length}개로 맞췄습니다.")
            
            # Session State에 분석 결과 저장
            st.session_state.analysis_data = {
                'topic': topic,
                'sentences': sentences,
                'translations': translations,
                'words': words,
                'word_meanings': word_meanings,
                'word_sentences': word_sentences,
                'difficulty_level': calculate_difficulty(sentences)
            }
            st.session_state.current_url = url_input
            
        except json.JSONDecodeError as e:
            st.error("결과 파싱 중 오류가 발생했습니다. 다시 시도해주세요.")
            st.write("원본 결과:")
            st.code(result)
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
            st.write("원본 결과:")
            st.code(result)

# ====== 분석 결과 표시 (Session State 사용) ======
if st.session_state.analysis_data:
    data = st.session_state.analysis_data
    
    # 기사 주제 표시
    st.info(f"📰 **기사 주제**: {data['topic']}")
    
    # 통계 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 문장 수", f"{len(data['sentences'])}개")
    with col2:
        st.metric("📚 단어 수", f"{len(data['words'])}개")
    with col3:
        color = "🟢" if data['difficulty_level'] in ["A1", "A2"] else "🟡" if data['difficulty_level'] in ["B1", "B2"] else "🔴"
        st.metric("🎯 난이도", f"{color} {data['difficulty_level']}")
    
    st.divider()
    
    # 탭으로 페이지 구분
    tab1, tab2 = st.tabs(["📖 문장별 번역", "📚 단어장"])
    
    with tab1:
        st.subheader("문장별 번역")
        
        # 문장과 번역 표시
        for i, (sentence, translation) in enumerate(zip(data['sentences'], data['translations']), 1):
            st.write(f"**문장 {i}:**")
            st.write(f"**원문:** {sentence}")
            st.write(f"**번역:** {translation}")
            st.write("---")
        
        # PDF 다운로드 버튼
        if data['sentences'] and data['translations']:
            try:
                pdf_buffer = create_pdf(data['sentences'], data['translations'], data['topic'])
                st.download_button(
                    label="📄 문장별 번역 PDF 다운로드",
                    data=pdf_buffer.getvalue(),
                    file_name=f"sentences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )
            except Exception as e:
                st.error(f"PDF 생성 중 오류가 발생했습니다: {str(e)}")
    
    with tab2:
        st.subheader("단어장")
        
        # 단어장 테이블 생성
        if data['words'] and data['word_meanings'] and data['word_sentences']:
            vocab_df = pd.DataFrame({
                '단어': data['words'],
                '뜻': data['word_meanings'],
                '예문': data['word_sentences']
            })
            
            # 테이블 표시
            st.dataframe(vocab_df, use_container_width=True)
            
            # CSV 다운로드 버튼 (UTF-8 인코딩)
            csv = vocab_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 단어장 CSV 다운로드",
                data=csv.encode('utf-8'),
                file_name=f"vocabulary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                key="csv_download"
            )
            
            st.success(f"✅ 총 {len(data['words'])}개의 단어를 추출했습니다!")
        else:
            st.warning("단어를 추출할 수 없습니다.")
    
    # 새 분석 버튼
    if st.button("🔄 새로운 기사 분석하기", type="secondary"):
        st.session_state.analysis_data = None
        st.session_state.current_url = ""
        st.rerun()