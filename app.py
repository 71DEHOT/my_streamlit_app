import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- 페이지 설정 ---
st.set_page_config(
    page_title="경기도 일자리 재단 주변 음식점 대시보드",
    page_icon="🍔",
    layout="wide",
)

# --- 데이터 로드 ---
# CSV 파일 로드, 캐싱을 통해 성능 향상
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # 평점, 리뷰수, 블로그수가 0 또는 NaN인 경우 처리
    for col in ['평점', '리뷰수', '블로그수']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # 하위 카테고리 NaN 값 처리
    df['하위카테고리'] = df['하위카테고리'].fillna('없음')
    return df

df = load_data("/Users/treasure/Documents/MCP/last.csv")

# --- 사이드바 (필터 옵션) ---
with st.sidebar:
    st.header("🔍 Filter Options")

    # 1. 카테고리 선택
    all_categories = sorted(df['카테고리'].unique())
    selected_categories = st.multiselect(
        "카테고리 선택 (복수 가능)",
        all_categories,
        default=all_categories # 기본으로 전체 선택
    )

    # 2. 하위 카테고리 선택
    if selected_categories:
        sub_categories = sorted(df[df['카테고리'].isin(selected_categories)]['하위카테고리'].unique())
    else:
        sub_categories = sorted(df['하위카테고리'].unique())
        
    selected_sub_categories = st.multiselect(
        "하위 카테고리 선택 (복수 가능)",
        sub_categories,
        default=sub_categories # 기본으로 전체 선택
    )

    # 3. 거리 선택
    distance = st.slider(
        "거리 선택 (m)",
        min_value=500,
        max_value=2000,
        value=2000, # 기본값
        step=100
    )

    # 4. 정렬 기준 선택
    sort_by = st.selectbox(
        "정렬 기준",
        ('평점', '리뷰수', '블로그수')
    )

    # 5. 상위 N개 선택
    top_n = st.slider(
        f"상위 {sort_by} 맛집 수",
        min_value=5,
        max_value=50,
        value=20, # 기본값
        step=5
    )

# --- 데이터 필터링 ---
# 1. 카테고리 필터
if not selected_categories:
    filtered_df = df.copy() # 선택 없으면 전체
else:
    filtered_df = df[df['카테고리'].isin(selected_categories)]

# 2. 하위 카테고리 필터
if selected_sub_categories:
    filtered_df = filtered_df[filtered_df['하위카테고리'].isin(selected_sub_categories)]

# 3. 거리 필터
filtered_df = filtered_df[filtered_df['distance'] <= distance]

# 4. 정렬 및 상위 N개 필터
sorted_df = filtered_df.sort_values(by=sort_by, ascending=False).head(top_n)


# --- 세션 상태 초기화 ---
if 'selected_place_id' not in st.session_state:
    st.session_state.selected_place_id = None

# --- 메인 대시보드 ---
st.title("경기도 일자리 재단 주변 음식점 대시보드")

col1, col2 = st.columns([0.6, 0.4]) # 지도와 상세정보 영역 분할

with col1:
    st.subheader("📍 지도")
    
    # 지도 중심점 계산 (필터링된 데이터 기준, 없으면 전체 데이터 기준)
    map_center_df = sorted_df if not sorted_df.empty else df
    map_center = [map_center_df['y'].mean(), map_center_df['x'].mean()]
    
    m = folium.Map(location=map_center, zoom_start=14)

    # 지도에 마커 추가
    for idx, row in sorted_df.iterrows():
        # 평점에 따라 원의 크기 조절 (최소 5, 최대 15)
        radius = max(5, min(15, row['평점'] * 2.5)) # 평점 1~5 -> 반지름 2.5~12.5, min/max로 범위 제한
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=radius,
            color='blue', # 테두리 색상
            fill_color='blue', # 채우기 색상
            fill_opacity=0.6, # 불투명도
            weight=1, # 테두리 두께
            popup=f"{row['place_name']}<br>ID:{row['id']}",
            tooltip=f"<b>{row['place_name']}</b><br>평점: {row['평점']}<br>리뷰: {row['리뷰수']}"
        ).add_to(m)

    # 경기도 일자리 재단 마커 추가 (빨간색 세모)
    # 경기도 일자리 재단 실제 좌표를 확인하여 아래 값을 수정해주세요.
    # 임시 좌표: 경기도 부천시 부천로 136번길 27 (경기도 일자리 재단 본원)
    gyeonggi_center_coords = [37.2534784863982, 127.031564444057]
    folium.Marker(
        location=gyeonggi_center_coords,
        icon=folium.Icon(color='red', icon='caret-up', prefix='fa'), # 빨간색 세모 아이콘
        popup="경기도 일자리 재단"
    ).add_to(m)
        
    # 지도 출력 및 상호작용
    map_data = st_folium(m, width=700, height=500)

    # 마커 클릭 시 ID 저장
    if map_data and map_data.get("last_object_clicked_popup"):
        popup_content = map_data["last_object_clicked_popup"]
        try:
            # popup 내용에서 ID 추출 (예: "가게이름<br>ID:12345")
            st.session_state.selected_place_id = int(popup_content.split("ID:")[1])
        except (IndexError, ValueError):
            # ID를 찾을 수 없는 경우
            pass


with col2:
    st.subheader("ℹ️ 음식점 상세 정보")
    
    if st.session_state.selected_place_id:
        selected_place_info = df[df['id'] == st.session_state.selected_place_id].iloc[0]
        
        st.markdown(f"### {selected_place_info['place_name']}")
        st.markdown(f"**주소**: {selected_place_info['address_name']}")
        st.markdown(f"**평점**: {selected_place_info['평점']} ⭐")
        st.markdown(f"**리뷰 수**: {int(selected_place_info['리뷰수'])} 개")
        st.markdown(f"**블로그 수**: {int(selected_place_info['블로그수'])} 개")
        st.markdown(f"**카카오맵 링크**: [바로가기]({selected_place_info['place_url']})")
    else:
        st.info("지도에서 음식점을 클릭하면 상세 정보가 표시됩니다.")


st.divider()

# --- 필터링된 음식점 리스트 ---
st.subheader(f"📋 '{sort_by}' 기준 상위 {top_n}개 음식점 리스트")

# Create a copy to modify for display
display_df = sorted_df.copy()

# Add a new column for the pin emoji
display_df['_indicator'] = ''
if st.session_state.selected_place_id is not None:
    selected_place_id = st.session_state.selected_place_id
    # Set the indicator for the selected row
    display_df.loc[display_df['id'] == selected_place_id, '_indicator'] = '📍'

st.dataframe(
    display_df[['_indicator', 'place_name', '카테고리', '하위카테고리', '평점', '리뷰수', '블로그수', 'distance']],
    use_container_width=True,
    hide_index=True # Hide the default index
)