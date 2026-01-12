import streamlit as st
import time
import os
import glob
from openai import OpenAI

# 导入自定义模块
from video_utils import extract_url, download_video_logic
from prompts import PROMPT_MAP, TEMP_MAP

# 尝试导入核心 AI 库
try:
    import whisper
    import torch
    import zhconv
except ImportError:
    st.error("⚠️ 检测到缺少必要库！请运行: pip install yt-dlp openai-whisper torch zhconv")
    st.stop()

# ==========================================
# 1. 页面基础配置与精美 CSS 
# ==========================================
st.set_page_config(
    page_title="DeepFlow v8.3 (Modular)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {background-color: #FFFFFF !important; color: #000000 !important;}
    section[data-testid="stSidebar"] {background-color: #F8F9FA !important; border-right: 1px solid #E9ECEF !important;}
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #212529 !important; line-height: 1.6 !important; }
    
    /* 进度块样式 (Chunk Boxes) */
    .chunk-container { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; }
    .chunk-box { width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-weight: bold; font-size: 14px; transition: all 0.3s ease; }
    .chunk-pending { background-color: #ffffff; border: 2px solid #dee2e6; color: #adb5bd; }
    .chunk-active { background-color: #e7f1ff; border: 2px solid #0d6efd; color: #0d6efd; box-shadow: 0 0 8px rgba(13, 110, 253, 0.3); }
    .chunk-done { background-color: #198754; border: 2px solid #198754; color: #ffffff; }

    button[kind="primary"] {background-color: #198754 !important; border-color: #198754 !important; color: #FFFFFF !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心辅助函数 
# ==========================================
def render_chunk_visualizer(total, current_index, container):
    html = '<div class="chunk-container">'
    for i in range(total):
        state = "chunk-done" if i < current_index else ("chunk-active" if i == current_index else "chunk-pending")
        html += f'<div class="chunk-box {state}">{i+1}</div>'
    html += '</div>'
    container.markdown(html, unsafe_allow_html=True)

def get_device_status():
    if torch.cuda.is_available():
        return "cuda", f"✅ GPU ({torch.cuda.get_device_name(0)})"
    return "cpu", "⚠️ CPU (Slow)"

@st.cache_resource
def load_whisper_model(model_size="base"):
    device, _ = get_device_status()
    return whisper.load_model(model_size, device=device)

def transcribe_logic(file_path, model_size="base"):
    model = load_whisper_model(model_size)
    result = model.transcribe(file_path, language="zh", initial_prompt="以下是普通话内容。")
    return zhconv.convert(result["text"], 'zh-cn')

def smart_split_text(text, max_chars=1200):
    chunks, current = [], ""
    for p in text.split('\n'):
        if len(current) + len(p) < max_chars: current += p + "\n"
        else:
            if current: chunks.append(current)
            current = p + "\n"
    if current: chunks.append(current)
    return chunks

# ==========================================
# 3. 侧边栏与状态初始化 
# ==========================================
with st.sidebar:
    st.markdown("## 🌊 DeepFlow v8.3")
    _, device_msg = get_device_status()
    st.info(device_msg)
    
    app_mode = st.radio("功能导航", ["📝 文本智能润色", "🎬 视频下载与转录"])
    st.divider()

    if app_mode == "📝 文本智能润色":
        api_key = st.text_input("API Key", type="password")
        base_url = st.text_input("Base URL", value="https://api.deepseek.com")
        model_name = st.selectbox("模型", ["deepseek-chat", "gpt-4o-mini"])
        
        selected_preset = st.selectbox("任务预设", list(PROMPT_MAP.keys()))
        sys_prompt = st.text_area("系统提示词", value=PROMPT_MAP[selected_preset], height=150)
        temp = st.slider("创意温度", 0.0, 1.5, value=TEMP_MAP[selected_preset])
        chunk_size = st.number_input("分段字符数", 500, 4000, 1500)
    else:
        w_size = st.selectbox("Whisper模型", ["tiny", "base", "small", "medium"], index=1)

# ==========================================
# 4. 主界面逻辑 
# ==========================================
if app_mode == "📝 文本智能润色":
    st.subheader("📄 文本处理")
    user_input = st.text_area("请输入原始文本", height=300)
    
    if st.button("🚀 开始处理", type="primary", use_container_width=True):
        if not api_key or not user_input:
            st.warning("请检查 API Key 和输入内容")
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
            chunks = smart_split_text(user_input, max_chars=chunk_size)
            full_res, vis_place = "", st.empty()
            
            for idx, chunk in enumerate(chunks):
                render_chunk_visualizer(len(chunks), idx, vis_place)
                st.caption(f"正在处理第 {idx+1}/{len(chunks)} 段...")
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":chunk}],
                        temperature=temp, stream=True
                    )
                    full_res += st.write_stream(resp) + "\n\n"
                except Exception as e:
                    st.error(f"出错: {e}"); break
            
            render_chunk_visualizer(len(chunks), len(chunks), vis_place)
            st.success("全部处理完成！")
            st.text_area("合并结果", full_res, height=300)

elif app_mode == "🎬 视频下载与转录":
    st.subheader("🔗 视频链接转录")
    url_input = st.text_input("粘贴 B站/YouTube 链接")
    
    if st.button("⬇️ 下载并转录", type="primary"):
        real_url = extract_url(url_input)
        if real_url:
            with st.status("处理中...") as s:
                res = download_video_logic(real_url, mode="audio")
                if res["status"] == "success":
                    s.write("下载成功，开始转录...")
                    text = transcribe_logic(res["file_path"], w_size)
                    st.session_state.last_text = text
                    s.update(label="✅ 完成", state="complete")
                    st.text_area("转录结果", text, height=300)
                else: st.error(res["msg"])
        else: st.error("无效链接")