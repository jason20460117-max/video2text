import streamlit as st
import time
import re
import os
import glob
from openai import OpenAI

# 尝试导入新功能库
try:
    import yt_dlp
    import whisper
    import torch
    import zhconv
except ImportError:
    st.error("⚠️ 检测到缺少必要库！请运行: pip install yt-dlp openai-whisper torch zhconv")
    st.stop()

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="DeepFlow v8.3 (UI Fix)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 强制白底黑字 CSS + 进度块样式
# ==========================================
st.markdown("""
<style>
    /* 全局白底黑字，并增加行高防止 Emoji 被切断 */
    .stApp {background-color: #FFFFFF !important; color: #000000 !important;}
    section[data-testid="stSidebar"] {background-color: #F8F9FA !important; border-right: 1px solid #E9ECEF !important;}
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #212529 !important; 
        line-height: 1.6 !important; /* 关键修复：增加行高 */
    }
    
    /* 输入框 */
    .stTextInput input, .stTextArea textarea {
        background-color: #FFFFFF !important; color: #000000 !important; border: 1px solid #CED4DA !important;
    }
    
    /* 进度块样式 (Chunk Boxes) */
    .chunk-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .chunk-box {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    /* 状态：等待中 (灰色描边) */
    .chunk-pending {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        color: #adb5bd;
    }
    /* 状态：处理中 (蓝色呼吸) */
    .chunk-active {
        background-color: #e7f1ff;
        border: 2px solid #0d6efd;
        color: #0d6efd;
        box-shadow: 0 0 8px rgba(13, 110, 253, 0.3);
    }
    /* 状态：已完成 (绿色填充) */
    .chunk-done {
        background-color: #198754;
        border: 2px solid #198754;
        color: #ffffff;
    }

    /* 按钮 (绿色) */
    button[kind="primary"] {background-color: #198754 !important; border-color: #198754 !important; color: #FFFFFF !important;}
    
    .block-container {padding-top: 2rem !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 核心功能函数库
# ==========================================

def render_chunk_visualizer(total, current_processing_index, container):
    """
    渲染可视化的进度块
    """
    html_content = '<div class="chunk-container">'
    for i in range(total):
        display_num = i + 1
        if i < current_processing_index:
            state_class = "chunk-done" # 已完成
        elif i == current_processing_index:
            state_class = "chunk-active" # 正在处理
        else:
            state_class = "chunk-pending" # 等待中
            
        html_content += f'<div class="chunk-box {state_class}">{display_num}</div>'
    html_content += '</div>'
    container.markdown(html_content, unsafe_allow_html=True)

def extract_url(text):
    pattern = r'(https?://\S+)'
    match = re.search(pattern, text)
    if match: return match.group(1)
    return None

def get_device_status():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", f"✅ GPU ({gpu_name})"
    else:
        return "cpu", "⚠️ CPU (Slow)"

@st.cache_resource
def load_whisper_model(model_size="base"):
    device, _ = get_device_status()
    try:
        return whisper.load_model(model_size, device=device)
    except:
        return whisper.load_model(model_size, device="cpu")

def smart_split_text(text, max_chars=1200):
    if not text: return []
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
            else:
                current_chunk = p + "\n"
    if current_chunk: chunks.append(current_chunk)
    return chunks

def download_video_logic(url, mode="video"):
    download_dir = "downloads"
    if not os.path.exists(download_dir): os.makedirs(download_dir)
    ydl_opts = {
        'outtmpl': f'{download_dir}/%(title)s.%(ext)s',
        'quiet': True, 'no_warnings': True, 'restrictfilenames': False, 'updatetime': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    if mode == "audio":
        ydl_opts.update({'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}]})
    else:
        ydl_opts.update({'format': 'bestvideo+bestaudio/best'})
    
    if os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            valid_extensions = ['*.mp4', '*.mkv', '*.webm', '*.mp3', '*.m4a', '*.wav']
            list_of_files = []
            for ext in valid_extensions: list_of_files.extend(glob.glob(f'{download_dir}/{ext}'))
            if not list_of_files: return {"status": "error", "msg": "下载未找到文件"}
            latest_file = max(list_of_files, key=os.path.getmtime)
            return {"status": "success", "file_path": latest_file, "title": os.path.basename(latest_file), "thumbnail": None}
    except Exception as e: return {"status": "error", "msg": str(e)}

def transcribe_logic(file_path, model_size="base"):
    model = load_whisper_model(model_size)
    prompt = "以下是简体中文的对话内容，包含标点符号，逻辑清晰。"
    result = model.transcribe(file_path, language="zh", initial_prompt=prompt)
    return zhconv.convert(result["text"], 'zh-cn')

# ==========================================
# 4. 侧边栏配置
# ==========================================
with st.sidebar:
    st.markdown("## 🌊 DeepFlow v8.3")
    st.caption("Visual Chunking Edition")
    st.markdown("---")
    
    _, device_msg = get_device_status()
    if "GPU" in device_msg: st.success(device_msg)
    else: st.warning(device_msg)
    st.markdown("---")
    
    app_mode = st.radio("Navigation", ["📝 文本智能润色", "🎬 视频下载与转录"], index=0)
    st.markdown("---")

    if app_mode == "📝 文本智能润色":
        st.markdown("#### 🔑 API设置")
        api_key = st.text_input("API Key", type="password")
        base_url = st.text_input("Base URL", value="https://api.deepseek.com")
        model_name = st.selectbox("Model", ["deepseek-chat", "deepseek-coder"], index=0)
        
        st.markdown("---")
        st.markdown("#### 🛠️ 任务预设")
        
        PROMPT_MAP = {
            "录音稿深度整理 (Deep Clean)": """你是一个专业的文字编辑。你的任务是将用户输入的【口语录音文本】转写成去除口语化的文本。

请严格遵守以下规则：
1. **去口语化**：消除录音文字中的停顿、重复和口语化语气词。
2. **仅仅校准**：仅校准，不改写任何原文，确保文字内容忠实于原文本。
3. **保持原意**：严格保持原文原意。
4. **禁止闲聊**：直接输出结果，不要解释，不要包含任何前缀。
5. **合理分段**：合理分段，避免单段过长。
6. **翻译功能**：如果监测到主要内容为中文以外的内容，同样以保留原文原意的方式翻译为中文。""",
            
            "通用助手 (General)": """你是一个知识渊博、逻辑严密的智能助手。

请遵循以下回答原则：
1. **准确性优先**：确保提供的信息准确无误。
2. **逻辑清晰**：回答复杂问题时，请分步骤、分点进行阐述。
3. **格式规范**：适当使用 Markdown 格式（如加粗、列表）。""",
            
            "文本润色 (Polishing)": """你是一位资深出版编辑。请对用户提供的文本进行深度润色。

目标：让文章读起来更专业、更流畅、更有文采。

操作：
1. **修正语病**：修复所有语法错误。
2. **提升词汇**：替换口语化词汇。
3. **优化句式**：增强语言节奏感。""",
            
            "代码解释 (Code Expert)": """你是一位资深软件架构师。请分析代码。

输出结构：
1. **功能解读**：解释代码在做什么。
2. **代码优化**：指出问题并提供优化后的代码。
3. **关键注释**：添加详细中文注释。""",
            
            "会议纪要总结 (Summarization)": """你是一位高效的行政秘书。请将文本整理为会议纪要。

输出格式：
1. **📜 核心议题**
2. **🗣️ 详细摘要** (Bullet points)
3. **✅ 待办事项**
4. **📌 结论/决策**"""
        }

        TEMP_MAP = {
            "录音稿深度整理 (Deep Clean)": 0.1,
            "通用助手 (General)": 1.0,
            "文本润色 (Polishing)": 1.0,
            "代码解释 (Code Expert)": 0.2,
            "会议纪要总结 (Summarization)": 0.5
        }

        if "user_system_prompt" not in st.session_state:
            st.session_state.user_system_prompt = PROMPT_MAP["录音稿深度整理 (Deep Clean)"]
        if "user_temperature" not in st.session_state:
            st.session_state.user_temperature = TEMP_MAP["录音稿深度整理 (Deep Clean)"]

        def on_preset_change():
            selected = st.session_state.preset_selector
            st.session_state.user_system_prompt = PROMPT_MAP[selected]
            st.session_state.user_temperature = TEMP_MAP[selected]

        st.selectbox("选择预设", list(PROMPT_MAP.keys()), key="preset_selector", on_change=on_preset_change)
        temperature = st.slider("创意温度", 0.0, 1.5, key="user_temperature", step=0.1)
        
        st.markdown("---")
        st.markdown("#### 📏 长文优化")
        enable_chunking = st.checkbox("启用分段处理 (Chunking)", value=True)
        chunk_size = st.number_input("分段字符数", min_value=500, max_value=4000, value=1500, step=100)

    elif app_mode == "🎬 视频下载与转录":
        st.markdown("#### ⚙️ Whisper 设置")
        whisper_model = st.selectbox("模型大小", ["tiny", "base", "small", "medium", "large"], index=1)
        st.markdown("#### 📥 下载设置")
        download_format = st.radio("文件格式", ["视频 (MP4)", "纯音频 (MP3)"], index=0)

# ==========================================
# 5. 主界面逻辑
# ==========================================
if app_mode == "📝 文本智能润色":
    # 【修复重点】使用 HTML 直接渲染标题，强制设置对齐和边距，防止 Emoji 被切断
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
            <span style="font-size: 1.4rem;">🧠</span>
            <h4 style="margin: 0; padding: 0;">系统指令控制 (System Prompt)</h4>
        </div>
    """, unsafe_allow_html=True)
    
    system_prompt_input = st.text_area("System Prompt", height=100, key="user_system_prompt", label_visibility="collapsed")
    
    st.markdown("---")
    col_in, col_out = st.columns([1, 1])
    with col_in:
        user_input_temp = st.session_state.get("user_input_temp", "")
        count_str = f"{len(user_input_temp)} 字" if user_input_temp else "0 字"
        
        # 同样优化"原始文本"的标题显示
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
                <h4 style="margin: 0;">📄 原始文本</h4>
                <span style='font-size:0.9em;color:#6c757d; font-family: monospace;'>{count_str}</span>
            </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area("Input", height=500, label_visibility="collapsed", placeholder="输入文本...", key="user_input_temp")
        
        if "last_transcription" in st.session_state and st.session_state.last_transcription:
            if st.button("📥 填入刚刚转录的文本"):
                st.info("请复制下方代码块内容到输入框 (Streamlit 安全限制无法直接写入)") 
                st.code(st.session_state.last_transcription, language=None)
        
        start_btn = st.button("🚀 启动处理任务", type="primary", use_container_width=True)
        
    with col_out:
        st.markdown("#### 🤖 DeepSeek 响应")
        
        result_container = st.container(border=True, height=500)
        
        if start_btn:
            if not api_key: st.error("请配置 API Key")
            elif not user_input: st.warning("请输入内容")
            else:
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                if not enable_chunking or len(user_input) < chunk_size:
                    with st.spinner("思考中..."):
                        with result_container:
                            try:
                                stream = client.chat.completions.create(
                                    model=model_name,
                                    messages=[{"role":"system","content":system_prompt_input},{"role":"user","content":user_input}],
                                    temperature=temperature, stream=True
                                )
                                st.write_stream(stream)
                            except Exception as e: st.error(f"Error: {e}")
                            
                else:
                    chunks = smart_split_text(user_input, max_chars=chunk_size)
                    total_chunks = len(chunks)
                    full_response_text = ""
                    
                    with result_container:
                        vis_placeholder = st.empty()
                        
                        render_chunk_visualizer(total_chunks, -1, vis_placeholder)
                        
                        for idx, chunk in enumerate(chunks):
                            render_chunk_visualizer(total_chunks, idx, vis_placeholder)
                            
                            st.caption(f"📝 正在处理: Part {idx+1} / {total_chunks}")
                            
                            try:
                                stream = client.chat.completions.create(
                                    model=model_name,
                                    messages=[{"role":"system","content":system_prompt_input},{"role":"user","content":chunk}],
                                    temperature=temperature, stream=True
                                )
                                chunk_resp = st.write_stream(stream)
                                full_response_text += chunk_resp + "\n\n"
                                st.markdown("---")
                                
                            except Exception as e:
                                st.error(f"Error in chunk {idx+1}: {e}")
                                break
                        
                        render_chunk_visualizer(total_chunks, total_chunks, vis_placeholder)
                    
                    with st.expander("📥 获取完整合并文本", expanded=True):
                        st.text_area("Full Result", value=full_response_text, height=200)
                        st.success(f"✅ 处理完成！共 {len(full_response_text)} 字")

elif app_mode == "🎬 视频下载与转录":
    st.markdown("#### 🔗 视频链接解析")
    url_input = st.text_area("Input URL", height=100, placeholder="在此粘贴 B站/抖音/Youtube 链接...", label_visibility="collapsed")
    col_dl_1, col_dl_2 = st.columns([1, 4])
    
    dl_btn_label = "⬇️ 下载视频" if "视频" in download_format else "⬇️ 下载音频 (MP3)"
    with col_dl_1: analyze_btn = st.button(dl_btn_label, type="primary", use_container_width=True)
    st.markdown("---")
    
    if "current_video_path" not in st.session_state: st.session_state.current_video_path = None
    
    if analyze_btn and url_input:
        real_url = extract_url(url_input)
        if not real_url: st.error("❌ 未能在文本中找到有效的 http 链接")
        else:
            mode_code = "audio" if "MP3" in download_format else "video"
            with st.status(f"正在下载{'音频' if mode_code == 'audio' else '视频'}...", expanded=True) as status:
                st.write(f"🔗 解析链接: `{real_url}`")
                result = download_video_logic(real_url, mode=mode_code)
                if result["status"] == "success":
                    status.update(label="✅ 下载成功！", state="complete", expanded=False)
                    st.session_state.current_video_path = result["file_path"]
                    st.session_state.current_video_title = result["title"]
                    st.rerun()
                else:
                    status.update(label="❌ 下载失败", state="error")
                    st.error(result["msg"])

    if st.session_state.current_video_path and os.path.exists(st.session_state.current_video_path):
        st.success(f"📂 文件已就绪: `{st.session_state.current_video_path}`")
        col_v_1, col_v_2 = st.columns([1, 1])
        with col_v_1:
            st.markdown("#### 📺 文件预览")
            if st.session_state.current_video_path.endswith(".mp4"):
                st.video(st.session_state.current_video_path)
            else:
                st.audio(st.session_state.current_video_path)
                
        with col_v_2:
            st.markdown("#### 📝 语音转录 (Whisper)")
            transcribe_btn = st.button("🎙️ 开始转录为文字", type="primary", use_container_width=True)
            
            if transcribe_btn:
                with st.status("Whisper 模型正在运行中...", expanded=True) as t_status:
                    try:
                        text_result = transcribe_logic(st.session_state.current_video_path, whisper_model)
                        t_status.update(label="✅ 转录完成！", state="complete", expanded=False)
                        st.text_area("转录结果", value=text_result, height=300)
                        st.session_state.last_transcription = text_result
                        st.info("💡 提示：转录内容已保存。")
                    except Exception as e:
                        t_status.update(label="❌ 转录失败", state="error")
                        st.error(f"Whisper 错误: {str(e)}")