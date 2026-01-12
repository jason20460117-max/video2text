import os
import re
import glob
import yt_dlp

def extract_url(text):
    """从复杂文本中提取 http 链接"""
    pattern = r'(https?://\S+)'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def download_video_logic(url, mode="video"):
    """执行下载逻辑"""
    download_dir = "downloads"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    ydl_opts = {
        'outtmpl': f'{download_dir}/%(title)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    if mode == "audio":
        ydl_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }]
        })
    else:
        ydl_opts.update({'format': 'bestvideo+bestaudio/best'})
    
    if os.path.exists('cookies.txt'):
        ydl_opts['cookiefile'] = 'cookies.txt'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            valid_extensions = ['*.mp4', '*.mkv', '*.webm', '*.mp3', '*.m4a', '*.wav']
            list_of_files = []
            for ext in valid_extensions:
                list_of_files.extend(glob.glob(f'{download_dir}/{ext}'))
            
            if not list_of_files:
                return {"status": "error", "msg": "下载未找到文件"}
            
            latest_file = max(list_of_files, key=os.path.getmtime)
            return {
                "status": "success", 
                "file_path": latest_file, 
                "title": os.path.basename(latest_file)
            }
    except Exception as e:
        return {"status": "error", "msg": str(e)}