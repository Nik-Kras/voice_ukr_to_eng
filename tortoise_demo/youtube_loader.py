import yt_dlp as youtube_dl

def get_audio_from_youtube_video(url: str, filename: str = "original_audio"):
    """ Download audio from youtube link and save as `filename`.wav """
    path = f'results/{filename}'
    
    ydl_opts = {
        'format': 'bestaudio/best',         # prioritization options
        # 'ffmpeg_location': r'C:\Users\tprok\Downloads\ffmpeg-7.0.2-essentials_build\ffmpeg-7.0.2-essentials_build\bin',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',    # extracts audio from the video
            'preferredcodec': 'wav',        # format
            'preferredquality': '192',      # prefered bitrate quality in kb/s
        }],
        'outtmpl': path,                    # Change the output filename
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url]) 
        
    return path + ".wav"

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=M_lwC5zPkyo"
    get_audio_from_youtube_video(url, filename="tortoise_demo/china_original_audio")
