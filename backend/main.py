import os
import re
import wave
import json
import uuid
import shutil
import asyncio
import zipfile
import subprocess
import warnings
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('urllib3').setLevel(logging.ERROR)

import numpy as np
import whisper
import edge_tts
import requests as http_requests

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = 'output'
MUSIC_DIR  = 'music_presets'
UPLOAD_DIR = 'uploads'
STATIC_DIR = 'static'
FONTS_DIR  = os.getcwd()

for d in [OUTPUT_DIR, MUSIC_DIR, UPLOAD_DIR, STATIC_DIR]:
    os.makedirs(d, exist_ok=True)

FONT_PATH = os.path.join(FONTS_DIR, 'font.ttf')

def _ensure_font():
    if os.path.exists(FONT_PATH):
        return
    try:
        import urllib.request
        url = 'https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf'
        urllib.request.urlretrieve(url, FONT_PATH)
        print(f"Downloaded Bangers font -> {FONT_PATH}")
    except Exception as e:
        print(f"Could not download font ({e}) — subtitles will use Arial")

_ensure_font()

model = whisper.load_model('tiny')

# In-memory job store — simple for single-user local deployment
JOBS: dict = {}          # job_id -> {status, message, clips, hooks}
LAST_TRANSCRIPTS: dict = {}
LAST_API_KEY: str = ''

PLATFORM_OPTIONS = ['TikTok', 'YouTube Shorts']
NICHE_OPTIONS = [
    'General', 'Fitness & Health', 'Finance & Investing',
    'Comedy & Entertainment', 'Education & How-To',
    'Tech & AI', 'Motivation & Mindset', 'Food & Cooking',
    'Travel & Lifestyle', 'Beauty & Fashion', 'Gaming',
]
VOICE_OPTIONS = {
    'Female US (Jenny)':   'en-US-JennyNeural',
    'Male US (Guy)':       'en-US-GuyNeural',
    'Female UK (Sonia)':   'en-GB-SoniaNeural',
    'Male UK (Ryan)':      'en-GB-RyanNeural',
    'Female AU (Natasha)': 'en-AU-NatashaNeural',
    'Male AU (William)':   'en-AU-WilliamNeural',
}
LANGUAGE_OPTIONS = {
    'English':    ('en-US-JennyNeural',    'en'),
    'Spanish':    ('es-ES-ElviraNeural',   'es'),
    'French':     ('fr-FR-DeniseNeural',   'fr'),
    'German':     ('de-DE-KatjaNeural',    'de'),
    'Portuguese': ('pt-BR-FranciscaNeural','pt'),
    'Italian':    ('it-IT-ElsaNeural',     'it'),
    'Japanese':   ('ja-JP-NanamiNeural',   'ja'),
    'Korean':     ('ko-KR-SunHiNeural',    'ko'),
    'Chinese':    ('zh-CN-XiaoxiaoNeural', 'zh'),
    'Hindi':      ('hi-IN-SwaraNeural',    'hi'),
    'Arabic':     ('ar-SA-ZariyahNeural',  'ar'),
    'Russian':    ('ru-RU-SvetlanaNeural', 'ru'),
}
MUSIC_PRESETS = {
    'None': None, 'Lo-fi': 'lofi', 'Upbeat': 'upbeat',
    'Cinematic': 'cinematic', 'Chill': 'chill',
}
WATERMARK_POSITIONS = {
    'Top Left': 'overlay=10:10', 'Top Right': 'overlay=W-w-10:10',
    'Bottom Left': 'overlay=10:H-h-10', 'Bottom Right': 'overlay=W-w-10:H-h-10',
}

# Subtitle position: Alignment locked to 2 (bottom-center, most reliable in
# libass force_style mode). MarginV controls vertical position on 1280px frame.
SUBTITLE_POSITION_MAP = {
    'Bottom': {'alignment': 2, 'margin_v': 40},
    'Middle': {'alignment': 2, 'margin_v': 580},
    'Top':    {'alignment': 2, 'margin_v': 1160},
}
SUBTITLE_FONTS = ['Bangers', 'Arial', 'Impact', 'Helvetica', 'Times New Roman']
HOOK_KEYWORDS = [
    'why', 'how', 'secret', 'best', 'never', 'crazy', 'watch', 'truth',
    'nobody', 'everyone', 'stop', 'start', 'mistake', 'change', 'think',
    'actually', 'honest', 'real', 'shocking', 'warning', 'proof',
]
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODEL   = 'claude-haiku-4-5'


# =============================================================================
# HELPERS
# =============================================================================

def safe_id():
    return uuid.uuid4().hex[:8]

def ffmpeg_path(path):
    return path.replace('\\', '/').replace(':', '\\:')

def format_time(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def get_media_duration(path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', path]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:    return float(r.stdout.strip())
    except: return 0.0

def run_ffmpeg(cmd_str):
    os.system(cmd_str + ' -loglevel error')

def claude_call(api_key, system_prompt, user_prompt, max_tokens=512):
    if not api_key or not api_key.strip():
        return None
    try:
        resp = http_requests.post(
            ANTHROPIC_API_URL,
            headers={'x-api-key': api_key.strip(), 'anthropic-version': '2023-06-01',
                     'content-type': 'application/json'},
            json={'model': ANTHROPIC_MODEL, 'max_tokens': max_tokens,
                  'system': system_prompt,
                  'messages': [{'role': 'user', 'content': user_prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()['content'][0]['text'].strip()
    except Exception:
        return None

def build_subtitle_filter(srt_path, subtitle_size, subtitle_position, subtitle_font):
    pos      = SUBTITLE_POSITION_MAP.get(subtitle_position, SUBTITLE_POSITION_MAP['Bottom'])
    marginv  = pos['margin_v']
    fontname = 'Bangers' if (subtitle_font == 'Bangers' and os.path.exists(FONT_PATH)) else subtitle_font
    fdir     = ffmpeg_path(FONTS_DIR)
    style = (f"Fontname={fontname},Fontsize={subtitle_size},"
             f"PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
             f"Outline=3,Shadow=0,Bold=0,"
             f"Alignment=2,MarginV={marginv},MarginL=0,MarginR=0")
    return f"subtitles='{srt_path}':fontsdir='{fdir}':force_style='{style}'"

def resolve_music_path(music_preset, music_file_path):
    if music_preset and music_preset != 'None' and MUSIC_PRESETS.get(music_preset):
        p = os.path.join(MUSIC_DIR, f'{MUSIC_PRESETS[music_preset]}.mp3')
        if os.path.exists(p):
            return p
    if music_file_path and os.path.exists(music_file_path):
        return music_file_path
    return None


# =============================================================================
# SCORING
# =============================================================================

def score_segments_keywords(segments, max_clips):
    scored = []
    for seg in segments:
        text = seg['text'].lower(); dur = seg['end'] - seg['start']; score = 0.0
        for k in HOOK_KEYWORDS:
            if k in text: score += 5
        if 5 <= dur <= 20: score += 5
        score += min(len(text) * 0.02, 5)
        if dur < 3: score -= 5
        scored.append((score, seg))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(round(s, 1), seg) for s, seg in scored[:max_clips]]

def score_segments_ai(segments, max_clips, api_key, platform, niche):
    if not api_key:
        return score_segments_keywords(segments, max_clips)
    system = (f"Score transcript segments for virality on {platform} in the {niche} niche. "
              'Respond ONLY with JSON: [{"index": int, "score": float 0-10}, ...].')
    seg_list = [f'{i}: [{round(seg["end"]-seg["start"],1)}s] {seg["text"].strip()}'
                for i, seg in enumerate(segments[:200])]
    raw = claude_call(api_key, system, "Score these:\n" + "\n".join(seg_list), 600)
    if not raw:
        return score_segments_keywords(segments, max_clips)
    try:
        clean = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
        sm    = {item['index']: float(item['score']) for item in json.loads(clean)}
        scored = [(sm.get(i, 0.0), seg) for i, seg in enumerate(segments)]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [(round(s, 1), seg) for s, seg in scored[:max_clips]]
    except Exception:
        return score_segments_keywords(segments, max_clips)


# =============================================================================
# SMART MODE
# =============================================================================

def _compute_rms_map(wav_path):
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_ch = wf.getnchannels(); sampwidth = wf.getsampwidth()
            framerate = wf.getframerate(); raw_data = wf.readframes(wf.getnframes())
    except Exception:
        return {}
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
    if n_ch > 1: samples = samples.reshape(-1, n_ch).mean(axis=1)
    samples /= max(np.iinfo(dtype).max, 1)
    win = int(framerate * 0.5); n = len(samples) // win
    return {i: float(np.sqrt(np.mean(samples[i*win:(i+1)*win]**2))) for i in range(n)}

def _snap_to_boundary(t, all_words, look_back=3.0):
    if not all_words: return t
    candidates = [t]
    for idx, w in enumerate(all_words):
        wt = w.get('start', 0)
        if not (t - look_back <= wt <= t): continue
        text = w.get('word', '').strip()
        if text and text[-1] in '.!?,;':
            candidates.append(wt + w.get('end', wt) - wt)
        if idx > 0 and wt - all_words[idx-1].get('end', 0) > 0.3:
            candidates.append(wt)
    valid = [c for c in candidates if c <= t]
    return max(valid) if valid else t

def _energy_bonus(t, duration, rms_map):
    if not rms_map: return 0.0
    vals = [rms_map[i] for i in range(int(t/0.5), int((t+duration)/0.5)+1) if i in rms_map]
    if not vals: return 0.0
    return round(min(sum(vals)/len(vals) / 0.15 * 2.0, 2.0), 2)

def smart_select_segments(segments, all_words, wav_path, max_clips,
                           clip_duration, video_duration, api_key, platform, niche):
    rms_map = _compute_rms_map(wav_path)
    candidates = []
    for seg in segments:
        ss = _snap_to_boundary(seg['start'], all_words)
        adj = dict(seg); adj['start'] = ss
        text = seg['text'].lower(); dur = seg['end'] - seg['start']
        base = sum(5 for k in HOOK_KEYWORDS if k in text)
        if 5 <= dur <= 20: base += 5
        base += min(len(text)*0.02, 5)
        if dur < 3: base -= 5
        energy = _energy_bonus(ss, min(clip_duration, dur+2), rms_map)
        candidates.append({'seg': adj, 'score': round(base+energy, 2),
                            'energy': energy, 'pos_pct': round(ss/max(video_duration,1)*100,1),
                            'dur': round(dur,1)})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    deduped, used = [], []
    for c in candidates:
        t = c['seg']['start']
        if any(abs(t-u) < 10.0 for u in used): continue
        used.append(t); deduped.append(c)
        if len(deduped) >= max_clips * 3: break
    if api_key and deduped:
        pool = deduped[:20]
        system = (f"Re-score clip candidates for virality on {platform} in {niche}. "
                  "Each has text, dur, pos%, energy(0-2). "
                  'Respond ONLY: [{"index": int, "score": float 0-10}, ...].')
        lines = [f'{i}: text="{c["seg"]["text"].strip()[:120]}" dur={c["dur"]}s '
                 f'pos={c["pos_pct"]}% energy={c["energy"]}' for i, c in enumerate(pool)]
        raw = claude_call(api_key, system, "\n".join(lines), 500)
        if raw:
            try:
                clean = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
                sm = {item['index']: float(item['score']) for item in json.loads(clean)}
                for i, c in enumerate(pool):
                    if i in sm: c['score'] = sm[i]
                pool.sort(key=lambda x: x['score'], reverse=True)
                deduped = pool
            except Exception: pass
    return [(round(c['score'],1), c['seg']) for c in deduped[:max_clips]]


# =============================================================================
# GAMING MODE
# =============================================================================

def detect_gaming_moments(wav_path, max_clips, clip_duration, video_duration):
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_ch=wf.getnchannels(); sw=wf.getsampwidth(); fr=wf.getframerate()
            raw=wf.readframes(wf.getnframes())
    except Exception: return []
    dtype = {1:np.int8,2:np.int16,4:np.int32}.get(sw,np.int16)
    samples = np.frombuffer(raw,dtype=dtype).astype(np.float32)
    if n_ch>1: samples=samples.reshape(-1,n_ch).mean(axis=1)
    samples /= np.iinfo(dtype).max
    ws=int(fr*0.5); nw=len(samples)//ws
    if nw==0: return []
    rms=np.array([np.sqrt(np.mean(samples[i*ws:(i+1)*ws]**2)) for i in range(nw)])
    rms=np.convolve(rms,np.ones(5)/5,mode='same')
    thr=np.percentile(rms,75)
    used,peaks=[],[]
    for idx in np.argsort(rms)[::-1]:
        t=idx*0.5
        if rms[idx]<thr: break
        if any(abs(t-u)<clip_duration*0.8 for u in used): continue
        if t+clip_duration>video_duration: continue
        used.append(t); peaks.append((rms[idx],t))
        if len(peaks)>=max_clips: break
    return [(round(float(s)*10,1),{'start':t,'end':min(t+clip_duration,video_duration),
             'text':f'[Gaming moment @ {t:.1f}s]','words':[]}) for s,t in peaks]


# =============================================================================
# FILM CUT
# =============================================================================

def detect_scene_cuts(video_path, clip_start, clip_duration, sensitivity=0.3):
    cmd = ['ffprobe','-v','quiet','-ss',str(clip_start),'-i',video_path,
           '-t',str(clip_duration),'-vf',f'scdet=threshold={sensitivity}',
           '-show_frames','-select_streams','v',
           '-show_entries','frame=pts_time','-of','csv=p=0']
    try:
        r = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
        cuts=[]
        for line in r.stderr.splitlines():
            m=re.search(r'lavfi\.scd\.time:\s*([\d.]+)',line)
            if m: cuts.append(float(m.group(1)))
        return sorted(cuts)
    except Exception: return []


# =============================================================================
# HOOK GENERATION
# =============================================================================

def generate_hooks_basic(text):
    words=text.strip().split()
    return [' '.join(words[:5]).upper().rstrip('.,!?'),'WAIT FOR THIS PART','YOU NEED TO SEE THIS']

def generate_hooks_ai(text, api_key, platform, niche):
    if not api_key: return generate_hooks_basic(text)
    system=(f"Viral {platform} strategist in {niche}. "
            "Write 3 ultra-short hook overlays (3-6 words, ALL CAPS). "
            "Respond ONLY as JSON array of 3 strings.")
    raw=claude_call(api_key,system,f"Transcript: {text[:300]}",150)
    if not raw: return generate_hooks_basic(text)
    try:
        clean=raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
        hooks=json.loads(clean)
        if isinstance(hooks,list) and hooks:
            return [' '.join(str(h).upper().split()[:6]) for h in hooks[:3]]
    except Exception: pass
    return generate_hooks_basic(text)


# =============================================================================
# SRT GENERATION
# =============================================================================

def create_clip_srt(words, clip_start, duration, uid):
    filename=os.path.join(OUTPUT_DIR,f'sub_{uid}.srt')
    clip_end=clip_start+duration
    with open(filename,'w',encoding='utf-8') as f:
        count=1
        for w in words:
            ws=w.get('start',0); we=w.get('end',0)
            if clip_start<=ws<=clip_end:
                f.write(f"{count}\n")
                f.write(f"{format_time(ws-clip_start)} --> {format_time(min(we-clip_start,duration))}\n")
                f.write(w['word'].strip().upper()+'\n\n')
                count+=1
    return filename

def create_script_srt(script, audio_duration, uid):
    outfile=os.path.join(OUTPUT_DIR,f'scriptsub_{uid}.srt')
    words=script.split()
    if not words: return None
    pw=max(audio_duration/len(words),0.18)
    with open(outfile,'w',encoding='utf-8') as f:
        for i,word in enumerate(words,1):
            f.write(f"{i}\n{format_time((i-1)*pw)} --> {format_time(i*pw)}\n{word.strip().upper()}\n\n")
    return outfile

def translate_srt(srt_path, target_language, api_key):
    if not api_key or target_language=='English': return srt_path
    try:
        with open(srt_path,'r',encoding='utf-8') as f: content=f.read()
    except Exception: return srt_path
    system=(f"Translate ONLY subtitle text lines to {target_language}. "
            "Preserve SRT numbering, timestamps, blank lines. Keep text SHORT.")
    translated=claude_call(api_key,system,content,2000)
    if not translated: return srt_path
    out=srt_path.replace('.srt',f'_{target_language.lower()}.srt')
    try:
        with open(out,'w',encoding='utf-8') as f: f.write(translated)
        return out
    except Exception: return srt_path


# =============================================================================
# CLIP RENDERING HELPERS
# =============================================================================

def _build_hook_filter(hook_text, hook_style, hook_size):
    if not hook_text: return ''
    safe=(hook_text.replace('\\','').replace(':','').replace("'",'')
                   .replace('"','').replace('%',''))
    if hook_style=='Vertical': safe='\n'.join(safe.split())
    fp=(f"fontfile='{ffmpeg_path(FONT_PATH)}'" if os.path.exists(FONT_PATH) else "font='Arial'")
    return (f"drawtext={fp}:text='{safe}':fontcolor=white:fontsize={hook_size}:"
            f"x=(w-text_w)/2:y=80:box=1:boxcolor=black@0.75:boxborderw=12:borderw=3:bordercolor=black")

def _vf_base(crop_mode):
    return ('crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=720:1280' if crop_mode
            else 'scale=720:-2,pad=720:1280:(ow-iw)/2:(oh-ih)/2:black')

def _render_scenes(video_path, seg_start, total_duration, boundaries,
                   crop_mode, subtitles_on, all_words,
                   subtitle_size, subtitle_position, subtitle_font,
                   hook_enabled, hook_text, hook_style, hook_size,
                   mute_original, uid, scene_dir_prefix):
    scene_dir=os.path.join(OUTPUT_DIR,f'{scene_dir_prefix}_{uid}')
    os.makedirs(scene_dir,exist_ok=True)
    vfb=_vf_base(crop_mode)
    hf=_build_hook_filter(hook_text if hook_enabled else '',hook_style,hook_size)
    af='-an' if mute_original else '-c:a aac -b:a 128k'
    scene_files=[]
    for n in range(len(boundaries)-1):
        t_start=seg_start+boundaries[n]; t_dur=boundaries[n+1]-boundaries[n]
        if t_dur<0.1: continue
        sout=os.path.join(scene_dir,f's{n:04d}.mp4')
        filters=[vfb]
        if hf: filters.append(hf)
        if subtitles_on and all_words:
            srt=create_clip_srt(all_words,t_start,t_dur,f'{uid}_n{n}')
            sp=ffmpeg_path(os.path.abspath(srt))
            filters.append(build_subtitle_filter(sp,subtitle_size,subtitle_position,subtitle_font))
        run_ffmpeg(f'ffmpeg -y -ss {t_start:.3f} -i "{video_path}" -t {t_dur:.3f} '
                   f'-vf "{",".join(filters)}" -c:v libx264 -preset veryfast -crf 23 '
                   f'{af} "{sout}"')
        if os.path.exists(sout): scene_files.append(sout)
    return scene_dir, scene_files

def _concat_scenes(scene_files, output_path, scene_dir):
    if not scene_files: return None
    ctxt=os.path.join(scene_dir,'concat.txt')
    with open(ctxt,'w') as f:
        for sf in scene_files: f.write(f"file '{os.path.abspath(sf)}'\n")
    run_ffmpeg(f'ffmpeg -y -f concat -safe 0 -i "{ctxt}" -c copy "{output_path}"')
    for sf in scene_files:
        try: os.remove(sf)
        except Exception: pass
    try: os.remove(ctxt); os.rmdir(scene_dir)
    except Exception: pass
    return output_path if os.path.exists(output_path) else None


# =============================================================================
# CLIP GENERATION
# =============================================================================

def generate_clip(args):
    (i, score, seg, video_path, video_duration, all_words,
     crop_mode, subtitles_on, clip_duration,
     subtitle_size, subtitle_position, subtitle_font,
     hook_enabled, hook_text, hook_style, hook_size,
     watermark_path, watermark_position, watermark_opacity, watermark_scale,
     mute_original, cut_style, scene_length, film_sensitivity) = args

    uid=f"{i}_{safe_id()}"
    start=max(min(seg['start']-1.5, video_duration-10), 0)
    duration=min(clip_duration, video_duration-start-0.5)
    if duration<=1: return None
    final_out=os.path.join(OUTPUT_DIR,f'clip_{i}.mp4')

    if cut_style=='Rapid-cut' and scene_length>0:
        n=max(int(duration//scene_length),1)
        bounds=[k*scene_length for k in range(n)]+[duration]
        sd,sf=_render_scenes(video_path,start,duration,bounds,
                              crop_mode,subtitles_on,all_words,
                              subtitle_size,subtitle_position,subtitle_font,
                              hook_enabled,hook_text,hook_style,hook_size,
                              mute_original,uid,'rc')
        raw_out=os.path.join(OUTPUT_DIR,f'rc_out_{uid}.mp4')
        if not _concat_scenes(sf,raw_out,sd): return None

    elif cut_style=='Film cut':
        cuts=detect_scene_cuts(video_path,start,duration,film_sensitivity)
        bounds=[0.0]+[t for t in cuts if 0<t<duration]+[duration]
        bounds=sorted(set(bounds))
        sd,sf=_render_scenes(video_path,start,duration,bounds,
                              crop_mode,subtitles_on,all_words,
                              subtitle_size,subtitle_position,subtitle_font,
                              hook_enabled,hook_text,hook_style,hook_size,
                              mute_original,uid,'fc')
        raw_out=os.path.join(OUTPUT_DIR,f'fc_out_{uid}.mp4')
        if not _concat_scenes(sf,raw_out,sd): return None

    else:
        raw_out=os.path.join(OUTPUT_DIR,f'raw_{uid}.mp4')
        filters=[_vf_base(crop_mode)]
        hf=_build_hook_filter(hook_text if hook_enabled else '',hook_style,hook_size)
        if hf: filters.append(hf)
        if subtitles_on:
            srt=create_clip_srt(all_words,start,duration,uid)
            sp=ffmpeg_path(os.path.abspath(srt))
            filters.append(build_subtitle_filter(sp,subtitle_size,subtitle_position,subtitle_font))
        af='-an' if mute_original else '-c:a aac -b:a 128k'
        run_ffmpeg(f'ffmpeg -y -ss {start:.3f} -i "{video_path}" -t {duration:.3f} '
                   f'-vf "{",".join(filters)}" -c:v libx264 -preset veryfast -crf 23 '
                   f'{af} "{raw_out}"')

    if not os.path.exists(raw_out): return None

    wm_out=raw_out
    if watermark_path and os.path.exists(watermark_path):
        wm_tmp=os.path.join(OUTPUT_DIR,f'wm_{uid}.mp4')
        pos=WATERMARK_POSITIONS.get(watermark_position,'overlay=10:10')
        sf2=(f"scale=iw*{watermark_scale}:-1,format=rgba,"
             f"colorchannelmixer=aa={watermark_opacity}")
        run_ffmpeg(f'ffmpeg -y -i "{raw_out}" -i "{watermark_path}" '
                   f'-filter_complex "[1:v]{sf2}[wm];[0:v][wm]{pos}" '
                   f'-c:v libx264 -preset veryfast -crf 23 -c:a copy "{wm_tmp}"')
        if os.path.exists(wm_tmp): wm_out=wm_tmp

    if wm_out!=final_out: os.replace(wm_out,final_out)
    elif raw_out!=final_out and os.path.exists(raw_out): os.replace(raw_out,final_out)
    for tmp in [raw_out]:
        if os.path.exists(tmp) and tmp!=final_out:
            try: os.remove(tmp)
            except Exception: pass

    LAST_TRANSCRIPTS[f'clip_{i}.mp4']=seg.get('text','').strip()
    return final_out if os.path.exists(final_out) else None


# =============================================================================
# MAIN PIPELINE (runs in background thread)
# =============================================================================

def run_pipeline(job_id, video_path, watermark_path, params):
    global LAST_TRANSCRIPTS, LAST_API_KEY

    def update(msg): JOBS[job_id]['message'] = msg

    LAST_TRANSCRIPTS = {}
    LAST_API_KEY     = params.get('api_key','')
    api_key          = params.get('api_key','')

    try:
        update('Getting video duration...')
        video_duration=get_media_duration(video_path)
        if video_duration<=0:
            JOBS[job_id].update({'status':'error','message':'Could not read video.'}); return

        update('Extracting audio...')
        wav_tmp=os.path.join(OUTPUT_DIR,f'audio_{safe_id()}.wav')
        run_ffmpeg(f'ffmpeg -y -i "{video_path}" -t {min(video_duration,3600):.1f} '
                   f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{wav_tmp}"')

        subtitles_on  = params.get('subtitles_on',True)
        gaming_mode   = params.get('gaming_mode',False)
        smart_mode    = params.get('smart_mode',False)
        clip_duration = float(params.get('clip_duration',45))
        max_clips     = int(params.get('max_clips',10))
        platform      = params.get('platform','TikTok')
        niche         = params.get('niche','General')

        if gaming_mode:
            update('Detecting gaming moments...')
            scored_segments=detect_gaming_moments(wav_tmp,max_clips,clip_duration,video_duration)
            if not scored_segments:
                update('No peaks found, falling back to keyword scoring...')
                try:
                    segs=model.transcribe(wav_tmp,word_timestamps=subtitles_on,
                                          fp16=False,language='en',verbose=False)['segments']
                except Exception as e:
                    if os.path.exists(wav_tmp): os.remove(wav_tmp)
                    JOBS[job_id].update({'status':'error','message':f'Transcription failed: {e}'}); return
                scored_segments=score_segments_keywords(segs,max_clips)
            all_words=[]
        else:
            update('Transcribing audio...')
            try:
                result=model.transcribe(wav_tmp,word_timestamps=subtitles_on,
                                        fp16=False,language='en',verbose=False)
                segments=result['segments']
            except Exception as e:
                if os.path.exists(wav_tmp): os.remove(wav_tmp)
                JOBS[job_id].update({'status':'error','message':f'Transcription failed: {e}'}); return

            if not segments:
                if os.path.exists(wav_tmp): os.remove(wav_tmp)
                JOBS[job_id].update({'status':'error','message':'No speech detected.'}); return

            all_words=[]
            for seg in segments: all_words.extend(seg.get('words',[]))

            if smart_mode:
                update('Smart clip analysis...')
                scored_segments=smart_select_segments(
                    segments,all_words,wav_tmp,max_clips,clip_duration,video_duration,
                    api_key,platform,niche)
            else:
                update('Scoring segments...')
                scored_segments=score_segments_ai(segments,max_clips,api_key,platform,niche)

        if os.path.exists(wav_tmp): os.remove(wav_tmp)

        update('Generating hooks...')
        hooks_map={}
        for _,seg in scored_segments:
            key=seg['text'][:50]
            hooks_map[key]=generate_hooks_ai(seg['text'],api_key,platform,niche)

        update('Rendering clips...')
        args_list=[]
        for i,(score,seg) in enumerate(scored_segments):
            key=seg['text'][:50]
            hooks=hooks_map.get(key,generate_hooks_basic(seg['text']))
            args_list.append((
                i,score,seg,video_path,video_duration,all_words,
                params.get('crop_mode',True), subtitles_on,
                clip_duration,
                int(params.get('subtitle_size',34)),
                params.get('subtitle_position','Bottom'),
                params.get('subtitle_font','Bangers'),
                params.get('hook_enabled',True),
                hooks[0] if hooks else '',
                params.get('hook_style','Horizontal'),
                int(params.get('hook_size',72)),
                watermark_path,
                params.get('watermark_position','Bottom Right'),
                float(params.get('watermark_opacity',0.8)),
                float(params.get('watermark_scale',0.12)),
                params.get('mute_original',False),
                params.get('cut_style','Traditional'),
                float(params.get('scene_length',3)),
                float(params.get('film_sensitivity',0.3)),
            ))

        with ThreadPoolExecutor(max_workers=3) as ex:
            results=list(ex.map(generate_clip,args_list))

        clips=[]
        for res,(score,seg) in zip(results,scored_segments):
            if res and os.path.exists(res):
                name=os.path.basename(res)
                clips.append({'name':name,'score':score})
                key=seg['text'][:50]
                LAST_TRANSCRIPTS[name+'_hooks']=hooks_map.get(key,[])

        if not clips:
            JOBS[job_id].update({'status':'error','message':'No clips generated.'}); return

        JOBS[job_id].update({'status':'done','message':f'Done — {len(clips)} clip(s) generated.','clips':clips})

    except Exception as e:
        JOBS[job_id].update({'status':'error','message':f'Pipeline error: {e}'})


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title='Auto Clipper V13')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.mount('/output', StaticFiles(directory=OUTPUT_DIR), name='output')
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get('/api/config')
def get_config():
    return {
        'platforms':  PLATFORM_OPTIONS,
        'niches':     NICHE_OPTIONS,
        'voices':     list(VOICE_OPTIONS.keys()),
        'languages':  list(LANGUAGE_OPTIONS.keys()),
        'music':      list(MUSIC_PRESETS.keys()),
        'fonts':      SUBTITLE_FONTS,
        'wm_pos':     list(WATERMARK_POSITIONS.keys()),
        'sub_pos':    list(SUBTITLE_POSITION_MAP.keys()),
        'api_key_set': bool(LAST_API_KEY),
    }


@app.post('/api/generate')
async def api_generate(
    video:               UploadFile = File(...),
    watermark:           UploadFile = File(None),
    music_upload:        UploadFile = File(None),
    api_key:             str = Form(''),
    platform:            str = Form('TikTok'),
    niche:               str = Form('General'),
    crop_mode:           str = Form('true'),
    subtitles_on:        str = Form('true'),
    hook_enabled:        str = Form('true'),
    mute_original:       str = Form('false'),
    gaming_mode:         str = Form('false'),
    smart_mode:          str = Form('false'),
    clip_duration:       str = Form('45'),
    max_clips:           str = Form('10'),
    subtitle_size:       str = Form('34'),
    subtitle_position:   str = Form('Bottom'),
    subtitle_font:       str = Form('Bangers'),
    hook_style:          str = Form('Horizontal'),
    hook_size:           str = Form('72'),
    watermark_position:  str = Form('Bottom Right'),
    watermark_opacity:   str = Form('0.8'),
    watermark_scale:     str = Form('0.12'),
    cut_style:           str = Form('Traditional'),
    scene_length:        str = Form('3'),
    film_sensitivity:    str = Form('0.3'),
    music_preset:        str = Form('None'),
):
    uid = safe_id()

    video_path = os.path.join(UPLOAD_DIR, f'vid_{uid}_{video.filename}')
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(video.file, f)

    watermark_path = None
    if watermark and watermark.filename:
        watermark_path = os.path.join(UPLOAD_DIR, f'wm_{uid}_{watermark.filename}')
        with open(watermark_path, 'wb') as f:
            shutil.copyfileobj(watermark.file, f)

    music_upload_path = None
    if music_upload and music_upload.filename:
        music_upload_path = os.path.join(UPLOAD_DIR, f'mus_{uid}_{music_upload.filename}')
        with open(music_upload_path, 'wb') as f:
            shutil.copyfileobj(music_upload.file, f)

    def b(s): return s.lower() in ('true','1','yes','on')

    params = {
        'api_key':            api_key,
        'platform':           platform,
        'niche':              niche,
        'crop_mode':          b(crop_mode),
        'subtitles_on':       b(subtitles_on),
        'hook_enabled':       b(hook_enabled),
        'mute_original':      b(mute_original),
        'gaming_mode':        b(gaming_mode),
        'smart_mode':         b(smart_mode),
        'clip_duration':      float(clip_duration),
        'max_clips':          int(max_clips),
        'subtitle_size':      int(subtitle_size),
        'subtitle_position':  subtitle_position,
        'subtitle_font':      subtitle_font,
        'hook_style':         hook_style,
        'hook_size':          int(hook_size),
        'watermark_position': watermark_position,
        'watermark_opacity':  float(watermark_opacity),
        'watermark_scale':    float(watermark_scale),
        'cut_style':          cut_style,
        'scene_length':       float(scene_length),
        'film_sensitivity':   float(film_sensitivity),
        'music_preset':       music_preset,
        'music_upload_path':  music_upload_path,
    }

    job_id = safe_id()
    JOBS[job_id] = {'status': 'running', 'message': 'Starting...', 'clips': []}
    t = threading.Thread(target=run_pipeline, args=(job_id, video_path, watermark_path, params), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get('/api/job/{job_id}')
def api_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


@app.post('/api/voiceover')
async def api_voiceover(
    clip_name:          str = Form(...),
    voice_text:         str = Form(''),
    selected_voice:     str = Form('Female US (Jenny)'),
    selected_language:  str = Form('English'),
    add_subtitles:      str = Form('true'),
    subtitle_size:      str = Form('36'),
    subtitle_position:  str = Form('Bottom'),
    subtitle_font:      str = Form('Bangers'),
    voice_speed:        str = Form('0'),
    voice_pitch:        str = Form('0'),
    music_preset:       str = Form('None'),
    music_volume:       str = Form('0.15'),
    music_upload:       UploadFile = File(None),
    api_key:            str = Form(''),
):
    clip_path = os.path.join(OUTPUT_DIR, clip_name)
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail='Clip not found')

    if not voice_text.strip():
        voice_text = "Check this out — you are not going to believe what happens next."

    uid        = safe_id()
    voice_file = os.path.join(OUTPUT_DIR, f'tts_{uid}.mp3')
    final_out  = os.path.join(OUTPUT_DIR, f'voiced_{uid}_{clip_name}')

    lang_name  = selected_language or 'English'
    voice_name = (LANGUAGE_OPTIONS[lang_name][0] if lang_name != 'English'
                  else VOICE_OPTIONS.get(selected_voice, 'en-US-JennyNeural'))

    try:
        async def gen():
            await edge_tts.Communicate(
                voice_text, voice_name,
                rate=f'{int(voice_speed):+d}%',
                pitch=f'{int(voice_pitch):+d}Hz'
            ).save(voice_file)
        asyncio.run(gen())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'TTS failed: {e}')

    if not os.path.exists(voice_file):
        raise HTTPException(status_code=500, detail='TTS file not created')

    audio_dur = get_media_duration(voice_file)
    clip_dur  = get_media_duration(clip_path)
    cmd = ['ffmpeg', '-y', '-i', clip_path, '-i', voice_file, '-map', '0:v:0', '-map', '1:a:0']

    if voice_text and add_subtitles.lower() in ('true','1'):
        srt_file = create_script_srt(voice_text, audio_dur, uid)
        if lang_name != 'English' and srt_file:
            srt_file = translate_srt(srt_file, lang_name, api_key)
        if srt_file and os.path.exists(srt_file):
            sp    = ffmpeg_path(os.path.abspath(srt_file))
            sub_f = build_subtitle_filter(sp, int(subtitle_size), subtitle_position, subtitle_font)
            cmd  += ['-vf', sub_f, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
        else:
            cmd += ['-c:v', 'copy']
    else:
        cmd += ['-c:v', 'copy']

    cmd += ['-c:a', 'aac', '-b:a', '128k', '-shortest', final_out]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'FFmpeg error: {e.stderr.decode()[:200]}')
    finally:
        if os.path.exists(voice_file): os.remove(voice_file)

    if not os.path.exists(final_out):
        raise HTTPException(status_code=500, detail='Output not created')

    music_upload_path = None
    if music_upload and music_upload.filename:
        music_upload_path = os.path.join(UPLOAD_DIR, f'mus_{uid}_{music_upload.filename}')
        with open(music_upload_path, 'wb') as f:
            shutil.copyfileobj(music_upload.file, f)

    music_path = resolve_music_path(music_preset, music_upload_path)
    if music_path:
        mx_out = os.path.join(OUTPUT_DIR, f'mx_{uid}_{clip_name}')
        run_ffmpeg(
            f'ffmpeg -y -i "{final_out}" -stream_loop -1 -i "{music_path}" '
            f'-filter_complex "[1:a]volume={music_volume},atrim=0:{clip_dur:.3f},'
            f'asetpts=PTS-STARTPTS[bg];[0:a][bg]amix=inputs=2:duration=first:'
            f'weights=1 {music_volume}" -c:v copy -c:a aac -b:a 128k -shortest "{mx_out}"'
        )
        if os.path.exists(mx_out): os.replace(mx_out, final_out)

    return {'filename': os.path.basename(final_out)}


@app.post('/api/captions')
async def api_captions(
    clip_name: str = Form(...),
    platform:  str = Form('TikTok'),
    niche:     str = Form('General'),
    api_key:   str = Form(''),
):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail='API key required')
    transcript = LAST_TRANSCRIPTS.get(clip_name, '')
    hooks      = LAST_TRANSCRIPTS.get(clip_name + '_hooks', [])
    lh = {'TikTok':'under 150 chars, 5-8 hashtags','YouTube Shorts':'under 100 chars, 3-5 hashtags'}.get(platform,'under 150 chars')
    system = (f"You are a {platform} content strategist in {niche}. "
              f"Write a viral caption. {lh}. Hook first, hashtags on a new line. "
              "Return ONLY: caption, blank line, hashtags.")
    result = claude_call(api_key, system,
                         f"Transcript: {transcript[:400]}\nHook: {hooks[0] if hooks else ''}",
                         300)
    if not result:
        raise HTTPException(status_code=500, detail='Could not generate caption')
    return {'caption': result}


@app.get('/api/hooks/{clip_name}')
def api_hooks(clip_name: str):
    hooks = LAST_TRANSCRIPTS.get(clip_name + '_hooks', [])
    return {'hooks': hooks}


@app.get('/api/export/zip')
def api_export_zip():
    clips = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
             if f.startswith('clip_') and f.endswith('.mp4')]
    if not clips:
        raise HTTPException(status_code=404, detail='No clips to export')
    zip_path = os.path.join(OUTPUT_DIR, 'clips_export.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for c in clips:
            if os.path.exists(c): zf.write(c, os.path.basename(c))
    return FileResponse(zip_path, filename='clips_export.zip')


@app.get('/api/script/{clip_name}')
def api_auto_script(clip_name: str, platform: str = 'TikTok', niche: str = 'General', api_key: str = ''):
    if not api_key.strip():
        raise HTTPException(status_code=400, detail='API key required')
    transcript = LAST_TRANSCRIPTS.get(clip_name, '')
    if not transcript:
        raise HTTPException(status_code=404, detail='No transcript found')
    system = (f"Viral {platform} scriptwriter in {niche}. "
              "Punchy voiceover script, 40-80 words, hook in 3s, strong CTA. "
              "Return ONLY the script text.")
    result = claude_call(api_key, system, f"Transcript: {transcript}", 250)
    if not result:
        raise HTTPException(status_code=500, detail='Could not generate script')
    return {'script': result}


# =============================================================================
# YOUTUBE DOWNLOAD
# =============================================================================

YT_QUALITY_MAP = {
    '720p':       'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
    '480p':       'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]',
    '1080p':      'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]',
    'audio_only': 'bestaudio[ext=m4a]/bestaudio',
}


def download_youtube(url: str, quality: str = '720p') -> dict:
    """
    Download a YouTube video using yt-dlp.
    Returns dict with keys: path, title, duration_secs, filename.
    Raises ValueError on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ValueError('yt-dlp is not installed. Run: pip install yt-dlp')

    uid      = safe_id()
    out_tmpl = os.path.join(UPLOAD_DIR, f'yt_{uid}.%(ext)s')
    fmt      = YT_QUALITY_MAP.get(quality, YT_QUALITY_MAP['720p'])

    ydl_opts = {
        'format':            fmt,
        'outtmpl':           out_tmpl,
        'merge_output_format': 'mp4',
        'quiet':             True,
        'no_warnings':       True,
        'noplaylist':        True,          # single video only
        'postprocessors': [{
            'key':            'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    # Audio-only: save as mp3
    if quality == 'audio_only':
        ydl_opts['postprocessors'] = [{
            'key':             'FFmpegExtractAudio',
            'preferredcodec':  'mp3',
            'preferredquality': '192',
        }]
        ydl_opts.pop('merge_output_format', None)

    info = {}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise ValueError(f'Download failed: {e}')

    # Find the output file (yt-dlp writes the final ext)
    ext      = 'mp3' if quality == 'audio_only' else 'mp4'
    out_path = os.path.join(UPLOAD_DIR, f'yt_{uid}.{ext}')

    if not os.path.exists(out_path):
        # Fallback: search for any file matching the uid
        matches = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(f'yt_{uid}')]
        if matches:
            out_path = os.path.join(UPLOAD_DIR, matches[0])
        else:
            raise ValueError('Downloaded file not found after yt-dlp completed')

    title    = info.get('title', 'YouTube video')
    duration = info.get('duration', 0) or 0

    return {
        'path':          out_path,
        'filename':      os.path.basename(out_path),
        'title':         title,
        'duration_secs': int(duration),
    }


@app.post('/api/yt-download')
async def api_yt_download(
    url:          str = Form(...),
    quality:      str = Form('720p'),
    auto_generate: str = Form('false'),
    # All generate params (only used when auto_generate=true)
    api_key:             str = Form(''),
    platform:            str = Form('TikTok'),
    niche:               str = Form('General'),
    crop_mode:           str = Form('true'),
    subtitles_on:        str = Form('true'),
    hook_enabled:        str = Form('true'),
    mute_original:       str = Form('false'),
    gaming_mode:         str = Form('false'),
    smart_mode:          str = Form('false'),
    clip_duration:       str = Form('45'),
    max_clips:           str = Form('10'),
    subtitle_size:       str = Form('34'),
    subtitle_position:   str = Form('Bottom'),
    subtitle_font:       str = Form('Bangers'),
    hook_style:          str = Form('Horizontal'),
    hook_size:           str = Form('72'),
    watermark_position:  str = Form('Bottom Right'),
    watermark_opacity:   str = Form('0.8'),
    watermark_scale:     str = Form('0.12'),
    cut_style:           str = Form('Traditional'),
    scene_length:        str = Form('3'),
    film_sensitivity:    str = Form('0.3'),
    music_preset:        str = Form('None'),
):
    # Run yt-dlp in a thread so we don't block the event loop
    import concurrent.futures
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: download_youtube(url.strip(), quality)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = {
        'filename':      result['filename'],
        'title':         result['title'],
        'duration_secs': result['duration_secs'],
        'path':          result['path'],
        'job_id':        None,
    }

    # Optionally kick off the clip pipeline immediately
    def b(s): return s.lower() in ('true', '1', 'yes', 'on')
    if b(auto_generate):
        params = {
            'api_key':            api_key,
            'platform':           platform,
            'niche':              niche,
            'crop_mode':          b(crop_mode),
            'subtitles_on':       b(subtitles_on),
            'hook_enabled':       b(hook_enabled),
            'mute_original':      b(mute_original),
            'gaming_mode':        b(gaming_mode),
            'smart_mode':         b(smart_mode),
            'clip_duration':      float(clip_duration),
            'max_clips':          int(max_clips),
            'subtitle_size':      int(subtitle_size),
            'subtitle_position':  subtitle_position,
            'subtitle_font':      subtitle_font,
            'hook_style':         hook_style,
            'hook_size':          int(hook_size),
            'watermark_position': watermark_position,
            'watermark_opacity':  float(watermark_opacity),
            'watermark_scale':    float(watermark_scale),
            'cut_style':          cut_style,
            'scene_length':       float(scene_length),
            'film_sensitivity':   float(film_sensitivity),
            'music_preset':       music_preset,
            'music_upload_path':  None,
        }
        job_id = safe_id()
        JOBS[job_id] = {'status': 'running', 'message': 'Starting pipeline...', 'clips': []}
        t = threading.Thread(
            target=run_pipeline,
            args=(job_id, result['path'], None, params),
            daemon=True,
        )
        t.start()
        response['job_id'] = job_id

    return response


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import uvicorn
    print('Auto Clipper V13 Studio')
    print(f'  Output  : {os.path.abspath(OUTPUT_DIR)}')
    print(f'  UI      : http://localhost:8000')
    print(f'  Music   : place lofi/upbeat/cinematic/chill.mp3 in {MUSIC_DIR}/')
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
