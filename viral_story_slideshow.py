#!/usr/bin/env python3
"""
viral_story_slideshow.py
Generate Shorts-ready slideshows from today’s viral niches.
pip install google-generativeai yt-dlp pandas requests opencv-python scenedetect pillow tqdm openai python-dotenv
.env:
GOOGLE_API_KEY=YOUR_GEMINI_KEY
OPENAI_API_KEY=sk-YOUR_DALLE_KEY
"""
import argparse, os, json, textwrap, requests, io, shutil
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
from scenedetect import open_video, SceneManager, ContentDetector
from PIL import Image
from yt_dlp import YoutubeDL
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI = genai.GenerativeModel("gemini-1.5-flash")
DALLE = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_URL = "https://datasets.1of10.co/outliers_daily.csv"

NICHE_KEYWORDS = {
    # Storytelling
    "MiniDoc": ["story","true story","mini doc","documentary","real story"],
    "POV": ["pov","day in the life","skit","roleplay","sketch"],
    "PlotTwist": ["plot twist","ai generated","twist ending","short film"],
    "DarkMystery": ["mystery","dark psychology","creepy pasta","scary"],
    "MicroHistory": ["history","lost in time","then vs now","forgotten"],
    "VoiceNote": ["voice note","confession","secret","voicemail"],
    "NPCComedy": ["npc","street comedy","green screen","sketch"],
    # Animals
    "Cats": ["cat","kitten","meow","funny cat","cute cat"],
    "Dogs": ["dog","puppy","doggo","funny dog"],
    "Wildlife": ["wildlife","animal rescue","lion","elephant"],
    "Farm": ["farm","cow","pig","goat","chicken"],
    # Tech
    "Gadgets": ["iphone","android","gadget","unboxing","tech"],
    "AI": ["ai art","ai tool","chatgpt","midjourney"],
    "Crypto": ["crypto","bitcoin","nft","altcoin","web3"],
    "VR": ["vr","metaverse","oculus","virtual reality"],
    # Money
    "MoneyHack": ["side hustle","make money","passive income","earn online"],
    "Budget": ["budget","save money","frugal","debt payoff"],
    "Investing": ["investing","stock market","dividends","roth ira"],
    # Fitness & Health
    "Gym": ["gym","workout","muscle","fat loss","bodybuilding"],
    "Yoga": ["yoga","stretch","meditation","mindfulness"],
    "Recipe": ["recipe","meal prep","air fryer","healthy food"],
    # Beauty & Fashion
    "Makeup": ["makeup","tutorial","beauty hack","cosmetic"],
    "Skincare": ["skincare","routine","anti aging","dermatologist"],
    "Streetwear": ["streetwear","outfit","sneaker","haul"],
    "Thrift": ["thrift flip","thrift haul","diy fashion"],
    # Gaming & Pop
    "Gaming": ["gaming","fortnite","minecraft","valorant"],
    "Anime": ["anime","manga","one piece","naruto"],
    "Meme": ["meme","dank meme","trend","viral meme"],
    # Life-Hacks
    "LifeHack": ["life hack","hack","diy","household trick"],
    "ASMR": ["asmr","satisfying","crunch","oddly satisfying"],
    "Cleaning": ["cleaning","speed clean","before after clean"],
    # Travel & Luxury
    "Travel": ["travel","airbnb","budget travel","passport"],
    "Luxury": ["luxury","luxury lifestyle","rich life","supercar"],
    # Education
    "Fact": ["fact","did you know","mind blown","random fact"],
    "Exam": ["study tips","exam hack","student life","motivation"],
}

# ---------- helpers ----------
def fetch_outliers(limit=2000):
    df = pd.read_csv(CSV_URL).head(limit)
    df["uploadDate"] = pd.to_datetime(df["uploadDate"])
    return df

def detect_hot_niches(df, top_k=5):
    scores = {n: df["title"].str.contains("|".join(kws), case=False, na=False).sum()
              for n, kws in NICHE_KEYWORDS.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def dl_video(video_id, path):
    ydl_opts = {"outtmpl": path, "format": "best[height<=720]"}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

def scene_split(video_path):
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]

def extract_frames(video_path, scenes, out_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i, (start, _) in enumerate(scenes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        ret, frame = cap.read()
        if ret:
            img_path = f"{out_dir}/frame_{i:02d}.jpg"
            cv2.imwrite(img_path, frame)
            frames.append(img_path)
    cap.release()
    return frames

def get_transcript(video_id):
    ydl_opts = {"skip_download": True, "writeautomaticsub": True, "sub_lang": "en"}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            sub_file = info.get("requested_subtitles", {}).get("en", {}).get("filepath")
            if sub_file:
                with open(sub_file) as f:
                    return f.read()
    except Exception:
        pass
    return ""

def ai_script_per_image(transcript, frames, niche):
    prompt = f"""
You are a viral slideshow Shorts creator for the niche "{niche}".
Transcript (trimmed):
{transcript[:2000]}

Break it into {len(frames)} image cards.
Return JSON list: [{{"vo":"≤12 word voice-over","caption":"≤5 word on-screen text"}}, ...]
"""
    res = GEMINI.generate_content(prompt).text
    try:
        return json.loads(res.strip().strip("```json").strip("```"))
    except:
        return [{"vo": "Wow!", "caption": "Nice"} for _ in frames]

def gen_dalle_images(prompts, out_dir):
    thumbs = []
    for i, p in enumerate(prompts):
        prompt = f"Viral 9:16 YouTube Shorts frame, bold text overlay, {p}"
        res = DALLE.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",
            quality="standard",
            n=1
        )
        url = res.data[0].url
        img = Image.open(io.BytesIO(requests.get(url, timeout=30).content))
        path = f"{out_dir}/ai_img_{i:02d}.png"
        img.save(path)
        thumbs.append(path)
    return thumbs

# ---------- main ----------
def run(top_niches=3, vids_per_niche=2):
    os.makedirs("output", exist_ok=True)
    df = fetch_outliers(limit=2000)
    hot = detect_hot_niches(df, top_niches)
    print("🔥 Hottest niches today:", [n for n, _ in hot])

    for niche, _ in hot:
        print(f"\n📂 {niche}")
        mask = df["title"].str.contains("|".join(NICHE_KEYWORDS[niche]), case=False, na=False)
        vids = df[mask].sort_values("viewVelocity", ascending=False).head(vids_per_niche)

        for _, vid in tqdm(vids.iterrows(), total=len(vids)):
            vid_dir = f"output/{niche}/{vid['videoId']}"
            os.makedirs(vid_dir, exist_ok=True)

            video_path = f"{vid_dir}/orig.mp4"
            dl_video(vid["videoId"], video_path)

            scenes = scene_split(video_path)
            frames = extract_frames(video_path, scenes, vid_dir)

            transcript = get_transcript(vid["videoId"])
            script = ai_script_per_image(transcript, frames, niche)

            prompts = [s["vo"] for s in script]
            ai_imgs = gen_dalle_images(prompts, vid_dir)

            story = pd.DataFrame([{"frame": f, "ai_img": a, **s}
                                  for f, a, s in zip(frames, ai_imgs, script)])
            story.to_csv(f"{vid_dir}/storyboard.csv", index=False)

            with open(f"{vid_dir}/slideshow.md", "w") as f:
                for row in story.itertuples():
                    f.write(f"![{row.caption}]({os.path.basename(row.ai_img)})\n")
                    f.write(f"**VO:** {row.vo}\n\n---\n\n")
    print("✅ Done – check output/")

# ---------- entry ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_niches", type=int, default=3)
    ap.add_argument("--vids_per_niche", type=int, default=2)
    args = ap.parse_args()
    run(args.top_niches, args.vids_per_niche)
