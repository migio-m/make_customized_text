import json
import os
import re
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Scout Message Generator")

DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MEDIA_LABELS = {
    "bizreach": "ビズリーチ",
    "linkedin": "LinkedIn",
    "wantedly": "Wantedly",
    "green": "Green",
    "doda": "doda",
    "other": "その他",
}


def load_json(filename: str) -> list:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(filename: str, data: list) -> None:
    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def gemini_generate(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
    if res.status_code == 400:
        raise HTTPException(status_code=400, detail=f"APIエラー: {res.json().get('error', {}).get('message', res.text)}")
    if res.status_code == 401 or res.status_code == 403:
        raise HTTPException(status_code=401, detail="APIキーが無効です。.envファイルのGOOGLE_API_KEYを確認してください。")
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ---------- Models ----------

class GenerateRequest(BaseModel):
    resume: str
    job_id: str
    template_id: str
    media: str


class JobCreate(BaseModel):
    title: str
    company: str
    description: str


class JobUpdate(BaseModel):
    title: str
    company: str
    description: str


class ChallengeCreate(BaseModel):
    job_id: str
    content: str


class ChallengeUpdate(BaseModel):
    job_id: str
    content: str


class TemplateCreate(BaseModel):
    job_id: str
    media: str
    name: str
    subject: str
    body: str


class TemplateUpdate(BaseModel):
    job_id: str
    media: str
    name: str
    subject: str
    body: str


# ---------- Jobs CRUD ----------

@app.get("/api/jobs")
def get_jobs() -> list:
    return load_json("jobs.json")


@app.post("/api/jobs", status_code=201)
def create_job(body: JobCreate) -> dict:
    jobs = load_json("jobs.json")
    new_job = {"id": f"job_{uuid.uuid4().hex[:8]}", **body.model_dump()}
    jobs.append(new_job)
    save_json("jobs.json", jobs)
    return new_job


@app.put("/api/jobs/{job_id}")
def update_job(job_id: str, body: JobUpdate) -> dict:
    jobs = load_json("jobs.json")
    idx = next((i for i, j in enumerate(jobs) if j["id"] == job_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs[idx] = {"id": job_id, **body.model_dump()}
    save_json("jobs.json", jobs)
    return jobs[idx]


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    jobs = load_json("jobs.json")
    new_jobs = [j for j in jobs if j["id"] != job_id]
    if len(new_jobs) == len(jobs):
        raise HTTPException(status_code=404, detail="Job not found")
    save_json("jobs.json", new_jobs)
    save_json("templates.json", [t for t in load_json("templates.json") if t["job_id"] != job_id])
    save_json("challenges.json", [c for c in load_json("challenges.json") if c["job_id"] != job_id])


# ---------- Challenges CRUD ----------

@app.get("/api/challenges")
def get_all_challenges() -> list:
    return load_json("challenges.json")


@app.get("/api/challenges/{job_id}")
def get_challenges_for_job(job_id: str) -> list:
    return [c for c in load_json("challenges.json") if c["job_id"] == job_id]


@app.post("/api/challenges", status_code=201)
def create_challenge(body: ChallengeCreate) -> dict:
    if not any(j["id"] == body.job_id for j in load_json("jobs.json")):
        raise HTTPException(status_code=404, detail="Job not found")
    challenges = load_json("challenges.json")
    new_c = {"id": f"chl_{uuid.uuid4().hex[:8]}", **body.model_dump()}
    challenges.append(new_c)
    save_json("challenges.json", challenges)
    return new_c


@app.put("/api/challenges/{chl_id}")
def update_challenge(chl_id: str, body: ChallengeUpdate) -> dict:
    challenges = load_json("challenges.json")
    idx = next((i for i, c in enumerate(challenges) if c["id"] == chl_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    challenges[idx] = {"id": chl_id, **body.model_dump()}
    save_json("challenges.json", challenges)
    return challenges[idx]


@app.delete("/api/challenges/{chl_id}", status_code=204)
def delete_challenge(chl_id: str) -> None:
    challenges = load_json("challenges.json")
    new_c = [c for c in challenges if c["id"] != chl_id]
    if len(new_c) == len(challenges):
        raise HTTPException(status_code=404, detail="Challenge not found")
    save_json("challenges.json", new_c)


# ---------- Templates CRUD ----------

@app.get("/api/templates")
def get_all_templates() -> list:
    return load_json("templates.json")


@app.get("/api/templates/{job_id}/{media}")
def get_templates_for_job_media(job_id: str, media: str) -> list:
    templates = [
        t for t in load_json("templates.json")
        if t["job_id"] == job_id and t.get("media") == media
    ]
    if not templates:
        raise HTTPException(status_code=404, detail="No templates found")
    return templates


@app.post("/api/templates", status_code=201)
def create_template(body: TemplateCreate) -> dict:
    if not any(j["id"] == body.job_id for j in load_json("jobs.json")):
        raise HTTPException(status_code=404, detail="Job not found")
    templates = load_json("templates.json")
    new_tpl = {"id": f"tpl_{uuid.uuid4().hex[:8]}", **body.model_dump()}
    templates.append(new_tpl)
    save_json("templates.json", templates)
    return new_tpl


@app.put("/api/templates/{tpl_id}")
def update_template(tpl_id: str, body: TemplateUpdate) -> dict:
    templates = load_json("templates.json")
    idx = next((i for i, t in enumerate(templates) if t["id"] == tpl_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Template not found")
    templates[idx] = {"id": tpl_id, **body.model_dump()}
    save_json("templates.json", templates)
    return templates[idx]


@app.delete("/api/templates/{tpl_id}", status_code=204)
def delete_template(tpl_id: str) -> None:
    templates = load_json("templates.json")
    new_tpl = [t for t in templates if t["id"] != tpl_id]
    if len(new_tpl) == len(templates):
        raise HTTPException(status_code=404, detail="Template not found")
    save_json("templates.json", new_tpl)


# ---------- Media list ----------

@app.get("/api/media")
def get_media() -> list:
    return [{"value": k, "label": v} for k, v in MEDIA_LABELS.items()]


# ---------- Scout message generation ----------

@app.post("/api/generate")
def generate_scout_message(req: GenerateRequest) -> dict:
    jobs = load_json("jobs.json")
    templates = load_json("templates.json")
    challenges = load_json("challenges.json")

    job = next((j for j in jobs if j["id"] == req.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    template = next((t for t in templates if t["id"] == req.template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    job_challenges = [c for c in challenges if c["job_id"] == req.job_id]
    challenge_text = "\n".join(c["content"] for c in job_challenges) if job_challenges else "（課題情報なし）"

    media_label = MEDIA_LABELS.get(req.media, req.media)

    prompt = f"""あなたはビズリーチのスカウト文作成の専門家です。
以下の情報をもとに、候補者に刺さる個別最適化されたスカウト文（件名と本文）を作成してください。

## 候補者レジュメ
{req.resume}

## 求人票
タイトル: {job['title']}
会社名: {job['company']}
詳細:
{job['description']}

## ポジション課題・採用背景
{challenge_text}

## スカウト文の雛形
件名: {template['subject']}
本文:
{template['body']}

## 送付媒体
{media_label}

## 指示
1. 雛形の構成・トーンを維持しながら、候補者のレジュメから読み取れる具体的なスキル・経験・実績を盛り込んでカスタマイズしてください
2. プレースホルダー（{{candidate_name}}, {{highlight_point}}, {{skill_match}}, {{company_appeal}}）を適切な内容に置き換えてください
3. {{candidate_name}} はレジュメから氏名を読み取り、不明な場合は「ご担当者様」としてください
4. ポジション課題・採用背景を踏まえ、「なぜ今このポジションが必要か」を自然に盛り込んでください
5. 媒体（{media_label}）の文化・文字数感覚に合わせた文体にしてください（例: LinkedInは英語混じりでグローバル感、Wantedlyはカジュアルで想いを重視）
6. 自然な日本語ビジネス文書として仕上げてください

以下のJSON形式で出力してください（コードブロック不要）:
{{"subject": "件名テキスト", "body": "本文テキスト"}}"""

    try:
        content = gemini_generate(prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"APIエラー: {str(e)}")

    # Remove markdown code fences if present
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    result = None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*?\}(?=\s*$)', content)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass
    if result is None:
        raise HTTPException(status_code=500, detail=f"Claude応答のJSON解析に失敗しました。応答: {content[:200]}")

    return {
        "subject": result.get("subject", ""),
        "body": result.get("body", ""),
        "job_title": job["title"],
        "company": job["company"],
        "media": media_label,
    }


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
def index():
    with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def admin():
    with open(STATIC_DIR / "admin.html", encoding="utf-8") as f:
        return f.read()
