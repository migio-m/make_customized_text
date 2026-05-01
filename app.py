import json
import os
from pathlib import Path

import anthropic
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


def load_json(filename: str) -> list:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


class GenerateRequest(BaseModel):
    resume: str
    job_id: str
    template_id: str


class Job(BaseModel):
    id: str
    title: str
    company: str
    description: str


class Template(BaseModel):
    id: str
    job_id: str
    name: str
    subject: str
    body: str


@app.get("/api/jobs")
def get_jobs() -> list[Job]:
    return load_json("jobs.json")


@app.get("/api/templates/{job_id}")
def get_templates_for_job(job_id: str) -> list[Template]:
    templates = load_json("templates.json")
    matched = [t for t in templates if t["job_id"] == job_id]
    if not matched:
        raise HTTPException(status_code=404, detail="No templates found for this job")
    return matched


@app.post("/api/generate")
def generate_scout_message(req: GenerateRequest) -> dict:
    jobs = load_json("jobs.json")
    templates = load_json("templates.json")

    job = next((j for j in jobs if j["id"] == req.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    template = next((t for t in templates if t["id"] == req.template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""あなたはビズリーチのスカウト文作成の専門家です。
以下の情報をもとに、候補者に刺さる個別最適化されたスカウト文（件名と本文）を作成してください。

## 候補者レジュメ
{req.resume}

## 求人票
タイトル: {job['title']}
会社名: {job['company']}
詳細:
{job['description']}

## スカウト文の雛形
件名: {template['subject']}
本文:
{template['body']}

## 指示
1. 雛形の構成・トーンを維持しながら、候補者のレジュメから読み取れる具体的なスキル・経験・実績を盛り込んでカスタマイズしてください
2. プレースホルダー（{{candidate_name}}, {{highlight_point}}, {{skill_match}}, {{company_appeal}}）を適切な内容に置き換えてください
3. {{candidate_name}} はレジュメから氏名を読み取り、不明な場合は「ご担当者様」としてください
4. 候補者の強みを具体的に言及し、「なぜこの方にスカウトするのか」が伝わる文章にしてください
5. 自然な日本語ビジネス文書として仕上げてください

以下のJSON形式で出力してください（コードブロック不要）:
{{"subject": "件名テキスト", "body": "本文テキスト"}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    content = message.content[0].text.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            result = json.loads(match.group())
        else:
            raise HTTPException(status_code=500, detail="Failed to parse Claude response")

    return {
        "subject": result.get("subject", ""),
        "body": result.get("body", ""),
        "job_title": job["title"],
        "company": job["company"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
        return f.read()
