import base64
import json
import os
import re
import uuid
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


def save_json(filename: str, data: list) -> None:
    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_claude_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------- Models ----------

class GenerateRequest(BaseModel):
    resume: str
    job_id: str
    template_id: str


class ExtractResumeRequest(BaseModel):
    image_base64: str  # data URL (data:image/png;base64,...)


class JobCreate(BaseModel):
    title: str
    company: str
    description: str


class JobUpdate(BaseModel):
    title: str
    company: str
    description: str


class TemplateCreate(BaseModel):
    job_id: str
    name: str
    subject: str
    body: str


class TemplateUpdate(BaseModel):
    job_id: str
    name: str
    subject: str
    body: str


# ---------- Resume extraction ----------

@app.post("/api/extract-resume")
def extract_resume(req: ExtractResumeRequest) -> dict:
    client = get_claude_client()

    # Strip data URL prefix to get raw base64
    raw = req.image_base64
    if "," in raw:
        media_type_part, raw = raw.split(",", 1)
        media_type = media_type_part.split(":")[1].split(";")[0]
    else:
        media_type = "image/png"

    is_pdf = media_type == "application/pdf"
    extract_text = (
        "このPDFはビズリーチの候補者レジュメです。"
        if is_pdf else
        "この画像はビズリーチの候補者レジュメページです。"
    ) + (
        "候補者の情報（氏名、職歴、スキル、学歴、資格など）をすべて読み取り、"
        "構造化されたテキストとして出力してください。"
        "テキストをできるだけ忠実に抽出し、レジュメとして読みやすい形式でまとめてください。"
        "余計な説明は不要です。抽出したレジュメテキストだけを出力してください。"
    )

    source_block = {
        "type": "base64",
        "media_type": media_type,
        "data": raw,
    }

    content_block = (
        {"type": "document", "source": source_block}
        if is_pdf else
        {"type": "image", "source": source_block}
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    content_block,
                    {"type": "text", "text": extract_text},
                ],
            }
        ],
    )

    return {"resume_text": message.content[0].text.strip()}


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
    # Cascade: remove linked templates
    templates = load_json("templates.json")
    save_json("templates.json", [t for t in templates if t["job_id"] != job_id])


# ---------- Templates CRUD ----------

@app.get("/api/templates")
def get_all_templates() -> list:
    return load_json("templates.json")


@app.get("/api/templates/{job_id}")
def get_templates_for_job(job_id: str) -> list:
    templates = load_json("templates.json")
    matched = [t for t in templates if t["job_id"] == job_id]
    if not matched:
        raise HTTPException(status_code=404, detail="No templates found for this job")
    return matched


@app.post("/api/templates", status_code=201)
def create_template(body: TemplateCreate) -> dict:
    jobs = load_json("jobs.json")
    if not any(j["id"] == body.job_id for j in jobs):
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


# ---------- Scout message generation ----------

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

    client = get_claude_client()

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


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
def index():
    with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def admin():
    with open(STATIC_DIR / "admin.html", encoding="utf-8") as f:
        return f.read()
