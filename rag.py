import json
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq, APIStatusError, APIConnectionError

load_dotenv()  # loads GROQ_API_KEY from .env

# Tried in order. Check console.groq.com/docs/models for current names.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Groq()  # reads GROQ_API_KEY automatically


def retrieve(question, k=4, file_filter=None):
    """
    Retrieve top-k chunks matching `question`.
    Optionally filter by `file_filter` (single filename string or list of filenames).
    """
    client = chromadb.PersistentClient(path="db")
    try:
        col = client.get_collection("notes")
    except Exception:
        return []

    q = model.encode([question]).tolist()

    where_clause = None
    if file_filter:
        if isinstance(file_filter, str):
            where_clause = {"source": file_filter}
        elif isinstance(file_filter, (list, tuple, set)):
            files = list(file_filter)
            if len(files) == 1:
                where_clause = {"source": files[0]}
            elif len(files) > 1:
                where_clause = {"source": {"$in": files}}

    try:
        r = col.query(query_embeddings=q, n_results=k, where=where_clause)
    except Exception:
        # Fallback without where clause if filter query fails
        r = col.query(query_embeddings=q, n_results=k)

    if not r or not r.get("documents") or not r["documents"][0]:
        return []

    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


def generate(prompt):
    """
    Generate completion using fallback Groq models.
    Catches errors without exposing API keys.
    """
    last_error = None
    for model_name in MODELS:
        for attempt in range(3):
            try:
                resp = llm.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content
            except APIConnectionError as e:
                message = str(e)
                last_error = f"{model_name}: {message}"
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except APIStatusError as e:
                status_code = e.status_code
                message = e.message
                last_error = f"{model_name} (status {status_code}): {message}"
                if status_code == 429 or (500 <= status_code < 600):
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                break  # e.g. 404/400: try the next model
            except Exception as e:
                message = str(e)
                last_error = f"{model_name}: {message}"
                break
    return f"AI service failed. {last_error or 'Please verify your Groq API configuration.'}"


def rewrite_query(question, history=None):
    """
    Rewrite follow-up questions into standalone queries using the last 4 messages.
    If no history or question is already standalone, returns the original question.
    """
    if not history or len(history) == 0:
        return question

    recent = history[-4:]
    history_lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        # Truncate content to keep prompt compact and focused
        content_snippet = content[:250] + ("..." if len(content) > 250 else "")
        history_lines.append(f"{role}: {content_snippet}")

    history_text = "\n".join(history_lines)

    prompt = f"""You are an expert search assistant.
Given the recent conversation history and a user's follow-up question, rewrite the follow-up question into a single standalone, self-contained search query suitable for semantic vector retrieval over study materials.

Guidelines:
- Retain all relevant technical context, subject matter, and entities from the conversation.
- Do NOT answer the question.
- Do NOT include any introductory or explanatory text.
- If the question is already fully self-contained, return it as-is.
- Output ONLY the standalone rewritten search query.

Conversation History:
{history_text}

Follow-up Question: {question}

Standalone Query:"""

    try:
        rewritten = generate(prompt).strip()
        # Sanity check: verify rewritten query is valid text
        if not rewritten or rewritten.startswith("AI service failed") or len(rewritten) > 300:
            return question
        # Strip potential wrapping quotes
        rewritten = rewritten.strip('"\'`')
        return rewritten
    except Exception:
        return question


def answer(question, retrieval_query=None, file_filter=None):
    """
    Answer the question using retrieved context.
    Optionally accepts a separate `retrieval_query` (e.g. standalone query) and `file_filter`.
    """
    search_query = retrieval_query or question
    hits = retrieve(search_query, k=4, file_filter=file_filter)
    if not hits:
        return "I couldn't find any relevant information in your notes for this question.", []

    context = "\n\n".join(
        f"[{m['source']}, p.{m['page']}]\n{doc}" for doc, m, _ in hits
    )
    prompt = f"""You are a study assistant. Answer the question using ONLY the context below.
Cite sources like [file, p.X]. If the context does not contain the answer,
say "I couldn't find this in your notes."

Context:
{context}

Question: {question}"""

    return generate(prompt), hits


def generate_quiz(topic=None, file_filter=None, num_questions=5):
    """
    Generate multiple-choice questions with answers and explanations from retrieved chunks.
    Returns list of dicts: [{"question": str, "options": [str, str, str, str], "answer": int, "explanation": str}]
    """
    query = topic.strip() if topic and topic.strip() else "key concepts, definitions, core principles, exam questions"
    hits = retrieve(query, k=max(6, num_questions + 2), file_filter=file_filter)

    if not hits and file_filter:
        # Fallback to general retrieval if filter was too restrictive
        hits = retrieve(query, k=max(6, num_questions + 2))

    if not hits:
        raise ValueError("No study notes found to generate quiz questions from. Please verify your documents and index.")

    context = "\n\n".join(
        f"[{m['source']}, p.{m['page']}]\n{doc}" for doc, m, _ in hits
    )

    prompt = f"""You are an expert exam creator.
Based ONLY on the provided study context, generate exactly {num_questions} high-quality Multiple Choice Questions (MCQs).

Context:
{context}

Requirements:
- Each question must have exactly 4 options.
- Exactly 1 option must be correct.
- Provide a clear explanation citing the source and page, e.g. [source.pdf, p.X].
- Return strictly a valid JSON array of objects. Do not include markdown or text outside the JSON.

Expected JSON schema:
[
  {{
    "question": "Question text here?",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "answer": 0,
    "explanation": "Explanation citing [source, p.X]."
  }}
]
Note: 'answer' must be the 0-based integer index of the correct option (0, 1, 2, or 3)."""

    response_text = generate(prompt)

    # Robust JSON parsing
    cleaned = response_text.strip()
    # Strip markdown code blocks if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    # Find the JSON array boundary
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1]

    try:
        questions = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse quiz response into JSON: {e}. Raw response: {response_text[:200]}")

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Invalid quiz format received from model.")

    formatted_questions = []
    for q in questions[:num_questions]:
        q_text = str(q.get("question", "")).strip()
        raw_options = q.get("options", [])
        if len(raw_options) != 4:
            continue

        raw_ans = q.get("answer", 0)
        # Parse answer if given as 'A', 'B', 'C', 'D' or string number
        ans_idx = 0
        if isinstance(raw_ans, int) and 0 <= raw_ans < 4:
            ans_idx = raw_ans
        elif isinstance(raw_ans, str):
            clean_ans = raw_ans.strip().upper()
            if clean_ans.startswith("A") or clean_ans == "0":
                ans_idx = 0
            elif clean_ans.startswith("B") or clean_ans == "1":
                ans_idx = 1
            elif clean_ans.startswith("C") or clean_ans == "2":
                ans_idx = 2
            elif clean_ans.startswith("D") or clean_ans == "3":
                ans_idx = 3

        explanation = str(q.get("explanation", "Refer to the referenced study notes.")).strip()

        formatted_questions.append({
            "question": q_text,
            "options": [str(opt).strip() for opt in raw_options],
            "answer": ans_idx,
            "explanation": explanation,
        })

    if not formatted_questions:
        raise ValueError("Could not extract valid MCQs from AI response. Please try again.")

    return formatted_questions


def generate_summary(file_name):
    """
    Generate key points and likely exam topics for a selected PDF.
    """
    client = chromadb.PersistentClient(path="db")
    try:
        col = client.get_collection("notes")
        records = col.get(where={"source": file_name})
    except Exception as e:
        raise ValueError(f"Could not load index for document: {e}")

    docs = records.get("documents", [])
    metas = records.get("metadatas", [])

    if not docs:
        raise ValueError(f"No indexed text found for '{file_name}'. Please rebuild the index in the sidebar.")

    # Select representative chunks across the document (up to 12 chunks)
    total = len(docs)
    if total <= 12:
        selected_indices = list(range(total))
    else:
        step = total / 12.0
        selected_indices = [int(i * step) for i in range(12)]

    context_pieces = []
    for idx in selected_indices:
        p_num = metas[idx].get("page", "?")
        context_pieces.append(f"[{file_name}, p.{p_num}]\n{docs[idx]}")

    context = "\n\n".join(context_pieces)

    prompt = f"""You are an elite academic professor and exam strategist.
Analyze the following material extracted from '{file_name}'.
Generate a comprehensive, structured study guide and exam prep summary.

Structure your response with clear markdown headers:
1. 📌 **Overview & Core Subject Scope**: High-level summary of the document's purpose.
2. 🔑 **Key Points & Fundamental Concepts**: Thorough bullet points of key topics and principles. Cite pages like [p.X].
3. 🎯 **Likely Exam Topics & Practice Questions**: 4 to 6 critical exam topics with sample questions and concise model answer hints.
4. ⚡ **Definitions & Key Formulas / Rules**: Essential terminology and rules to memorize for the test.

Context:
{context}"""

    return generate(prompt)