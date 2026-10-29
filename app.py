from pathlib import Path
import streamlit as st
import chromadb
from ingest import build_index
from rag import (
    answer,
    rewrite_query,
    generate_quiz,
    generate_summary,
    model as embedding_model,
)

# Page configuration
st.set_page_config(
    page_title="AskMyNotes",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_current_index" not in st.session_state:
    st.session_state.quiz_current_index = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "quiz_submitted_current" not in st.session_state:
    st.session_state.quiz_submitted_current = False
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# Show toast notification after rerun if index was rebuilt
if "rebuild_toast" in st.session_state:
    st.toast(st.session_state.pop("rebuild_toast"), icon="✅")

# Theme color palettes (Default: Light mode, with preserved purple accents)
if st.session_state.dark_mode:
    bg_color = "#0F0B1E"
    sidebar_bg = "#18122B"
    card_bg = "#1E1838"
    text_color = "#F3F0FF"
    subtext_color = "#CBD5E1"
    heading_color = "#C084FC"
    accent_color = "#A855F7"
    border_color = "rgba(168, 85, 247, 0.25)"
    input_bg = "#1A1530"
    welcome_bg = "linear-gradient(135deg, rgba(168, 85, 247, 0.18) 0%, rgba(126, 34, 206, 0.12) 100%)"
    welcome_border = "rgba(168, 85, 247, 0.35)"
    welcome_text = "#F3F0FF"
    welcome_head = "#FFFFFF"
    tagline_color = "#C4B5FD"
    snippet_bg = "rgba(168, 85, 247, 0.12)"
else:
    bg_color = "#F8F7FC"
    sidebar_bg = "#F3EFFC"
    card_bg = "#FFFFFF"
    text_color = "#1E1B4B"
    subtext_color = "#475569"
    heading_color = "#4C1D95"
    accent_color = "#7C3AED"
    border_color = "rgba(124, 58, 237, 0.2)"
    input_bg = "#FFFFFF"
    # Solid purple card in light mode so white text stays readable
    welcome_bg = "linear-gradient(135deg, #7C3AED 0%, #A855F7 100%)"
    welcome_border = "rgba(124, 58, 237, 0.6)"
    welcome_text = "#FFFFFF"
    welcome_head = "#FFFFFF"
    tagline_color = "#9329F6"
    snippet_bg = "#F3E8FF"

# Inject dynamic CSS block for light/dark theme
st.markdown(
    f"""
    <style>
    /* App background and base typography */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}

    /* Sidebar background and borders */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
        background-color: {sidebar_bg} !important;
        color: {text_color} !important;
        border-right: 1px solid {border_color} !important;
    }}

    /* Headings and labels */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {heading_color} !important;
    }}
    .stApp p, .stApp span, .stApp label, [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}
    .stApp small {{
        color: {subtext_color} !important;
    }}
    .stApp code {{
        background-color: {snippet_bg} !important;
        color: {heading_color} !important;
    }}

    /* Inputs, textareas, and selectboxes */
    .stApp input, .stApp textarea, [data-baseweb="select"] > div,
    [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    .stApp input::placeholder, .stApp textarea::placeholder {{
        color: {subtext_color} !important;
        opacity: 0.8;
    }}

    /* Dropdown menus (selectbox / multiselect popups) */
    [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-baseweb="menu"] li, ul[role="listbox"] li {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
    [data-baseweb="menu"] li:hover, ul[role="listbox"] li:hover {{
        background-color: {snippet_bg} !important;
    }}

    /* File uploader (dropzone + uploaded file chips) */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {{
        background-color: {input_bg} !important;
        border: 1px solid {accent_color} !important;
        border-style: solid !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: {text_color} !important;
    }}
    [data-testid="stFileUploader"] small {{
        color: {subtext_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {card_bg} !important;
        border: 1px solid {accent_color} !important;
        border-radius: 8px !important;
    }}

    /* Chat input box (outer bar, inner wrapper, text area, send button) */
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {{
        background-color: {bg_color} !important;
    }}
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {{
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stChatInput"] {{
        border: 1px solid {border_color} !important;
        border-radius: 14px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        caret-color: {accent_color} !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {subtext_color} !important;
        opacity: 0.85;
    }}
    [data-testid="stChatInputSubmitButton"] {{
        background-color: {accent_color} !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }}
    [data-testid="stChatInputSubmitButton"] svg {{
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }}

    /* Chat message cards */
    [data-testid="stChatMessage"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        margin-bottom: 0.75rem;
    }}

    /* Multiselect tags */
    [data-baseweb="tag"] {{
        background-color: {accent_color} !important;
    }}
    [data-baseweb="tag"] span {{
        color: #FFFFFF !important;
    }}

    /* Expander cards */
    [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{
        color: {text_color} !important;
    }}

    /* Tabs styling */
    [data-testid="stTabs"] button {{
        color: {subtext_color} !important;
    }}
    [data-testid="stTabs"] button p {{
        color: inherit !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {accent_color} !important;
        border-bottom-color: {accent_color} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {accent_color} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {subtext_color} !important;
    }}

    /* Tagline between title and subtitle */
    .tagline-mid {{
        font-size: 1.3rem;
        font-style: italic;
        font-weight: 500;
        color: {tagline_color} !important;
        margin: -0.3rem 0 0.35rem 0;
        line-height: 1.3;
    }}

    /* Purple welcome / explanation card (text stays white on purple) */
    .welcome-card {{
        background: {welcome_bg} !important;
        border: 1px solid {welcome_border} !important;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }}
    .stApp .welcome-card, .stApp .welcome-card * {{
        color: {welcome_text} !important;
    }}
    .stApp .welcome-card h3, .stApp .welcome-card h4 {{
        color: {welcome_head} !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.4rem;
    }}
    .stApp .welcome-card p {{
        color: {welcome_text} !important;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }}

    /* Source snippet container */
    .source-snippet {{
        background-color: {snippet_bg} !important;
        border-left: 3px solid {accent_color} !important;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-top: 4px;
        margin-bottom: 8px;
        white-space: pre-wrap;
        word-break: break-word;
        color: {text_color} !important;
    }}

    /* Buttons */
    .stButton button {{
        border-radius: 8px;
    }}
    .stButton button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {{
        background-color: {accent_color} !important;
        border: none !important;
    }}
    /* White text on every purple button, including the label inside it */
    .stApp .stButton button[kind="primary"],
    .stApp .stButton button[kind="primary"] *,
    .stApp button[data-testid="baseButton-primary"],
    .stApp button[data-testid="baseButton-primary"] *,
    .stApp button[data-testid="stBaseButton-primary"],
    .stApp button[data-testid="stBaseButton-primary"] * {{
        color: #FFFFFF !important;
    }}
    .stButton button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# Helper to check whether vector index exists and retrieve counts
def get_index_metrics():
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        return "empty_data", pdfs, 0, 0
    try:
        client = chromadb.PersistentClient(path="db")
        col = client.get_collection("notes")
        chunks_count = col.count()
        if chunks_count == 0:
            return "missing_index", pdfs, 0, 0
        metadatas = col.get(include=["metadatas"]).get("metadatas", [])
        unique_pages = len(set((m.get("source"), m.get("page")) for m in metadatas if m))
        return "ready", pdfs, unique_pages, chunks_count
    except Exception:
        return "missing_index", pdfs, 0, 0


index_state, current_pdfs, total_pages_count, total_chunks_count = get_index_metrics()
current_pdfs = sorted(current_pdfs, key=lambda p: p.name.lower())
pdf_names = [p.name for p in current_pdfs]

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("📚 AskMyNotes")
    st.caption("Chat with your notes. Quiz yourself. Study smarter.")
    st.markdown("---")

    st.header("📂 Document Management")

    # Multiple PDF file uploader
    uploaded_files = st.file_uploader(
        "Upload Study Notes (PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, or slides in PDF format",
    )
    if uploaded_files:
        newly_saved = 0
        for f in uploaded_files:
            target = DATA_DIR / f.name
            if not target.exists() or target.stat().st_size != f.size:
                try:
                    target.write_bytes(f.getbuffer())
                    newly_saved += 1
                except Exception as e:
                    st.error(f"Error saving {f.name}: {e}")
        if newly_saved > 0:
            st.success(f"Saved {newly_saved} new PDF{'s' if newly_saved > 1 else ''}!")
            st.rerun()

    # List current PDFs with a delete button each
    st.subheader(f"Saved PDFs ({len(current_pdfs)})")
    if not current_pdfs:
        st.info("No PDFs uploaded yet. Upload above to get started.")
    else:
        for pdf in current_pdfs:
            col_name, col_del = st.columns([0.8, 0.2])
            size_kb = pdf.stat().st_size / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
            col_name.write(
                f"📄 **{pdf.name}**  \n<small>{size_str}</small>",
                unsafe_allow_html=True,
            )
            if col_del.button("🗑️", key=f"del_{pdf.name}", help=f"Delete {pdf.name}"):
                try:
                    pdf.unlink()
                    st.success(f"Deleted {pdf.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete: {e}")

    st.markdown("---")
    # File Filter Multiselect
    st.subheader("🔍 Scope Filter")
    selected_file_filter = st.multiselect(
        "Filter Retrieval by Document:",
        options=pdf_names,
        default=[],
        help="Leave empty to retrieve from all documents, or select specific files to constrain search and quiz generation.",
    )

    st.markdown("---")
    # Knowledge Base Metrics
    st.subheader("📊 Knowledge Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Files", len(current_pdfs))
    m_col2.metric("Pages", total_pages_count)
    m_col3.metric("Chunks", total_chunks_count)

    st.markdown("---")
    st.subheader("⚙️ Vector Index")

    # Rebuild index button
    if st.button("🔄 Rebuild index", use_container_width=True, type="primary"):
        if not current_pdfs:
            st.warning("Cannot rebuild index: no PDFs found in data/.")
        else:
            try:
                with st.spinner("Processing documents & building vector index..."):
                    pages_count, chunks_count = build_index(folder=str(DATA_DIR), model=embedding_model)
                if pages_count == 0 or chunks_count == 0:
                    st.warning("No readable text found. Verify that your PDFs are not scanned images.")
                else:
                    st.session_state["rebuild_toast"] = f"Index rebuilt successfully! ({pages_count} pages, {chunks_count} chunks indexed)"
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to rebuild index: {e}")

    # Warning status banners in sidebar
    if index_state == "empty_data":
        st.warning("⚠️ `data/` is empty. Upload PDFs to get started.")
    elif index_state == "missing_index":
        st.warning("⚠️ Index is missing or empty. Please click 'Rebuild index'.")


# ==================== MAIN HEADER ====================
col_title, col_theme, col_clear = st.columns([0.62, 0.22, 0.16])
with col_title:
    st.title("📚 AskMyNotes")
    st.markdown(
        '<div class="tagline-mid">Your Personalized Study Assistant</div>',
        unsafe_allow_html=True,
    )
    st.caption("Chat with your notes. Quiz yourself. Study smarter.")

with col_theme:
    st.write("")  # Spacing for visual alignment
    # Light/Dark toggle in top-right, default Light (False)
    is_dark_selected = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
    if is_dark_selected != st.session_state.dark_mode:
        st.session_state.dark_mode = is_dark_selected
        st.rerun()

with col_clear:
    st.write("")  # Spacing for visual alignment
    if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear current conversation"):
        st.session_state.messages = []
        st.rerun()

# ==================== WELCOME CARD (above the tabs) ====================
st.markdown(
    f"""
    <div class="welcome-card" style="color: {welcome_text} !important;">
        <h3 style="color: {welcome_head} !important;">👋 Welcome to AskMyNotes!</h3>
        <p style="color: {welcome_text} !important;">I can help you review your course materials, answer follow-up questions with conversation context, explain difficult concepts, and cite exact source pages. Ask anything or pick a sample question below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Index warning banners on main page
if index_state == "empty_data":
    st.warning("⚠️ **No study materials found.** Please upload your PDFs in the sidebar to get started.")
elif index_state == "missing_index":
    st.warning("⚠️ **Index is missing.** Please click **'Rebuild index'** in the sidebar to index your notes.")


# ==================== TABS ====================
tab_chat, tab_quiz, tab_summary = st.tabs([
    "💬 Chat",
    "📝 Quiz",
    "📑 Summary",
])

# -------------------- TAB 1: CHAT --------------------
with tab_chat:
    # Filter alert if active
    if selected_file_filter:
        st.info(f"🎯 **Active Document Filter:** Searching within: {', '.join(selected_file_filter)}")

    # Clickable example question buttons when chat is empty
    if not st.session_state.messages:
        st.markdown("##### 💡 Try asking:")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        ex1 = "What are the core concepts covered in the notes?"
        ex2 = "Summarize the key definitions and formulas."
        ex3 = "What are the advantages and disadvantages mentioned?"

        if ex_col1.button(f"📌 {ex1}", use_container_width=True):
            st.session_state.pending_prompt = ex1
            st.rerun()
        if ex_col2.button(f"📌 {ex2}", use_container_width=True):
            st.session_state.pending_prompt = ex2
            st.rerun()
        if ex_col3.button(f"📌 {ex3}", use_container_width=True):
            st.session_state.pending_prompt = ex3
            st.rerun()

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("standalone_query") and msg["standalone_query"] != msg.get("original_question"):
                st.caption(f"🔍 *Standalone Search Query:* `{msg['standalone_query']}`")
            if msg.get("hits"):
                with st.expander(f"📖 Sources ({len(msg['hits'])} citations)"):
                    for doc, m, d in msg["hits"]:
                        source_name = m.get("source", "Document")
                        page_num = m.get("page", "?")
                        st.markdown(f"**📄 {source_name}** — Page {page_num}")
                        snippet = doc[:400] + ("..." if len(doc) > 400 else "")
                        st.markdown(f'<div class="source-snippet">{snippet}</div>', unsafe_allow_html=True)

    # Process user input via chat_input or clicked example button
    prompt_from_example = st.session_state.pop("pending_prompt", None)
    chat_input_val = st.chat_input("Ask a question about your study notes...")
    user_question = prompt_from_example or chat_input_val

    if user_question:
        # Rewriting follow-up question into standalone query using the last 4 messages
        standalone_query = user_question
        if len(st.session_state.messages) > 0:
            with st.spinner("Understanding conversation context..."):
                try:
                    standalone_query = rewrite_query(user_question, st.session_state.messages)
                except Exception:
                    standalone_query = user_question

        # Append user question
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "standalone_query": standalone_query,
            "original_question": user_question,
        })
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            if index_state != "ready":
                warn_msg = "Please upload PDFs and click **'Rebuild index'** in the sidebar before asking questions."
                st.warning(warn_msg)
                st.session_state.messages.append({"role": "assistant", "content": warn_msg})
            else:
                with st.spinner("Searching notes and generating answer..."):
                    try:
                        if standalone_query != user_question:
                            st.caption(f"🔍 *Standalone Search Query:* `{standalone_query}`")

                        effective_filter = selected_file_filter if selected_file_filter else None
                        answer_text, hits = answer(
                            question=user_question,
                            retrieval_query=standalone_query,
                            file_filter=effective_filter,
                        )
                        st.markdown(answer_text)
                        if hits:
                            with st.expander(f"📖 Sources ({len(hits)} citations)"):
                                for doc, m, d in hits:
                                    source_name = m.get("source", "Document")
                                    page_num = m.get("page", "?")
                                    st.markdown(f"**📄 {source_name}** — Page {page_num}")
                                    snippet = doc[:400] + ("..." if len(doc) > 400 else "")
                                    st.markdown(f'<div class="source-snippet">{snippet}</div>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "hits": hits,
                            "standalone_query": standalone_query,
                            "original_question": user_question,
                        })
                    except Exception as e:
                        err_msg = f"Sorry, an error occurred while generating the answer: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})


# -------------------- TAB 2: QUIZ --------------------
with tab_quiz:
    st.subheader("📝 Practice Quiz Generator")
    st.caption("Generate 5 multiple-choice questions grounded in your course notes, answer them one at a time, and track your score.")

    quiz_col1, quiz_col2 = st.columns([0.45, 0.55])
    with quiz_col1:
        quiz_pdf_choice = st.selectbox(
            "Pick a PDF (or All Documents):",
            options=["All Documents"] + pdf_names,
            key="quiz_pdf_select",
        )
    with quiz_col2:
        quiz_topic_input = st.text_input(
            "Topic or Focus Area (optional):",
            placeholder="e.g. Biological neurons, Cloud models, Risk and return",
            key="quiz_topic_input",
        )

    btn_col1, btn_col2 = st.columns([0.3, 0.7])
    with btn_col1:
        generate_quiz_btn = st.button("🎯 Generate 5 MCQs", type="primary", use_container_width=True)
    with btn_col2:
        if st.session_state.quiz_questions:
            if st.button("🔄 Reset Quiz", use_container_width=False):
                st.session_state.quiz_questions = None
                st.session_state.quiz_current_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_user_answers = {}
                st.session_state.quiz_submitted_current = False
                st.rerun()

    if generate_quiz_btn:
        if index_state != "ready":
            st.error("Vector index is not ready. Please upload PDFs and rebuild the index in the sidebar first.")
        else:
            with st.spinner("Extracting relevant concepts and generating 5 MCQs..."):
                try:
                    file_filter_param = None
                    if quiz_pdf_choice != "All Documents":
                        file_filter_param = quiz_pdf_choice
                    elif selected_file_filter:
                        file_filter_param = selected_file_filter

                    questions = generate_quiz(
                        topic=quiz_topic_input,
                        file_filter=file_filter_param,
                        num_questions=5,
                    )
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_user_answers = {}
                    st.session_state.quiz_submitted_current = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not generate quiz: {e}")

    # Render active quiz
    if st.session_state.quiz_questions:
        questions = st.session_state.quiz_questions
        curr_idx = st.session_state.quiz_current_index
        total_q = len(questions)

        # Still taking the quiz (questions 0 to total_q - 1)
        if curr_idx < total_q:
            q_data = questions[curr_idx]

            # Header with question counter and score
            header_col1, header_col2 = st.columns([0.7, 0.3])
            header_col1.markdown(f"#### ❓ Question {curr_idx + 1} of {total_q}")
            header_col2.markdown(f"### Current Score: **{st.session_state.quiz_score} / {len(st.session_state.quiz_user_answers)}**")

            st.progress(curr_idx / total_q)
            st.markdown("---")

            st.markdown(f"### {q_data['question']}")

            # Radio options for single selection
            selected_choice = st.radio(
                "Choose the correct answer:",
                options=list(range(len(q_data["options"]))),
                format_func=lambda i: q_data["options"][i],
                key=f"quiz_radio_{curr_idx}",
                disabled=st.session_state.quiz_submitted_current,
            )

            # Submit answer or advance
            if not st.session_state.quiz_submitted_current:
                if st.button("Submit Answer", type="primary", key=f"sub_btn_{curr_idx}"):
                    is_correct = (selected_choice == q_data["answer"])
                    if is_correct:
                        st.session_state.quiz_score += 1
                    st.session_state.quiz_user_answers[curr_idx] = {
                        "selected": selected_choice,
                        "is_correct": is_correct,
                        "correct_answer": q_data["answer"],
                        "explanation": q_data["explanation"],
                    }
                    st.session_state.quiz_submitted_current = True
                    st.rerun()
            else:
                user_record = st.session_state.quiz_user_answers.get(curr_idx, {})
                if user_record.get("is_correct"):
                    st.success("🎉 **Correct!** Great job.")
                else:
                    correct_opt_text = q_data["options"][q_data["answer"]]
                    st.error(f"❌ **Incorrect.** The correct answer is: **{correct_opt_text}**")

                st.markdown(
                    f"""
                    <div class="welcome-card" style="margin-top: 10px;">
                        <h4 style="color: {welcome_head} !important;">📖 Explanation & Source</h4>
                        <p style="color: {welcome_text} !important;">{q_data['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                next_label = "Next Question ➡️" if curr_idx < total_q - 1 else "Finish Quiz & View Results 🏆"
                if st.button(next_label, type="primary", key=f"next_btn_{curr_idx}"):
                    st.session_state.quiz_current_index += 1
                    st.session_state.quiz_submitted_current = False
                    st.rerun()

        # Quiz completed screen
        else:
            final_score = st.session_state.quiz_score
            score_pct = (final_score / total_q) * 100

            if score_pct >= 80:
                st.balloons()
                st.success(f"## 🏆 Outstanding! You scored {final_score} / {total_q} ({score_pct:.0f}%)")
            elif score_pct >= 60:
                st.info(f"## 👏 Well done! You scored {final_score} / {total_q} ({score_pct:.0f}%)")
            else:
                st.warning(f"## 📚 Good practice session! You scored {final_score} / {total_q} ({score_pct:.0f}%)")

            st.markdown("### 📋 Detailed Question Review")
            for i, q in enumerate(questions):
                user_record = st.session_state.quiz_user_answers.get(i, {})
                is_correct = user_record.get("is_correct", False)
                status_icon = "✅ Correct" if is_correct else "❌ Incorrect"

                with st.expander(f"Question {i + 1}: {status_icon} — {q['question'][:80]}...", expanded=(not is_correct)):
                    st.markdown(f"**Question:** {q['question']}")
                    for opt_idx, opt_text in enumerate(q["options"]):
                        prefix = ""
                        if opt_idx == q["answer"]:
                            prefix = "✅ **[Correct Answer]** "
                        elif opt_idx == user_record.get("selected"):
                            prefix = "👉 **[Your Answer]** "
                        st.markdown(f"- {prefix}{opt_text}")
                    st.markdown(f"**Explanation:** {q['explanation']}")

            if st.button("🔄 Retake or Start New Quiz", type="primary"):
                st.session_state.quiz_questions = None
                st.session_state.quiz_current_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_user_answers = {}
                st.session_state.quiz_submitted_current = False
                st.rerun()


# -------------------- TAB 3: SUMMARY --------------------
with tab_summary:
    st.subheader("📑 Document Summarizer & Exam Prep")
    st.caption("Generate structured executive summaries, core concepts, and high-yield exam questions from any uploaded study material.")

    if not current_pdfs:
        st.info("No study notes found in `data/`. Upload a PDF in the sidebar to generate summaries.")
    else:
        sum_col1, sum_col2 = st.columns([0.7, 0.3])
        with sum_col1:
            selected_pdf_for_summary = st.selectbox(
                "Pick a PDF to summarize:",
                options=pdf_names,
                key="summary_pdf_picker",
            )
        with sum_col2:
            st.write("")  # Spacing
            generate_summary_btn = st.button("⚡ Generate Summary", type="primary", use_container_width=True)

        if generate_summary_btn:
            if index_state != "ready":
                st.error("Index is not ready. Please rebuild the index in the sidebar first.")
            else:
                with st.spinner(f"Analyzing '{selected_pdf_for_summary}' and extracting exam topics..."):
                    try:
                        summary_md = generate_summary(selected_pdf_for_summary)
                        st.session_state.summaries[selected_pdf_for_summary] = summary_md
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")

        # Display saved summary if available
        if selected_pdf_for_summary in st.session_state.summaries:
            summary_content = st.session_state.summaries[selected_pdf_for_summary]
            st.markdown("---")
            st.markdown(summary_content)

            st.download_button(
                label="💾 Download Summary (.md)",
                data=summary_content,
                file_name=f"{Path(selected_pdf_for_summary).stem}_Summary.md",
                mime="text/markdown",
            )