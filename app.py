"""
Policy Review Agent — Phase 1 MVP
Streamlit multi-page application.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

from src.ingestion.chunker import ContextualChunker
from src.ingestion.parser import PolicyDocumentParser
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.pipeline import PolicyReviewPipeline

st.set_page_config(
    page_title="Policy Review Agent",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever: HybridRetriever | None = None

if "pipeline" not in st.session_state:
    st.session_state.pipeline: PolicyReviewPipeline | None = None

if "current_doc" not in st.session_state:
    st.session_state.current_doc: dict | None = None  # filename, content, metadata

if "history_filenames" not in st.session_state:
    st.session_state.history_filenames: list[str] = []

if "result" not in st.session_state:
    st.session_state.result = None  # FinalPolicyPackage

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "📤 Upload",
    "🔍 Compare",
    "✏️ Rewrite",
    "⭐ Rate",
    "📝 Final Review",
])

# ===========================================================================
# Tab 1 — Upload
# ===========================================================================
with tabs[0]:
    st.header("Upload Policy Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Policy")
        current_file = st.file_uploader(
            "Upload the policy to review",
            type=["pdf", "docx", "md", "txt"],
            key="current_policy",
        )
        if current_file:
            st.success(f"✅ Loaded: {current_file.name}")

    with col2:
        st.subheader("Historical Policies")
        history_files = st.file_uploader(
            "Upload reference policies (one or more)",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            key="history_policies",
        )
        if history_files:
            st.success(f"✅ Loaded {len(history_files)} historical document(s)")

    st.divider()

    with st.expander("Optional Metadata"):
        policy_type = st.selectbox(
            "Policy Type",
            ["governance", "compliance", "operational", "security", "hr", "financial", "other"],
        )
        department = st.text_input("Department")
        jurisdiction = st.text_input("Jurisdiction")

    if st.button("🚀 Initialize System & Index Documents", type="primary"):
        if not current_file:
            st.error("Please upload a current policy document.")
        else:
            with st.spinner("Initializing retriever and indexing documents…"):
                temp_files_to_cleanup: list[Path] = []
                try:
                    # Use a unique collection name per session to prevent cross-contamination
                    session_collection = f"policy_documents_{uuid.uuid4().hex[:12]}"
                    retriever = HybridRetriever(collection_name=session_collection)
                    parser = PolicyDocumentParser()
                    chunker = ContextualChunker()

                    history_names: list[str] = []
                    total = len(history_files) if history_files else 0
                    progress = st.progress(0)

                    for i, hfile in enumerate(history_files or []):
                        suffix = Path(hfile.name).suffix or ".txt"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(hfile.read())
                            tmp_path = Path(tmp.name)
                        temp_files_to_cleanup.append(tmp_path)

                        metadata, content, sections = parser.parse_document(
                            tmp_path,
                            metadata_overrides={
                                "policy_type": policy_type,
                                **({"department": department} if department else {}),
                                **({"jurisdiction": jurisdiction} if jurisdiction else {}),
                            },
                            original_filename=hfile.name,
                        )
                        chunks = chunker.chunk_document(content, metadata, sections)
                        retriever.add_documents(chunks)
                        history_names.append(hfile.name)
                        progress.progress((i + 1) / max(total, 1))

                    # Parse current document
                    suffix = Path(current_file.name).suffix or ".txt"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(current_file.read())
                        tmp_path = Path(tmp.name)
                    temp_files_to_cleanup.append(tmp_path)

                    curr_meta, curr_content, _ = parser.parse_document(
                        tmp_path,
                        metadata_overrides={
                            "policy_type": policy_type,
                            **({"department": department} if department else {}),
                            **({"jurisdiction": jurisdiction} if jurisdiction else {}),
                        },
                        original_filename=current_file.name,
                    )

                    st.session_state.retriever = retriever
                    st.session_state.pipeline = PolicyReviewPipeline(retriever=retriever)
                    st.session_state.current_doc = {
                        "filename": current_file.name,
                        "content": curr_content,
                        "metadata": curr_meta.model_dump(mode="json"),
                    }
                    st.session_state.history_filenames = history_names
                    st.session_state.result = None  # Reset previous run

                    n_chunks = retriever.collection_count()
                    st.success(
                        f"✅ System initialized! "
                        f"Indexed {n_chunks} chunks from {len(history_names)} historical doc(s). "
                        "Navigate to **Compare** to run analysis."
                    )
                except Exception as exc:
                    st.error("Initialization failed. Check your API key and Qdrant connection.")
                    logger.exception("Initialization error: %s", exc)
                finally:
                    for tf in temp_files_to_cleanup:
                        try:
                            os.remove(tf)
                        except OSError:
                            pass

# ===========================================================================
# Tab 2 — Compare (also triggers full pipeline)
# ===========================================================================
with tabs[1]:
    st.header("Document Comparison & Analysis")

    if st.session_state.current_doc is None:
        st.info("Upload documents in the **Upload** tab first.")
    else:
        st.markdown(
            f"**Current policy:** `{st.session_state.current_doc['filename']}`  \n"
            f"**Historical docs indexed:** {len(st.session_state.history_filenames)}"
        )

        if st.button("🔍 Run Full Analysis Pipeline", type="primary"):
            with st.spinner("Running policy review pipeline (this may take a minute)…"):
                try:
                    result = st.session_state.pipeline.run(
                        policy_content=st.session_state.current_doc["content"],
                        policy_filename=st.session_state.current_doc["filename"],
                        policy_metadata=st.session_state.current_doc["metadata"],
                        historical_policy_filenames=st.session_state.history_filenames,
                    )
                    st.session_state.result = result
                    st.success("✅ Analysis complete! Explore results in the other tabs.")
                except Exception as exc:
                    st.error("Pipeline error. Check your API key and try again.")
                    logger.exception("Pipeline error: %s", exc)

        result = st.session_state.result
        if result:
            st.subheader("📊 Similarity Report")
            if result.similarity_report:
                for item in result.similarity_report:
                    score = item.get("rrf_score", 0.0)
                    label = item.get("section_title", "Section")
                    doc_id = item.get("document_id", "")
                    with st.expander(f"📄 {label}  (RRF: {score:.4f})  — `{doc_id}`"):
                        st.markdown(item.get("excerpt", ""))
            else:
                st.info("No similar historical sections found.")

            st.subheader("🗂️ Positioning Recommendation")
            p = result.positioning
            st.markdown(f"**Recommended:** `{p.recommended_positioning.upper()}`")
            st.markdown(p.summary)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Rationale**")
                for r in p.rationale:
                    st.markdown(f"- {r}")
            with col2:
                st.markdown("**Retained Strengths**")
                for s in p.retained_strengths:
                    st.markdown(f"- {s}")
            with col3:
                st.markdown("**Top Risks**")
                for r in p.top_risks:
                    st.markdown(f"- ⚠️ {r}")

# ===========================================================================
# Tab 3 — Rewrite
# ===========================================================================
with tabs[2]:
    st.header("Policy Rewrite Workspace")

    result = st.session_state.result
    current_doc = st.session_state.current_doc

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Policy")
        if current_doc:
            st.text_area(
                "Original",
                current_doc["content"][:4000],
                height=450,
                disabled=True,
                key="orig_text",
            )
        else:
            st.info("Upload a policy document first.")

    with col2:
        st.subheader("Improved Draft")
        if result:
            st.text_area(
                "Improved",
                result.improved_draft_markdown[:4000],
                height=450,
                disabled=True,
                key="improved_text",
            )
        else:
            st.info("Run analysis in the **Compare** tab.")

    if result and result.issues:
        st.subheader("⚠️ Issues Addressed")
        for issue in result.issues:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                issue.severity, "⚪"
            )
            with st.expander(
                f"{severity_icon} [{issue.issue_type.upper()}] {issue.section_title}"
            ):
                st.markdown(f"**Description:** {issue.description}")
                st.markdown(f"**Recommendation:** {issue.recommendation}")

# ===========================================================================
# Tab 4 — Rate
# ===========================================================================
with tabs[3]:
    st.header("Quality Rating Dashboard")

    result = st.session_state.result

    if result:
        sc = result.scorecard
        col1, col2 = st.columns([1, 2])

        with col1:
            overall = sc.overall_score
            label = sc.overall_label.upper()
            delta_color = "normal" if overall >= 70 else "inverse"
            st.metric("Overall Score", f"{overall:.0f} / 100", label, delta_color=delta_color)

        with col2:
            dimensions = [
                ("Structure (20%)", sc.structure_score),
                ("Clarity (25%)", sc.clarity_score),
                ("Consistency (20%)", sc.consistency_score),
                ("Policy Alignment (25%)", sc.policy_alignment_score),
                ("Language Quality (10%)", sc.language_quality_score),
            ]
            for dim_name, dim_score in dimensions:
                st.progress(
                    min(dim_score / 100, 1.0),
                    text=f"{dim_name}: {dim_score:.0f}/100",
                )

        if sc.weaknesses_cited:
            st.subheader("⚠️ Areas for Improvement")
            for weakness in sc.weaknesses_cited:
                st.markdown(f"- {weakness}")

        if sc.dimension_notes:
            st.subheader("📝 Dimension Notes")
            for dim, note in sc.dimension_notes.items():
                st.markdown(f"**{dim}:** {note}")
    else:
        st.info("Run analysis in the **Compare** tab to see the rating.")

# ===========================================================================
# Tab 5 — Final Review
# ===========================================================================
with tabs[4]:
    st.header("Final Review & Download")

    result = st.session_state.result

    if result:
        # Grammar fixes
        if result.grammar_fixes:
            st.subheader(f"📝 {len(result.grammar_fixes)} Language Correction(s)")
            for fix in result.grammar_fixes:
                with st.expander(
                    f"[{fix.issue_type.upper()}] {fix.original_text[:60]}…"
                ):
                    st.markdown(f"**Original:** {fix.original_text}")
                    st.markdown(f"**Corrected:** {fix.corrected_text}")
                    st.caption(fix.explanation)
        else:
            st.info("No language corrections suggested.")

        st.divider()

        st.subheader("Final Copyedited Policy")
        st.text_area(
            "Copyedited",
            result.copyedited_draft_markdown[:6000],
            height=400,
            disabled=True,
            key="final_text",
        )

        st.download_button(
            label="📥 Download Final Policy (Markdown)",
            data=result.copyedited_draft_markdown,
            file_name="final_policy.md",
            mime="text/markdown",
        )

        st.divider()
        st.subheader("Approval")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve", type="primary"):
                result.approval_status = "approved"
                st.success("Policy approved!")
        with col2:
            if st.button("❌ Reject"):
                result.approval_status = "rejected"
                st.error("Policy rejected.")
        with col3:
            if st.button("🔄 Request Revision"):
                result.approval_status = "revision_requested"
                st.warning("Revision requested.")

        notes = st.text_area("Reviewer Notes")
        if st.button("Add Note") and notes.strip():
            result.reviewer_notes.append(notes.strip())
            st.success("Note added.")
    else:
        st.info("Run analysis in the **Compare** tab first.")
