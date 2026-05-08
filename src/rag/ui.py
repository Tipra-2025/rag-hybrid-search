"""Optional Streamlit UI. Lazy-imports streamlit so it stays in the [ui] extra."""

from __future__ import annotations

from .config import load_settings


def render() -> None:  # pragma: no cover — streamlit-only
    import asyncio

    import streamlit as st

    from .cli import _build_state

    settings = load_settings()
    if "state" not in st.session_state:
        st.session_state.state = _build_state(settings)
    state = st.session_state.state

    st.set_page_config(page_title="rag-hybrid-search", layout="wide")
    st.title("rag-hybrid-search — ask")
    st.caption(
        f"Strategy: {settings.chunking_strategy.value} · "
        f"Index dir: {settings.index_dir} · "
        f"Dense chunks: {len(state.dense)} · Sparse chunks: {len(state.sparse)}"
    )

    question = st.text_input("Your question", value="")
    if not question:
        return
    with st.spinner("Retrieving and answering..."):
        answer = asyncio.run(state.engine.answer(question))

    st.subheader("Answer")
    st.write(answer.text)

    cols = st.columns(3)
    cols[0].metric("Retrieval confidence", f"{answer.retrieval_confidence:.2f}")
    cols[1].metric("Citation accuracy", f"{answer.citation_accuracy:.2f}")
    cols[2].metric("Composite", f"{answer.composite_confidence:.2f}")

    st.subheader("Citations")
    if not answer.citations:
        st.info("No citations.")
    else:
        for c in answer.citations:
            with st.expander(f"{c.marker}  {c.title or c.source}"):
                st.code(c.quote)

    st.subheader("Top chunks (post-rerank)")
    for h in answer.used_chunks:
        with st.expander(
            f"score={h.score:.3f}  rrf={h.rrf_score:.3f}  "
            f"dense_rank={h.dense_rank}  sparse_rank={h.sparse_rank}"
        ):
            st.write(f"**{h.chunk.title}** — `{h.chunk.source}`")
            st.code(h.chunk.text[:1500])


if __name__ == "__main__":  # pragma: no cover
    render()
