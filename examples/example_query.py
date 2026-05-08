"""End-to-end example. Run after `rag ingest --path docs/` with a real OpenAI key."""

from __future__ import annotations

import asyncio

from rag.cli import _build_state


async def main() -> None:
    state = _build_state()
    try:
        ans = await state.engine.answer(
            "What does the system do when retrieval confidence is below the threshold?"
        )
        print(ans.text)
        print(
            f"\n[retrieval={ans.retrieval_confidence:.2f} "
            f"citations={ans.citation_accuracy:.2f} "
            f"composite={ans.composite_confidence:.2f}]"
        )
        for c in ans.citations:
            print(f"  {c.marker} {c.title} ({c.source})")
    finally:
        await state.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
