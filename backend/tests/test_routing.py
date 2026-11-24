"""Adaptive query routing unit tests."""

from app.services.generation import classify_intent
from app.services.retrieval import _breadth_first_chunks


def test_classify_specific_question() -> None:
    assert classify_intent("Which method is described on page 4?") == "SPECIFIC_QA"
    assert classify_intent("How does the second approach differ?") == "SPECIFIC_QA"


def test_classify_global_summary() -> None:
    assert classify_intent("Summarize the main argument") == "GLOBAL_SUMMARY"
    assert classify_intent("Give me an overview of the whole document") == "GLOBAL_SUMMARY"
    assert classify_intent("What are the key takeaways?") == "GLOBAL_SUMMARY"


def test_global_breadth_sample_is_strided() -> None:
    points = []
    for index in range(1, 31):
        point = type("Point", (), {
            "id": f"chunk-{index}",
            "payload": {
                "text": f"Chunk {index}",
                "page_number": (index - 1) // 3 + 1,
                "chunk_index": index,
            },
        })()
        points.append(point)

    selected = _breadth_first_chunks(points, limit=15)
    assert len(selected) == 15
    pages = [item["chunk"]["page_num"] for item in selected]
    assert pages[0] == 1
    assert pages[-1] == 10
