from pathlib import Path


def test_static_segment_viewer_does_not_render_json_data_with_inner_html():
    template = Path("index.html").read_text()

    assert "innerHTML" not in template
    assert "textContent" in template
