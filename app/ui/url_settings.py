from typing import Any

WILDCARD_BIND_HOSTS = {"0.0.0.0", "::", "[::]"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def local_http_url(host: Any, port: Any) -> str:
    host_text = _clean_text(host) or "127.0.0.1"
    port_text = _clean_text(port)

    if host_text in WILDCARD_BIND_HOSTS:
        host_text = "127.0.0.1"
    elif ":" in host_text and not host_text.startswith("["):
        host_text = f"[{host_text}]"

    return f"http://{host_text}:{port_text}"
