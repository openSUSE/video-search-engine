from typing import Any

WILDCARD_BIND_HOSTS = {"0.0.0.0", "::", "[::]"}
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tcp_port(value: Any) -> str:
    port_text = _clean_text(value)
    invalid_message = f"port must be a TCP port between {MIN_TCP_PORT} and {MAX_TCP_PORT}"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(invalid_message) from exc

    if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ValueError(invalid_message)
    return str(port)


def local_http_url(host: Any, port: Any) -> str:
    host_text = _clean_text(host) or "127.0.0.1"
    port_text = _tcp_port(port)

    if host_text in WILDCARD_BIND_HOSTS:
        host_text = "127.0.0.1"
    elif ":" in host_text and not host_text.startswith("["):
        host_text = f"[{host_text}]"

    return f"http://{host_text}:{port_text}"
