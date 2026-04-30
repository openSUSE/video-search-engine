import ipaddress
from typing import Any

WILDCARD_BIND_HOSTS = {"0.0.0.0", "::", "[::]"}
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535
HOST_INVALID_MESSAGE = (
    "host must be a hostname or IP address without scheme, path, or port"
)


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


def _browser_host(value: Any) -> str:
    host_text = _clean_text(value) or "127.0.0.1"

    if host_text in WILDCARD_BIND_HOSTS:
        return "127.0.0.1"

    if (
        "://" in host_text
        or any(char in host_text for char in "/\\?#@")
        or any(char.isspace() for char in host_text)
    ):
        raise ValueError(HOST_INVALID_MESSAGE)

    if host_text.startswith("[") or host_text.endswith("]"):
        if not (host_text.startswith("[") and host_text.endswith("]")):
            raise ValueError(HOST_INVALID_MESSAGE)
        try:
            ip_address = ipaddress.ip_address(host_text[1:-1])
        except ValueError as exc:
            raise ValueError(HOST_INVALID_MESSAGE) from exc
        if ip_address.version != 6:
            raise ValueError(HOST_INVALID_MESSAGE)
        return host_text

    if ":" in host_text:
        try:
            ip_address = ipaddress.ip_address(host_text)
        except ValueError as exc:
            raise ValueError(HOST_INVALID_MESSAGE) from exc
        if ip_address.version != 6:
            raise ValueError(HOST_INVALID_MESSAGE)
        return f"[{host_text}]"

    return host_text


def local_http_url(host: Any, port: Any) -> str:
    host_text = _browser_host(host)
    port_text = _tcp_port(port)

    return f"http://{host_text}:{port_text}"
