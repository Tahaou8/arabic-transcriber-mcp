"""Remote MCP server for Arabic transcription with Groq Whisper."""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gdown
import httpx
from fastmcp import FastMCP
from groq import Groq
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


MODEL = os.environ.get("GROQ_MODEL", "whisper-large-v3")
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")
MCP_ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("MCP_ALLOWED_HOSTS", "*").split(",")
    if host.strip()
] or ["*"]
PORT = int(os.environ.get("PORT", "8000"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", str(500 * 1024 * 1024)))

mcp = FastMCP("Arabic Transcriber")


def _groq_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def _is_drive_source(source: str) -> bool:
    parsed = urlparse(source)
    return (
        parsed.hostname in {"drive.google.com", "docs.google.com"}
        or (not parsed.scheme and bool(re.fullmatch(r"[A-Za-z0-9_-]{20,}", source)))
    )


def _drive_file_id(source: str) -> str:
    """Extract a Drive file ID without relying on gdown's removed fuzzy option."""
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", source):
        return source

    parsed = urlparse(source)
    query_id = parse_qs(parsed.query).get("id", [""])[0]
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", query_id):
        return query_id

    path_match = re.search(r"/(?:file/)?d/([A-Za-z0-9_-]{20,})(?:/|$)", parsed.path)
    if path_match:
        return path_match.group(1)

    raise ValueError(
        "Could not extract a Google Drive file ID. Use a Drive file share link "
        "or the file ID, and share it as 'Anyone with the link'."
    )


def _validate_public_https_url(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("source must be a public HTTPS URL, Google Drive share link, or Drive file ID")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("local or private hosts are not allowed")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve source host: {hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("local, private, reserved, or link-local source addresses are not allowed")
    return source


def _download_source(source: str, destination: Path) -> None:
    source = source.strip()
    if not source:
        raise ValueError("source is required")

    if _is_drive_source(source):
        file_id = _drive_file_id(source)
        result = gdown.download(
            id=file_id,
            output=str(destination),
            quiet=True,
        )
        if not result or not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(
                "Google Drive download failed. Share the file as 'Anyone with the link'."
            )
        if destination.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("source exceeds the configured download size limit")
        return

    url = _validate_public_https_url(source)
    total = 0
    timeout = httpx.Timeout(connect=20.0, read=300.0, write=30.0, pool=20.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("source exceeds the configured download size limit")
                    output.write(chunk)

    if total == 0:
        raise RuntimeError("the media URL returned an empty file")


def _extract_audio(media_path: Path, audio_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(audio_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while extracting audio") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "ffmpeg could not read this media").strip()[-2000:]
        raise RuntimeError(f"audio extraction failed: {detail}") from exc


@mcp.tool
def transcriber_status() -> dict[str, object]:
    """Report whether Groq and ffmpeg are ready for transcription."""
    key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    ffmpeg_present = shutil.which("ffmpeg") is not None
    groq_reachable = False
    groq_error = ""

    if key_set:
        try:
            _groq_client().models.list()
            groq_reachable = True
        except Exception as exc:  # status must report failures instead of raising
            groq_error = f"{type(exc).__name__}: {exc}"

    return {
        "groq_api_key_set": key_set,
        "ffmpeg_on_path": ffmpeg_present,
        "model": os.environ.get("GROQ_MODEL", "whisper-large-v3"),
        "groq_reachable": groq_reachable,
        "groq_error": groq_error,
    }


@mcp.tool
def transcribe_arabic(source: str, language: str = "ar", prompt: str = "") -> str:
    """Transcribe speech from a public HTTPS media URL or Google Drive link/file ID."""
    with tempfile.TemporaryDirectory(prefix="arabic-transcriber-") as temp_dir:
        temp = Path(temp_dir)
        media_path = temp / "source-media"
        audio_path = temp / "audio.flac"

        _download_source(source, media_path)
        _extract_audio(media_path, audio_path)

        with audio_path.open("rb") as audio_file:
            result = _groq_client().audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=os.environ.get("GROQ_MODEL", "whisper-large-v3"),
                language=language or None,
                prompt=prompt or None,
                response_format="text",
            )

        if isinstance(result, str):
            return result
        return getattr(result, "text", str(result))


app = mcp.http_app(
    path=MCP_PATH,
    transport="http",
    allowed_hosts=MCP_ALLOWED_HOSTS,
    allowed_origins=["*"],
)


class _MCPHealthCheckMiddleware(BaseHTTPMiddleware):
    """Let platform GET health checks coexist with Streamable HTTP MCP."""

    async def dispatch(self, request: Request, call_next):
        accepts_sse = "text/event-stream" in request.headers.get("accept", "").lower()
        if request.method == "GET" and request.url.path == MCP_PATH and not accepts_sse:
            return JSONResponse({"ok": True, "service": "Arabic Transcriber"})
        return await call_next(request)


app.add_middleware(_MCPHealthCheckMiddleware)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
