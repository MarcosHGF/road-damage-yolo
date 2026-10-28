import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rdd.download import DownloadError, download_file, fetch_countries, safe_extract, verify_zip


def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class Server:
    """Tiny HTTP server with Range support and an optional first-request cut-off."""

    def __init__(self, routes: dict[str, bytes], cut_first_at: float | None = None):
        self.routes = routes
        self.cut_first_at = cut_first_at
        self.requests: list[tuple[str, str | None]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                body = outer.routes.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(body):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                if outer.cut_first_at is not None and len(outer.requests) == 1:
                    self.wfile.write(chunk[: int(len(chunk) * outer.cut_first_at)])
                    self.wfile.flush()
                    self.close_connection = True
                    return  # connection drops mid-body
                self.wfile.write(chunk)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}{path}"


def test_download_resumes_after_dropped_connection(tmp_path):
    payload = make_zip_bytes({"Japan/train/images/a.txt": b"x" * 200_000})
    with Server({"/f.zip": payload}, cut_first_at=0.4) as srv:
        out = download_file(srv.url("/f.zip"), tmp_path / "f.zip", backoff=0, quiet=True)
    assert out.read_bytes() == payload
    assert srv.requests[0][1] is None  # first request: full file
    assert srv.requests[1][1] is not None and srv.requests[1][1].startswith("bytes=")  # second: resumed
    assert not (tmp_path / "f.zip.part").exists()


def test_http_404_points_to_manual_download(tmp_path):
    with Server({}) as srv:
        with pytest.raises(DownloadError, match="browser"):
            download_file(srv.url("/missing.zip"), tmp_path / "m.zip", backoff=0, quiet=True)


def test_verify_zip_rejects_html_error_page_and_truncated_zip(tmp_path):
    html = tmp_path / "x.zip"
    html.write_text("<html>Access denied</html>")
    with pytest.raises(DownloadError, match="not a valid zip"):
        verify_zip(html)
    good = make_zip_bytes({"a.txt": b"hello" * 1000})
    trunc = tmp_path / "t.zip"
    trunc.write_bytes(good[: len(good) // 2])
    with pytest.raises(DownloadError):
        verify_zip(trunc)


def test_safe_extract_blocks_path_traversal(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "boom")
    with pytest.raises(DownloadError, match="unsafe"):
        safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_fetch_countries_downloads_extracts_and_is_idempotent(paths):
    payload = make_zip_bytes({"Czech/train/images/Czech_000001.jpg": b"jpg", "Czech/train/annotations/xmls/Czech_000001.xml": b"<a/>"})
    with Server({"/Czech.zip": payload}) as srv:
        urls = {"Czech": srv.url("/Czech.zip")}
        assert fetch_countries(["Czech"], paths, urls=urls) == ["Czech"]
        assert (paths.raw / "Czech" / "Czech" / "train" / "images" / "Czech_000001.jpg").exists()
        assert (paths.raw / "Czech" / ".extracted_ok").exists()
        assert not (paths.downloads / "Czech.zip").exists()  # removed after extraction
        n_requests = len(srv.requests)
        fetch_countries(["Czech"], paths, urls=urls)
        assert len(srv.requests) == n_requests  # second run: nothing downloaded


def test_manual_zip_in_downloads_is_used_without_network(paths):
    paths.downloads.mkdir(parents=True)
    (paths.downloads / "RDD2022_India.zip").write_bytes(make_zip_bytes({"India/train/images/India_000001.jpg": b"jpg"}))
    # A URL that would fail instantly if it were contacted.
    fetch_countries(["India"], paths, urls={"India": "http://127.0.0.1:9/never.zip"})
    assert (paths.raw / "India" / ".extracted_ok").exists()
    assert (paths.downloads / "RDD2022_India.zip").exists()  # user's file is never deleted


def test_unknown_country_rejected(paths):
    with pytest.raises(DownloadError, match="unknown countries"):
        fetch_countries(["Atlantis"], paths)
