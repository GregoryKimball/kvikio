# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http
import multiprocessing as mp
import re
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest
import utils

import kvikio
import kvikio.defaults
from kvikio.utils import LargeBacklogHTTPServer

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason=(
        "RemoteFile not available, please build KvikIO "
        "with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
    ),
)

ACCOUNT = "devstoreaccount1"


class AzureBlobHandler(BaseHTTPRequestHandler):
    """Minimal emulator of anonymous access to Azure Blob Storage.

    Uses the Azurite path-style URL "http://<host>:<port>/<account>/<container>/<blob>".
    Containers are created with ``PUT ...?restype=container`` and are private unless
    the ``x-ms-blob-public-access`` header is set. Anonymous reads of private
    containers return 404, as Azure does. Reads without the ``x-ms-version`` header
    are rejected so that the tests check KvikIO sends it.
    """

    containers: dict[str, bool] = {}
    blobs: dict[tuple[str, str], bytes] = {}

    def log_message(self, *args):
        pass

    def _parse(self):
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.lstrip("/").split("/", 2)
        if len(parts) < 2 or parts[0] != ACCOUNT:
            return None, None, parsed.query
        return parts[1], (parts[2] if len(parts) == 3 else None), parsed.query

    def _send_empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self):
        container, blob, query = self._parse()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if container is None:
            return self._send_empty(http.HTTPStatus.BAD_REQUEST)
        if blob is None and "restype=container" in query:
            public = self.headers.get("x-ms-blob-public-access") is not None
            self.containers[container] = public
            return self._send_empty(http.HTTPStatus.CREATED)
        if container not in self.containers:
            return self._send_empty(http.HTTPStatus.NOT_FOUND)
        self.blobs[(container, blob)] = body
        self._send_empty(http.HTTPStatus.CREATED)

    def do_DELETE(self):
        container, blob, _ = self._parse()
        self.blobs.pop((container, blob), None)
        self._send_empty(http.HTTPStatus.ACCEPTED)

    def _read(self, send_body: bool):
        if "x-ms-version" not in self.headers:
            return self._send_empty(http.HTTPStatus.BAD_REQUEST)
        container, blob, _ = self._parse()
        data = self.blobs.get((container, blob))
        if not self.containers.get(container, False) or data is None:
            return self._send_empty(http.HTTPStatus.NOT_FOUND)

        status = http.HTTPStatus.OK
        start, end = 0, len(data) - 1
        range_header = self.headers.get("Range")
        if range_header is not None:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            if match is None or int(match[1]) >= len(data):
                return self._send_empty(http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            start, end = int(match[1]), min(int(match[2]), len(data) - 1)
            status = http.HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Length", str(end - start + 1))
        if status == http.HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("x-ms-blob-type", "BlockBlob")
        self.end_headers()
        if send_body:
            self.wfile.write(data[start : end + 1])

    def do_HEAD(self):
        self._read(send_body=False)

    def do_GET(self):
        self._read(send_body=True)


@pytest.fixture(scope="session")
def endpoint_ip() -> str:
    return utils.localhost()


@pytest.fixture(scope="session")
def endpoint_port():
    return utils.find_free_port()


def start_azure_server(ip_address, port, ready):
    server = LargeBacklogHTTPServer((ip_address, port), AzureBlobHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ready.set()
    time.sleep(600)
    print("Azure blob emulator shutting down because of timeout (10min)")


@pytest.fixture(scope="session")
def azure_base(endpoint_ip, endpoint_port):
    """Fixture to set up the Azure blob emulator in separate process"""
    ready = mp.Event()
    p = mp.Process(target=start_azure_server, args=(endpoint_ip, endpoint_port, ready))
    p.start()
    try:
        if not ready.wait(timeout=30):
            pytest.fail(
                "Azure blob emulator did not become ready within 30 seconds "
                f"(exit code: {p.exitcode})"
            )
        yield f"http://{endpoint_ip}:{endpoint_port}/{ACCOUNT}"
    finally:
        if p.is_alive():
            p.kill()
        p.join()


def _request(method, url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        return resp.status


@contextmanager
def azure_context(azure_base, container, files=None, public=True):
    if files is None:
        files = {}
    headers = {"x-ms-blob-public-access": "blob"} if public else {}
    _request("PUT", f"{azure_base}/{container}?restype=container", b"", headers)
    for f, data in files.items():
        _request(
            "PUT",
            f"{azure_base}/{container}/{f}",
            data,
            {"x-ms-blob-type": "BlockBlob"},
        )
    yield azure_base
    for f in files:
        try:
            _request("DELETE", f"{azure_base}/{container}/{f}")
        except Exception:  # nosec B110
            pass


def test_read_access(azure_base):
    container_name = "container"
    blob_name = "Data"
    data = b"file content"
    with azure_context(
        azure_base=azure_base, container=container_name, files={blob_name: data}
    ) as server_address:
        with kvikio.RemoteFile.open_azure(
            f"{server_address}/{container_name}/{blob_name}"
        ) as f:
            assert f.nbytes() == len(data)
            assert f.remote_endpoint_type() == kvikio.RemoteEndpointType.AZURE
            got = bytearray(len(data))
            assert f.read(got) == len(got)
            assert bytes(got) == data

        with pytest.raises(RuntimeError, match="URL returned error: 404"):
            kvikio.RemoteFile.open_azure(
                f"{server_address}/unknown-container/{blob_name}"
            )

        with pytest.raises(RuntimeError, match="URL returned error: 404"):
            kvikio.RemoteFile.open_azure(
                f"{server_address}/{container_name}/unknown-file"
            )

        with pytest.raises(ValueError, match="must be an Azure Blob Storage url"):
            kvikio.RemoteFile.open_azure(f"unknown://{container_name}/{blob_name}")

    # Anonymous reads of a private container are rejected
    with azure_context(
        azure_base=azure_base,
        container="private",
        files={blob_name: data},
        public=False,
    ) as server_address:
        with pytest.raises(RuntimeError, match="URL returned error: 404"):
            kvikio.RemoteFile.open_azure(f"{server_address}/private/{blob_name}")


@pytest.mark.parametrize("size", [10, 100, 1000])
@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("tasksize", [99, 999])
@pytest.mark.parametrize("buffer_size", [101, 1001])
def test_read(azure_base, xp, size, nthreads, tasksize, buffer_size):
    # The MULTI_POLL backend requires `task_size <= bounce_buffer_size` for device
    # buffers. Skip the combos that violate it under this configuration.
    if (
        kvikio.defaults.get("remote_io_backend") == kvikio.RemoteIOBackend.MULTI_POLL
        and xp.__name__ != "numpy"
        and tasksize > buffer_size
    ):
        pytest.skip(
            "MULTI_POLL + device buffer requires task_size <= bounce_buffer_size"
        )

    container_name = "test-read"
    blob_name = "Aa1"
    a = xp.arange(size)
    with azure_context(
        azure_base=azure_base, container=container_name, files={blob_name: bytes(a)}
    ) as server_address:
        with kvikio.defaults.set(
            {
                "num_threads": nthreads,
                "task_size": tasksize,
                "bounce_buffer_size": buffer_size,
            }
        ):
            with kvikio.RemoteFile.open_azure(
                f"{server_address}/{container_name}/{blob_name}"
            ) as f:
                assert f.nbytes() == a.nbytes
                b = xp.empty_like(a)
                assert f.read(buf=b) == a.nbytes
                xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 10 * 4096),
        (1, int(1.3 * 4096)),
        (int(2.1 * 4096), int(5.6 * 4096)),
        (42, int(2**20)),
    ],
)
def test_read_with_file_offset(azure_base, xp, start, end):
    container_name = "test-read-with-file-offset"
    blob_name = "Aa1"
    a = xp.arange(end, dtype=xp.int64)
    with azure_context(
        azure_base=azure_base, container=container_name, files={blob_name: bytes(a)}
    ) as server_address:
        url = f"{server_address}/{container_name}/{blob_name}"
        with kvikio.RemoteFile.open_azure(url) as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)


# `RemoteFile.open` only identifies Azure hostnames as Azure, so these tests pass
# `nbytes` to avoid sending requests to the dummy URLs.
@pytest.mark.parametrize(
    "url",
    [
        "https://account.blob.core.windows.net/container/blob_name",
        "https://account.dfs.core.windows.net/container/blob_name",
        "abfss://container@account.dfs.core.windows.net/blob_name",
    ],
)
@pytest.mark.parametrize(
    "remote_endpoint_type",
    [kvikio.RemoteEndpointType.AUTO, kvikio.RemoteEndpointType.AZURE],
)
@pytest.mark.parametrize("allow_list", [None, [kvikio.RemoteEndpointType.AZURE]])
def test_open_valid(url, remote_endpoint_type, allow_list):
    nbytes = 1
    with kvikio.RemoteFile.open(url, remote_endpoint_type, allow_list, nbytes) as f:
        assert f.nbytes() == nbytes
        assert f.remote_endpoint_type() == kvikio.RemoteEndpointType.AZURE


def test_open_invalid():
    container_name = "container"
    blob_name = "blob_name"

    # Specified URL not in the allowlist
    url = f"https://account.blob.core.windows.net/{container_name}/{blob_name}"
    with pytest.raises(RuntimeError, match="not in the allowlist"):
        kvikio.RemoteFile.open(
            url, kvikio.RemoteEndpointType.AZURE, [kvikio.RemoteEndpointType.WEBHDFS]
        )

    # Invalid URLs
    url = f"abfss://{container_name}@account.dfs.core.windows.net"
    with pytest.raises(RuntimeError, match="Unsupported endpoint URL"):
        kvikio.RemoteFile.open(url)
    with pytest.raises(RuntimeError, match="Invalid URL"):
        kvikio.RemoteFile.open(url, kvikio.RemoteEndpointType.AZURE)


test_azure_io_under_easy_threadpool = utils.rerun_in_new_process(
    __file__, {"KVIKIO_REMOTE_IO_BACKEND": "EASY_THREADPOOL"}
)
