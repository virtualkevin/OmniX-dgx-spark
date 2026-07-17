FROM omnix-dgx-spark:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/OmniX:/opt/Deformable_DETR \
    OMX_TEMP_DIRECTORY=/tmp/omnix-viewer \
    TMPDIR=/tmp/omnix-viewer \
    TEMP=/tmp/omnix-viewer \
    TMP=/tmp/omnix-viewer \
    HOME=/tmp/omnix-viewer \
    XDG_CACHE_HOME=/tmp/omnix-viewer/cache

WORKDIR /workspace/OmniX

COPY viewer/server/requirements.txt /tmp/omnix-viewer-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/omnix-viewer-requirements.txt \
    && groupadd --gid 10001 omnix-viewer \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp/omnix-viewer --shell /usr/sbin/nologin omnix-viewer \
    && install -d -o 10001 -g 10001 -m 0700 /tmp/omnix-viewer

COPY --chown=10001:10001 viewer/server /workspace/OmniX/viewer/server

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"]

CMD ["uvicorn", "viewer.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
