#!/usr/bin/env bash
# Download the sample drone footage listed in resources.md into datasets/.
# YouTube throttles each connection, so videos are fetched in parallel, one
# yt-dlp process per URL. Re-running resumes partial files and skips finished
# ones (tracked in datasets/.yt-archive.txt).
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS="${JOBS:-6}"
mkdir -p datasets logs/yt-dlp
grep -oE 'https://www\.youtube\.com/(watch\?v=|shorts/)[A-Za-z0-9_-]+' resources.md | sort -u |
  xargs -P "$JOBS" -I{} bash -c '
    id=$(basename "{}" | sed "s/.*v=//")
    for attempt in 1 2 3 4 5; do
      uvx --from "yt-dlp[default]@latest" yt-dlp --no-progress --js-runtimes node \
        --retries 20 --fragment-retries 50 \
        -f "bv*[ext=webm]+ba[ext=webm]/bv*+ba/b" --merge-output-format webm \
        -o "datasets/%(title)s [%(id)s].%(ext)s" \
        --download-archive datasets/.yt-archive.txt "{}" >> "logs/yt-dlp/$id.log" 2>&1 && exit 0
      sleep $((attempt * 10))
    done
    echo "FAILED {}" >> logs/yt-dlp/failed.txt'
echo "done: $(wc -l < datasets/.yt-archive.txt) videos in archive"
