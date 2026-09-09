#!/usr/bin/env bash
# fetch the small multimodal model used to exercise --mmproj-backend (Task 3).
# ungated, ~600 MB for the f16 pair
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MMPROJ_REPO="${MMPROJ_REPO:-ggml-org/SmolVLM-256M-Instruct-GGUF}"
MMPROJ_REVISION="${MMPROJ_REVISION:-main}"
DEST="${MMPROJ_DIR:-$GGUF_DIR/smolvlm}"
LOG="$LOGS_DIR/task3/07_fetch_mmproj_model.log"
PROVENANCE="$LOGS_DIR/task3/mmproj_provenance.txt"

run_logged "$LOG" "$PYTHON_BIN" - "$DEST" "$MMPROJ_REPO" "$MMPROJ_REVISION" "$PROVENANCE" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

dest, repo, revision, prov = sys.argv[1:5]

path = snapshot_download(repo, revision=revision, local_dir=dest,
                         allow_patterns=["*f16.gguf", "README.md", ".gitattributes"])
sha = HfApi().model_info(repo, revision=revision).sha

Path(prov).parent.mkdir(parents=True, exist_ok=True)
Path(prov).write_text(f"source repo:     {repo}\nsource revision: {sha}\n")

print(f"source repo:     {repo}")
print(f"source revision: {sha}")
print(f"local dir:       {path}")
PY

ls -l "$DEST" | tee -a "$LOG"
