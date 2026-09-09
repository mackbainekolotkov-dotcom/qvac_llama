#!/usr/bin/env bash
# download the source model from huggingface and record the resolved revision
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="$LOGS_DIR/task1/00_fetch_model.log"
DEST="$HF_DIR/$MODEL_NAME"

PROVENANCE="$LOGS_DIR/task1/model_provenance.txt"

run_logged "$LOG" "$PYTHON_BIN" - "$DEST" "$MODEL_REPO" "$MODEL_REPO_FALLBACK" "$MODEL_REVISION" "$PROVENANCE" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

dest, repo, fallback, revision = sys.argv[1:5]
ignore = ["original/*", "*.pth", "*.gguf"]

try:
    path = snapshot_download(repo, revision=revision, local_dir=dest, ignore_patterns=ignore)
    used = repo
except Exception as e:
    print(f"WARNING: {repo} is gated or unavailable ({type(e).__name__}), falling back to {fallback}")
    path = snapshot_download(fallback, revision=revision, local_dir=dest, ignore_patterns=ignore)
    used = fallback

from huggingface_hub import HfApi
sha = HfApi().model_info(used, revision=revision).sha
Path(dest, ".qvac_repo").write_text(f"{used}\n{sha}\n")

# the model tree is gitignored, so keep a tracked copy of the provenance next to the logs
prov = Path(sys.argv[5])
prov.parent.mkdir(parents=True, exist_ok=True)
prov.write_text(f"source repo:     {used}\nsource revision: {sha}\nrequested repo:  {repo}\n")

print(f"source repo:     {used}")
print(f"source revision: {sha}")
print(f"local dir:       {path}")
PY
