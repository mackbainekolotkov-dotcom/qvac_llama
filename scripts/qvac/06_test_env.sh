#!/usr/bin/env bash
# Prepare the two ctest fixtures that are not part of the build, so the suite reaches 60/60.
# Both are environment, not code: without them test-jinja-py and test-tokenizers-ggml-vocabs fail
# on a clean machine regardless of which branch is checked out.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VOCAB_REPO="${VOCAB_REPO:-https://huggingface.co/ggml-org/vocabs}"
VOCAB_DIR="${VOCAB_DIR:-$QVAC_ROOT/models/ggml-vocabs}"
PYTHON_TEST_BIN="${PYTHON_TEST_BIN:-python3}"   # test-jinja spawns plain "python3" from PATH

# 1. test-jinja -py renders its expected output with the reference jinja2 implementation
echo "### ensuring jinja2 for $($PYTHON_TEST_BIN -c 'import sys; print(sys.executable)')"
if ! "$PYTHON_TEST_BIN" -c 'import jinja2' 2>/dev/null; then
    "$PYTHON_TEST_BIN" -m pip install --user jinja2
fi
"$PYTHON_TEST_BIN" -c 'import jinja2; print("### jinja2", jinja2.__version__)'

# 2. the vocab gguf files are git-lfs objects. Rather than requiring git-lfs, fetch the blobs
#    straight from the HuggingFace resolve endpoint: a clone without lfs leaves 132 byte pointers
#    that the tokenizer test then rejects with "invalid magic characters: 'vers'"
# check for the files themselves, not for a .git directory: this tree ships without one,
# and the blobs are fetched over https below anyway
if [ -z "$(find "$VOCAB_DIR" -name '*.gguf' 2>/dev/null | head -1)" ]; then
    echo "### cloning $VOCAB_REPO"
    rm -rf "$VOCAB_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$VOCAB_REPO" "$VOCAB_DIR"
fi

cd "$VOCAB_DIR"
for f in $(find . -name '*.gguf'); do
    if head -c 20 "$f" | grep -q '^version https'; then
        rel="${f#./}"
        echo "### fetching $rel"
        curl -fsSL -o "$f.tmp" "$VOCAB_REPO/resolve/main/$rel"
        mv "$f.tmp" "$f"
    fi
    magic=$(head -c 4 "$f")
    [ "$magic" = "GGUF" ] || { echo "### $f is still not a gguf (magic=$magic)"; exit 1; }
done

echo "### all vocab files present, ctest should now be 60/60"
