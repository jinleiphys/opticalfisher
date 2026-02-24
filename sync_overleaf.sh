#!/bin/bash
# Sync paper/figures to Overleaf
# Usage: ./sync_overleaf.sh

set -euo pipefail

# Config
OVERLEAF_DIR=${OVERLEAF_DIR:-"/tmp/overleaf_opticalfisher"}
OVERLEAF_URL=${OVERLEAF_URL:-"https://git@git.overleaf.com/6951e717227815c0789a012a"}

# Resolve repo root (directory containing this script is project root)
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FIG_SRC="$REPO_ROOT/paper/figures"
TEX_SRC="$REPO_ROOT/paper"

echo "Syncing figures and LaTeX to Overleaf..."

# Clone if not exists, otherwise pull
if [ ! -d "$OVERLEAF_DIR/.git" ]; then
    echo "Cloning Overleaf repo..."
    git clone "$OVERLEAF_URL" "$OVERLEAF_DIR"
fi
cd "$OVERLEAF_DIR"
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "master")
echo "Pulling latest from Overleaf (branch: $BRANCH)..."
git pull origin "$BRANCH"

# Copy figures (PDF only for paper)
echo "Copying figures..."
mkdir -p "$OVERLEAF_DIR/figures"
shopt -s nullglob
pdf_files=("$FIG_SRC"/*.pdf)
shopt -u nullglob
if [ ${#pdf_files[@]} -gt 0 ]; then
    cp "${pdf_files[@]}" "$OVERLEAF_DIR/figures/"
else
    echo "  No PDF files found in $FIG_SRC"
fi

# Copy LaTeX files
echo "Copying LaTeX files..."
for f in "$TEX_SRC"/*.tex "$TEX_SRC"/*.bib "$TEX_SRC"/*.bst "$TEX_SRC"/*.sty; do
    [ -f "$f" ] && cp "$f" "$OVERLEAF_DIR/"
done

# Commit and push if there are changes
cd "$OVERLEAF_DIR"
if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Sync from local: $(date '+%Y-%m-%d %H:%M')"
    git push origin "$BRANCH"
    echo "Done! All files synced."
else
    echo "No changes to sync."
fi
echo "Overleaf project: $OVERLEAF_URL"
