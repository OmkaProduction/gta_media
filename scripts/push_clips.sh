#!/usr/bin/env bash
# Usage: push_clips.sh <source dir> <folder name>
# Copies the clips (mp4, txt, md) from <source dir> to <folder name>/ on the `clips` branch,
# so they can be fetched with plain git and shared outside GitHub.
set -euo pipefail
src=$(realpath "$1"); folder=$2
work=$(mktemp -d) && cd "$work"
git init -q
git config user.name "clipper-bot"
git config user.email "github-actions[bot]@users.noreply.github.com"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if git fetch -q --depth 1 origin clips; then git checkout -q -b clips FETCH_HEAD; else git checkout -q --orphan clips; fi
rm -rf "$folder" && mkdir "$folder"
find "$src" -type f \( -name '*.mp4' -o -name '*.txt' -o -name '*.md' \) -exec cp {} "$folder/" \;
ls -la "$folder"
git add "$folder"
git diff --cached --quiet && { echo "nothing new"; exit 0; }
git commit -q -m "Clips: $folder"
for i in 1 2 3; do
  git push -q origin clips && exit 0
  git fetch -q --depth 1 origin clips && git rebase -q FETCH_HEAD || true
done
exit 1
