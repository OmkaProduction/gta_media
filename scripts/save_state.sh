#!/usr/bin/env bash
# Commits state/processed.json without clobbering state that another run pushed meanwhile:
# only the entries this run changed are applied on top of the latest branch state.
set -euo pipefail
f=state/processed.json
git config user.name "clipper-bot"
git config user.email "github-actions[bot]@users.noreply.github.com"
git diff --quiet -- "$f" && { echo "state unchanged"; exit 0; }
cp "$f" /tmp/ours.json
git show "HEAD:$f" > /tmp/base.json 2>/dev/null || echo '{}' > /tmp/base.json
branch=${GITHUB_REF_NAME}
for i in 1 2 3 4; do
  git fetch -q origin "$branch"
  git checkout -q -f -B "$branch" FETCH_HEAD
  python3 - "$f" <<'PY'
import json, sys
path = sys.argv[1]
load = lambda p: json.loads(open(p, encoding="utf-8").read() or "{}")
ours, base = load("/tmp/ours.json"), load("/tmp/base.json")
try:
    latest = load(path)
except FileNotFoundError:
    latest = {}
for key, value in ours.items():
    if base.get(key) != value:
        latest[key] = value
open(path, "w", encoding="utf-8").write(json.dumps(latest, indent=2, sort_keys=True) + "\n")
PY
  git add "$f"
  git diff --cached --quiet && { echo "state already up to date"; exit 0; }
  git commit -q -m "Update processed VODs"
  git push -q origin "HEAD:$branch" && exit 0
  sleep $((i * 5))
done
exit 1
