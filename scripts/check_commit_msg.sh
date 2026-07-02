#!/bin/sh
# Conventional-commit check (pre-commit commit-msg stage). $1 = message file.
# type(scope): summary — types/scopes per CLAUDE.md.
pattern='^(feat|fix|test|refactor|docs|chore|build|ci|perf)(\([a-z0-9-]+\))?!?: .+'
if ! head -1 "$1" | grep -qE "$pattern"; then
    echo "✖ commit message must follow conventional commits: type(scope): summary" >&2
    echo "  types: feat fix test refactor docs chore build ci perf" >&2
    echo "  got: $(head -1 "$1")" >&2
    exit 1
fi
