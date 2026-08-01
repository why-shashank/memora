#!/bin/sh
# Conventional-commit check (pre-commit commit-msg stage). $1 = message file.
# Rules mirror CLAUDE.md: `type(scope): summary`, one line, <= 72 chars, body after a blank line.
#
# The length and blank-line guards are not decoration. Git folds a second line into the
# subject when no blank line separates the two, so a wrapped paste once produced an 89-char
# subject with a gap in the middle of it and a prefix-only check called it valid.

MAX=72

# Git's own comment lines — the editor template, and the diff under `commit -v` — are not
# part of the message.
subject=$(grep -v '^#' "$1" | sed -n '1p')
second=$(grep -v '^#' "$1" | sed -n '2p')

reject() {
    echo "✖ $1" >&2
    echo "  subject: $subject" >&2
    exit 1
}

pattern='^(feat|fix|test|refactor|docs|chore|build|ci|perf)(\([a-z0-9-]+\))?!?: .+'
printf '%s\n' "$subject" | grep -qE "$pattern" || reject "not a conventional commit — want type(scope): summary
  types: feat fix test refactor docs chore build ci perf"

[ "${#subject}" -le "$MAX" ] || reject "subject is ${#subject} chars, limit $MAX — move the detail into a body"

[ -z "$second" ] || reject "line 2 must be blank — a body needs a blank line after the subject"
