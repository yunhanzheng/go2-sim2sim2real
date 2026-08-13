#!/usr/bin/env bash
# Record git provenance for a run so it's reproducible even if the working tree
# had uncommitted changes (the commit hash alone does NOT capture those).
#
#   ./deploy_mac/lib/log_git_state.sh <RUN_DIR>
#
# Appends a git block to <RUN_DIR>/meta.txt and writes:
#   <RUN_DIR>/git_diff.patch   `git diff HEAD` — tracked changes (staged+unstaged)
#                              vs the recorded commit. Empty file => tree was clean.
#                              Re-apply with:  git checkout <commit> && git apply git_diff.patch
# Untracked files under run_logs/ are ignored so the run's own output never marks the
# tree dirty.
RUN="$1"
[ -n "$RUN" ] || { echo "usage: log_git_state.sh <RUN_DIR>" >&2; exit 2; }
mkdir -p "$RUN"
META="$RUN/meta.txt"

SHA="$(git rev-parse HEAD 2>/dev/null)"
BR="$(git branch --show-current 2>/dev/null)"
# --quiet HEAD checks TRACKED files only, so the freshly-created (untracked) run
# folder does not count as a change.
if git diff --quiet HEAD 2>/dev/null; then STATE="clean"; else STATE="dirty"; fi

# Tracked diff vs HEAD -> patch file (empty when clean).
git diff HEAD > "$RUN/git_diff.patch" 2>/dev/null
STAT="$(git diff HEAD --shortstat 2>/dev/null | sed 's/^ *//')"
[ -n "$STAT" ] || STAT="no tracked changes"

# Untracked files that would be needed to reproduce (excludes run_logs/ output).
UNTRACKED="$(git ls-files --others --exclude-standard -- ':(exclude)run_logs' 2>/dev/null)"

{ echo "git_commit:  ${SHA:-unknown}"
  echo "git_branch:  ${BR:-unknown}"
  echo "git_state:   $STATE ($STAT)"
  echo "git_diff:    git_diff.patch"
  if [ -n "$UNTRACKED" ]; then
    echo "git_untracked (NOT in git_diff.patch):"
    printf '  %s\n' $UNTRACKED
  fi
} >> "$META"
