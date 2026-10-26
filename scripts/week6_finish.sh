#!/usr/bin/env bash
# Week 6: wait for a pod list to finish, copy its results to the network volume, verify the
# copy, and only then remove this pod. If anything is missing or differs, the pod stays up.
#
#   nohup bash scripts/week6_finish.sh A > week6_finish_A.out 2>&1 &
#
# Requires `runpodctl config --apiKey <key>` done on the pod beforehand (checked at start).
# Results land in /workspace/week6_results/<list>/ (data/week6 and analysis/week6, adapters
# and checkpoints included) plus a copy of the run log and this script's own log.
set -uo pipefail
cd "$(dirname "$0")/.."
LIST=${1:?usage: week6_finish.sh A|B|C}
DEST=/workspace/week6_results/$LIST
LOG=/root/week6_finish_$LIST.log   # outside the tree that gets copied and verified
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if ! runpodctl get pod "$RUNPOD_POD_ID" > /dev/null 2>&1; then
  log "runpodctl is not authorised on this pod; run 'runpodctl config --apiKey <key>' first. Not starting."
  exit 2
fi
if [ ! -d /workspace ] || ! touch /workspace/.week6_write_test 2>/dev/null; then
  log "/workspace is not a writable volume; not starting."
  exit 2
fi
rm -f /workspace/.week6_write_test
log "authorised; waiting for list $LIST to finish (pod $RUNPOD_POD_ID)"

# Wait for the list's process to exit, then for its closing line in the run log.
while pgrep -f "scripts/week6_pod.sh $LIST" > /dev/null; do sleep 60; done
if ! grep -q "pod list $LIST finished" data/week6/logs/run.log 2>/dev/null; then
  log "the list process is gone but run.log has no 'finished' line; leaving the pod up."
  exit 1
fi
log "list $LIST finished: $(grep "pod list $LIST finished" data/week6/logs/run.log | tail -n 1)"

# Copy, then verify byte for byte.
mkdir -p "$DEST"
rm -rf "$DEST/data_week6" "$DEST/analysis_week6"
cp -rT data/week6 "$DEST/data_week6" && { [ ! -d analysis/week6 ] || cp -rT analysis/week6 "$DEST/analysis_week6"; }
cp week6_$LIST.out "$DEST/" 2>/dev/null
if diff -rq data/week6 "$DEST/data_week6" > "$DEST/verify_data.txt" 2>&1 && \
   { [ ! -d analysis/week6 ] || diff -rq analysis/week6 "$DEST/analysis_week6" > "$DEST/verify_analysis.txt" 2>&1; }; then
  sync
  log "copy verified: $(du -sh "$DEST" | cut -f1) at $DEST"
else
  log "copy differs from source (see $DEST/verify_*.txt); leaving the pod up."
  exit 1
fi
cp "$LOG" "$DEST/" 2>/dev/null
log "removing pod $RUNPOD_POD_ID in 60 s"
sleep 60
runpodctl remove pod "$RUNPOD_POD_ID"
