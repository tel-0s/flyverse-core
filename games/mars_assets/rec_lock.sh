#!/usr/bin/env bash
# One recording of games/mars.py under an ownership lock (for cluster jobs that a scheduler may preempt and retry).
#
#     bash games/mars_assets/rec_lock.sh OUTDIR NAME [games/mars.py arguments ...]
#
# writes OUTDIR/NAME.mp4, OUTDIR/NAME.json (the run log) and OUTDIR/NAME.console.txt. The first attempt makes
# OUTDIR/.own_NAME and records. A retry of the same job finds that directory and waits for the owner's run log
# instead of writing a second copy beside it. Only if the owner's console has been silent for LOCK_STALE_S seconds
# (default 1200; a live recording prints a line every brain second) does the retry take over: it moves the partial
# files to OUTDIR/preempted/ (kept, never used) and records from zero, and says so in OUTDIR/.own_NAME/owner.txt.
set -u
out=$1; name=$2; shift 2
stale=${LOCK_STALE_S:-1200}
mkdir -p "$out"
lock="$out/.own_$name"; con="$out/$name.console.txt"; js="$out/$name.json"; mp4="$out/$name.mp4"
record() {
    echo "$(date -u +%FT%TZ) $(hostname) pid $$ records $name: python games/mars.py $* --record $mp4" >> "$lock/owner.txt"
    python games/mars.py "$@" --record "$mp4" > "$con" 2>&1
}
if mkdir "$lock" 2>/dev/null; then
    record "$@"
    exit $?
fi
echo "mars-rec: an earlier attempt owns $name ($(tail -n 1 "$lock/owner.txt" 2>/dev/null)); waiting for $js"
while [ ! -f "$js" ]; do
    sleep 60
    [ -f "$js" ] && break
    if [ -f "$con" ]; then age=$(( $(date +%s) - $(stat -c %Y "$con") )); else age=$stale; fi
    if [ "$age" -ge "$stale" ]; then
        echo "mars-rec: the owner's console has been silent for $age s; taking over $name"
        mkdir -p "$out/preempted"
        for f in "$mp4" "$con"; do
            [ -f "$f" ] && mv "$f" "$out/preempted/$(basename "$f").$(date +%s)"
        done
        record "$@"
        exit $?
    fi
done
sleep 15
[ -f "$js" ]
