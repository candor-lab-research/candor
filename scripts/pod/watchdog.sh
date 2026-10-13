#!/bin/bash
# Deletes this pod once its deadline passes, so a pod nobody deletes can't keep billing.
#
# The pod's own start command runs it from boot (scripts/pod.sh args), so it has the pod's Runpod
# key and id, which SSH sessions don't see. The deadline, in seconds since 1970, starts as
# POD_DELETE_AT and lives in the deadline file, where scripts/pod.sh moves it. The file is on the
# container disk: a restart wipes it, and the watchdog starts again from POD_DELETE_AT. A deadline
# that isn't a plain number of seconds counts as passed.

file=${POD_DEADLINE_FILE:-/pod-delete-at}
interval=${POD_WATCHDOG_INTERVAL:-30}
seconds='^[1-9][0-9]{0,10}$'

[[ -s $file ]] || echo "$POD_DELETE_AT" >"$file"

while true; do
    deadline=$(<"$file")
    if [[ ! $deadline =~ $seconds ]] || (($(date +%s) >= deadline)); then
        echo "$(date -u +%FT%TZ) deadline '$deadline' passed: deleting pod $RUNPOD_POD_ID"
        runpodctl pod delete "$RUNPOD_POD_ID" ||
            runpodctl remove pod "$RUNPOD_POD_ID" ||
            curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
                "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID"
    fi
    sleep "$interval"
done
