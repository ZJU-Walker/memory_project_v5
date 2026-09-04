#!/usr/bin/env bash
# Run on a LOGIN node (fast NFS, Kerberos): wait until the A5 ckpt-999 is protected on iris-hgx-2, then stream
# its params to the NFS worktree path the A6 config's audited loader reads, and drop a marker for the hgx-1 queue.
export HOME=/iris/u/kewalk KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
dst=/iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageA5/v5_stageA5_20260903_r1/999
log=$diag/copy_a5_999.log
until grep -q "A5 ckpt-999 protected" $diag/queue_a5_b5a.log 2>/dev/null; do sleep 60; done
echo "A5 999 protected; copying to NFS $(date '+%m/%d %H:%M')" >> $log
mkdir -p $dst
t0=$(date +%s)
ssh -o BatchMode=yes iris-hgx-2 'tar -C /scr/kewalk_v5/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageA5/v5_stageA5_20260903_r1/keep_999 -cf - params' 2>/dev/null | tar -C $dst -xf -
echo "copy exit=${PIPESTATUS[0]}/${PIPESTATUS[1]} $(( $(date +%s) - t0 ))s size=$(du -sh $dst/params | cut -f1)" >> $log
[ -e $dst/params/manifest.ocdbt ] && touch $dst/.copied && echo "marker written $(date '+%m/%d %H:%M')" >> $log
