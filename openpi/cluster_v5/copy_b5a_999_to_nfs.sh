#!/usr/bin/env bash
# Login node: when B5a ckpt-999 is protected on iris-hgx-2, stream params+assets to NFS so the robot server on
# iris-hgx-1 can load it (cluster_v5/serve_v5_hgx1.sh <dir> pi05_yam_mem_v5_stageB5a).
export HOME=/iris/u/kewalk KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
diag=/iris/u/kewalk/memory_project_v5/v5/diagnostics
dst=/iris/u/kewalk/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_999
log=$diag/copy_b5a_999.log
until grep -q "B5a ckpt-999 protected" $diag/queue_a5_b5a.log 2>/dev/null; do sleep 60; done
mkdir -p $dst; t0=$(date +%s)
ssh -o BatchMode=yes iris-hgx-2 'tar -C /scr/kewalk_v5/memory_project_v5/v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1/keep_999 -cf - params assets' 2>/dev/null | tar -C $dst -xf -
echo "B5a keep_999 -> NFS exit=${PIPESTATUS[0]}/${PIPESTATUS[1]} $(( $(date +%s) - t0 ))s size=$(du -sh $dst | cut -f1) $(date '+%m/%d %H:%M')" >> $log
[ -e $dst/params/manifest.ocdbt ] && touch $dst/.copied
