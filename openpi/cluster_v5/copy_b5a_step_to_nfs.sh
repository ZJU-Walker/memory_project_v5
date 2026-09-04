#!/usr/bin/env bash
# Login node: copy one B5a checkpoint (params+assets only) from iris-hgx-2 /scr to NFS as keep_<STEP>.
#   STEP=2999 bash cluster_v5/copy_b5a_step_to_nfs.sh
export HOME=/iris/u/kewalk KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
STEP=${STEP:-2999}
run=v5/checkpoints/pi05_yam_mem_v5_stageB5a/v5_stageB5a_20260903_r1
src=/scr/kewalk_v5/memory_project_v5/$run/$STEP
dst=/iris/u/kewalk/memory_project_v5/$run/keep_$STEP
log=/iris/u/kewalk/memory_project_v5/v5/diagnostics/copy_b5a_$STEP.log
mkdir -p $dst; t0=$(date +%s)
ssh -o BatchMode=yes iris-hgx-2 "tar -C $src -cf - params assets" 2>/dev/null | tar -C $dst -xf -
rc="${PIPESTATUS[0]}/${PIPESTATUS[1]}"
echo "B5a $STEP -> keep_$STEP exit=$rc $(( $(date +%s) - t0 ))s size=$(du -sh $dst | cut -f1) $(date '+%m/%d %H:%M')" >> $log
[ "$rc" = "0/0" ] && [ -e $dst/params/manifest.ocdbt ] && touch $dst/.copied && echo "copied OK" >> $log
