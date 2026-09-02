#!/usr/bin/env bash
# Stage a COMPLETE copy of the v5 project on iris-hgx-2's local disk (/scr/kewalk_v5/memory_project_v5)
# and run training from it: the node reads /iris at ~1 MB/s under load, a stream to its local
# disk moves ~100 MB/s. Everything the v3.5 runtime-path contract pins to the project root
# (caches, data, assets, checkpoints) then resolves inside the local root -- no code change.
# Run FROM a fast-NFS host (iris-ws-18). Re-runnable (tar overwrites). ~140 GB, ~15-25 min.
#   bash cluster_v5/stage_local_project_hgx2.sh
# Layout: /scr/kewalk_v5/python (uv CPython), /scr/kewalk_v5/memory_project_v5/{openpi,i2rt,
#   openpi_test_mem,data,v35,v5,.git,README.md,.gitignore}; the venv is relocated to the local
#   python and its editable .pth files to the local src. Marker: <root>/.staged
set -u
export HOME=/iris/u/kewalk
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_claude
src=/iris/u/kewalk/memory_project_v5
v4=/iris/u/kewalk/memory_project_v4
dst=/scr/kewalk_v5/memory_project_v5
log=$src/v5/diagnostics/stage_local_hgx2.log
errlog=$src/v5/diagnostics/stage_local_hgx2.err
S() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR iris-hgx-2 "$@" 2>> "$errlog"; }
export -f S; export errlog dst
stamp() { echo "$(date +%H:%M:%S) $*" >> "$log"; }
stream() {  # label src-dir dst-subdir [tar-excludes...]
  local label="$1" from="$2" to="$3"; shift 3
  local t0=$(date +%s)
  tar -C "$from" "$@" -cf - . | S "mkdir -p $dst/$to && tar -C $dst/$to -xf -"
  stamp "$label: exit=${PIPESTATUS[1]} in $(( $(date +%s) - t0 ))s"
}
export -f stream stamp; export log
rm -f "$errlog"; stamp "stage start"
S "mkdir -p $dst && rm -f $dst/.staged && mkdir -p $dst/v35/{assets,checkpoints,diagnostics,tmp,wandb} $dst/v35/cache/{jax,uv,huggingface,openpi} $dst/v5/{assets,checkpoints,diagnostics} $dst/data $dst/openpi/.venv"
# --- group 1: code + venv (small files: skeleton stream, then site-packages in 24 parallel streams)
(
  stream code "$src" . --exclude='./openpi/.venv' --exclude='./v35' --exclude='./v5' --exclude='./data' --exclude='./.claude'
  stream venv-skeleton "$src/openpi/.venv" openpi/.venv --exclude='./lib/python3.11/site-packages'
  t0=$(date +%s); sp=$src/openpi/.venv/lib/python3.11/site-packages; export sp
  S "mkdir -p $dst/openpi/.venv/lib/python3.11/site-packages"
  ( cd "$sp" && ls -A ) | xargs -d '\n' -P 6 -n 40 bash -c 'tar -C "$sp" -cf - "$@" | S "tar -C $dst/openpi/.venv/lib/python3.11/site-packages -xf -"' _
  stamp "venv-site-packages: exit=$? in $(( $(date +%s) - t0 ))s"
  stream v5-assets "$src/v5/assets" v5/assets
  stream hf-hub "$src/v35/cache/huggingface/hub" v35/cache/huggingface/hub
  stream hf-modules "$src/v35/cache/huggingface/modules" v35/cache/huggingface/modules
  stream data-raw "$v4/data" data --exclude='./lerobot'
) &
# --- group 2..5: big streams, concurrent
( stream weights "$src/v35/cache/openpi" v35/cache/openpi ) &
( stream arrow-A "$src/v35/cache/huggingface/datasets/parquet/default-6bd6369c39f4ed85" v35/cache/huggingface/datasets/parquet/default-6bd6369c39f4ed85 ) &
( stream arrow-B "$src/v35/cache/huggingface/datasets/parquet/default-eeef94aaa0a4c792" v35/cache/huggingface/datasets/parquet/default-eeef94aaa0a4c792 ) &
( stream lerobot "$v4/data/lerobot" data/lerobot ) &
wait
cp "$src/v35/cache/huggingface/datasets/"*.lock /dev/null 2>/dev/null
tar -C "$src/v35/cache/huggingface/datasets" -cf - --exclude='./parquet' . | S "tar -C $dst/v35/cache/huggingface/datasets -xf -"
# --- relocate the venv to the local python and the local source tree
S "cd $dst/openpi/.venv && sed -i 's#^home = .*#home = /scr/kewalk_v5/python/bin#' pyvenv.cfg && for b in python python3 python3.11; do ln -sfn /scr/kewalk_v5/python/bin/python3.11 bin/\$b; done && cd lib/python3.11/site-packages && sed -i 's#/iris/u/kewalk/memory_project_v5/#$dst/#g' _editable_impl_openpi.pth _editable_impl_openpi_client.pth && cat _editable_impl_openpi.pth _editable_impl_openpi_client.pth"
S "cd $dst/openpi && MEMORY_PROJECT_ROOT=$dst .venv/bin/python -c 'import sys, jax, openpi.shared.project_paths as pp; pp.configure_v35_runtime_environment(); pp.validate_executing_openpi_checkout(); print(\"local project ok\", sys.executable, jax.__version__, pp.memory_project_root())' && touch $dst/.staged && du -sh $dst/* $dst/data/* $dst/v35/cache/* 2>/dev/null && df -h /scr | tail -1" >> "$log" 2>&1
stamp "stage done staged=$(S "test -e $dst/.staged && echo yes || echo NO")"
