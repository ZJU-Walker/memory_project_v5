#!/usr/bin/env bash
# ssh to the v5 node with every user file under /iris/u/kewalk (the passwd home is a read-only
# AFS directory; plain ssh tries to create ~/.ssh there and spams warnings). Kerberos/GSSAPI only.
#   cluster_v5/ssh_hgx2.sh [host] '<remote command>'      (host defaults to iris-hgx-2)
export HOME=/iris/u/kewalk
export KRB5CCNAME="${KRB5CCNAME:-FILE:/tmp/krb5cc_24706_claude}"
host=iris-hgx-2.stanford.edu
if [[ $# -ge 2 ]]; then host="$1"; shift; fi
mkdir -p /iris/u/kewalk/.ssh
exec ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR \
  -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o StrictHostKeyChecking=no \
  -o IdentitiesOnly=yes -o IdentityFile=/iris/u/kewalk/.ssh/none -o GSSAPIAuthentication=yes -o GSSAPITrustDNS=yes \
  "$host" "$@"
