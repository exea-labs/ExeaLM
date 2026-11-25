#!/bin/bash
# Destroy the MI300X droplet once everything on it is saved. Run it yourself, on your own machine:
#
#     export DIGITALOCEAN_TOKEN=<your DigitalOcean API token>      # never commit it; revoke it when you are done
#     bash experiments/finalize/destroy_box.sh                        # checks, then asks before destroying
#
# Destroying deletes the droplet and its disk for good, and stops the billing (a powered-off droplet is still
# billed). Before it asks, it checks:
#   1. the data slices are on Hugging Face (exea-labs/exealm-pretrain: slice-30b and, if built, slice-30b-r2);
#   2. the MI300X results and models are on Hugging Face (exea-labs/exealm-nairr-mi300x);
#   3. the box's own watchdog log says all work finished (if the box is still reachable).
# Then it shows the droplet it found by its address and asks you to type the droplet's name. Nothing else confirms.
# BOX_IP picks the droplet (default 165.245.134.58). FORCE_CHECKS=0 skips the checks (still asks for the name).
set -uo pipefail
BOX_IP=${BOX_IP:-165.245.134.58}
API=https://api.digitalocean.com/v2
: "${DIGITALOCEAN_TOKEN:?export DIGITALOCEAN_TOKEN=<your API token> first}"
auth=(-H "Authorization: Bearer $DIGITALOCEAN_TOKEN" -H "Content-Type: application/json")

hf_has() {                                   # hf_has <repo type> <repo> <path prefix>: does that path exist on main
  curl -s -m 60 "https://huggingface.co/api/${1}s/$2/tree/main/$3" | python3 -c "
import sys, json
try: t = json.load(sys.stdin)
except Exception: sys.exit(1)
sys.exit(0 if isinstance(t, list) and t else 1)"
}

if [ "${FORCE_CHECKS:-1}" = 1 ]; then
  ok=1
  for rel in slice-30b slice-30b-r2; do
    if hf_has dataset exea-labs/exealm-pretrain "$rel"; then echo "ok: exea-labs/exealm-pretrain/$rel is on Hugging Face"
    else echo "MISSING: exea-labs/exealm-pretrain/$rel"; [ "$rel" = slice-30b-r2 ] || ok=0; fi
  done
  # the results repo is private: the public API answers only with a token, so ask with the local HF login if any
  if python3 -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('exea-labs/exealm-nairr-mi300x')
need = ['results/finalize/', 'runs/finalize/']
missing = [n for n in need if not any(f.startswith(n) for f in files)]
print('ok: exea-labs/exealm-nairr-mi300x has results and models' if not missing else f'MISSING in exea-labs/exealm-nairr-mi300x: {missing}')
raise SystemExit(1 if missing else 0)" 2>/dev/null; then :; else
    echo "could not confirm exea-labs/exealm-nairr-mi300x (log in with huggingface-cli to check a private repo)"; ok=0
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=15 root@"$BOX_IP" 'tail -2 /root/ExeaLM/logs/poweroff.log' 2>/dev/null; then
    echo "(the last lines of the box's watchdog log, above)"
  else
    echo "the box does not answer (powered off by its watchdog, or unreachable)"
  fi
  if [ "$ok" != 1 ]; then
    echo; echo "Not everything is confirmed saved. Fix that first, or rerun with FORCE_CHECKS=0 if you are sure."; exit 1
  fi
fi

droplet=$(curl -s -m 60 "${auth[@]}" "$API/droplets?per_page=200" | python3 -c "
import sys, json
ip = sys.argv[1]
for d in json.load(sys.stdin).get('droplets', []):
    if any(n.get('ip_address') == ip for n in d['networks'].get('v4', [])):
        print(d['id'], d['name'], d['size_slug'], d['status'], d['region']['slug'])
" "$BOX_IP")
[ -n "$droplet" ] || { echo "no droplet with address $BOX_IP in this account"; exit 1; }
read -r id name size status region <<< "$droplet"
echo
echo "Droplet: $name (id $id), size $size, $status, region $region, address $BOX_IP"
echo "This deletes the droplet and its disk permanently. Type the droplet's name to destroy it:"
read -r answer
[ "$answer" = "$name" ] || { echo "names differ: nothing destroyed"; exit 1; }
code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "${auth[@]}" "$API/droplets/$id")
if [ "$code" = 204 ]; then echo "destroyed droplet $name ($id). Revoke the API token now if you are done with it."
else echo "DigitalOcean answered HTTP $code: not destroyed"; exit 1; fi
