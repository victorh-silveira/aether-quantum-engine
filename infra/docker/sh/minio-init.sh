#!/bin/sh
set -eu

MC_ALIAS="${MC_ALIAS:-aether}"
ENDPOINT="${MINIO_ENDPOINT:-http://minio:9000}"
ACCESS="${MINIO_ROOT_USER:-aether}"
SECRET="${MINIO_ROOT_PASSWORD:-aethersecret}"
BUCKET="${AETHER_MINIO_BUCKET:-dl-models}"
ILM_PREFIX="${AETHER_MINIO_ILM_PREFIX:-optuna/}"
ILM_DAYS="${AETHER_MINIO_ILM_DAYS:-7}"

i=0
until mc alias set "$MC_ALIAS" "$ENDPOINT" "$ACCESS" "$SECRET" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 60 ]; then
    echo "minio-init: alias timeout" >&2
    exit 1
  fi
  sleep 1
done

mc mb --ignore-existing "${MC_ALIAS}/${BUCKET}"
mc anonymous set none "${MC_ALIAS}/${BUCKET}" >/dev/null 2>&1 || true

cat >/tmp/aether-ilm.json <<EOF
{
  "Rules": [
    {
      "ID": "aether-optuna-expire-${ILM_DAYS}d",
      "Status": "Enabled",
      "Filter": { "Prefix": "${ILM_PREFIX}" },
      "Expiration": { "Days": ${ILM_DAYS} }
    }
  ]
}
EOF

mc ilm import "${MC_ALIAS}/${BUCKET}" </tmp/aether-ilm.json
echo "minio-init: bucket=${BUCKET} ilm_prefix=${ILM_PREFIX} days=${ILM_DAYS}"
