#!/bin/sh
exec /bin/sh "$(dirname "$0")/sh/minio-init.sh" "$@"
