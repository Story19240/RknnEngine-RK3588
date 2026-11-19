#!/bin/bash
# Container entrypoint. Mode selected by first arg:
#   docker run rknn-convert convert --weights ...     # one-shot pipeline
#   docker run -it rknn-convert                        # interactive bash
#   docker run rknn-convert <cmd...>                   # exec arbitrary cmd
set -e

mode="${1:-}"
case "$mode" in
  convert|run)
    shift
    exec /opt/convert/run_convert.sh "$@"
    ;;
  "")
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac