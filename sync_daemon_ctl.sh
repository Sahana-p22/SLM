#!/bin/bash
# chat/backend/sync_daemon_ctl.sh start|stop|status
# Manages the mongo_sqlite_sync.py background daemon for this deployment.
set -e
REPO=/home/wgtech/slm-llama3b-sqlite
VENV=/home/wgtech/slm-main/.venv
PIDFILE="$REPO/sync_daemon.pid"
LOGFILE="$REPO/sync_daemon.log"

cd "$REPO"
source "$VENV/bin/activate"

case "$1" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "sync daemon already running, pid $(cat "$PIDFILE")"
      exit 0
    fi
    setsid nohup python3 -u chat/backend/mongo_sqlite_sync.py > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    disown
    sleep 1
    echo "sync daemon started, pid $(cat "$PIDFILE")"
    ;;
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "sync daemon stopped"
    else
      echo "sync daemon not running"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running, pid $(cat "$PIDFILE")"
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 start|stop|status"
    exit 1
    ;;
esac
