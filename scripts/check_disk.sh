#!/bin/bash
# Check disk usage under /tmp and /tmp/linguangming

echo "=========================================="
echo "  Disk Space"
echo "=========================================="
df -h /tmp

echo ""
echo "=========================================="
echo "  /tmp/linguangming Top-level"
echo "=========================================="
du -h --max-depth=1 /tmp/linguangming/ 2>/dev/null | sort -rh

echo ""
echo "=========================================="
echo "  /tmp/linguangming/slime-workspace"
echo "=========================================="
du -h --max-depth=1 /tmp/linguangming/slime-workspace/ 2>/dev/null | sort -rh

echo ""
echo "=========================================="
echo "  /tmp/linguangming/slime-workspace/slime"
echo "=========================================="
du -h --max-depth=1 /tmp/linguangming/slime-workspace/slime/ 2>/dev/null | sort -rh

echo ""
echo "=========================================="
echo "  /tmp/linguangming/slime-workspace/slime_env"
echo "=========================================="
du -h --max-depth=1 /tmp/linguangming/slime-workspace/slime_env/ 2>/dev/null | sort -rh
