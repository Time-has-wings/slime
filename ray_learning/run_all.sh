#!/bin/bash
# Ray Learning 全套示例运行脚本
# 按顺序运行所有示例

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=========================================="
echo "  Ray Learning 示例"
echo "=========================================="
echo

run_example() {
    echo "=========================================="
    echo "  运行: $1"
    echo "=========================================="
    python3 "$1"
    echo
    echo "  完成: $1"
    echo "=========================================="
    echo
}

# 清理残留
ray stop --force 2>/dev/null || true
sleep 1

run_example "ray_learning/01_quickstart.py"
run_example "ray_learning/02_tasks.py"
run_example "ray_learning/03_actors.py"
run_example "ray_learning/04_objects.py"
run_example "ray_learning/05_resources.py"
run_example "ray_learning/06_placement_group.py"
run_example "ray_learning/07_ray_init.py"
run_example "ray_learning/08_ray_job.py"
run_example "ray_learning/09_distributed_training_pattern.py"

echo "=========================================="
echo "  全部示例运行完毕!"
echo "=========================================="
