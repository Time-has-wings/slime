"""
Ray Job —— ray job submit 的工作原理与替代方案

ray job submit 干了什么:
  1. 连接 Ray Dashboard (默认 http://127.0.0.1:8265)
  2. 把代码和依赖打包上传
  3. 在 Ray 集群上启动一个 job
  4. job 会自动调用 ray.init()

为什么 ray job submit 可能卡住:
  - Dashboard 端口 (8265) 没正确暴露
  - Ray 集群状态不健康
  - 网络问题

替代方案对比:
  ray job submit           vs  直接 python3 train.py
  需要 Ray Dashboard             不需要 Dashboard
  更复杂的依赖管理               依赖由 shell 管理
  适合生产多机部署               适合开发调试
  自带日志收集                   日志直接输出到终端

运行: python3 ray_learning/08_ray_job.py
"""

import ray
import subprocess
import sys
import os

ray.init(address="auto", log_to_driver=False)
print("Ray 已启动 (嵌入式模式)")
print()

# ========== 1. 模拟 ray job submit 的效果 ==========
# ray job submit 本质上就是启动一个 Python 进程,
# 这个进程连接到已有的 Ray 集群

print("=" * 60)
print("ray job submit 的本质")
print("=" * 60)
print("""
  ray job submit  -- python3 train.py
         │
         ├── 1. 打包当前环境
         ├── 2. 通过 Dashboard API 提交到集群
         ├── 3. 集群启动新 Python 进程
         ├── 4. 新进程自动执行 ray.init(address="auto")
         └── 5. 开始执行 train.py

  直接运行 python3 train.py 需要:
         ├── 1. ray.init() 自己启动 Ray
         │      (或者保证 Ray 已经 ray start 好了)
         ├── 2. 设置环境变量 (PYTHONPATH, CUDA 等)
         └── 3. 开始执行 train.py
""")

# ========== 2. 对比: 两种启动方式 ==========
print("=" * 60)
print("两种启动方式对比")
print("=" * 60)

print("""
  方式 A (原始脚本 — 卡在 ray job submit):
    ray start --head --num-gpus 1
    ray job submit --address="http://127.0.0.1:8265" \\
      --runtime-env-json='{"env_vars": {...}}' -- \\
      python3 train.py <ARGS>

  方式 B (修改后 — 直接运行):
    (不需要 ray start)
    export PYTHONPATH=... CUDA_DEVICE_MAX_CONNECTIONS=1
    python3 train.py <ARGS>
    # train.py 里调用 ray.init()
""")

# ========== 3. 模拟: subprocess 提交任务 ==========
print("=" * 60)
print("模拟 job submit: subprocess 启动任务")
print("=" * 60)

# 创建一个临时脚本
tmp_script = "/tmp/ray_job_example.py"
with open(tmp_script, "w") as f:
    f.write("""
import ray
ray.init(address="auto", log_to_driver=False)

@ray.remote
def hello():
    import socket
    return f"Hello from {socket.gethostname()}, GPU: {ray.get_gpu_ids()}"

print(ray.get(hello.remote()))
print("Job done!")
ray.shutdown()
""")

print(f"创建临时任务脚本: {tmp_script}")
print("启动方式: subprocess.run(['python3', tmp_script])")
print()

# 其实可以用 subprocess 启动
# 但这里直接演示 ray.init(address="auto") 在当前进程
# ========== 4. 检查是否连接到已有集群 ==========
print("=" * 60)
print("检查当前 Ray 集群状态")
print("=" * 60)

resources = ray.cluster_resources()
print(f"  节点数: {len(ray.nodes())}")
print(f"  总 GPU: {resources.get('GPU', 0)}")
print(f"  总 CPU: {resources.get('CPU', 0)}")

ray.shutdown()
print()

# ========== 5. 总结 ==========
print("=" * 60)
print("总结: 什么时候用什么方式")
print("=" * 60)
print("""
  场景                       推荐方式
  ─────────────────────────────────────────────
  单机开发/调试              ray.init()
  单机但需要独立 Ray 进程    ray start + ray.init(address="auto")
  多机分布式训练             ray start + ray.init(address="auto")
  生产环境 (需要日志管理)    ray job submit (正确配置下)
  CI/CD 测试                 ray.init() (最简洁)

  你的情况:
  8 张 H20, 单机训练
  → 推荐 ray.init(), 不需要 ray job submit
""")
