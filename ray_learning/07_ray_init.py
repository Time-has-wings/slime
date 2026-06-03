"""
Ray Init —— 不同启动方式详解

这是理解 Ray 的关键: 不同的 ray.init() 方式决定了
Ray 集群的架构和运行模式。

四种主要模式:
  1. ray.init()              → 嵌入式单机集群
  2. ray.init(address="auto") → 连接已有集群
  3. ray.init(address="node:port") → 连接指定集群
  4. ray.init(num_gpus=N)     → 手动指定资源 (旧版)

运行: python3 ray_learning/07_ray_init.py

先跑 ray start --head --num-gpus 1 再跑这个脚本
可以对比 address="auto" 的效果
"""

# ========== 模式 1: 嵌入式单机集群 ==========
# 最常用: 不需要外部 Ray 进程, 在当前进程里启动
# Ray 会自己检测所有硬件资源
print("=" * 60)
print("模式 1: ray.init() — 嵌入式单机集群")
print("=" * 60)

import ray

ray.init(address="auto", log_to_driver=False)
print(f"  地址: {ray.get_runtime_context().gcs_address}")
print(f"  GPU:  {ray.cluster_resources().get('GPU', 0)}")
print(f"  CPU:  {ray.cluster_resources().get('CPU', 0)}")
print(f"  运行模式: {'单机' if '127.0.0.1' in ray.get_runtime_context().gcs_address else '集群'}")

ray.shutdown()
print()

# ========== 模式 2: 连接到已有集群 ==========
# 需要先执行: ray start --head ...
print("=" * 60)
print("模式 2: ray.init(address='auto') — 连接已有集群")
print("=" * 60)
print("  (需要先运行 'ray start --head' 才有效)")
print("  如果没运行, 会报错: ConnectionError")
print()

try:
    ray.init(address="auto", log_to_driver=False)
    print(f"  连接成功! 地址: {ray.get_runtime_context().gcs_address}")
    ray.shutdown()
except Exception as e:
    print(f"  没有运行中的 Ray 集群: {type(e).__name__}")
print()

# ========== 模式对比 ==========
print("=" * 60)
print("四种启动模式对比")
print("=" * 60)
print("""
  ray.init()
    启动方式: 当前 Python 进程中启动 Ray
    适用场景: 单机开发/测试
    优缺点:   简单, 自动检测资源; 进程退出 Ray 也退出

  ray.init(address="auto")
    启动方式: 连接已有 Ray 集群 (需要先 ray start)
    适用场景: 多机训练 / slurm 集群
    优缺点:   可以跨节点, 但需要预先启动 Ray

  ray.init(address="node:6379")
    启动方式: 连接指定地址的 GCS
    适用场景: 明确知道集群地址
    优缺点:   精准控制, 但不够灵活

  ray.init() 内部做了什么?
    1. 启动 GCS (Global Control Store) 进程
    2. 启动 Object Store (共享内存)
    3. 启动 Worker 进程池
    4. 注册当前节点
""")

# ========== 实际情况：slime 项目 ==========
print("=" * 60)
print("slime 项目中 Ray 的使用方式")
print("=" * 60)
print("""
  原始方式 (你之前遇到的问题):
    ray start --head --num-gpus 1   (启动外部 Ray 集群)
    ray job submit ... -- python3 train.py   (提交任务)
    # train.py 里不需要 ray.init(), 因为 ray job submit 会自动初始化

  你的需求: 去掉 ray job submit
    方案 A: ray start --head + train.py 里加 ray.init(address="auto")
            → 连接已有集群运行

    方案 B: 去掉 ray start, train.py 里用 ray.init()
            → 嵌入式启动, 自动检测所有 GPU
            → 最简单, 推荐

  建议:
    - 开发/调试用方案 B (ray.init())
    - 生产多机用方案 A (ray start + ray.init(address="auto"))
""")

ray.shutdown()
print("\nRay 已关闭")
