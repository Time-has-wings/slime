"""
Placement Group —— 资源分组调度

为什么需要 Placement Group?
  在分布式训练中, 我们通常需要 "一组资源" 而不是单个 GPU。
  比如训练需要 2 张 GPU, rollout 需要 2 张 GPU。
  用 Placement Group 可以保证这些资源在物理上靠近 (PACK) 或分散 (SPREAD)。

三种策略:
  PACK:  尽量把 bundle 放在同一台机器上 (适合多 GPU 互通)
  SPREAD: bundle 尽量分散到不同机器 (适合高可用)
  STRICT_PACK: 强制放在同一台机器, 否则失败

运行: python3 ray_learning/06_placement_group.py
"""

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time

ray.init(address="auto", log_to_driver=False)
print(f"集群资源: {ray.cluster_resources()}")
print()

# ========== 1. 基础 Placement Group ==========
print("=== 基础 PG: 2 个 bundle, 各 1 GPU (或 1 CPU) ===")

# 每个 bundle 是一组资源需求
# 这里创建 2 个 bundle, 每个需要 1 CPU
bundles = [
    {"CPU": 1},
    {"CPU": 1},
]

pg = placement_group(bundles, strategy="PACK")
ray.get(pg.ready(), timeout=30)
print(f"PG ID: {pg.id}")
print(f"PG 已就绪, bundles: {bundles}")
print()

# ========== 2. 在 PG 上调度 Actor ==========
# 用 PlacementGroupSchedulingStrategy 把 Actor 调度到特定 bundle


@ray.remote
class BundleWorker:
    def __init__(self, name):
        self.name = name
        self.node = ray.util.get_node_ip_address()

    def report(self):
        return f"{self.name} @ {self.node}"


# 调度到 bundle 0
worker_a = BundleWorker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
).remote("worker-A")

# 调度到 bundle 1
worker_b = BundleWorker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=1,
    )
).remote("worker-B")

print(f"A: {ray.get(worker_a.report.remote())}")
print(f"B: {ray.get(worker_b.report.remote())}")
print()

# ========== 3. PG 生命周期 ==========
# 不手动删除的话, PG 会一直存在
# 删除 PG 会杀死上面所有 Actor

print("=== PG 生命周期 ===")
print("删除 PG...")
ray.util.remove_placement_group(pg)
print("PG 已删除")
print()

# ========== 4. 训练场景模拟 ==========
# 模拟训练模式: 1 Actor + 1 Critique + 1 Rollout (colocate 模式)
print("=== 模拟 colocate 训练 PG ===")
print("  总 GPU 数: 1 (train + colocate rollout)")
print()

total_gpus = int(ray.cluster_resources().get("GPU", 0))
if total_gpus >= 1:
    # 创建 1 个 GPU bundle
    colocate_bundles = [{"GPU": 1, "CPU": 1}]
    colocate_pg = placement_group(colocate_bundles, strategy="PACK")
    ray.get(colocate_pg.ready(), timeout=30)

    print("创建训练 Actor (共享 GPU)...")


    @ray.remote(num_gpus=1)
    class TrainActor:
        def __init__(self):
            self.gpu_ids = ray.get_gpu_ids()
            print(f"  TrainActor 获得 GPU: {self.gpu_ids}")

        def train(self, step):
            import time
            time.sleep(0.1)
            return f"  train step {step} done"

    train_actor = TrainActor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=colocate_pg,
            placement_group_bundle_index=0,
        )
    ).remote()

    # 训练一步
    result = ray.get(train_actor.train.remote(1))
    print(result)
    print()

    ray.util.remove_placement_group(colocate_pg)
else:
    # 没有 GPU 时, 用 CPU 版本演示
    print("(没有 GPU, 用 CPU 版本演示)")


    @ray.remote(num_cpus=1)
    class DummyActor:
        def work(self, x):
            return f"work {x} done"

    cpu_bundles = [{"CPU": 1}]
    cpu_pg = placement_group(cpu_bundles, strategy="PACK")
    ray.get(cpu_pg.ready(), timeout=30)

    dummy = DummyActor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=cpu_pg,
            placement_group_bundle_index=0,
        )
    ).remote()

    print(ray.get(dummy.work.remote(1)))
    ray.util.remove_placement_group(cpu_pg)
    print()

# ========== 5. PG 三种策略对比 ==========
print("=== 三种 PG 策略对比 ===")
print("  PACK:        尽量放到同一节点 (GPU 间通信快)")
print("  SPREAD:      尽量分散到不同节点 (高可用)")
print("  STRICT_PACK: 必须同一节点, 否则失败")

ray.shutdown()
print("\nRay 已关闭")
