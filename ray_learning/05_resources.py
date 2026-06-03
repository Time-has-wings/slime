"""
Ray Resource Management —— GPU/CPU 资源管理

重点是理解 Ray 的资源调度机制:
  1. ray.init() 时会检测集群资源
  2. @ray.remote(num_gpus=N, num_cpus=M) 声明需要多少资源
  3. Ray 调度器确保不超额分配

运行: python3 ray_learning/05_resources.py
"""

import ray
import time

ray.init(address="auto", log_to_driver=False)
print("集群资源:", ray.cluster_resources())
print()

# ========== 1. GPU 资源管理 ==========
print("=== GPU 资源 ===")
total_gpus = ray.cluster_resources().get("GPU", 0)
print(f"总 GPU 数: {total_gpus}")


@ray.remote(num_gpus=1)
class GpuConsumer:
    """每个实例占用 1 张 GPU"""

    def __init__(self, idx):
        self.idx = idx
        self.gpu_ids = ray.get_gpu_ids()

    def report(self):
        return {
            "actor": self.idx,
            "assigned_gpu_ids": self.gpu_ids,
        }


# 创建尽量多的 GPU Actor
gpu_actors = []
actual_gpus = int(total_gpus) if total_gpus > 0 else 0
num_actors = min(actual_gpus, 4)

for i in range(num_actors):
    try:
        actor = GpuConsumer.remote(i)
        info = ray.get(actor.report.remote())
        gpu_actors.append(info)
        print(f"  Actor {i}: GPU ID = {info['assigned_gpu_ids']}")
    except Exception as e:
        print(f"  Actor {i} 创建失败: {e}")

if not gpu_actors:
    print("  (当前环境没有 GPU, 跳过 GPU Actor 测试)")

print()

# ========== 2. GPU 超订 ==========
# 如果在同一张 GPU 上执行多个任务, 需要设置 num_gpus=0 或者用 fractional GPU
print("=== GPU 超订 / 共享 GPU ===")


@ray.remote(num_gpus=0.5)
class HalfGpuActor:
    def __init__(self, idx):
        self.idx = idx

    def info(self):
        return f"Actor {self.idx} on {ray.util.get_node_ip_address()}"

# 注意: 如果实际 GPU 数 = 0, 这也会失败
if actual_gpus > 0:
    half_actors = []
    for i in range(int(actual_gpus * 2)):
        try:
            a = HalfGpuActor.remote(i)
            half_actors.append(a)
        except Exception:
            pass
    print(f"用 num_gpus=0.5 创建了 {len(half_actors)} 个 Actor (共享 GPU)")
print()

# ========== 3. CPU 资源 ==========
print("=== CPU 资源 ===")


@ray.remote(num_cpus=2)
class CpuBoundActor:
    def compute(self, n):
        return sum(i * i for i in range(n))


total_cpus = ray.cluster_resources().get("CPU", 0)
print(f"总 CPU 数: {total_cpus}")

# 创建一些 CPU-bound Actor
cpu_actors = []
for i in range(min(3, int(total_cpus) // 2 if total_cpus else 0)):
    try:
        a = CpuBoundActor.remote()
        cpu_actors.append(a)
    except Exception:
        break

print(f"创建了 {len(cpu_actors)} 个 CPU-bound Actor (每个 2 CPU)")
if cpu_actors:
    refs = [a.compute.remote(1000000) for a in cpu_actors]
    results = ray.get(refs)
    print(f"计算结果: {[r // 1000000 for r in results]}M ...")
print()

# ========== 4. 资源超订 (Oversubscription) ==========
# 有时候希望创建比硬件更多的 worker, 用 num_cpus=0
print("=== 超订: num_cpus=0 ===")


@ray.remote(num_cpus=0)
class LightweightActor:
    """不占用 CPU 资源的 Actor, 可以创建很多个"""

    def __init__(self, idx):
        self.idx = idx

    def ping(self):
        return f"lightweight-{self.idx}"


many_actors = [LightweightActor.remote(i) for i in range(100)]
results = ray.get([a.ping.remote() for a in many_actors[:5]])
print(f"100 个 lightweight Actor 已创建, 前 5 个: {results}")
print()

# ========== 5. 调度策略 ==========
# PlacementGroupSchedulingStrategy 可以控制 Actor 调度到哪个 bundle
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

print("=== 调度策略 ===")

# 创建一个简单的 placement group
bundles = [{"CPU": 1, "GPU": 0} for _ in range(2)]
pg = placement_group(bundles, strategy="PACK")
ray.get(pg.ready())
print(f"Placement group ready: {pg}")


@ray.remote
class PinnedActor:
    def __init__(self, idx):
        self.idx = idx

    def get_node(self):
        return ray.util.get_node_ip_address()


# 把 Actor 调度到 placement group 的指定 bundle
actor_on_bundle0 = PinnedActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
).remote(0)

actor_on_bundle1 = PinnedActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=1,
    )
).remote(1)

print(f"Actor on bundle 0: {ray.get(actor_on_bundle0.get_node.remote())}")
print(f"Actor on bundle 1: {ray.get(actor_on_bundle1.get_node.remote())}")

ray.util.remove_placement_group(pg)
print()

ray.shutdown()
print("Ray 已关闭")
