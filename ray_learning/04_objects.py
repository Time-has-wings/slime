"""
Ray Objects —— 对象存储

核心 API:
  - ray.put(obj)    → 把对象存入共享内存, 返回 ObjectRef
  - ray.get(ref)    → 通过 ObjectRef 取出对象
  - ray.wait(refs)  → 等待部分对象就绪

为什么要用 ray.put?
  Tasks/Actor 之间传递大对象时, 用 ray.put 可以避免重复序列化。
  ray.put 把对象存进 Ray 的分布式共享内存 (Object Store), 所有 worker 都能快速访问。

运行: python3 ray_learning/04_objects.py
"""

import ray
import time
import numpy as np

ray.init(address="auto", log_to_driver=False)  # 连接到已有的 Ray 集群

# ========== 1. ray.put 基础 ==========
print("=== ray.put 基础 ===")

# 直接存一个对象到共享内存
data = {"hello": "world", "list": [1, 2, 3]}
ref = ray.put(data)

print(f"ObjectRef: {ref}")
print(f"类型: {type(ref)}")

# 通过 ObjectRef 取出来
retrieved = ray.get(ref)
print(f"取出数据: {retrieved}")
print(f"id(data) = {id(data)}, id(retrieved) = {id(retrieved)} (不同进程, 不同对象)")
print()

# ========== 2. 大对象共享 ==========
print("=== 大对象共享 ===")

# 创建一个大的 numpy 数组
big_array = np.random.rand(1000000)
ref = ray.put(big_array)

# 多个任务都可以通过同一个 ObjectRef 读取


@ray.remote
def process_chunk(data):
    return f"处理了 {len(data)} 个元素, 均值: {data.mean():.4f}"


start = time.time()
refs = [process_chunk.remote(ref) for _ in range(4)]
results = ray.get(refs)
elapsed = time.time() - start

for r in results:
    print(r)
print(f"4 个任务共用大对象耗时: {elapsed:.3f} 秒")
print()

# ========== 3. ray.put vs 直接传参 ==========
print("=== ray.put vs 直接传参 ===")


@ray.remote
def direct_pass(data):
    return len(data)


@ray.remote
def ref_pass(data):
    return len(data)


large_list = list(range(100000))

# 直接传: 每次调用都需要序列化再传
start = time.time()
refs = [direct_pass.remote(large_list) for _ in range(5)]
ray.get(refs)
direct_time = time.time() - start

# 用 ray.put: 只序列化一次
data_ref = ray.put(large_list)
start = time.time()
refs = [ref_pass.remote(data_ref) for _ in range(5)]
ray.get(refs)
ref_time = time.time() - start

print(f"直接传参: {direct_time:.3f} 秒 (序列化 5 次)")
print(f"ray.put:   {ref_time:.3f} 秒 (序列化 1 次)")
print()

# ========== 4. 对象生命周期 ==========
# ObjectRef 以引用计数方式管理
# 当没有引用指向它时, 对象会被自动回收

print("=== 对象生命周期 ===")

# 创建对象
ref = ray.put("temporary data")
print(f"创建对象: {ref}")

# 删除引用 → 对象会被回收
del ref

# 手动删除引用
ref2 = ray.put("more data")
print(f"手动删除: {ref2}")
ray.internal.free([ref2])  # 立即释放
print("已释放")
print()

# ========== 5. 超大对象 — 注意限制 ==========
# Ray 对象存储有内存上限
# 超出会 spill 到磁盘

print("=== 对象存储限制 ===")
object_store_memory = ray.cluster_resources().get("object_store_memory", 0)
print(f"对象存储内存: {object_store_memory / 1024**3:.2f} GB")

# 查看可用资源
print(f"所有资源: {ray.cluster_resources()}")
print()

ray.shutdown()
print("Ray 已关闭")
