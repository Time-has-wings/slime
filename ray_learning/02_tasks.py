"""
Ray Tasks —— 远程任务进阶

涵盖:
  - 批量任务与 ray.get 超时
  - ray.wait 等待部分完成
  - ray.cancel 取消任务
  - 任务重试与异常处理
  - 任务资源指定

运行: python3 ray_learning/02_tasks.py
"""

import ray
import time

ray.init(address="auto", log_to_driver=False)

# ========== 1. ray.wait —— 先到先得 ==========
# 当你只需要部分结果时, 可以用 ray.wait
# ray.wait(refs, num_returns=2, timeout=5)
#   返回: (ready_refs, remaining_refs)


@ray.remote
def random_sleep(x):
    import random
    t = random.uniform(0.5, 3)
    time.sleep(t)
    return f"任务 {x} 睡了 {t:.1f} 秒"


print("=== ray.wait 示例: 先到先得 ===")
refs = [random_sleep.remote(i) for i in range(5)]

ready, remaining = ray.wait(refs, num_returns=2, timeout=10)
print(f"最先完成的 {len(ready)} 个: {ray.get(ready)}")
print(f"还剩 {len(remaining)} 个未完成")
print()

# ========== 2. 批量获取 ==========
print("=== ray.get 批量获取 ===")
results = ray.get(refs)
print(f"全部结果: {results}")
print()

# ========== 3. 任务取消 ==========
# 可以用 ray.cancel() 取消正在运行的任务


@ray.remote
def long_task(x):
    import time
    for i in range(10):
        time.sleep(1)
    return x


print("=== ray.cancel 示例 ===")
ref = long_task.remote(42)
time.sleep(2)
ray.cancel(ref, force=False)  # force=False 是优雅取消
try:
    result = ray.get(ref, timeout=5)
    print(f"结果: {result}")
except ray.exceptions.TaskCancelledError:
    print("任务已被取消")
except ray.exceptions.GetTimeoutError:
    print("获取超时")
print()

# ========== 4. 任务重试 ==========
# @ray.remote(max_retries=N) 可以设置任务失败后重试次数


@ray.remote(max_retries=2)
def unreliable_task(x):
    import random
    if random.random() < 0.7:
        raise ValueError("随机失败!")
    return x * 2


print("=== 任务重试 示例 ===")
for i in range(5):
    try:
        result = ray.get(unreliable_task.remote(i))
        print(f"任务 {i} 成功: {result}")
    except Exception as e:
        print(f"任务 {i} 最终失败: {e}")
print()

# ========== 5. 指定资源 ==========
# 可以给任务指定需要的 CPU 数


@ray.remote(num_cpus=2)
def cpu_intensive_task(n):
    """这个任务需要 2 个 CPU"""
    return sum(i * i for i in range(n))


print("=== 指定资源 示例 ===")
ref = cpu_intensive_task.remote(10000)
result = ray.get(ref)
print(f"CPU 密集型任务结果: {result}")
print()

# ========== 6. 嵌套任务 (任务调用任务) ======
# 远程任务内部可以再调用其他远程任务


@ray.remote
def leaf(x):
    return x + 1


@ray.remote
def branch(x):
    # 在远程任务里调用其他远程任务
    refs = [leaf.remote(i) for i in range(x)]
    return ray.get(refs)


print("=== 嵌套任务 示例 ===")
result = ray.get(branch.remote(5))
print(f"嵌套任务结果: {result}")
print()

ray.shutdown()
print("Ray 已关闭")
