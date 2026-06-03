"""
Ray 快速入门 —— 最核心的三个概念
  1. ray.init()      —— 启动 Ray 集群
  2. @ray.remote      —— 把函数变成远程任务
  3. ray.get()        —— 获取远程任务的结果

运行: python3 ray_learning/01_quickstart.py
"""

import ray

# ========== 1. 启动 Ray ==========
# 无参数 = 在当前进程内启动一个单机 Ray 集群
# 也可以 ray.init(address="auto") 连接到已有的 Ray 集群
ray.init(address="auto", log_to_driver=False)
print(f"Ray 集群已启动, 地址: {ray.get_runtime_context().gcs_address}")
print(f"可用 GPU 数: {ray.cluster_resources().get('GPU', 0)}")
print(f"可用 CPU 数: {ray.cluster_resources().get('CPU', 0)}")
print()

# ========== 2. 定义远程任务 ==========
# @ray.remote 把普通函数变成远程任务
# 远程任务在 Ray 的 worker 进程里执行, 可以分布在不同机器上


@ray.remote
def add(a, b):
    """这个函数会在 Ray worker 中执行"""
    return a + b


@ray.remote
def slow_square(x):
    """模拟一个耗时计算"""
    import time
    time.sleep(1)
    return x * x


# ========== 3. 调用远程任务 ==========
# 调用远程函数会立即返回一个 ObjectRef (类似 future/句柄)
# 真正的计算在后台进行

obj_ref = add.remote(1, 2)
print(f"add.remote(1, 2) 返回 ObjectRef: {obj_ref}")
print(f"类型: {type(obj_ref)}")

# 用 ray.get() 获取实际结果 (会阻塞直到任务完成)
result = ray.get(obj_ref)
print(f"ray.get() 获取结果: {result}")
print()

# ========== 4. 并行执行 ==========
import time

start = time.time()

# 串行执行三个耗时任务 (3秒)
# slow_square(1)
# slow_square(2)
# slow_square(3)

# 并行执行三个耗时任务 (~1秒)
refs = [slow_square.remote(i) for i in range(3)]
results = ray.get(refs)

elapsed = time.time() - start
print(f"3 个 slow_square 并行执行耗时: {elapsed:.2f} 秒")
print(f"结果: {results}")
print()

# ========== 5. 任务依赖 ==========
# ObjectRef 可以直接传给另一个远程任务
# Ray 会自动调度: 等依赖的任务完成后, 再执行下游任务
# 这叫 任务依赖图 / 计算图


@ray.remote
def add_ref(x, y):
    return x + y


# 传统写法 (等出结果再传):
# a = ray.get(slow_square.remote(3))
# b = ray.get(slow_square.remote(4))
# c = add(a, b)

# Ray 写法 (ObjectRef 直接传, 自动等):
ref_a = slow_square.remote(3)
ref_b = slow_square.remote(4)
ref_c = add_ref.remote(ref_a, ref_b)  # 直接传 ObjectRef!
result_c = ray.get(ref_c)
print(f"依赖计算: 3^2 + 4^2 = {result_c}")
print()

ray.shutdown()
print("Ray 已关闭")
