"""
Ray Actors —— 有状态的计算

Tasks 是无状态的 (每次调用都是独立进程)
Actors 是有状态的 (在同一个进程中, 可以维持状态)

与 @ray.remote 装饰函数不同:
  @ray.remote 装饰类 → 这个类变成 Actor
  每次 .remote() 调用都是在同一个 Actor 实例上执行

运行: python3 ray_learning/03_actors.py
"""

import ray
import time

ray.init(address="auto", log_to_driver=False)  # 连接到已有的 Ray 集群


# ========== 1. 基础 Actor ==========
@ray.remote
class Counter:
    """一个简单的计数器 Actor"""

    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count


print("=== 基础 Actor ===")
counter = Counter.remote()  # 创建 Actor (返回 ActorHandle)

# 调用 Actor 方法, 返回 ObjectRef
ref1 = counter.increment.remote()
ref2 = counter.increment.remote()
ref3 = counter.increment.remote()

print(f"count: {ray.get([ref1, ref2, ref3])}")  # [1, 2, 3]
print()

# ========== 2. Actor 有状态 ==========
# Actor 的 state 在方法调用之间保留


@ray.remote
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.history = []

    def deposit(self, amount):
        self.balance += amount
        self.history.append(f"存入 {amount}")
        return self.balance

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            self.history.append(f"取出 {amount}")
            return self.balance
        else:
            return f"余额不足! 当前余额: {self.balance}"

    def get_history(self):
        return self.history


print("=== Actor 有状态 ===")
acc = BankAccount.remote("Alice", 100)

ray.get(acc.deposit.remote(50))
ray.get(acc.withdraw.remote(30))
ray.get(acc.deposit.remote(200))
result = ray.get(acc.withdraw.remote(500))

print(f"取款结果: {result}")
print(f"操作历史: {ray.get(acc.get_history.remote())}")
print()

# ========== 3. 多个 Actor 实例 ==========
# 每个 .remote() 调用创建一个独立 Actor


@ray.remote
class Worker:
    def __init__(self, name):
        self.name = name
        self.tasks_done = 0

    def work(self, task_id):
        import time
        time.sleep(0.1)
        self.tasks_done += 1
        return f"{self.name} 完成 task-{task_id}, 总共完成 {self.tasks_done} 个"


print("=== 多个 Actor ===")
workers = [Worker.remote(f"Worker-{i}") for i in range(3)]

# 将 6 个任务分配给 3 个 Actor
refs = []
for i in range(6):
    worker = workers[i % 3]
    refs.append(worker.work.remote(i))

results = ray.get(refs)
for r in results:
    print(r)
print()

# ========== 4. Actor 资源指定 ==========
# 可以给 Actor 指定 GPU/CPU


@ray.remote(num_gpus=1, num_cpus=1)
class GpuWorker:
    """这个 Actor 会占用 1 张 GPU"""

    def __init__(self):
        import torch
        self.gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else -1

    def get_info(self):
        return {
            "gpu_ids": ray.get_gpu_ids(),
            "node_ip": ray.util.get_node_ip_address(),
        }


print("=== GPU Actor ===")
gpu_worker = GpuWorker.remote()
info = ray.get(gpu_worker.get_info.remote())
print(f"GPU Actor 信息: {info}")
print()

# ========== 5. Actor 句柄传递 ==========
# ActorHandle 可以传给其他任务或其他 Actor


@ray.remote
class TaskProducer:
    def __init__(self, counter_actor):
        self.counter = counter_actor

    def do_work(self):
        # 通过句柄调用其他 Actor
        return ray.get(self.counter.increment.remote())


counter_actor = Counter.remote()
producer = TaskProducer.remote(counter_actor)
result = ray.get(producer.do_work.remote())
print(f"Actor 句柄传递: {result}")
print()

# ========== 6. Actor 池 ==========
# 用多个 Actor 实现简单的任务分发


@ray.remote
class PoolWorker:
    def __init__(self, wid):
        self.wid = wid

    def compute(self, x):
        return f"Worker-{self.wid} 计算结果: {x * x}"


class ActorPool:
    def __init__(self, size):
        self.workers = [PoolWorker.remote(i) for i in range(size)]

    def map(self, items):
        # 轮询分发
        refs = []
        for i, item in enumerate(items):
            worker = self.workers[i % len(self.workers)]
            refs.append(worker.compute.remote(item))
        return ray.get(refs)


print("=== Actor 池 ===")
pool = ActorPool(3)
results = pool.map([1, 2, 3, 4, 5, 6, 7])
for r in results:
    print(r)

ray.shutdown()
print("\nRay 已关闭")
