"""
分布式训练模式 —— slime 项目中的 Ray 用法解析

以 slime 的 train.py 为例子, 拆解 Ray 在训练中的角色:

  @ray.remote(num_gpus=1)
  class InfoActor: ...      → 检测 GPU 分配

  placement_group(bundles)  → 确保 GPU 资源一起分配

  RolloutManager             → 管理 SGLang 推理引擎
  RayTrainGroup              → 管理 Megatron 训练

  ray.get()                  → 等待训练/推理完成

运行: python3 ray_learning/09_distributed_training_pattern.py
"""

import ray

# ========== 1. 训练流程 ==========
print("=" * 60)
print("slime 训练流程 (Ray 视角)")
print("=" * 60)
print("""
  train.py 启动
      │
      ├── ray.init()
      │     (启动 Ray, 检测所有 GPU)
      │
      ├── create_placement_groups(args)
      │     (为 train + rollout 预留 GPU)
      │     └── _create_placement_group(N)
      │           ├── placement_group(bundles, strategy="PACK")
      │           └── InfoActor → 检测每个 bundle 对应的 GPU ID
      │
      ├── create_rollout_manager(args, pg)
      │     (启动 SGLang 推理引擎)
      │     └── RolloutManager.remote()
      │
      ├── create_training_models(args, pg, rollout_manager)
      │     (启动 Megatron 训练)
      │     └── RayTrainGroup(...).async_init(...)
      │
      └── 训练循环
            ├── rollout_manager.generate.remote()   → 推理
            ├── actor_model.async_train.remote()     → 训练
            └── rollout_manager.eval.remote()        → 评估
""")

# ========== 2. 资源分配图解 ==========
print("=" * 60)
print("资源分配图解 (colocate 模式)")
print("=" * 60)
print("""
  --colocate 模式下:
  1 个 GPU 同时给训练和 rollout 用

  placement_group:
    bundle 0: {"GPU": 1, "CPU": 1}  ← 这块 GPU 上:
        ├── InfoActor (检测 GPU)
        ├── TrainActor (Megatron 训练)
        └── RolloutManager (SGLang 推理)
             (分时复用: 训练完 → 推理 → 训练完 → 推理)

  非 colocate 模式:
    bundle 0: {"GPU": 1, "CPU": 1}  ← 训练用
    bundle 1: {"GPU": 2, "CPU": 2}  ← rollout 用 (2 张 GPU)
""")

# ========== 3. Ray API 映射 ==========
print("=" * 60)
print("slime 中的 Ray API 映射")
print("=" * 60)
print("""
  API                         用途
  ─────────────────────────────────────────────────
  @ray.remote(num_gpus=1)     声明需要 GPU 的 Actor
  @ray.remote(num_gpus=0)     声明不需要 GPU 的 Actor (如 RolloutManager)
  placement_group(bundles)    预留一组 GPU 资源
  ray.get(pg.ready())         等资源分配完成
  ray.get(actor.method.remote())  调用 Actor 方法并等结果
  actor.method.remote()       异步调用 Actor 方法 (立即返回)
  ray.kill(actor)             终止 Actor
  ray.shutdown()              关闭 Ray

  关键区别:
    .remote()  → 异步, 立即返回 ObjectRef
    ray.get()  → 同步, 阻塞直到结果就绪
""")

# ========== 4. 简化版训练循环 ==========
print("=" * 60)
print("简化版训练循环 (理解流程)")
print("=" * 60)

ray.init(address="auto", log_to_driver=False)

# 用一个简单例子模拟 train.py 的训练循环


@ray.remote
class SimpleRollout:
    def generate(self, step):
        import time
        time.sleep(0.1)
        return {"step": step, "data": f"rollout_data_{step}"}

    def eval(self, step):
        return {"step": step, "metric": 0.95 - step * 0.01}


@ray.remote
class SimpleTrainer:
    def train(self, step, data):
        import time
        time.sleep(0.1)
        return {"step": step, "loss": 0.1 / (step + 1)}


# 创建 Actor
rollout = SimpleRollout.remote()
trainer = SimpleTrainer.remote()

# 模拟训练循环
print("模拟训练 3 步:")
for step in range(3):
    # 1. Rollout (推理)
    data = ray.get(rollout.generate.remote(step))
    print(f"  Step {step}: 推理完成 → {data['data']}")

    # 2. Train (训练)
    result = ray.get(trainer.train.remote(step, data))
    print(f"  Step {step}: 训练完成 → loss={result['loss']:.4f}")

    # 3. Eval (评估)
    metric = ray.get(rollout.eval.remote(step))
    print(f"  Step {step}: 评估 → metric={metric['metric']:.3f}")

ray.shutdown()
print()
print("简化版训练流程演示完毕")
