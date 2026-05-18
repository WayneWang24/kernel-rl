# PolyU 训练同步快照

- 时间: 2026-05-18 07:55:42 UTC
- 主机: kvmise-Standard-PC-Q35-ICH9-2009
- 项目: /home/vm/workspace/kernel-rl

## 日志
- 源文件: `logs/grpo_polyu.log`
- 大小: 1 MB
- 总行数: 945
- 同步尾部行数: 945（上限 120000）

## 正在运行的训练进程
```
(没有匹配到 verl/main_ppo/ray 进程)
```

## GPU 状态
```
index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu
0, NVIDIA A100 80GB PCIe, 0 %, 10 MiB, 81920 MiB, 34
1, NVIDIA A100 80GB PCIe, 0 %, 10 MiB, 81920 MiB, 33
2, NVIDIA A100 80GB PCIe, 0 %, 10 MiB, 81920 MiB, 31
```

## 日志最后 40 行
```
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/torch/serialization.py", line 759, in _open_file_like
[36m(TaskRunner pid=49607)[0m     return _open_file(name_or_buffer, mode)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/torch/serialization.py", line 740, in __init__
[36m(TaskRunner pid=49607)[0m     super().__init__(open(name, mode))
[36m(TaskRunner pid=49607)[0m FileNotFoundError: [Errno 2] No such file or directory: '/home/vm/workspace/kernel-rl/checkpoints/grpo_3b_cuda/global_step_75/actor/model_world_size_3_rank_2.pt'
[36m(TaskRunner pid=49607)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): [36mray::WorkerDict.actor_rollout_load_checkpoint()[39m (pid=50460, ip=10.22.63.111, actor_id=61042a3ba5dc4516dfada0ef01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x73ab23e18070>)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/concurrent/futures/_base.py", line 451, in result
[36m(TaskRunner pid=49607)[0m     return self.__get_result()
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
[36m(TaskRunner pid=49607)[0m     raise self._exception
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/verl/single_controller/ray/base.py", line 932, in func
[36m(TaskRunner pid=49607)[0m     return getattr(self.worker_dict[key], name)(*args, **kwargs)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/verl/single_controller/base/decorator.py", line 427, in inner
[36m(TaskRunner pid=49607)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/verl/workers/fsdp_workers.py", line 1256, in load_checkpoint
[36m(TaskRunner pid=49607)[0m     self.checkpoint_manager.load_checkpoint(
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/verl/utils/checkpoint/fsdp_checkpoint_manager.py", line 141, in load_checkpoint
[36m(TaskRunner pid=49607)[0m     model_state_dict = torch.load(local_model_path, weights_only=False)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/torch/serialization.py", line 1484, in load
[36m(TaskRunner pid=49607)[0m     with _open_file_like(f, "rb") as opened_file:
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/torch/serialization.py", line 759, in _open_file_like
[36m(TaskRunner pid=49607)[0m     return _open_file(name_or_buffer, mode)
[36m(TaskRunner pid=49607)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/site-packages/torch/serialization.py", line 740, in __init__
[36m(TaskRunner pid=49607)[0m     super().__init__(open(name, mode))
[36m(TaskRunner pid=49607)[0m FileNotFoundError: [Errno 2] No such file or directory: '/home/vm/workspace/kernel-rl/checkpoints/grpo_3b_cuda/global_step_75/actor/model_world_size_3_rank_1.pt'
[36m(vLLMHttpServer pid=52671)[0m ERROR 05-07 21:12:03 [core_client.py:564] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
[36m(vLLMHttpServer pid=52671)[0m [1;36m(Worker pid=53401)[0;0m /home/vm/miniconda3/envs/kernel-rl/lib/python3.10/multiprocessing/resource_tracker.py:104: UserWarning: resource_tracker: process died unexpectedly, relaunching.  Some resources might leak.
[36m(vLLMHttpServer pid=52671)[0m [1;36m(Worker pid=53401)[0;0m   warnings.warn('resource_tracker: process died unexpectedly, '
[36m(vLLMHttpServer pid=52671)[0m Traceback (most recent call last):
[36m(vLLMHttpServer pid=52671)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/multiprocessing/resource_tracker.py", line 209, in main
[36m(vLLMHttpServer pid=52671)[0m     cache[rtype].remove(name)
[36m(vLLMHttpServer pid=52671)[0m KeyError: '/psm_790b75f7'
[36m(vLLMHttpServer pid=52671)[0m Traceback (most recent call last):
[36m(vLLMHttpServer pid=52671)[0m   File "/home/vm/miniconda3/envs/kernel-rl/lib/python3.10/multiprocessing/resource_tracker.py", line 209, in main
[36m(vLLMHttpServer pid=52671)[0m     cache[rtype].remove(name)
[36m(vLLMHttpServer pid=52671)[0m KeyError: '/mp-u3ld0_go'
[36m(vLLMHttpServer pid=52673)[0m [1;36m(Worker pid=53415)[0;0m WARNING 05-07 21:11:59 [cudagraph_dispatcher.py:106] cudagraph dispatching keys are not initialized. No cudagraph will be used.[32m [repeated 2x across cluster][0m
[36m(vLLMHttpServer pid=52673)[0m WARNING 05-07 21:12:00 [model.py:1389] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.[32m [repeated 2x across cluster][0m
[36m(vLLMHttpServer pid=52673)[0m ERROR 05-07 21:12:03 [core_client.py:564] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.[32m [repeated 2x across cluster][0m
[36m(vLLMHttpServer pid=52673)[0m [1;36m(EngineCore_DP0 pid=53186)[0;0m The tokenizer you are loading from '/home/vm/workspace/kernel-rl/checkpoints/sft_3b_merged' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.[32m [repeated 2x across cluster][0m
```
