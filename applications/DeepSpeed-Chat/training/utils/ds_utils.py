# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    # the zero stage and offload
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",

        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    # lora with deepspeed
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count(
            )
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        # fp16 is false means we use fp32, since we change a lot on model architecture, 
        # the representation of fp16 is [-256, 256], the representation of bf16 is [-65536, 65536]
        # if A100, use bf16, bf16 is more useful for deep learning, we need more parameter value range.
        "bf16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            # tensor parallelism, usually use in inference stage if token length is very long
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
