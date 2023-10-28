# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch.nn as nn

# 在rank0也就是master rank打印信息，防止每个机器或GPU都打印消息造成大量重复信息
def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)
# 这个函数的作用是把一个数据批次移动到指定的设备上。在PyTorch中，数据默认是在CPU上的，
# 如果要在GPU上进行运算，就需要先把数据移到GPU上。这个函数通过遍历批次中的所有元素并
# 调用to(device)方法来实现这一点。如果某个元素不能被移到指定的设备上
#（例如，这个元素是一个字符串或者其他不能在GPU上使用的类型），那么就直接保留这个元素，不进行任何操作。

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    return tokenizer

# 这段代码的功能是将训练好的模型以Hugging Face格式保存，这样之后就可以使用Hugging Face库的from_pretrained方法加载了。

def save_hf_format(model, tokenizer, args, sub_folder=""):
     # used to save huggingface format, so we can use it for hf.from_pretrained
    # 首先，这行代码检查模型是否有'module'这个属性。这在PyTorch中是一个常见的模式，
    # 当模型被封装在torch.nn.DataParallel或torch.nn.parallel.DistributedDataParallel时，
    # 模型的所有属性都会被存储在'module'属性中。所以这行代码的目的是确保我们总是在原始模型上进行操作，而不是并行化的包装器。
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    # "lora"可能是某种特定的模型组件或参数，这里将其排除在保存的模型权重之外。
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
def set_random_seed(seed):
    # 首先检查种子是否是None。如果是None，那么就跳过这段代码，随机数生成器将会用一个随机的种子。
    if seed is not None: 
        set_seed(seed)# 这一行调用transformers库的set_seed的函数，将种子设定为指定的值。
        random.seed(seed) # 将Python内置的随机数生成器的种子设定为指定的值。
        np.random.seed(seed) # 将NumPy的随机数生成器的种子设定为指定的值。
        torch.manual_seed(seed) # 将PyTorch的随机数生成器的种子设定为指定的值。
        torch.cuda.manual_seed_all(seed) # 将PyTorch的所有GPU随机数生成器的种子设定为指定的值



# 这段代码是在分布式训练环境中进行平均值计算的函数，通过这段代码，
# 所有的处理器（或者叫节点）上的同一个tensor都会被加和起来，然后除以总的处理器数，得到平均值。
def get_all_reduce_mean(tensor):
    # 这行代码执行一个分布式的reduce操作。reduce操作是指所有处理器中的同一个tensor都被某种方式结合起来。
    # 在这个例子中，torch.distributed.ReduceOp.SUM表示所有处理器上的tensor将被加和起来。
    # 加和的结果会在所有处理器上都可用。
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # 这行代码将前一步得到的加和结果除以处理器的数量（也叫作 world size）。
    # 这样，tensor就变成了所有处理器上原始tensor的平均值。
    tensor = tensor / torch.distributed.get_world_size()
    # 最后，这个平均值tensor被返回。在所有处理器上，这个函数返回的tensor都是相同的，
    # 等于所有处理器上原始tensor的平均值。
    return tensor


# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs
# 这段代码的作用是将模型中的参数分组以便于在优化器中使用。它将模型参数分为两组：
# 一组需要进行权重衰减（L2正则化）的参数，另一组不需要进行权重衰减的参数。

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    # 它定义了一个列表 optimizer_grouped_parameters，其中包含两个字典。每个字典都对应一个参数组，包含 "params" 和 "weight_decay" 这两个关键字。
    optimizer_grouped_parameters = [
        # 在第一个字典中，它从模型参数中选出那些名称不包含 "bias" 或 "LayerNorm.weight" 
        # 且需要求梯度的参数。这些参数在优化过程中会应用 weight_decay 作为权重衰减项。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        # 在第二个字典中，它选出那些名称包含 "bias" 或 "LayerNorm.weight" 且需要求梯度的参数。
        # 这些参数在优化过程中不会应用权重衰减，即其 "weight_decay" 值为0。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
