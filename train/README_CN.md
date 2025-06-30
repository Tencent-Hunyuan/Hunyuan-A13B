<p align="left">
   <a href="train/README.md">English</a>  ｜ 中文</a>&nbsp
</p>
<br><br>

# 模型训练

Hunyuan-A13B 提供了模型训练相关流程，您可以在此章节对训练数据格式进行处理以供模型训练使用。

### 训练数据格式及处理 

**Hunyuan-A13B-Instruct 同时支持慢思考与快思考两种模式，模型的默认输出是慢思考模式，若想让模型进行快思考，可在 query 前附加上 "/no_think"。**

训练数据按照以下形式处理为 messages 格式，训练和推理的默认 system prompt 为空，可以根据自己的需求进行设定。

```python
# think_pattern
think = ""
answer = ""
think_pattern = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"

# fast think pattern
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "/no_think海水为什么是咸的" },
    {"role": "assistant", "content": "<think>\n\n</think>\n<answer>\n海水是咸的主要是因为其中含有许多溶解在水中的盐类和矿物质。这些盐类和矿物质来自于地球表面的岩石和土壤中的化学物质，随着时间的推移，它们被带到了海洋中。当海水蒸发时，水分蒸发掉了，但盐类和矿物质仍然留在水中，导致海水变得更加咸味。因此，海水的咸度是由其中的盐类和矿物质的含量决定的。\n</answer>"}
]

# slow think pattern
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1+1=" },
    {"role": "assistant", "content": "<think>\n嗯，用户问的是1加1等于多少。首先，我需要确认这是一个基本的算术问题。1加1在十进制的数学体系中，通常的结果是2。不过，可能需要考虑是否有其他情况，比如二进制或者其他数制，但用户没有特别说明，所以默认应该是十进制。另外，有时候可能会有脑筋急转弯的情况，比如在某些语境下1+1可能等于1（比如1滴水加1滴水还是1滴水），但通常数学问题中都是2。所以最准确的回答应该是2。</think>\n<answer>\n在十进制的基本算术运算中，1加1的结果是2。这是数学中最基础的加法运算之一，遵循自然数的加法规则。因此，1 + 1 = 2。\n</answer>"}
]

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models", use_fast=False, trust_remote_code=True)
ids = tokenizer.apply_chat_template(messages)
```

&nbsp;

## 快速开始

您可以参照<a href="../examples/README.md">快速开始文档</a>中的内容进行快速上手。

## 模型训练 

### 硬件需求

经过测试，不开 make_moe_param_leaf_module 以及 zero3+offload，max_seq_length 为 2048，全量微调最少需要单机 8 卡（显存至少80G）。

### 启动方式

参考：[HuggingFace Transformers Trainer](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/trainer)

#### 单机启动训练

在`train`目录下，执行：

```sh
pip install -r requirements.txt
bash train.sh
```

#### 多机启动训练

如果要用多台机器启动训练，请按照以下步骤执行，并保证多台机器在一个集群内。

##### 配置机器间免密 ssh 登录

以下操作以两个机器为例，两台机器的 ip 分别以`${ip1}`和`${ip2}`标识，以下操作均在 docker container 内执行。

首先，配置多机container免密，在每台机器上执行。

```sh
ssh-keygen			# 生成id_rsa和id_rsa.pub，用于免密登录
ssh-keygen -t rsa -A    # 生成/etc/ssh/ssh_host_rsa_key和ssh_host_ecdsa_key， 用于后面启动ssh listen
/usr/sbin/sshd -p 36005 -o ListenAddress=0.0.0.0        # 启动Listen
echo "Port 36005" > ~/.ssh/config   # ssh 连接端口修改为 36005
passwd root    # 需要配置root密码，否则监测平台会报警
```

注意：这里的`36005`是一个示例端口，可以选用任意端口，但需要保证使用的端口**开放**且**不被其他的进程占用**。

接下来，在每台机器的 container 内，执行：

```sh
cat ~/.ssh/id_rsa.pub
```

**将输出的 ssh 公钥复制并粘贴到`~/.ssh/authorized_keys`文件中，每行一个公钥，每台机器上都要做这个操作**。最终每台机器上的`~/.ssh/authorized_keys`文件内容应当是一致的，并且包含了所有机器的公钥。

需要注意，多节点训练时，每个节点上执行的代码都得一致，建议挂载一个共享的网络盘，如果无法挂载共享网盘，则需要手动将数据集、脚本、代码复制在多台机器的相同目录下。

##### 启动多机训练

在以上准备步骤准备好了之后，以及确认依赖已经安装完成（如未安装，请执行`pip install -r requirements.txt`安装），就可以在`train.sh`中的开头增加以下配置：

```shell
export HOST_GPU_NUM=8
# 当前机器ip
export LOCAL_IP=${ip1}
# 多节点机器ip，逗号隔开
export NODE_IP_LIST="${ip1}:8,${ip2}:8"
# 机器节点个数
export NODES=2
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))
```

注意：将以上的`${ip1}`和`${ip2}`替换为真实的 ip 地址！

然后，在`${ip1}`的机器上，在`train/`目录下，执行`bash train.sh`即可，注意第一次启动时可能会看见以下的输出：

```ssh
The authenticity of host '[ip]:36005 ([ip]:36005)' can't be established.
ECDSA key fingerprint is xxxxxx.
ECDSA key fingerprint is MD5:xxxxxx.
Are you sure you want to continue connecting (yes/no)?
```

此时输入`yes`即可继续。

##### 关键参数

脚本中的关键参数如下：

- `--deepspeed`: 此参数应当指向一个 deepspeed 的配置文件，`train`文件夹下提供了三种 DeepSpeed 的默认配置文件：`ds_zero2_no_offload.json`, `ds_zero3_no_offload.json`, `ds_zero3_offload.json`，这三个配置文件所需显存依次减少
- `--model_name_or_path`: 要加载的 HF 预训练模型权重，确保这个路径下包含了 `modeling_hunyuan.py` 和 `configuration_hunyuan.py` 文件，否则无法加载
- `--tokenizer_name_or_path`: tokenizer 文件夹路径，确保这个路径下包含了`tokenization_hy.py` 文件，否则无法加载
- `--train_data_file`: 训练文件路径，应该为一个 jsonl 文件
- `--output_dir`: 输出文件夹，log、tensorboard 和权重都会存储在这个路径下
- `--per_device_train_batch_size`: 每张卡上的 batch size
- `--gradient_accumulation_steps`: 梯度累计次数，`per_device_train_batch_size * gradient_accumulation_steps * dp_size`为 global_batch_size
- `--max_steps`: 训练的总步数
- `--save_steps`: 每多少个 step 存储一个 checkpoint
- `--use_lora`: 是否用 lora 训练，同时接收`--lora_rank`，`--lora_alpha`和`--lora_dropout`参数。lora 默认应用于 "q_proj", "k_proj", "v_proj", "o_proj" 四个参数，如果需要改变的话在代码中修改即可。注意：**使用 lora 训练时，只会保存 lora 的权重，而不会保存 base 模型的权重**，如果需要合并 lora 权重，看下面的“Lora 权重合并”一节
- `--make_moe_param_leaf_module`：当用 zero3 以及 MoE 训练时，将 MoE 模块视作一个 leaf module，即它的参数不进行 zero3 切分，这个选项预计会显著增加显存占用
- `--gradient_checkpointing`：开启梯度重计算
- `--train_attention_params_only`: 是否只训练 attention 参数
- `--learning_rate`: 训练时的最大学习率
- `--min_lr`: 训练时的最小学习率
- `--use_flash_attn`: 开启 flash-attention 进行训练加速

**注意：**

- 如果想从一个中途保存的 ckpt 继续训练，而不是加载一个预训练的权重，直接指定`--resume_from_checkpoint`为之前训练保存的 ckpt 路径，不要指定`--model_name_or_path`，这样只会加载权重，而不会加载训练状态
- 从 ckpt 继续训练时，loss 可能会有微小的偏差，这是由一些非确定性算法带来的随机性，是正常现象。参考：[HuggingFace Transformers Trainer Randomness](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/trainer#randomness)
- 当 `--model_name_or_path` 有效时，所有模型相关的参数都会被忽略
- 一个 batch 内的样本会通过 padding 对齐 batch 内最长的样本，而每条样本的长度最长为 max_seq_length，超出的部分会被裁剪
- 如果报出 bias 权重没有 load 的 warning，忽略即可，Hunyuan-Large 中不会用到 bias

#### 显存不足怎么办？

参考：[DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

可以尝试修改 ds config，去掉这几个参数的 auto 属性，改小试试看：

- `stage3_param_persistence_threshold`
- `stage3_prefetch_bucket_size`
- `stage3_max_reuse_distance`


#### Lora 模型合并

保存下来的 lora 权重没法在训练运行时合并到 zero3 模型中，因为 zero3 开启时模型权重会切分到各 dp rank 上。因此如果想把 lora 权重合并到 base 模型上，可以通过离线的方式合并后得到权重文件。执行`merge_lora_weight.sh`即可完成 lora 权重和 base 模型权重的合并，其中的参数有：

- `--base_model_path`：base 模型的权重目录
- `--adapter_model_path`：lora 权重目录
- `--output_path`：合并后的权重保存目录
- `--save_dtype`： 以什么数据格式存储合并后的权重，可选值：fp16，bf16，fp32

#### LLaMA-Factory 支持

如果对 LLaMA-Factory 较为熟悉，可使用 https://github.com/hiyouga/LLaMA-Factory/tree/main 进行微调，我们提供了 llama-factory 的训练示例配置文件 `./train/llama_factory_support/hunyuan_a13b_full_sft.yaml`文件。
