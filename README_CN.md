<p align="left">
   <a href="README.md">English</a>  ｜ 中文</a>&nbsp
</p>
<br><br>

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>

<p align="center">
    🫣&nbsp;<a href="https://huggingface.co/tencent/Hunyuan-A13B-Instruct"><b>Hugging Face</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🖥️&nbsp;<a href="https://llm.hunyuan.tencent.com/" style="color: red;"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕖&nbsp;<a href="https://cloud.tencent.com/product/hunyuan"><b>HunyuanAPI</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕹️&nbsp;<a href="https://hunyuan.tencent.com/?model=hunyuan-a13b"><b>Demo</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <img src="https://avatars.githubusercontent.com/u/109945100?s=200&v=4" width="16"/>&nbsp;<a href="https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct"><b>ModelScope</b></a>
</p>

<p align="center">
    <a href="https://github.com/Tencent/Hunyuan-A13B"><b>GITHUB</b></a>
</p>




## 模型介绍

随着人工智能技术的快速发展，大型语言模型（LLMs）在自然语言处理、计算机视觉和科学任务等领域取得了显著进展。然而，随着模型规模的扩大，如何在保持高性能的同时优化资源消耗成为一个关键挑战。为了应对这一挑战，我们研究了混合专家（MoE）模型，当前亮相的 Hunyuan-A13B 模型，拥有800亿总参数和130亿激活参数。不仅在效果上达到了高标准，而且在尺寸上也做到了极致的优化，成功平衡了模型性能与资源占用。


### 核心特性与优势
- ​**小参数量，高性能**​：仅激活130亿参数（总参数量800亿），即可在多样化基准任务中媲美更大规模模型的竞争力表现 
- ​**混合推理支持**​：同时支持快思考和慢思考两种模式，支持用户灵活选择 
- ​**超长上下文理解**​：原生支持256K上下文窗口，在长文本任务中保持稳定性能
- ​**增强Agent能力**​：优化Agent能力，在BFCL-v3、τ-Bench等智能体基准测试中领先
- ​**高效推理**​：采用分组查询注意力（GQA）策略，支持多量化格式，实现高效推理
    

### 为何选择Hunyuan-A13B？
作为兼具强大性能与计算效率的大模型，Hunyuan-A13B是研究者与开发者在资源受限条件下追求高性能的理想选择。无论学术研究、高性价比AI解决方案开发，还是创新应用探索，本模型都能提供强大的基础支持。


&nbsp;

## 新闻
<br>

* 2025.6.26 我们在Hugging Face开源了 **Hunyuan-A13B-Instruct**，**Hunyuan-A13B-Pretrain**, **Hunyuan-A13B-Instruct-FP8**， **Hunyuan-A13B-Instruct-GPTQ-Int4**。并发布了技术报告和训练推理操作手册，详细介绍了模型能力和训练与推理的操作。

## 模型结构

Hunyuan-A13B采用了细粒度混合专家（Fine-grained Mixture of Experts，Fine-grained MoE）架构，包含800亿参数和130亿激活参数，累计训练了超过 20T tokens。该模型支持 256K 的上下文长度，以下为模型结构细节:
* 总参数: 80B
* 激活参数: 13B
* 层数: 32
* Attention Heads: 32
* 共享专家数: 1
* 非共享专家数: 64
* 路由策略: Top-8
* 激活函数: SwiGLU
* 隐层维度: 4096
* 专家隐层维度: 3072 

## Benchmark评估榜单 

**Hunyuan-A13B-Pretrain** 在 12/14 个任务上超越了Hunyuan上一代52B激活参数的MoE模型Hunyuan-Large，证实了它在预训练任务上出色的能力。与业界更大参数量的Dense和MoE模型相比, Hunyuan-A13B在多个代码和数学任务上都取得了最高分数。在MMLU, MMLU-PRO等诸多聚合任务上, Hunyuan-A13B达到了与Qwen3-A22B模型同等的水平，表现出优秀的综合能力。

| Model            | Hunyuan-Large | Qwen2.5-72B  | Qwen3-A22B | Hunyuan-A13B |
|------------------|---------------|--------------|-------------|---------------|
| MMLU             | 88.40          | 86.10         | 87.81        | 88.17          |
| MMLU-Pro         | 60.20          | 58.10        | 68.18           | 67.23          |
| MMLU-Redux              |  87.47         | 83.90         | 87.40        | 87.67          |
| BBH        | 86.30             | 85.80            | 88.87        | 87.56          |
| SuperGPQA    |  38.90         | 36.20          | 44.06           | 41.32          |
| EvalPlus       | 75.69          | 65.93         | 77.60        | 78.64          |
| MultiPL-E             | 59.13             | 60.50            | 65.94        | 69.33          |
| MBPP | 72.60             | 76.00            | 81.40        | 83.86          |
| CRUX-I             | 57.00          | 57.63          | -        | 70.13          |
| CRUX-O             | 60.63          | 66.20          | 79.00        | 77.00          |
| MATH            | 69.80          | 62.12         | 71.84        | 72.35          |
| CMATH            | 91.30          | 84.80         | -        | 91.17          |
| GSM8k         | 92.80             | 91.50           | 94.39        | 91.83          |
| GPQA            | 25.18             | 45.90            | 47.47        | 49.12          |

**Hunyuan-A13B-Instruct** 在多项基准测试中取得了极具有竞争力的表现，尤其是在数学、科学、agent等领域。我们与一些强力模型进行了对比，结果如下所示。

| Topic               | Bench                         | OpenAI-o1-1217 | DeepSeek R1 | Qwen3-A22B | Hunyuan-A13B-Instruct |
|:-------------------:|:-----------------------------:|:-------------:|:------------:|:-----------:|:---------------------:|
| **Mathematics**     | AIME 2024<br>AIME 2025<br>MATH | 74.3<br>79.2<br>96.4 | 79.8<br>70<br>94.9 | 85.7<br>81.5<br>94.0 | 87.3<br>76.8<br>94.3 |
| **Science**         | GPQA-Diamond<br>OlympiadBench | 78<br>83.1 | 71.5<br>82.4 | 71.1<br>85.7 | 71.2<br>82.7 |
| **Coding**          | Livecodebench<br>Fullstackbench<br>ArtifactsBench | 63.9<br>64.6<br>38.6 | 65.9<br>71.6<br>44.6 | 70.7<br>65.6<br>44.6 | 63.9<br>67.8<br>43 |
| **Reasoning**       | BBH<br>DROP<br>ZebraLogic    | 80.4<br>90.2<br>81 | 83.7<br>92.2<br>78.7 | 88.9<br>90.3<br>80.3 | 89.1<br>91.1<br>84.7 |
| **Instruction<br>Following** | IF-Eval<br>SysBench  | 91.8<br>82.5 | 88.3<br>77.7 | 83.4<br>74.2 | 84.7<br>76.1 |
| **Text<br>Creation**| LengthCtrl<br>InsCtrl       | 60.1<br>74.8 | 55.9<br>69 | 53.3<br>73.7 | 55.4<br>71.9 |
| **NLU**             | ComplexNLU<br>Word-Task     | 64.7<br>67.1 | 64.5<br>76.3 | 59.8<br>56.4 | 61.2<br>62.9 |
| **Agent**           | BDCL v3<br> τ-Bench<br>ComplexFuncBench<br> C3-Bench | 67.8<br>60.4<br>47.6<br>58.8 | 56.9<br>43.8<br>41.1<br>55.3 | 70.8<br>44.6<br>40.6<br>51.7 | 78.3<br>54.7<br>61.2<br>63.5 |


## 数据

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

您可以参照<a href="examples/README.md">快速开始文档</a>中的内容进行快速上手。

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
- 从 ckpt 继续训练时，loss 可能会有微小的偏差，这是由一些非确定性算法带来的随机性，是正常现象。参考：[HuggingFace Transformers Trainer Randomness 
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


&nbsp;

## Agent 功能

Hunyuan-A13B 模型支持通过函数调用（Function Call）来实现 Agent 的搭建。[Agent示例](agent/README.md)


## 量化压缩

我们采用自研的`AngleSlim`压缩工具产出了FP8及INT4量化模型，`AngleSlim`压缩工具预计7月初开源，将支持大模型一键式量化压缩，敬请期待，现在可以直接下载我们的量化模型进行部署测试。

### FP8量化
我们采用`FP8-static`量化，FP8量化采用8位浮点格式，通过少量校准数据（无需训练）预先确定量化scale，将模型权重与激活值转换为FP8格式，提升推理效率并降低部署门槛。 
我们您可以使用`AngleSlim`量化，你也可以直接下载我们量化完成的开源模型使用[Hunyuan-A13B-Instruct-FP8](https://huggingface.co/tencent/Hunyuan-A13B-Instruct-FP8)。

#### FP8 Benchmark
本小节介绍 Hunyuan-A13B-Instruct-FP8 量化模型的Benchmark指标。

|   Bench   | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-FP8 | 
|:---------:|:---------------------:|:-------------------------:|
| AIME 2024 |         87.3          |           86.7            |
|   Gsm8k   |         94.39         |           94.01           |
|    BBH    |         89.1          |           88.34           |
|   DROP    |         91.1          |           91.1            |



### Int4量化
Int4量化我们采用[GPTQ](https://arxiv.org/abs/2210.17323 )算法实现W4A16量化，该算法逐层处理模型权重，利用少量校准数据最小化量化后的权重重构误差，通过近似Hessian逆矩阵的优化过程逐层调整权重。流程无需重新训练模型，仅需少量校准数据即可量化权重，提升推理效率并降低部署门槛。
您可以使用`AngleSlim`量化，你也可以直接下载我们量化完成的开源模型使用[Hunyuan-A13B-Instruct-Int4](https://huggingface.co/tencent/Hunyuan-A13B-Instruct-GPTQ-Int4)。

#### INT4 Benchmark
本小节介绍 Hunyuan-A13B-Instruct-GPTQ-Int4 量化模型的Benchmark指标。

|     Bench      | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-GPTQ-Int4 | 
|:--------------:|:---------------------:|:-------------------------------:|
| OlympiadBench  |         82.7          |              84.0               |
|   AIME 2024    |         87.3          |              86.7               |
|     Gsm8k      |         94.39         |              94.24              |
|      BBH       |         88.34         |              87.91              |
|      DROP      |         91.12         |              91.05              |

&nbsp;

## 推理和部署 

HunyuanLLM可以采用TensorRT-LLM, vLLM或sglang部署。为了简化部署过程HunyuanLLM提供了预构建docker镜像，详见一下章节。

镜像：https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags

## 使用TensorRT-LLM推理
### Docker:

为了简化部署过程，HunyuanLLM提供了预构建docker镜像 (注意： 该镜像要求Host的Cuda版本为12.8以上）：

[hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm](https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags) 。您只需要下载模型文件并用下面代码启动docker即可开始推理模型。
```shell
# 拉取
国内：
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-trtllm
国外：
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm

# 启动
docker run --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm     
```

注: Docker容器权限管理。以上代码采用特权模式（--privileged）启动Docker容器会赋予容器较高的权限，增加数据泄露和集群安全风险。建议在非必要情况下避免使用特权模式，以降低安全威胁。对于必须使用特权模式的场景，应进行严格的安全评估，并实施相应的安全监控、加固措施。

### BF16部署

#### Step1：执行推理

#### 方式1：命令行推理

下面我们展示一个代码片段，采用`TensorRT-LLM`快速请求chat model：
修改 examples/pytorch/quickstart_advanced.py 中如下代码：


```python
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                 MTPDecodingConfig)

prompt = "Write a short summary of the benefits of regular exercise"

def main():
    args = parse_arguments()

    llm, sampling_params = setup_llm(args)
    new_prompts = []
    if args.apply_chat_template:
        messages = [{"role": "user", "content": f"{prompt}"}]
        new_prompts.append(llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(new_prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

运行方式：

```shell
python3 quickstart_advanced.py --model_dir "HunyuanLLM模型路径" --tp_size 4 --apply_chat_template
```

#### 方式2：服务化推理

下面我们展示使用`TensorRT-LLM`服务化的方式部署模型和请求。

```shell
trtllm-serve \
  /path/to/HunYuan-moe-A13B \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 128 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```

服务启动成功后, 使用 OpenAI API 进行模型推理调用：
```
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "HunYuan/HunYuan-80B-A13B",
           "Max_tokens": 1024,
           "Temperature": 0,
    "messages": [
      {
        "role": "user",
        "content": "What is Tencent HunYuan?"
      }
    ]
  }'
```

#### FP8/Int4量化模型部署：
目前 TensorRT-LLM 的 fp8 和 int4 量化模型正在支持中，敬请期待。


## 使用vLLM推理
### Docker:

为了简化部署过程，HunyuanLLM提供了预构建docker镜像 (注意： 该镜像要求Host的Cuda版本为12.8以上）：

[hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm](https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags) 。您只需要下载模型文件并用下面代码启动docker即可开始推理模型。
```shell
# 下载模型：
# ModelScope: 
modelscope download --model Tencent-Hunyuan/Hunyuan-A13B-Instruct
# Huggingface: vllm 会自动下载

# 拉取
国内：
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm 
国外：
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm

# 使用 huggingface 起服务
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache:/root/.cache/ \
        --gpus=all -it --entrypoint python docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
         --tensor-parallel-size 4 --model tencent/Hunyuan-A13B-Instruct --trust-remote-code 

# 使用modelscope下载的模型起服务
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache/modelscope:/root/.cache/modelscope \
        --gpus=all -it --entrypoint python   docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --tensor-parallel-size 4 \
         --port 8000 --model /root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct/ --trust_remote_code           
```

注: Docker容器权限管理。以上代码采用特权模式（--privileged）启动Docker容器会赋予容器较高的权限，增加数据泄露和集群安全风险。建议在非必要情况下避免使用特权模式，以降低安全威胁。对于必须使用特权模式的场景，应进行严格的安全评估，并实施相应的安全监控、加固措施。


### BF16部署

BF16可以在2张显存超过80G的GPU卡上部署，如果长文推荐TP4。按如下步骤执行：

运行命令前请先设置如下环境变量：

```shell
export MODEL_PATH=PATH_TO_MODEL
```

#### Step1：执行推理

#### 方式1：命令行推理

下面我们展示一个代码片段，采用`vLLM`快速请求chat model：

注: vLLM组件远程代码执行防护。下列代码中vLLM组件的trust-remote-code配置项若被启用，将允许加载并执行来自远程模型仓库的代码，这可能导致恶意代码的执行。除非业务需求明确要求，否则建议该配置项处于禁用状态，以降低潜在的安全威胁。


```python
import os
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.inputs import PromptType
from transformers import AutoTokenizer

model_path=os.environ.get('MODEL_PATH')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

llm = LLM(model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9)

sampling_params = SamplingParams(
    temperature=0.7, top_p=0.8, max_tokens=4096, top_k=20, repetition_penalty=1.05)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {"role": "user", "content": "Write a short summary of the benefits of regular exercise"},
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

dummy_inputs: List[PromptType] = [{
    "prompt_token_ids": batch
} for batch in tokenized_chat.numpy().tolist()]

outputs = llm.generate(dummy_inputs, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

#### 方式2：服务化推理

下面我们展示使用`vLLM`服务化的方式部署模型并请求

在主节点上运行：

```shell
export VLLM_HOST_IP=${LOCAL_IP}
```
接着我们启动服务，运行 :
```shell
cd inference
sh run_server.sh
```

运行`run_server.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

注意修改`openapi.sh`中的`${LOCAL_IP}`和`${MODEL_PATH}`为服务对应值。


### 量化模型部署：

本部分介绍采用vLLM部署量化后模型的流程。

镜像：部署镜像同BF16。


#### Int8量化模型部署：
部署Int8-weight-only版本HunYuan-A13B模型只需设置`run_server_int8.sh`中的环境变量：
```SHELL
export MODEL_PATH=PATH_TO_BF16_MODEL
```

接着我们启动Int8服务。运行：
```shell
sh run_server_int8.sh
```

运行`run_server_int8.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

#### Int4量化模型部署：
部署Int4-weight-only版本HunYuan-A13B模型只需设置`run_server_int4.sh`中的环境变量，采用GPTQ方式：
```SHELL
export MODEL_PATH=PATH_TO_INT4_MODEL
```

接着我们启动Int4服务。运行：
```shell
sh run_server_int4.sh
```

运行`run_server_int4.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

#### FP8量化模型部署：
部署W8A8C8版本HunYuan-A13B模型只需设置`run_server_int8.sh`中的环境变量：
```shell
export MODEL_PATH=PATH_TO_FP8_MODEL
```

接着我们启动FP8服务。运行：
```shell
sh run_server_fp8.sh
```

运行`run_server_fp8.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

### 性能评估：

本部分介绍采用vLLM部署各个模型（原始模型和量化模型）的效率测试结果，包括不同Batchsize下的推理速度(tokens/s), 测试环境（腾讯云，H80（96G）GPU x 卡数）:

测试命令：
```python
python3 benchmark_throughput.py --backend vllm \
         --input-len 2048 \
         --output-len 14336 \
         --model $MODEL_PATH \
         --tensor-parallel-size $TP \
         --use-v2-block-manager \
         --async-engine \
         --trust-remote-code \
         --num_prompts $BATCH_SIZE \
         --max-num-seqs $BATCH_SIZE
```

| 推理框架 | 模型                          | 部署卡数   | input_length | batch=1             | batch=16              | batch=32       |
|------|-----------------------------|-----------|-------------------------|---------------------|----------------------|----------------------|
| vLLM | Hunyuan-A13B-Instruct                   |    8     | 2048                  |      190.84     |       1246.54      |       1981.99     |
| vLLM | Hunyuan-A13B-Instruct                   |    4     | 2048                  |     158.90      |       779.10       |    1301.75        |
| vLLM | Hunyuan-A13B-Instruct                   |    2     | 2048                  |    111.72       |      327.31        |    346.54         |
| vLLM | Hunyuan-A13B-Instruct(int8 weight only) |    2      | 2048                  |   109.10       |      444.17        |     721.93        |
| vLLM | Hunyuan-A13B-Instruct(W8A8C8-FP8)       |    2      | 2048                  |    91.83       |      372.01        |      617.70       |
| vLLM | Hunyuan-A13B-Instruct(W8A8C8-FP8)       |    1      | 2048                  |     60.07      |         148.80     |      160.41       |


## 使用sglang推理

### BF16部署

#### Step1: 拉取镜像


```
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang
或
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-sglang
```

- 启动 API server:

```
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ipc=host \
    docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang \
    -m sglang.launch_server --model-path hunyuan/huanyuan_A13B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

#### Step2：执行推理

#### 方式1：命令行推理

下面我们展示一个代码片段，采用`sglang`快速请求chat model：


```python
import sglang as sgl
from transformers import AutoTokenizer

model_path=os.environ.get('MODEL_PATH')


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {"role": "user", "content": "Write a short summary of the benefits of regular exercise"},
]
prompts = []
prompts.append(tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
))
print(prompts)

llm = sgl.Engine(
    model_path=model_path,
    tp_size=4,
    trust_remote_code=True,
    mem_fraction_static=0.7,
)

sampling_params = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 4096}
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

#### 方式2：服务化推理

下面我们展示使用`sglang`服务化的方式部署模型和请求。

```shell
model_path="HunyuanLLM模型路径"
python3 -u -m sglang.launch_server \
    --model-path $model_path \
    --tp 4 \
    --trust-remote-code \
```

服务启动成功后, 运行请求脚本：
```python
import openai
client = openai.Client(
    base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="default",
    messages= [
        {"role": "user", "content": "Write a short summary of the benefits of regular exercise"},
    ],
    temperature=0.7,
    max_tokens=4096,
    extra_body={"top_p": 0.8, "top_k": 20}
)
print(response)
```

#### FP8/Int4量化模型部署：
目前 sglang 的 fp8 和 int4 量化模型正在支持中，敬请期待。

## 交互式Demo Web 
hunyuan-A13B 现已开放网页demo。访问 https://hunyuan.tencent.com/?model=hunyuan-a13b 即可简单体验我们的模型。

<br>

## 引用
如果你觉得我们的工作对你有帮助，欢迎引用我们的<a href="report/Hunyuan_A13B_Technical_Report.pdf">技术报告</a>！

<br>


## 联系我们
如果你想给我们的研发和产品团队留言，欢迎联系我们腾讯混元LLM团队。你可以通过邮件（hunyuan_opensource@tencent.com）联系我们。
