<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ English</a>
</p>
<br><br>

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>


<p align="center">
    <img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4" width="16"/><a href="https://huggingface.co/tencent/Hunyuan-A13B-Instruct"><b>Hugging Face</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🖥️&nbsp;<a href="https://hunyuan.tencent.com" style="color: red;"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕖&nbsp;<a href="https://cloud.tencent.com/product/hunyuan"><b>HunyuanAPI</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕹️&nbsp;<a href="https://hunyuan.tencent.com/?model=hunyuan-a13b"><b>Demo</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <img src="https://avatars.githubusercontent.com/u/109945100?s=200&v=4" width="16"/>&nbsp;<a href="https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct"><b>ModelScope</b></a>
</p>


<p align="center">
    <a href="report/Hunyuan_A13B_Technical_Report.pdf"><b>Technical Report</b> </a> |
    <a href="https://github.com/Tencent-Hunyuan/Hunyuan-A13B"><b>GITHUB</b></a> | 
    <a href="https://cnb.cool/tencent/hunyuan/Hunyuan-A13B"><b>cnb.cool</b></a> | 
    <a href="https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/LICENSE"><b>LICENSE</b></a>
</p>


  
Welcome to the official repository of **Hunyuan-A13B**, an innovative and open-source large language model (LLM) built on a fine-grained Mixture-of-Experts (MoE) architecture. Designed for efficiency and scalability, Hunyuan-A13B delivers cutting-edge performance with minimal computational overhead, making it an ideal choice for advanced reasoning and general-purpose applications, especially in resource-constrained environments.

## Model Introduction

With the rapid advancement of artificial intelligence technology, large language models (LLMs) have achieved remarkable progress in natural language processing, computer vision, and scientific tasks. However, as model scales continue to expand, optimizing resource consumption while maintaining high performance has become a critical challenge. To address this, we have explored Mixture of Experts (MoE) architectures. The newly introduced Hunyuan-A13B model features a total of 80 billion parameters with 13 billion active parameters. It not only delivers high-performance results but also achieves optimal resource efficiency, successfully balancing computational power and resource utilization.

### Key Features and Advantages

- **Compact yet Powerful**: With only 13 billion active parameters (out of a total of 80 billion), the model delivers competitive performance on a wide range of benchmark tasks, rivaling much larger models.
- **Hybrid Reasoning Support**: Supports both fast and slow thinking modes, allowing users to flexibly choose according to their needs.
- **Ultra-Long Context Understanding**: Natively supports a 256K context window, maintaining stable performance on long-text tasks.
- **Enhanced Agent Capabilities**: Optimized for agent tasks, achieving leading results on benchmarks such as BFCL-v3, τ-Bench and C3-Bench.
- **Efficient Inference**: Utilizes Grouped Query Attention (GQA) and supports multiple quantization formats, enabling highly efficient inference.

### Why Choose Hunyuan-A13B?

As a powerful yet computationally efficient large model, Hunyuan-A13B is an ideal choice for researchers and developers seeking high performance under resource constraints. Whether for academic research, cost-effective AI solution development, or innovative application exploration, this model provides a robust foundation for advancement.

&nbsp;

## Related News
* 2025.6.27 We have open-sourced  **Hunyuan-A13B-Pretrain** , **Hunyuan-A13B-Instruct** , **Hunyuan-A13B-Instruct-FP8** , **Hunyuan-A13B-Instruct-GPTQ-Int4** on Hugging Face. In addition, we have released a <a href="report/Hunyuan_A13B_Technical_Report.pdf">technical report </a> and a training and inference operation manual, which provide detailed information about the model’s capabilities as well as the operations for training and inference.

<br>


## Benchmark

Note: The following benchmarks are evaluated by TRT-LLM-backend on several **base models**. 

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


Hunyuan-A13B-Instruct has achieved highly competitive performance across multiple benchmarks, particularly in mathematics, science, agent domains, and more. We compared it with several powerful models, and the results are shown below.

| Topic               |                        Bench                         | OpenAI-o1-1217 | DeepSeek R1 | Qwen3-A22B | Hunyuan-A13B-Instruct |
|:-------------------:|:----------------------------------------------------:|:-------------:|:------------:|:-----------:|:---------------------:|
| **Mathematics**     |            AIME 2024<br>AIME 2025<br>MATH            | 74.3<br>79.2<br>96.4 | 79.8<br>70<br>94.9 | 85.7<br>81.5<br>94.0 | 87.3<br>76.8<br>94.3 |
| **Science**         |            GPQA-Diamond<br>OlympiadBench             | 78<br>83.1 | 71.5<br>82.4 | 71.1<br>85.7 | 71.2<br>82.7 |
| **Coding**          |  Livecodebench<br>Fullstackbench<br>ArtifactsBench   | 63.9<br>64.6<br>38.6 | 65.9<br>71.6<br>44.6 | 70.7<br>65.6<br>44.6 | 63.9<br>67.8<br>43 |
| **Reasoning**       |              BBH<br>DROP<br>ZebraLogic               | 80.4<br>90.2<br>81 | 83.7<br>92.2<br>78.7 | 88.9<br>90.3<br>80.3 | 89.1<br>91.1<br>84.7 |
| **Instruction<br>Following** |                 IF-Eval<br>SysBench                  | 91.8<br>82.5 | 88.3<br>77.7 | 83.4<br>74.2 | 84.7<br>76.1 |
| **Text<br>Creation**|                LengthCtrl<br>InsCtrl                 | 60.1<br>74.8 | 55.9<br>69 | 53.3<br>73.7 | 55.4<br>71.9 |
| **NLU**             |               ComplexNLU<br>Word-Task                | 64.7<br>67.1 | 64.5<br>76.3 | 59.8<br>56.4 | 61.2<br>62.9 |
| **Agent**           | BFCL v3<br> τ-Bench<br>ComplexFuncBench<br> C3-Bench | 67.8<br>60.4<br>47.6<br>58.8 | 56.9<br>43.8<br>41.1<br>55.3 | 70.8<br>44.6<br>40.6<br>51.7 | 78.3<br>54.7<br>61.2<br>63.5 |


&nbsp;

## Use with transformers

Our model defaults to using slow-thinking reasoning, and there are two ways to disable CoT reasoning. 
1. Pass "enable_thinking=False" when calling apply_chat_template.
2. Adding "/no_think" before the prompt will force the model not to use perform CoT reasoning. Similarly, adding "/think" before the prompt will force the model to perform CoT reasoning.

The following code snippet shows how to use the transformers library to load and apply the model. It also demonstrates how to enable and disable the reasoning mode , and how to parse the reasoning process along with the final output.



```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

model_name_or_path = os.environ['MODEL_PATH']
# model_name_or_path = "tencent/Hunyuan-A13B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",trust_remote_code=True)  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": "Write a short summary of the benefits of regular exercise"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt",
                                                enable_thinking=True # Toggle thinking mode (default: True)
                                                )
                                                
outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=4096)

output_text = tokenizer.decode(outputs[0])

think_pattern = r'<think>(.*?)</think>'
think_matches = re.findall(think_pattern, output_text, re.DOTALL)

answer_pattern = r'<answer>(.*?)</answer>'
answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)

think_content = [match.strip() for match in think_matches][0]
answer_content = [match.strip() for match in answer_matches][0]
print(f"thinking_content:{think_content}\n\n")
print(f"answer_content:{answer_content}\n\n")
```



## Training Quick Start
Hunyuan-A13B provides processes related to model training. Please refer to [Training](train/README.md) for model training purposes.


## Quantization Compression

We used our own [AngelSlim](https://github.com/Tencent/AngelSlim) compression tool to produce FP8 and INT4 quantization models. [AngelSlim](https://github.com/Tencent/AngelSlim) support one-click quantization and compression of LLms  models, please refer to [AngelSlim documentation](https://angelslim.readthedocs.io/).

### FP8 Quantization
We use FP8-static quantization, FP8 quantization adopts 8-bit floating point format, through a small amount of calibration data (without training) to pre-determine the quantization scale, the model weights and activation values will be converted to FP8 format, to improve the inference efficiency and reduce the deployment threshold. We you can use AngleSlim quantization, you can also directly download our quantization completed open source model to use [Hunyuan-A13B-Instruct-FP8](https://huggingface.co/tencent/Hunyuan-A13B-Instruct-FP8).

#### FP8 Benchmark
This subsection describes the Benchmark metrics for the Hunyuan-80B-A13B-Instruct-FP8 quantitative model.

|   Bench   | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-FP8 | 
|:---------:|:---------------------:|:-------------------------:|
| AIME 2024 |         87.3          |           86.7            |
|   Gsm8k   |         94.39         |           94.01           |
|    BBH    |         89.1          |           88.34           |
|   DROP    |         91.1          |           91.1            |

### Int4 Quantization
We use the GPTQ algorithm to achieve W4A16 quantization, which processes the model weights layer by layer, uses a small amount of calibration data to minimize the reconfiguration error of the quantized weights, and adjusts the weights layer by layer by the optimization process of approximating the Hessian inverse matrix. The process eliminates the need to retrain the model and requires only a small amount of calibration data to quantize the weights, improving inference efficiency and lowering the deployment threshold. You can use `AngleSlim` quantization, you can also directly download our quantization completed open source model to use [Hunyuan-A13B-Instruct-Int4](https://huggingface.co/tencent/Hunyuan-A13B-Instruct-GPTQ-Int4).

#### Int4 Benchmark
This subsection describes the Benchmark metrics for the Hunyuan-80B-A13B-Instruct-GPTQ-Int4 quantitative model.

|     Bench      | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-GPTQ-Int4 | 
|:--------------:|:---------------------:|:-------------------------------:|
| OlympiadBench  |         82.7          |              84.0               |
|   AIME 2024    |         87.3          |              86.7               |
|     Gsm8k      |         94.39         |              94.24              |
|      BBH       |         89.1         |              87.91              |
|      DROP      |         91.1         |              91.05              |


## Deployment   

For deployment, you can use frameworks such as **TensorRT-LLM**, **vLLM**, or **SGLang** to serve the model and create an OpenAI-compatible API endpoint.

image: https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags 


### TensorRT-LLM

#### Docker Image 

We provide a pre-built Docker image based on the latest version of TensorRT-LLM.

- To get started:

https://hub.docker.com/r/hunyuaninfer/hunyuan-large/tags 

```
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm
```
```
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm
```

- Prepare Configuration file:

```
cat >/path/to/extra-llm-api-config.yml <<EOF
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
- 1
- 2
- 4
- 8
- 16
- 32
print_iter_log: true
EOF
```


- Start the API server:


```
trtllm-serve \
  /path/to/HunYuan-moe-A13B \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.6 \
  --trust_remote_code \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```


### vllm

#### Docker Image
We provide a pre-built Docker image containing vLLM 0.8.5 with full support for this model. The official vllm release is currently under development， **note: cuda 12.8 is require for this docker**.

- To get started:

```
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm 
or
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm
```

- Download Model file: 
  - Huggingface:  will download automicly by vllm.
  - ModelScope: `modelscope download --model Tencent-Hunyuan/Hunyuan-A13B-Instruct`
 

- Start the API server:

model download by huggingface:
```
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache:/root/.cache/ \
        --gpus=all -it --entrypoint python  hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm
 \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
         --tensor-parallel-size 4 --model tencent/Hunyuan-A13B-Instruct --trust-remote-code 

``` 

model downloaded by modelscope:
```
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache/modelscope:/root/.cache/modelscope \
        --gpus=all -it --entrypoint python   hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --tensor-parallel-size 4 --port 8000 \ 
         --model /root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct/ --trust_remote_code  
```

#### Quantitative model deployment
This section describes the process of deploying a post-quantization model using vLLM.

Image: Deploy image as in BF16.

##### Int8 quantitative model deployment
Deploying the Int8-weight-only version of the HunYuan-A13B model only requires setting the environment variables in `run_server_int8.sh`
```SHELL
export MODEL_PATH=PATH_TO_BF16_MODEL
```

Next we start the Int8 service. Run:
```shell
sh run_server_int8.sh
```

After running run_server_int8.sh successfully, run the request script
```shell
sh openapi.sh
```

##### Int4 quantitative model deployment
Deploying the Int4-weight-only version of the HunYuan-A13B model only requires setting the environment variables in `run_server_int4.sh`, using the GPTQ method
```SHELL
export MODEL_PATH=PATH_TO_INT4_MODEL
```

Next we start the Int4 service. Run
```shell
sh run_server_int4.sh
```

After running `run_server_int4.sh` successfully, run the request script
```shell
sh openapi.sh
```

##### FP8 quantitative model deployment
Deploying the W8A8C8 version of the HunYuan-A13B model only requires setting the environment variables in `run_server_int8.sh`
```shell
export MODEL_PATH=PATH_TO_FP8_MODEL
```

Next we start the FP8 service. Run
```shell
sh run_server_fp8.sh
```

After running `run_server_fp8.sh` successfully, run the request script
```shell
sh openapi.sh
```



### SGLang

#### Docker Image 

We also provide a pre-built Docker image based on the latest version of SGLang.

To get started:

- Pull the Docker image

```
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang
or
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-sglang
```

- Start the API server:

```
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ipc=host \
    docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang \
    -m sglang.launch_server --model-path hunyuan/huanyuan_A13B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

## Community Resources

- [Hunyuan-A13B Quick Start on CNB](https://cnb.cool/tencent/hunyuan/examples/Hunyuan-A13B-Quick-Start)

## Contact Us

If you would like to leave a message for our R&D and product teams, Welcome to contact our open-source team . You can also contact us via email (hunyuan_opensource@tencent.com).
