# Fine Tune LISA

### Prepare Dataset

JSON should be in this format under `data/train`

```json
{
  "type": "text2text",
  "instances": [
    {
      "input": "SAMPLE_INPUT_1",
      "output": "SAMPLE_OUTPUT_1"
    },
    {
      "input": "SAMPLE_INPUT_2",
      "output": "SAMPLE_OUTPUT_2"
    },
    {
      "input": "SAMPLE_INPUT_3",
      "output": "SAMPLE_OUTPUT_3"
    }
  ]
}
```

### Fine Tuning

[LISA](https://arxiv.org/abs/2403.17919) is a memory-efficient finetuning algorithm that allows tradeoff between memory and the number of randomly unfreezed layers. This script currently is only tested on single GPUs.

```sh
./run_finetune_with_lisa.sh \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_path data/train \
  --output_model_path output_models/finetuned_llama \
  --lisa_activated_layers 1 \
  --lisa_interval_steps 20
```
