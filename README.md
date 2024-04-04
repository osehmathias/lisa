# Fine Tune LISA - Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning

LISA (Layerwise Importance Sampled AdamW) is an innovative approach aimed at optimizing memory efficiency during the fine-tuning of Large Language Models (LLMs). By leveraging the concept of importance sampling across different layers of LLMs and allowing for selective freezing of middle layers, LISA achieves superior performance to both LoRA (Low-Rank Adaptation) and full parameter training, all while maintaining low memory usage.

## Table of Contents

- [Fine Tune LISA - Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](#fine-tune-lisa---layerwise-importance-sampling-for-memory-efficient-large-language-model-fine-tuning)
  - [Abstract](#abstract)
  - [Citation](#citation)
  - [Prerequisites](#prerequisites)
  - [Prepare Dataset](#prepare-dataset)
  - [Fine-Tuning with LISA](#fine-tuning-with-lisa)
  - [Performance and Benchmarks](#performance-and-benchmarks)
  - [Acknowledgments](#acknowledgments)

## Abstract

The machine learning community has continually evolved with the advent of LLMs, though the substantial memory consumption required has limited large-scale training and applications. Existing Parameter Efficient Fine-Tuning techniques like LoRA offer some relief, but often at a compromise to performance. Through extensive research, the LISA technique has been formulated, utilizing a layerwise importance sampling strategy that significantly enhances fine-tuning efficiency and effectiveness across various benchmarks.

## Citation

I am not the author of the paper.

If you use LISA in your research, please cite the paper:

```bib
@article{pan2024lisa,
title={LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning},
authors={Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, Tong Zhang},
journal={arXiv preprint arXiv:2403.17919},
year={2024}
}
```

## Prerequisites

Before starting, ensure your environment is set up with the necessary dependencies:

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.8+
- CUDA 11.0+ (for GPU acceleration)

## Prepare Dataset

Prepare your dataset in a JSON format and place it under `data/train`. The JSON structure should look like this:

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

## Fine-Tuning with LISA

Fine-tune your LLM with the LISA algorithm using the provided script. The script is tested on single GPU setups but can be adjusted for multi-GPU configurations.

```bash
./run_finetune_with_lisa.sh \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_path data/train \
  --output_model_path output_models/finetuned_llama \
  --lisa_activated_layers 1 \
  --lisa_interval_steps 20
```

## Performance and Benchmarks

LISA significantly outperforms existing fine-tuning methods in memory efficiency and model performance across various tasks and benchmarks. For detailed results and comparisons, refer to the paper.

## Acknowledgments

This project was made with LMFlow, an extensible toolkit designed for the fine-tuning and inference of large foundation models.
