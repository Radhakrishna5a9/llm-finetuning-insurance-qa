# LLM Fine-Tuning for Health Insurance Q&A

Fine-tuning 8B open-weight LLMs to answer questions about the **HDFC ERGO Optima Restore** health insurance policy, comparing supervised fine-tuning (SFT) with two preference-optimization methods: **ORPO** and **KTO**. All runs use LoRA adapters (PEFT) on a single NVIDIA A40, with experiment tracking in Weights & Biases.

The fine-tuned model is the generator in a streaming RAG pipeline (Client -> WebSocket -> RAG over Qdrant -> LLM on AWS EKS -> streamed response).

## Results

Each model was evaluated on the same 30 held-out policy questions. Every answer was scored manually against a reference answer: `1` = correct, `0.5` = partially correct, `0` = wrong.

- **Strict** = share of answers scored 1
- **Lenient** = share of answers scored 1 or 0.5
- **Weighted** = mean score

| Method | Base model | Key config | Strict | Lenient | Weighted |
|---|---|---|---|---|---|
| KTO | Meta-Llama-3-8B-Instruct | LoRA r=16, 6 epochs, effective batch 8, lr 5e-6, beta 0.1 | 40.00% | **83.33%** | **61.67%** |
| ORPO | DeepSeek-R1-Distill-Llama-8B | LoRA r=16 / alpha 32, 5 epochs, batch 2, lr 5e-6, beta 0.1 | **43.33%** | 76.67% | 60.00% |
| SFT (LoRA) | Meta-Llama-3-8B-Instruct | 6 epochs, batch 3, NEFTune 0.005, label smoothing 0.1 | 40.00% | 80.00% | 60.00% |
| ORPO | Meta-Llama-3-8B-Instruct | LoRA r=16 / alpha 32, 5 epochs, batch 2, lr 5e-6, beta 0.1 | **43.33%** | 73.33% | 58.33% |
| SFT (LoRA) | DeepSeek-R1-Distill-Llama-8B | LoRA r=32 / alpha 64, 5 epochs, batch 1, lr 5e-5, NEFTune 0.05 | - | - | - |

The SFT Llama-3-8B scores come from `results/`; its training notebook is not part of this repo. The SFT DeepSeek run was stopped at the training stage (see below).

Score breakdown out of 30 questions:

| Run | Correct (1) | Partial (0.5) | Wrong (0) |
|---|---|---|---|
| KTO Llama-3-8B | 12 | 13 | 5 |
| ORPO DeepSeek-8B | 13 | 10 | 7 |
| SFT Llama-3-8B | 12 | 12 | 6 |
| ORPO Llama-3-8B | 13 | 9 | 8 |

Training-side metrics:

- **ORPO Llama-3-8B**: eval loss 1.1565, reward accuracy 0.9245, 41,943,040 trainable parameters (0.52% of the model)
- **ORPO DeepSeek-8B**: eval loss 0.782, reward accuracy 0.9906
- **KTO Llama-3-8B**: eval loss 0.1078, reward margin 6.58
- **SFT DeepSeek-8B (r=32)**: train loss fell to 0.224 while eval loss rose from 3.11 to 4.11, a clear overfitting signal, so this run was not taken forward to the accuracy check

### Takeaways

- Preference optimization (KTO) gave the best lenient and weighted accuracy, mostly by turning wrong answers into partially correct ones.
- ORPO produced the most fully correct answers on both base models, and the DeepSeek distill edged out Llama-3 under identical settings.
- Plain SFT with a higher LoRA rank and learning rate overfit quickly on a ~2k-example dataset; low NEFTune noise plus label smoothing kept SFT competitive.
- Strict accuracy sits around 40-43% for every method, which is the motivation for pairing the model with retrieval (RAG) instead of relying on memorized policy details.

## Repository layout

```
notebooks/
  01_data_prep/
    convert_xlsx_to_parquet.ipynb      Convert the Excel Q&A sheet to Parquet
    create_dataset_in_hf.ipynb         Build and push the preference / KTO datasets to the Hugging Face Hub
  02_sft_lora/
    lora_deepseek_8b_loraRank32_ep5_bs1.ipynb   SFT with LoRA on DeepSeek-R1-Distill-Llama-8B
  03_orpo/
    orpo_llama3_8B_training_20260108_epochs_5_cw.ipynb     ORPO training, Llama-3-8B-Instruct
    accuracyCheck_llama3_8B_ORPO_ep5_bs2_code.ipynb         30-question evaluation of that adapter
    orpo_deepseek_8B_training_20251231_epoch_5_cw.ipynb     ORPO training, DeepSeek-R1-Distill-Llama-8B
    accuracyCheck_deepseek_8B_ORPO_ep3_bs2_code.ipynb       30-question evaluation of that adapter
  04_kto/
    kto_llama3_8B_training_20260122.ipynb                   KTO training, Llama-3-8B-Instruct
    accuracyCheck_llama3_8B_KTO_ep6_ebs8_code.ipynb         30-question evaluation of that adapter
results/
  eval_results.csv                                          Scored answers, SFT Llama-3-8B run
  accuracyCheck_llama38b_ep6_bs3_nn0.005_ls0.1.xlsx         Same scored answers as an Excel sheet
```

## Datasets

| Dataset | Format | Size | Used by |
|---|---|---|---|
| [`agi123/optima_restore_ha`](https://huggingface.co/datasets/agi123/optima_restore_ha) | `question` / `answer` (chosen) / `rejected` | 2,115 train, 30 test | SFT, ORPO |
| [`agi123/optima_restore_ha_kto`](https://huggingface.co/datasets/agi123/optima_restore_ha_kto) | `prompt` / `completion` / `label` (true/false) | 4,230 rows, 50/50 desirable vs undesirable | KTO |

The KTO dataset is derived from the preference dataset: every chosen answer becomes a `label=True` row and every rejected answer a `label=False` row.

## Trained adapters (Hugging Face Hub)

- [`agi123/Meta-Llama-3-8B-Instruct-orpo-orpo_health_insurance-adapters_epochs_5`](https://huggingface.co/agi123/Meta-Llama-3-8B-Instruct-orpo-orpo_health_insurance-adapters_epochs_5)
- [`agi123/DeepSeek-R1-Distill-Llama-8B-orpo-orpo_health_insurance_deepseek-epochs-5adapters`](https://huggingface.co/agi123/DeepSeek-R1-Distill-Llama-8B-orpo-orpo_health_insurance_deepseek-epochs-5adapters)
- [`agi123/Meta-Llama-3-8B-Instruct-kto-kto_health_insurance-effective_bactch_size-8-epochs-6-adapters`](https://huggingface.co/agi123/Meta-Llama-3-8B-Instruct-kto-kto_health_insurance-effective_bactch_size-8-epochs-6-adapters)
- [`agi123/DeepSeek-R1-Distill-Llama-8B-optima_restore-ep5_bs1-adapters`](https://huggingface.co/agi123/DeepSeek-R1-Distill-Llama-8B-optima_restore-ep5_bs1-adapters)

## Methods in one paragraph each

- **SFT + LoRA**: freeze the base model and train small low-rank matrices on the attention projections using (question, answer) pairs. NEFTune adds noise to embeddings during training to reduce memorization.
- **ORPO** (Odds Ratio Preference Optimization): a single-stage method that combines the SFT loss on the chosen answer with an odds-ratio penalty that pushes the model away from the rejected answer. No separate reference model is needed.
- **KTO** (Kahneman-Tversky Optimization): learns from individual answers labelled desirable or undesirable rather than from paired comparisons, using a prospect-theory style loss against a reference model.

## How to run

1. Use a CUDA GPU with at least 48 GB of memory (runs were done on an NVIDIA A40) and Python 3.10+.
2. Set credentials as environment variables; the notebooks read them with `os.environ` and never hard-code them:
   - `HF_TOKEN` (Hugging Face, write access needed to push adapters and datasets)
   - `WANDB_API_KEY` (Weights & Biases)
3. Each notebook installs its own pinned package versions in its first cell. `requirements.txt` holds the stack used by the Llama-3 ORPO and KTO training runs.
4. Run the notebooks in folder order: data prep -> training -> matching `accuracyCheck_*` notebook.

## Tech stack

PyTorch, Hugging Face Transformers, PEFT (LoRA), TRL (`SFTTrainer`, `ORPOTrainer`, `KTOTrainer`), Datasets, Accelerate, Flash-Attention 2, bf16, Weights & Biases.
