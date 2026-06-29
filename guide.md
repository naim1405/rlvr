
What you're trying to do is actually a good fit for PEFT/LoRA before RLVR.

The idea is:

1. **SFT (LoRA)** teaches the model *always emit the required XML/JSON structure*.
2. **RLVR** later optimizes the reasoning/answer quality while assuming the model already follows the output format.

I would **not** train the model to predict only `formatted_output`. Instead, train it as a normal instruction-tuned chat model.

---

## 1. Convert your dataset into an instruction dataset

Instead of

```json
{
  "question": "...",
  "formatted_output": "<think>...</think><answer>...</answer>"
}
```

make each sample look like

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You must always answer using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
        },
        {
            "role": "user",
            "content": "A wild tiger escapes..."
        },
        {
            "role": "assistant",
            "content": "<think>\nWorking through this now.\n</think>\n<answer>\n135\n</answer>"
        }
    ]
}
```

This is much closer to how Qwen was originally trained.

---

## 2. Use the Qwen chat template

Qwen3 expects conversations, not plain text.

The tokenizer will produce something like

```
<|im_start|>system
You must...
<|im_end|>

<|im_start|>user
Question...
<|im_end|>

<|im_start|>assistant
<think>
...
</think>
<answer>
135
</answer>
<|im_end|>
```

During training, only the assistant portion should contribute to the loss.

---

## 3. Dataset class

The HellaSwag dataset in the example is just a placeholder.

Replace

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
```

with your own dataset.

For example,

```yaml
dataset:
  _target_: my_project.datasets.MyDataset
  path: data/train.json
```

where

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.data[idx]

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["formatted_output"],
            },
        ]

        return {
            "messages": messages
        }
```

---

## 4. LoRA configuration

For Qwen, I'd increase the LoRA rank compared to NVIDIA's toy example.

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig

  target_modules: "*_proj"

  dim: 32
  alpha: 64
  dropout: 0.05

  use_triton: true
```

or

```yaml
dim: 64
alpha: 128
```

if your GPU memory allows.

The example's

```yaml
dim: 8
alpha: 32
```

is intentionally tiny.

---

## 5. Learning rate

For LoRA SFT on Qwen, a learning rate around

```yaml
lr: 2e-4
```

or

```yaml
lr: 1e-4
```

is more common.

`1e-5` is quite conservative and may converge slowly when only LoRA parameters are trainable.

---

## 6. Training objective

Your target sequence should simply be

```
<think>
Working through this now.
</think>
<answer>
135
</answer>
```

The model learns to predict every token of that assistant response.

No special loss is needed.

---

## 7. Why do SFT before RLVR?

RLVR assumes the policy already knows the output structure.

Suppose your verifier expects

```
<think>
...
</think>
<answer>
42
</answer>
```

If the model instead outputs

```
42
```

or

```
The answer is 42.
```

the verifier has nothing reliable to parse.

SFT first makes the format nearly deterministic.

Then RLVR only has to optimize

* reasoning quality,
* correctness,
* brevity,
* refusal behavior,

rather than also learning the syntax.

---

## 8. Your overall pipeline

```text
Raw dataset
        │
        ▼
Convert to chat format
        │
        ▼
LoRA SFT
        │
        ▼
Model reliably emits

<think>
...
</think>
<answer>
...
</answer>

        │
        ▼
RLVR
        │
        ▼
Verifier extracts only

<answer>...</answer>

and computes the reward.
```

This separation is generally much more stable than asking RLVR to learn both the response format and the task simultaneously.

One additional note: if your eventual RLVR setup will expose the model's reasoning, consider whether you actually want to train on explicit `<think>` content. Many modern reasoning pipelines instead keep reasoning internal and train the model to emit only a structured final answer (for example, `<answer>...</answer>` or JSON), because exposing chain-of-thought can have downsides. If your RLVR implementation truly requires `<think>` tags as part of the observable output, then your current approach is appropriate; otherwise, it's worth considering whether only the final structured answer needs to be generated.

[ChatSource][https://chatgpt.com/s/t_6a4273c9015c8191b97f92cd6d807414]
