## Structure
```
ASKBIZ
│   
├── data
│   └── dastaset                    (RAW-dataset)
├── utils
│   └── prompter.py                 (Prompt template)
├── embed
│   └── vector                      (RAG dataset)
├── model_file
│   └── lora_model_file             (Finetuning model)
│   
│   app.py
│   finetune.py
│   make_standalone_model.py
└   README.md
```

![asset1](/assets/askbiz_workflow.png)

## Installation

```
$ pip install -r requirements.txt
```

## Data

- 동아비즈니스리뷰(DBR)와 하버드비즈니스리뷰(HBR)을 활용하여 [FLAN](https://arxiv.org/abs/2109.01652) 데이터셋 생성

![asset2](/assets/dataset_generation_workflow.png)

## Model

- Backbone 모델 : [beomi/KoAlpaca-Polyglot-12.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B)


## Fine-tuning

### LoRA

- 파인튜닝 관련 모든 소스는 [tleon/alpaca-lora](https://github.com/tloen/alpaca-lora)에 있는 자료를 활용

Example usage:

```
$ python finetune.py \
    --base_model "{YOUR_BASE_MODEL}" \
    --data_path "{YOUR_DATA_PATH}" \
    --output_dir "{YOUR_OUTPUT_PATH}"
    --batch_size 16 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 3e-5 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --train_on_inputs \
    --group_by_length
```


## Streaming (transformers.TextIteratorStreamer)

```
$ python app.py
```

- default port : `7860`
