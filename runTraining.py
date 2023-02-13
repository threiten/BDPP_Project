import datasets
import transformers
import utils
import argparse
import copy
import torch
import numpy as np
import evaluate

def tokenize_inputs(examples: datasets.Dataset, tokenizer: transformers.BasicTokenizer, truncation: bool=True):
    tokenized_inputs = tokenizer(examples['text'], truncation=truncation)
    labels = examples['party']
    new_labels = []
    for label in labels:
        new_labels.append(utils.party_dict[label])
    tokenized_inputs['labels'] = new_labels

    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    metric = evaluate.load('accuracy')
    all_metrics = metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": all_metrics["accuracy"],
    }

def main(options: argparse.Namespace):
    raw_datasets = datasets.load_dataset('threite/Bundestag-v2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-base')

    tokenized_datasets = raw_datasets.map(tokenize_inputs, batched=True, remove_columns=raw_datasets['train'].column_names, fn_kwargs={'tokenizer': tokenizer, 'truncation': True})

    id2label={v: k for k, v in utils.party_dict.items() if k not in ['', 'independent']}
    id2label[6] = 'unknown'
    label2id = copy.deepcopy(utils.party_dict)
    label2id['unknown'] = 6
    del label2id['independent']
    del label2id['']

    train_dataset_by_label = [tokenized_datasets['train'].filter(lambda x: x['labels']==lab) for lab in id2label.keys()]
    probs = [.96/(len(id2label.keys())-2.)] * int(len(id2label.keys()))
    probs[4] = 0.02
    probs[6] = 0.02
    print(probs)
    tokenized_datasets['train'] = datasets.interleave_datasets(train_dataset_by_label, probabilities=probs)
    print(tokenized_datasets)

    model = transformers.AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', label2id=label2id, id2label=id2label)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    use_mps = torch.backends.mps.is_available()
    
    push_to_hub = options.hfToken is not None
    training_args = transformers.TrainingArguments(
        options.modelName,
        evaluation_strategy='steps',
        save_strategy='steps',
        learning_rate=2e-5,
        num_train_epochs=16,
        weight_decay=0.01,
        use_mps_device=use_mps,
        push_to_hub=push_to_hub,
        hub_token=options.hfToken,
        eval_steps=5000,
        save_steps=5000
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics
    )
    
    print('Resume Training:', options.resumeTraining)
    trainer.train(resume_from_checkpoint=options.resumeTraining)
    trainer.push_to_hub()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.add_argument_group()
    args.add_argument('--modelName', '-n', action='store', type=str, help='Name of the model used when pushed to the huggingface hub')
    args.add_argument('--hfToken', '-t', action='store', type=str, help='Acces Token for the huggingface hub')
    args.add_argument('--resumeTraining', '-r', action='store_true', default=False, help='Use this flag to resume training from a saved checkpoint.')
    options = parser.parse_args()
    main(options)