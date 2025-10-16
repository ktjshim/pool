import json
import argparse
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, IntervalStrategy, AutoModelForCausalLM, AutoTokenizer
import random
import torch
import numpy as np
import sys
from graph_mistral import GraphMistralForCausalLM
import wandb
sys.path.append("../")
from utils import prepare_training_ids, init_random_state


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100


LLM_INFER_PROMPT = """Please understand the user's request and generate task steps and task invocation graph to solve the request.\n""" \
                  + """## Requirements:\n1. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx to do xxx" ], "task_nodes": [{"task": "task name must be from TASK LIST", "arguments": [ {"name": "parameter name", "value": "parameter value"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n""" \
                  + """2. The generated task steps and task nodes can resolve the given user request perfectly. Task name must be selected from TASK LIST.\n""" \
                  + """3. The task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes.\n""" \
                  + """4. The task links (task_links) should reflect the dependencies among task nodes, i.e. the order in which the APIs are invoked, you should also understand each tool's input and output demands.\n""" \
                  + """## User Request:\n{{user_request}}\nNow please generate your response in a strict JSON format:\n## Response:\n"""



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val, torch.Tensor):
                item[key] = val[idx]
            else:
                item[key] = torch.tensor(val[idx])
        item['idx'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class GraphTrainer(Trainer):
    """Custom Trainer that passes graph-specific inputs to the model"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to pass token_type_ids and graph_attention_mask to the model
        """
        # Extract graph-specific inputs
        token_type_ids = inputs.pop("token_type_ids", None)
        graph_attention_mask = inputs.pop("graph_attention_mask", None)

        # Forward pass with graph inputs
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        if graph_attention_mask is not None:
            inputs["graph_attention_mask"] = graph_attention_mask

        # Get model outputs
        outputs = model(**inputs)

        # Extract logits and labels
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = inputs.get("labels")

        # Compute loss
        if labels is not None:
            # Shift tokens for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = None

        return (loss, outputs) if return_outputs else loss


def prepare_llm_training_data(dataset_name="huggingface", train_ids=None):
    """
    Prepare training data samples (without graph processing).
    Graph structure is handled separately for efficiency.
    """
    raw_data = f"../data/{dataset_name}/data.json"
    data_contents = []

    for line in open(raw_data, 'r'):
        content = json.loads(line)

        if train_ids and content["id"] not in train_ids:
            continue

        input = content['user_request']
        prompt = LLM_INFER_PROMPT

        output = {
                "task_steps": content["task_steps"],
                "task_nodes": content["task_nodes"],
                "task_links": content["task_links"]
        }

        data_contents.append({
            'id': content['id'],
            'prompt': prompt,
            "input": input,
            "output": json.dumps(output),
        })

    random.shuffle(data_contents)
    return data_contents


def prepare_graph_structure(dataset_name):
    """
    Prepare graph structure once for the entire dataset.
    Tokenizes the graph and builds the base attention mask that will be reused.

    Args:
        dataset_name: name of the dataset

    Returns:
        dict with:
            - graph_str: the formatted graph string
            - graph_token_ids: tokenized graph
            - graph_token_spans: dict mapping tool_name -> (start_idx, end_idx) relative to graph
            - base_graph_mask: [graph_len, graph_len] attention mask for graph region
    """
    tool_file = f"../data/{dataset_name}/tool_desc.json"
    tool_desc_data = json.load(open(tool_file, 'r'))
    all_tools = [tool['id'] for tool in tool_desc_data['nodes']]

    # Load graph edges
    try:
        graph_file = f"../data/{dataset_name}/graph_desc.json"
        graph_links = json.load(open(graph_file, 'r'))['links']
    except:
        # If no graph_desc.json, create empty links (no edges)
        graph_links = []

    # Create graph representation
    graph_str = "<start_graph>" + "".join([f"<node>{tool}" for tool in all_tools]) + "<end_graph>"

    # Tokenize the graph once
    graph_token_ids = tokenizer(graph_str, add_special_tokens=False).input_ids

    # Find token spans for each tool (relative positions within graph)
    graph_token_spans = {}
    for tool_name in all_tools:
        tool_marker = f"<node>{tool_name}"
        tool_tokens = tokenizer(tool_marker, add_special_tokens=False).input_ids

        # Find where this tool appears in graph_token_ids
        for i in range(len(graph_token_ids) - len(tool_tokens) + 1):
            if graph_token_ids[i:i+len(tool_tokens)] == tool_tokens:
                graph_token_spans[tool_name] = (i, i + len(tool_tokens))
                break

    # Build base graph attention mask (only for graph region)
    graph_len = len(graph_token_ids)
    base_graph_mask = torch.zeros(graph_len, graph_len, dtype=torch.bool)

    # Allow full self-attention within each node
    for tool_name, (start_idx, end_idx) in graph_token_spans.items():
        base_graph_mask[start_idx:end_idx, start_idx:end_idx] = True

    # Add edges from graph_links
    for link in graph_links:
        source_tool = link["source"]
        target_tool = link["target"]

        if source_tool in graph_token_spans and target_tool in graph_token_spans:
            source_start, source_end = graph_token_spans[source_tool]
            target_start, target_end = graph_token_spans[target_tool]

            # Bidirectional blocked attention
            base_graph_mask[source_start:source_end, target_start:target_end] = True
            base_graph_mask[target_start:target_end, source_start:source_end] = True

    return {
        'graph_str': graph_str,
        'graph_token_ids': graph_token_ids,
        'graph_token_spans': graph_token_spans,
        'base_graph_mask': base_graph_mask,
        'all_tools': all_tools
    }


def tokenizer_dataset(raw_data, graph_info):
    """
    Tokenize dataset using pre-built graph structure (EFFICIENT VERSION).

    Args:
        raw_data: list of data samples
        graph_info: dict from prepare_graph_structure() containing pre-built graph

    Returns:
        dict with input_ids, attention_mask, labels, token_type_ids, graph_attention_mask
    """
    bos_tokens = tokenizer(BOS, add_special_tokens=False)
    eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
    eos_tokens = tokenizer(EOS, add_special_tokens=False)

    # Extract pre-built graph components (computed ONCE for all samples)
    graph_str = graph_info['graph_str']
    graph_token_ids = graph_info['graph_token_ids']
    base_graph_mask = graph_info['base_graph_mask']
    graph_len = len(graph_token_ids)

    full_input_ids, full_attention_masks, full_labels = [], [], []
    full_token_type_ids, full_graph_attention_masks = [], []

    for sample in raw_data:
        # Build prompt with pre-tokenized graph
        prefix = "## Task Graph:\n"
        suffix = "\n" + sample["prompt"].replace("{{user_request}}", sample["input"])
        label = sample["output"]

        # Tokenize prefix and suffix only
        tokenized_prefix = tokenizer(prefix, add_special_tokens=False)
        tokenized_suffix = tokenizer(suffix, add_special_tokens=False)
        tokenized_label = tokenizer(label, add_special_tokens=False)

        # Build input_ids using pre-tokenized graph (NO re-tokenization!)
        prefix_ids = tokenized_prefix.input_ids
        suffix_ids = tokenized_suffix.input_ids

        combined_input_ids = prefix_ids + graph_token_ids + suffix_ids

        label_ids = tokenized_label.input_ids + eos_tokens.input_ids
        input_ids = bos_tokens.input_ids + combined_input_ids + eos_user_tokens.input_ids + label_ids
        final_label_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_ids)) + label_ids

        # Build token_type_ids (0 = text, 1 = graph)
        token_type_ids = [0] * len(bos_tokens.input_ids)  # BOS
        token_type_ids += [0] * len(prefix_ids)  # prefix text
        token_type_ids += [1] * graph_len  # GRAPH TOKENS (pre-computed length)
        token_type_ids += [0] * len(suffix_ids)  # suffix text
        token_type_ids += [0] * len(eos_user_tokens.input_ids)  # EOS_USER
        token_type_ids += [0] * len(label_ids)  # labels are text

        # Insert pre-built graph mask into full sequence mask (NO recomputation!)
        total_len = len(input_ids)
        graph_attention_mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Calculate where graph tokens start in the full sequence
        graph_start = len(bos_tokens.input_ids) + len(prefix_ids)
        graph_end = graph_start + graph_len

        # Insert the pre-built base_graph_mask (just copy, no computation!)
        graph_attention_mask[graph_start:graph_end, graph_start:graph_end] = base_graph_mask

        full_input_ids.append(input_ids)
        full_attention_masks.append([1] * len(input_ids))
        full_labels.append(final_label_ids)
        full_token_type_ids.append(token_type_ids)
        full_graph_attention_masks.append(graph_attention_mask)

    max_length = max([len(x) for x in full_input_ids])

    # Pad all sequences
    for i in range(len(full_input_ids)):
        current_length = len(full_input_ids[i])
        pad_length = max_length - current_length

        # Pad input sequences (left padding)
        full_input_ids[i] = [0] * pad_length + full_input_ids[i]
        full_attention_masks[i] = [0] * pad_length + full_attention_masks[i]
        full_labels[i] = [IGNORE_INDEX] * pad_length + full_labels[i]
        full_token_type_ids[i] = [0] * pad_length + full_token_type_ids[i]

        # Pad graph attention mask (2D)
        current_mask_size = full_graph_attention_masks[i].shape[0]
        if current_mask_size < max_length:
            padded_mask = torch.zeros(max_length, max_length, dtype=torch.bool)
            padded_mask[pad_length:, pad_length:] = full_graph_attention_masks[i]
            full_graph_attention_masks[i] = padded_mask
        elif current_mask_size > max_length:
            full_graph_attention_masks[i] = full_graph_attention_masks[i][:max_length, :max_length]

    input_ids = torch.tensor(full_input_ids).to(device)
    attention_mask = torch.tensor(full_attention_masks).to(device)
    label_input_ids = torch.tensor(full_labels).to(device)
    token_type_ids = torch.tensor(full_token_type_ids).to(device)
    graph_attention_masks = torch.stack(full_graph_attention_masks).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_input_ids,
        "token_type_ids": token_type_ids,
        "graph_attention_mask": graph_attention_masks
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='huggingface', choices=['huggingface', 'multimedia', 'dailylife'])
    parser.add_argument('--llm', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--wandb_project', type=str, default='graph-mistral-training')  # ADD THIS
    parser.add_argument('--wandb_run_name', type=str, default=None)  # ADD THIS
    args = parser.parse_args()
    print(args, "\n")
    init_random_state(args.seed)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    train_ids = prepare_training_ids(args.dataset, alignment_ids=alignment_ids)


    # ===================
    # EFFICIENT: Process graph ONCE for the entire dataset
    
    graph_info = prepare_graph_structure(args.dataset)
    data_contents = prepare_llm_training_data(args.dataset, train_ids=train_ids)
    encodings = tokenizer_dataset(data_contents, graph_info)
    # ===================

    dataset = TextDataset(encodings)
    train_num = int(0.85 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_num)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_num, len(dataset))))

    print(f"Dataset sizes: total={len(dataset)}, train={len(train_dataset)}, val={len(val_dataset)}")

    save_dir = f"output/{args.dataset}_{args.llm}_seed{args.seed}"
    
    # ===================
    # INITIALIZE WANDB
    run_name = args.wandb_run_name or f"Graph{args.llm}_{args.dataset}_seed{args.seed}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "dataset": args.dataset,
            "llm": args.llm,
            "seed": args.seed,
            "num_epoch": args.num_epoch,
            "learning_rate": 1e-5,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "total_size": len(dataset)
        }
    )
    # ===================

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    kwargs = {
        'max_memory': {0: '80GiB'},
        # if have 2 A100
        # 'max_memory': {0: '80GiB', 1: '80GiB'}
        'device_map': "auto",
    }


    # ===================================================================
    model = GraphMistralForCausalLM.from_pretrained(args.llm, **kwargs)
    # ===================================================================



    ft_model = get_peft_model(model, peft_config)
    ft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1e-5,
        per_device_eval_batch_size=2, # 1 for multimedia / dailylife
        per_device_train_batch_size=2, # 1 for multimedia / dailylife
        num_train_epochs=args.num_epoch,
        weight_decay=0.01,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=200,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",  
        logging_steps=50, 
    )

    # Use GraphTrainer to properly handle token_type_ids and graph_attention_mask
    trainer = GraphTrainer(
        model=ft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # trainer.train()
    # if the training have error
    # try `pip install markupsafe==2.0.1`
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
