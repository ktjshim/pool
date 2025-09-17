import os 
import wandb
from tqdm import tqdm 
import torch 
import json 
import numpy as np
from plan_dataset_kshot import TaskPlanningDatasetKshot
from plan_dataset import TaskPlanningDataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from glm_direct import GraphTokenDirect
# from glm_graph import GraphTokenGraph
# from glm_node import GraphTokenNode
# from glm_diffpool import GraphTokenPool
import argparse
from ckpt import save_checkpoint, reload_best_model
import sys
sys.path.append("../")
from utils import init_random_state, load_tool, get_cur_time
from torch_geometric.data import Data
from torch_geometric.utils import degree
from datetime import datetime

# 속도를 위해서 FP32 -> TF32
# torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="huggingface")
    parser.add_argument("--llm", type=str, default="Mistral-7B")
    parser.add_argument("--llm_model_path", type=str, default="")
    parser.add_argument("--seed", default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--max_txt_length", type=int, default=512)
    parser.add_argument("--max_ans_length", type=int, default=512)

    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_output_dim", type=int, default=4096) # Mistral-7B 4096, CodeLlama-13B 5120 #gpt-oss-20b 2880
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--gnn_type", type=str, default="SAGE", choices=["GCN", "SAGE", "GIN", "GAT", "TransformerConv"])
    parser.add_argument("--max_degree", type=int, default=300),
    parser.add_argument("--max_nodes_per_graph", type=int, default=23) # huggingface: 23, ultratool: 260
    parser.add_argument("--experiment", type=str, default="diffpool")


    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--grad_steps", type=int, default=4)
    parser.add_argument("--name", type=str, default="direct")

    args = parser.parse_args()
    
    today = datetime.now().strftime("%m%d")
    wandb.init(project="GraphToken", name=f"{args.dataset}_{args.llm}_{args.name}_{today}")
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)

    print(args, "\n")
    device = torch.device(args.device)
    init_random_state(args.seed)

    path_mapping = {
        "CodeLlama-13B": "codellama/CodeLlama-13b-Instruct-hf",
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "CodeLlama-7B": "codellama/CodeLlama-7b-Instruct-hf",
        "Vicuna-13B": "lmsys/vicuna-13b-v1.5",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "gemma-3-27b-it": "google/gemma-3-27b-it",
        "gemma-3-12b-it": "google/gemma-3-12b-it"
    }

    gnn_hidden_mapping = {"CodeLlama-13B": 5120, "Mistral-7B": 4096, "Vicuna-13B": 5120, "CodeLlama-7B": 4096, "gpt-oss-20b": 2880, "gemma-3-4b-it": 2560, "gemma-3-27b-it": 5376, "gemma-3-12b-it": 3840}
    
    args.llm_model_path = path_mapping[args.llm]
    args.gnn_output_dim = gnn_hidden_mapping[args.llm]
    
    plan_dataset = TaskPlanningDataset(args.dataset)
    train_ids, test_ids = plan_dataset.idxes_split["train"], plan_dataset.idxes_split["test"]
    train_dataset = [plan_dataset[i] for i in train_ids[: int(0.8 * len(train_ids))]]
    eval_dataset = [plan_dataset[i] for i in train_ids[int(0.8 * len(train_ids)): ]]
    test_dataset = [plan_dataset[i] for i in test_ids]
    print(f"# Train {len(train_dataset)}   # Val {len(eval_dataset)}   # Test {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)

    # 속도 빨라지나 테스트
    model = torch.compile(model = GraphTokenDirect(args))
    
    
    # 학습 중 결과 확인
    os.makedirs(f"prediction_train/{args.dataset}/{args.llm}_{args.name}_{today}", exist_ok=True)
    train_process = f"prediction_train/{args.dataset}/{args.llm}_{args.name}_{today}/GraphToken_{args.gnn_type}.json"
    
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )

    trainable_params, all_params = model.print_trainable_params()
    print(f"Trainable params {trainable_params} || all params {all_params} || trainable% {100 * trainable_params / all_params:.5f}")
    
    tool_texts, tool2index, index2tool, edge_index, _, adj_g = load_tool(dataset_name=args.dataset)

    # # each `process/{dataset}.npy` stores a processed task embedding matrix via e5-355M
    task_graph = Data(x=torch.FloatTensor(np.load(f"process/{args.dataset}.npy")), edge_index=edge_index).to(device)
    
    
    # add centrality
    # node num 추가해야됨
    out_degree = degree(task_graph.edge_index[0], dtype=torch.long).to(device)
    in_degree = degree(task_graph.edge_index[1], dtype=torch.long).to(device)
    task_graph.out_degree = out_degree
    task_graph.in_degree = in_degree
    

    
    

    # num_training_steps = args.num_epochs * len(train_loader)
    # progress_bar = tqdm(range(num_training_steps))

    # best_val_loss = float('inf')
    
    # model.model.gradient_checkpointing_enable() 
    # for epoch in range(args.num_epochs):
        
    #     epoch_loss, accum_loss = 0.0, 0.0 

    #     for step, batch in enumerate(train_loader):
    #         model.train()
            
    #         optimizer.zero_grad()

    #         loss, logits = model(batch, task_graph)
    #         loss.backward()

    #         clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

    #         optimizer.step()
    #         epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

    #         if (step + 1) % args.grad_steps == 0:
    #             lr = optimizer.param_groups[0]["lr"]
    #             wandb.log({'Accum Loss': accum_loss / args.grad_steps})
    #             wandb.log({'Step Loss': loss.item()})
    #             accum_loss = 0.

    #         progress_bar.update(1)
            
    #         if (step + 1) % 200 == 0:
    #             print(f"\n--- Running Inference Check at Step {step + 1} ---")
                
    #             model.eval()
                
    #             # 결과 저장
    #             with torch.no_grad():
    #                 id_list, predictions, requests = model.inference(batch, task_graph)
    #                 with open(train_process, 'a') as train_file:
    #                     train_file.write(json.dumps(predictions) + "\n")
                        
    #             # import code; code.interact(local=locals()) # debug here
                
    #     print(f"Epoch {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    #     wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

    #     val_loss = 0. 
    #     model.eval()

    #     with torch.no_grad():
    #         for step, batch in enumerate(val_loader):
    #             loss, logits = model(batch, task_graph)
    #             val_loss += loss.item()
                
                
    #         val_loss = val_loss / len(val_loader)
    #         wandb.log({'Val Loss (Epoch Mean)': val_loss})
    #         print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
        
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss 
    #         best_epoch = epoch 
    #         # save_checkpoint(model, optimizer, epoch, args, is_best=False)
    #         save_checkpoint(model, optimizer, epoch, args, is_best=True)
        
    #     if epoch - best_epoch >= args.patience:
    #         print(f"Early stop at epoch {epoch}")
    #         break 

    # torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()

    os.makedirs(f"prediction/{args.dataset}/{args.llm}_{args.name}_{today}", exist_ok=True)
    path = f"prediction/{args.dataset}/{args.llm}_{args.name}_{today}/GraphToken_{args.gnn_type}.json"
    
    
    os.makedirs(f"prediction_errors/{args.dataset}/{args.llm}_{args.name}_{today}", exist_ok=True)
    error_path = f"prediction_errors/{args.dataset}/{args.llm}_{args.name}_{today}/GraphToken_{args.gnn_type}.json"
    

    # model = reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, 'w') as file:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                id_list, predictions, requests = model.inference(batch, task_graph)

                for sample_id, pred, req in zip(id_list, predictions, requests):
                    idx = sample_id.item()
                    origin_id = plan_dataset.id_mapping[idx]
                    try:
                        pred = json.loads(pred)
                        write_obj = {
                            "id": origin_id,
                            "user_request": plan_dataset.raw_data_dictionary[origin_id]["request"],
                            "task_steps": pred["task_steps"],
                            "task_nodes": [{"task": node} for node in pred["task_nodes"]],
                            "task_links": pred["task_links"]
                        }
                        file.write(json.dumps(write_obj) + "\n")
                        file.flush()
                    except Exception as e:
                        print(f"[Test Error] Test-ID {plan_dataset.id_mapping[idx]} e pred")
                        with open(error_path, 'a') as error_file:
                            error_file.write(json.dumps(pred) + "\n")
                            
                progress_bar_test.update(1)
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
