import numpy as np 
import json 
import prettytable as pt
import os
import sys

sys.path.append("../")

def reformat_task_nodes(content):
    raw_nodes = content.get("task_nodes", [])
    nodes = [node["task"] for node in raw_nodes]

    if len(nodes) > 0 and not isinstance(nodes[0], str):
        nodes = [node.get("name", "") for node in nodes]
    return nodes


def reformat_task_links(content):
    """
    Corrected function to safely handle malformed links.
    It filters for links that are dictionaries and contain both 'source' and 'target' keys
    before attempting to format them, avoiding KeyErrors.
    """
    raw_links = content.get("task_links", [])
    links = [
        f"{link['source']}, {link['target']}"
        for link in raw_links
        if isinstance(link, dict) and "source" in link and "target" in link
    ]
    return links


def f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    
    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f 


def batch_f1_score(pred_list, gt_list):
    """
    Added a check to prevent a warning when calculating the mean of an empty list.
    """
    f1_score_list = [f1_score(pred, gt) for pred, gt in zip(pred_list, gt_list)]
    if not f1_score_list:
        return 0.0
    return round(np.mean(np.array(f1_score_list)), 4)


def batch_task_succ(pred_list, gt_list):
    """
    Corrected function to prevent ZeroDivisionError by checking if the scores list is empty.
    """
    scores = [float(f1_score(pred, gt) >= 0.99) for pred, gt in zip(pred_list, gt_list)]
    if not scores:
        return 0.0
    succ_rate = round(sum(scores) / len(scores) * 100, 2)
    return succ_rate


def node_hallucination_rate(solution, valid_tools):
    if len(solution) == 0:
        return [0.0, 0.0]
    
    hall_list = [1.0 if node not in valid_tools else 0.0 for node in solution]
    micro_hall = sum(hall_list) / len(solution)
    macro_hall = 1.0 if sum(hall_list) >= 1 else 0.0

    return [micro_hall, macro_hall]


def batch_node_hallucination(solutions, valid_tools):
    """
    Added a check to handle cases with no solutions to evaluate.
    """
    if not solutions:
        return np.array([0.0, 0.0])
    hall_scores = [node_hallucination_rate(sol, valid_tools) for sol in solutions]
    avg_score = np.round(np.mean(np.array(hall_scores), axis=0), 4)
    return avg_score


def prediction_loader(filename, content_type):
    return_data = {}
    with open(filename, 'r') as readfile:
        for line in readfile:
            try:
                data = json.loads(line)
                data_id = data["id"]

                if content_type == 'id':
                    retrieve_data = data_id
                elif content_type == "graph":
                    nodes, links = reformat_task_nodes(data), reformat_task_links(data)
                    retrieve_data = {"nodes": nodes, "links": links}
                return_data[data_id] = retrieve_data
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line in {filename}: {line.strip()} - Error: {e}")
    
    return return_data


def evaluate(dataset, llm_name, method):
    alignment = json.load(open(f"../data/{dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    
    gt_filename = f"../data/{dataset}/data.json"
    pred_filename = f"prediction/{dataset}/{llm_name}/{method}.json"

    if not os.path.exists(pred_filename):
        print(f"Prediction file not found, skipping: {pred_filename}")
        return 

    tool_desc_data = json.load(open(f"../data/{dataset}/tool_desc.json", 'r'))
    if isinstance(tool_desc_data, dict) and "nodes" in tool_desc_data:
        gt_tool_nodes = [tool["id"] for tool in tool_desc_data["nodes"]]
    else: # Handles list format like in ultratool
        gt_tool_nodes = [tool["id"] for tool in tool_desc_data]

    graph_desc_data = json.load(open(f"../data/{dataset}/graph_desc.json", 'r'))
    # Handles both dict with "links" and list of links format
    links_list = graph_desc_data.get("links", []) if isinstance(graph_desc_data, dict) else graph_desc_data
    gt_tool_links = [", ".join([link["source"], link["target"]]) for link in links_list]
    
    gt_graph_dict = prediction_loader(gt_filename, content_type="graph")
    pred_graph_dict = prediction_loader(pred_filename, content_type="graph")

    pred_align = prediction_loader(pred_filename, "id")
    alignment_ids = [data_id for data_id in alignment if data_id in pred_align]

    print(f"Evaluating {pred_filename} | Found {len(alignment_ids)} aligned predictions.")
    
    if not alignment_ids:
        # If there are no valid predictions, report zeros and avoid calculations
        table.add_row([dataset, llm_name, method, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return

    pred_graphs = [pred_graph_dict.get(data_id, {"nodes": [], "links": []}) for data_id in alignment_ids]
    gt_graphs = [gt_graph_dict[data_id] for data_id in alignment_ids]

    node_f1 = batch_f1_score([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
    link_f1 = batch_f1_score([pred_g["links"] for pred_g in pred_graphs], [gt_g["links"] for gt_g in gt_graphs]) 

    node_hr = batch_node_hallucination([pred_g["nodes"] for pred_g in pred_graphs], gt_tool_nodes)
    link_hr = batch_node_hallucination([pred_g["links"] for pred_g in pred_graphs], gt_tool_links)
    succ_rate = batch_task_succ([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
            
    table.add_row([dataset, llm_name, method, node_f1, link_f1, succ_rate, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])


if __name__ == "__main__":
    table = pt.PrettyTable()
    table.field_names = ['Dataset', 'LLM', 'Method', 'NF', 'LF', "Succ", 'NH-1', 'NH-2', 'LH-1', 'LH-2']
    
    prediction_dir = "prediction"
    llms = [name for name in os.listdir(prediction_dir) 
                if os.path.isdir(os.path.join(prediction_dir, name))]

    for dataset in os.listdir("prediction"):
        for llm in os.listdir(f"prediction/{dataset}"):
            for gnn_type in ["SAGE"]:
                method_name = f"GraphToken_{gnn_type}"
                evaluate(dataset, llm, method_name)
     
    print(table)
    
    csv_string = table.get_csv_string()
    with open("evaluation_results.csv", "w", newline="") as f_output:
        f_output.write(csv_string)