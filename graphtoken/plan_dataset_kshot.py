import json 
from torch.utils.data import Dataset
import numpy as np
import sys
sys.path.append("../")
from utils import prepare_training_ids, init_random_state
from transformers import AutoTokenizer


class TaskPlanningDatasetKshot(Dataset):
    def __init__(self, dataset_name, k_shot=1):
        super().__init__()
        
        self.dataset = dataset_name
        self.k_shot = k_shot
        self.raw_data_dictionary = self.load_raw_data()
        self.idxes_split = self.get_idx_split()
        self.prepare_prompt()

    def prepare_prompt(self):
        
        # 1. Prepare the tool list string
        tool_list = json.load(open(f"../data/{self.dataset}/tool_desc.json", "r"))["nodes"]
        tool_string = "# TASK LIST #:\n"
        for k, tool in enumerate(tool_list):
            tool_string += json.dumps(tool) + "\n"
        

        # 2. Prepare the k-shot example string (demo_string)
        demo_string = ""
        if self.k_shot > 0:
            # These are the hardcoded demo IDs from your inference script

            demos_id_list = {
                "huggingface": ["10523150", "14611002", "22067492"],
                "multimedia": ["30934207", "20566230", "19003517"],
                "dailylife": ["27267145", "91005535", "38563456"],
                "tmdb": [1],
                "ultratool": ["1355", "1307", "2311"]
            }
            
            # Select the first k demo IDs
            demo_ids_to_use = demos_id_list.get(self.dataset, [])[:self.k_shot]
            
            if demo_ids_to_use:
                for demo_id in demo_ids_to_use:
                    # Check if the demo data exists in our loaded raw data
                    if demo_id in self.raw_data_dictionary:
                        demo_data = self.raw_data_dictionary[demo_id]
                        user_request = demo_data["request"]
                        result = json.dumps(demo_data["label"])
                        demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #: {user_request}\n# RESULT #: {result}"""
        
        # 3. Define the main prompt structure
        main_prompt = ("""\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n""" 
        + """\n# REQUIREMENT #:\n""" 
        + """1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n""" 
        + """2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes;\n""" \
        + """3. the dependencies among task steps should align with the argument dependencies of the task nodes;\n""" 
        + """4. the tool arguments should be align with the input-type field of # TASK LIST #;\n""" 
        + """\nHere are provided examples for your reference.\n""")
        
        # 4. Combine all parts: tool list + main goal + demos + user request placeholder
        self.prompt = (tool_string 
                       + main_prompt 
                       + demo_string 
                       + """\n\n# USER REQUEST #: {{user_request}}\nNow please generate your result in a strict JSON format:\n# RESULT #:""")


    def __len__(self):
        return len(self.id_mapping)
    
    def __getitem__(self, index):
        origin_id = self.id_mapping[index]
        origin_data = self.raw_data_dictionary[origin_id]
        
        # The prompt template now includes the k-shot examples
        cur_request = self.prompt.replace("{{user_request}}", origin_data["request"])
        return {
            "id": index, 
            "origin_id": origin_id,
            "request": cur_request,
            "label": json.dumps(origin_data["label"])
        }

    def load_raw_data(self):
        data_file = f"../data/{self.dataset}/data.json"
        data_dict = {}
        for line in open(data_file, 'r'):
            content = json.loads(line)
            data_dict[content["id"]] = {
                "id": content["id"], # origin ID
                "request": content["user_request"],
                "label": {
                    "task_steps": content["task_steps"],
                    "task_nodes": [node["task"] for node in content["task_nodes"]],
                    "task_links": content["task_links"],
                }
            }
        return data_dict

    def get_idx_split(self):
        split_id_file = f"../data/{self.dataset}/split_ids.json"
        test_ids = json.load(open(split_id_file, 'r'))["test_ids"]["chain"]

        # IMPORTANT: Exclude demo IDs from the training set to prevent data leakage
        demos_id_list = {
            "huggingface": ["10523150", "14611002", "22067492"],
            "multimedia": ["30934207", "20566230", "19003517"],
            "dailylife": ["27267145", "91005535", "38563456"],
            "tmdb": [1],
            "ultratool": ["1355", "1307", "2311"]
        }
        demo_ids_to_exclude = demos_id_list.get(self.dataset, [])
        
        # Filter out demo ids from the potential training pool
        full_id_pool = list(self.raw_data_dictionary.keys())
        potential_train_ids = [id for id in full_id_pool if id not in test_ids and id not in demo_ids_to_exclude]

        # Now sample from the filtered list
        train_num = min(3000, len(potential_train_ids))
        train_ids = np.random.choice(potential_train_ids, train_num, replace=False).tolist()
        
        assert len(list(set(train_ids) & set(test_ids))) == 0
        assert len(list(set(train_ids) & set(demo_ids_to_exclude))) == 0

        full_ids = train_ids + test_ids 
        self.id_mapping = {idx: origin_id for idx, origin_id in enumerate(full_ids)}
        self.reverse_id_mapping = {origin_id: idx for idx, origin_id in enumerate(full_ids)}

        formatted_train_ids = [self.reverse_id_mapping[origin_id] for origin_id in train_ids]
        formatted_test_ids = [self.reverse_id_mapping[origin_id] for origin_id in test_ids]
        
        print(f"[Data Split] # Train {len(formatted_train_ids)}  # Test {len(formatted_test_ids)}")
        print(f"[Prompting] Using {self.k_shot}-shot examples in prompt.")

        return {
            "train": formatted_train_ids,
            "val": [],
            "test": formatted_test_ids
        }