vocab_size = {
    'starqe_2000':47688, 
    'starqe_100':47688, 
    'starqe': 47688,
    'wd50k_json': 46335,
    "new_wd50k_json": 47688,
    "fb15k_237_json": 14981,
    'wd50k_nfol': 46335,
    }
num_relations = {
    'starqe_2000':531,     
    'starqe_100':531, 
    'starqe': 531,
    "wd50k_json": 517,
    "new_wd50k_json": 531,
    "fb15k_237_json": 474,
    "wd50k_nfol": 517,
    }
max_seq_len = {
    'starqe_2000':7, 
    'starqe_100':7, 
    'starqe':7,
    "wd50k_json": 19,
    "new_wd50k_json": 19,
    "fb15k_237_json": 7,
    "wd50k_nfol": 19,
    }
max_arity = {
    'starqe_2000':4,     
    'starqe_100':4, 
    'starqe': 4,
    "wd50k_json": 10,
    "new_wd50k_json": 10,
    "fb15k_237_json": 4,
    "wd50k_nfol": 10,
    }



def process_qe_dataset(dataset_name, config):
    dataset_name = dataset_name.lower()
    config["dataset_dir"] = "./data/" + dataset_name + "/tasks"
    config["validation_gt_dir"] = "./data/" + dataset_name + config["relative_validation_gt_dir"]
    config["test_gt_dir"] = "./data/" + dataset_name + config["relative_test_gt_dir"]
    config["vocab_path"] = "./data/" + dataset_name + "/vocab.txt"
    config["vocab_size"] = vocab_size[dataset_name]
    config["num_relations"] = num_relations[dataset_name]
    config["max_seq_len"] = max_seq_len[dataset_name]
    config["max_arity"] = max_arity[dataset_name]

    return config