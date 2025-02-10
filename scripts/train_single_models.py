from typing import List, Optional
import warnings
from datetime import datetime

import numpy as np
import yaml
import torch
from torch.utils.data import Subset
from pytorch_lightning.loggers import CometLogger #, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer 
from src.datasets.datasets import CPDDatasets
from src.metrics.evaluation_pipelines import evaluation_pipeline
from src.metrics.metrics_utils import write_metrics_to_file
from src.models.model_utils import get_models_list
from src.utils.fix_seeds import fix_seeds

warnings.filterwarnings("ignore")

def train_single_models(
    model_type: str,
    experiments_name: str,
    loss_type: Optional[str] = None,
    ens_num_list: List[int] = [1, 2, 3],
    n_models: int = 10,
    verbose: bool = True,
    evaluate: bool = True,
    save_models: bool = True,
    boot_sample_size: Optional[int] = None,
):

    # read config file
    if experiments_name in ["human_activity", "yahoo"]:
        path_to_config = "configs/" + experiments_name + "_" + model_type + ".yaml"
        device = "cpu"
    elif experiments_name in ["explosion", "road_accidents"]:
        if model_type == "seq2seq":
            path_to_config = "configs/" + "video" + "_" + model_type + "_" + loss_type + ".yaml"
        else:
             path_to_config = "configs/" + "video" + "_" + model_type + ".yaml"
        device = "cuda"
    else:
        raise ValueError(f"Unknown experiments name {experiments_name}")
    
    with open(path_to_config, "r") as f:
        args_config = yaml.safe_load(f.read())
        
    if experiments_name == "yahoo" and model_type == "seq2seq":
        test_seq_len = 150
    else:
        test_seq_len = None

    args_config["experiments_name"] = experiments_name
    args_config["model_type"] = model_type

    args_config["num_workers"] = 2
    args_config["loss_type"] = loss_type

    # prepare datasets
    train_dataset, test_dataset = CPDDatasets(
        experiments_name=experiments_name, test_seq_len=test_seq_len
    ).get_dataset_()
    
    torch.manual_seed(42)

    for ens_num in ens_num_list:
        if boot_sample_size is None:
            train_datasets_list = [train_dataset] * n_models

        else:
            train_datasets_list = []
            for _ in range(n_models):
                # sample with replacement
                idxs = torch.randint(
                    len(train_dataset), size=(boot_sample_size,)
                )
                curr_train_data = Subset(train_dataset, idxs)
                train_datasets_list.append(curr_train_data)
            
        for s in range(n_models):
            seed = s + 10 * (ens_num - 1)
            
            curr_train_dataset = train_datasets_list[s] 

            fix_seeds(seed)
            model = get_models_list(args_config, curr_train_dataset, test_dataset)[-1]

            if model_type == "seq2seq":
                model_name = (
                    args_config["experiments_name"]
                    + "_"
                    + args_config["loss_type"]
                    + "_boot_sample_size_"
                    + str(boot_sample_size)
                    + "_model_num_"
                    + str(seed)
                )
            elif model_type in ["tscp", "ts2vec"]:
                model_name = (
                    args_config["experiments_name"]
                    + "_"
                    + args_config["model_type"]
                    + "_boot_sample_size_"
                    + boot_sample_size
                    + "_model_num_"
                    + str(seed)
                ) 

            # logger = TensorBoardLogger(
            #     save_dir=f'logs/{args_config["experiments_name"]}',
            #     name=f'{args_config["model_type"]}_{args_config["loss_type"]}',
            # )

            logger = CometLogger(
                save_dir=f"logs/{experiments_name}",
                api_key="agnHNC2vEt7tOxnnxT4LzYf7Y",
                project_name="indid",
                workspace="stalex2902",
                experiment_name=model_name,
            )
        
            trainer = Trainer(
                max_epochs=args_config["learning"]["epochs"],
                accelerator=device,
                devices=1,
                benchmark=True,
                check_val_every_n_epoch=1,
                gradient_clip_val=args_config["learning"]["grad_clip"],
                logger=logger,
                callbacks=EarlyStopping(**args_config["early_stopping"]),
            )

            trainer.fit(model)
            
            if model_type == "seq2seq":
                step, alpha = None, None
                path_to_models_folder = f"saved_models/{loss_type}/{experiments_name}/ens_{ens_num}"
                path_to_metrics = f"results/{experiments_name}/{args_config['loss_type']}/single_model_results.txt"
            
            elif model_type in ["tscp", "ts2vec"]:
                step, alpha = args_config["predictions"].values()
                path_to_models_folder = f"saved_models/{model_type}/{experiments_name}/ens_{ens_num}"
                path_to_metrics = f"results/{experiments_name}/{args_config['model_type']}/single_model_results.txt"

            if save_models:
                torch.save(
                    model.state_dict(), f"{path_to_models_folder}/{model_name}.pth",
                )

            if evaluate:   
                model.load_state_dict(
                    torch.load(f"{path_to_models_folder}/{model_name}.pth")
                )
                model.eval()

                threshold_number = 300
                threshold_list = np.linspace(-5, 5, threshold_number)
                threshold_list = 1 / (1 + np.exp(-threshold_list))
                threshold_list = [-0.001] + list(threshold_list) + [1.001]

                all_metrics = evaluation_pipeline(
                    model,
                    model.val_dataloader(),
                    threshold_list,
                    device=device,
                    model_type=model_type,
                    step=step, 
                    alpha=alpha,
                    verbose=verbose,
                    margin_list=args_config["evaluation"]["margin_list"],
                )
                
                write_metrics_to_file(
                    filename=path_to_metrics,
                    metrics=all_metrics,
                    seed=seed,
                    timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
                    comment=f"{experiments_name}, {args_config['model_type']}, {args_config['loss_type']}",
                )
