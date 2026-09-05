# -*- modified by wazenmai -*-
# @Time: 2024/07/03

# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

import numpy as np

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
# from lm_eval.tasks import initialize_tasks
from lm_eval.utils import make_table

TASK_TO_NUM_FEWSHOT = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa": 0,
    "mmlu": 5,
    "winogrande": 5,
    "gsm8k": 5
}


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def evaluate_fewshot(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        num_fewshot: int,
        eval_batch_size: Optional[int] = 4,
        log: Optional[bool] = True,
        output_path: Optional[str] = None,
):
    # initialize_tasks(verbosity="WARNING")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        device_map="auto"
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task,
        num_fewshot=num_fewshot,
        batch_size=eval_batch_size,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
    )

    if log:
        print(make_table(results))
        
        if "groups" in results:
            print(make_table(results, "groups"))
    
    if output_path:
        f = open(output_path, "a")
        print(make_table(results), file=f)
        if "groups" in results:
            print(make_table(results, "groups"), file=f)
        f.close() 
        


    return results
