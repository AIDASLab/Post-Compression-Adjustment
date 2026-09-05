# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import os
import time
from typing import Optional

import logging
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from hcsmoe.evaluation.minipile import get_calib_dataloder
from hcsmoe.merging.grouping_qwen import (
    ExpertsGrouperForQwen2MoE,
    merge_by_groups_with_usage_weighted,
    merge_by_groups_within_and_across_models,
)
from hcsmoe.models.qwen.compression import (
    compress_physical_experts_,
    save_compressed_qwen3_model,
)

logger = logging.getLogger(__name__)

###############################
#  Helper class and functions #
###############################
class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "zipit",
        mode: Optional[str] = "normal",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        try_oracle: Optional[bool] = False,
        random_start_center: Optional[bool] = False,
        weight: Optional[str] = None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        overlap_metric: Optional[str] = "kl-divergence",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.try_oracle = try_oracle
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=0,
    )

def get_grouper(args, config):
    return ExpertsGrouperForQwen2MoE(
                config=config,
                similarity_base=args.similarity_base,
                start_layer=args.start_layer,
                group_limit=args.group_limit,
                data_limit=args.data_limit,
                random_start_center=args.random_start_center,
                cluster=args.cluster,
                linkage=args.linkage,
                hierarchical_stopping_metric=args.hierarchical_stopping_metric,
                overlap_metric=args.overlap_metric,
                dynamic_group=args.dynamic_group,
            )

def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()


def run_hcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        dominant: Optional[str] = "knowledge", # random, frequency, knowledge
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, update, fix-dom, unmerge,ix-dom-same
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 1000000,
        random_start_center: Optional[bool] = False,
        num_fewshot: Optional[int] = 0,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        ingredient: Optional[str] = "act", # act, weight, act+weight
        overlap_metric: Optional[str] = "cosine", # kl-divergence, wasserstein, cosine
        dynamic_group: Optional[bool] = False,
        save_full_duplicate_experts: Optional[bool] = False,
        max_shard_size: Optional[str] = "5GB",
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} merge - {mode}, ingredient: {ingredient}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}, hierarchical_stopping_metric: {hierarchical_stopping_metric}, overlap_metric: {overlap_metric}, dynamic_group: {dynamic_group}")
    
    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        hierarchical_stopping_metric=hierarchical_stopping_metric,
        overlap_metric=overlap_metric,
        dynamic_group=dynamic_group,
    )
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader_for_merging = get_dataloader(args, tokenizer)
    grouper = get_grouper(args, model.config)

    # HC-SMoE!
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters())
    print(f"[HC-SMoE] Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge == "freq" or dominant == "frequency":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    if dynamic_group:
        grouper.compute_all_usages(model, dataloader_for_merging, mode=hierarchical_stopping_metric)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    

    ### 2. Get dominant experts
    dom_experts = None
    if dominant == "random":
        grouper.group_experts_randomly(num_groups=num_average_groups)
        dom_experts = None
    elif dominant == "frequency":
        if similarity_base != "no":
            grouper.compute_all_similarities(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "routing-score":
        grouper.compute_all_usages(model, dataloader_for_merging, mode="routing-score")
        print_usage_frequency(grouper._usage_frequency_state_dict)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "no":
        ### Clustering
        dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
    else:
        raise ValueError(f"Unknown dominant: {dominant}")   

    ### 3. Merging
    if merge == "freq":
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    else:
        model = merge_by_groups_within_and_across_models(
            qwen_model=model,
            grouper=grouper,
            dataloader=dataloader_for_merging,
            merge=merge,
            mode=mode,
            partition=partition,
            core_experts=dom_experts,
            dominant_alone=False,
            usage_weighted=False,
            ingredient=ingredient,
        )
        
    print(f"[HC-SMoE] Merging time: {time.time() - group_st:.2f} seconds")

    ### 4. Grouping results
    print(f"[HC-SMoE] ========= Grouping results ========= ")
    group_state_dict = grouper.group_state_dict()
    for name, state in group_state_dict.items():
        if dom_experts is None:
            print(f"Group {name}: {state.tolist()}")
        else:
            print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]}, {len(dom_experts[name])})")

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "outputs", "qwen3-30b-a3b-hcsmoe")

    if save_full_duplicate_experts:
        raise ValueError("save_full_duplicate_experts is disabled for Qwen3 compression runs.")
    else:
        mapping_metadata = compress_physical_experts_(
            model=model,
            group_state_dict=group_state_dict,
            expected_physical_num_experts=num_average_groups,
            base_model_name=model_name,
        )
        print("[HC-SMoE] Number of parameters after physical compression:", model.num_parameters())
        save_compressed_qwen3_model(
            model=model,
            tokenizer=tokenizer,
            output_path=output_path,
            metadata=mapping_metadata,
            max_shard_size=max_shard_size,
        )
        torch.save(group_state_dict, os.path.join(output_path, "group_state_dict.pt"))
        print(f"[HC-SMoE] Saved compressed model and expert mapping to {output_path}")

    del grouper

    ### 6. Evaluation
    # Evaluation is intentionally disabled for merge-only compression runs.


if __name__ == "__main__":
    Fire(run_hcsmoe)
