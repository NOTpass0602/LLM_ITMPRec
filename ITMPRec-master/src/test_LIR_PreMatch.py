# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from llm_itmprec.complex_feedback import ComplexFeedbackSimulator
from llm_itmprec.config import LIRConfig
from llm_itmprec.cot_bridge_planner import CachedCoTBridgePlanner
from llm_itmprec.profile_store import ProfileStore
from llm_itmprec.replanner import FeedbackAwareReplanner
from llm_itmprec.runner import LIRReranker
from simulators_original import RecSim
from test_models import SASRecModel
from utils import check_path, data_partition, data_partition2, data_partition3, get_user_seqs, set_seed


def show_args_info(args):
    print("--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {str(getattr(args, arg)):>35}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--data_name", default="ml-1m", type=str)
    parser.add_argument("--model_idx", default=0, type=int)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--n_prematch", default=200, type=int)
    parser.add_argument("--n_items", default=50, type=int)
    parser.add_argument("--num_users", default=945, type=int)
    parser.add_argument("--num_items", default=2782, type=int)
    parser.add_argument("--env_omega", default=0.8, type=float)
    parser.add_argument("--env_offset", default=0.8, type=float)
    parser.add_argument("--env_slope", default=10, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--model_scheme", default="proact", type=str)
    parser.add_argument("--target_mode", default="PreMatch", type=str)

    parser.add_argument("--noise_ratio", default=0.0, type=float)
    parser.add_argument("--training_data_ratio", default=1.0, type=float)
    parser.add_argument("--augment_type", default="random", type=str)
    parser.add_argument("--tao", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--contrast_type", default="Hybrid", type=str)
    parser.add_argument("--num_intent_clusters", "--num_intent_cluster", default="256", type=str)
    parser.add_argument("--seq_representation_type", default="mean", type=str)
    parser.add_argument("--seq_representation_instancecl_type", default="concatenate", type=str)
    parser.add_argument("--warm_up_epoches", type=float, default=0)
    parser.add_argument("--de_noise", action="store_true")
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--intent_cf_weight", type=float, default=0.1)
    parser.add_argument("--cf_weight", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--user_emb_path", type=str, default=None)
    parser.add_argument("--item_emb_path", type=str, default=None)

    parser.add_argument("--lir_ablation", default="full", choices=["baseline", "profiles", "cot", "feedback", "replan", "full"])
    parser.add_argument("--lir_candidate_k", default=50, type=int)
    parser.add_argument("--lir_profile_path", default=None, type=str)
    parser.add_argument("--lir_cot_cache_path", default=None, type=str)
    parser.add_argument("--lir_profile_embedding_path", default=None, type=str)
    parser.add_argument("--lir_alpha", default=0.20, type=float)
    parser.add_argument("--lir_beta", default=0.20, type=float)
    parser.add_argument("--lir_gamma", default=0.20, type=float)
    parser.add_argument("--lir_delta", default=0.30, type=float)
    parser.add_argument("--lir_cot_min_transition", default=0.50, type=float)
    parser.add_argument("--lir_cot_max_risk", default=0.50, type=float)
    parser.add_argument("--lir_disable_cot_bonus", action="store_true")
    parser.add_argument("--lir_cot_use_risk_penalty", action="store_true")
    parser.add_argument("--lir_cot_replan_filter", action="store_true")
    parser.add_argument("--lir_cot_replan_risk_min", default=0.75, type=float)
    parser.add_argument("--lir_cot_replan_transition_max", default=0.25, type=float)
    parser.add_argument("--lir_cot_replan_min_rejects", default=1, type=int)
    parser.add_argument("--lir_cot_replan_margin", default=0.0, type=float)
    parser.add_argument("--lir_cot_replan_proactive", action="store_true")
    parser.add_argument("--lir_dump_bridge_pairs_path", default=None, type=str)
    parser.add_argument("--lir_dump_pairs_per_target", default=50, type=int)
    return parser.parse_args()


def build_lir(args):
    use_profiles = args.lir_ablation in {"profiles", "cot", "full"}
    use_cot = args.lir_ablation in {"cot", "full"}
    use_feedback = args.lir_ablation in {"feedback", "replan", "full"}
    use_replanning = args.lir_ablation in {"replan", "full"}
    config = LIRConfig(
        alpha=args.lir_alpha,
        beta=args.lir_beta,
        gamma=args.lir_gamma,
        delta=args.lir_delta,
        candidate_k=args.lir_candidate_k,
        use_profiles=use_profiles,
        use_cot=use_cot,
        use_feedback=use_feedback,
        use_replanning=use_replanning,
        profile_embedding_path=args.lir_profile_embedding_path,
        cot_min_transition=args.lir_cot_min_transition,
        cot_max_risk=args.lir_cot_max_risk,
        cot_use_bonus=not args.lir_disable_cot_bonus,
        cot_use_risk_penalty=args.lir_cot_use_risk_penalty,
        cot_replan_filter=args.lir_cot_replan_filter,
        cot_replan_risk_min=args.lir_cot_replan_risk_min,
        cot_replan_transition_max=args.lir_cot_replan_transition_max,
        cot_replan_min_rejects=args.lir_cot_replan_min_rejects,
        cot_replan_margin=args.lir_cot_replan_margin,
        cot_replan_proactive=args.lir_cot_replan_proactive,
    )
    profile_path = args.lir_profile_path or os.path.join(args.data_dir, "lir_profiles.jsonl")
    cot_cache_path = args.lir_cot_cache_path or os.path.join(args.data_dir, "cot_bridge_cache.jsonl")
    profile_store = ProfileStore(profile_path)
    planner = CachedCoTBridgePlanner(cot_cache_path)
    replanner = FeedbackAwareReplanner()
    reranker = LIRReranker(config, profile_store, planner, replanner)
    feedback_simulator = ComplexFeedbackSimulator()
    print(f"LIR ablation: {args.lir_ablation}")
    print(f"LIR profiles loaded: {profile_store.describe()}")
    print(f"LIR CoT cache records: {planner.describe()}")
    if args.lir_profile_embedding_path:
        print(f"LIR semantic embedding path: {args.lir_profile_embedding_path}")
    return config, reranker, feedback_simulator, replanner


def load_partition(args):
    input_file_name = os.path.join(args.data_dir, "data.txt")
    if args.data_name in {"steam", "douban_movie"}:
        return data_partition2(input_file_name)
    if args.data_name == "office":
        return data_partition3(input_file_name)
    return data_partition(input_file_name)


def choose_target_items(args, model, user_train, interacted, seq_guidance, pre_target_items=None):
    if args.target_mode == "random":
        all_target_items = list(range(args.item_size))
        random.seed(2024)
        random.shuffle(all_target_items)
        target_items = all_target_items[: args.n_items]
        print(f"target items in random : {target_items}")
        return target_items

    batch_num = args.num_users // args.batch_size + 1
    score_of_target = np.zeros(len(pre_target_items))
    for batch in range(batch_num):
        if batch == batch_num - 1:
            input_seq = seq_guidance[batch * args.batch_size :]
            interacted_input = interacted[batch * args.batch_size :]
        else:
            input_seq = seq_guidance[batch * args.batch_size : (batch + 1) * args.batch_size]
            interacted_input = interacted[batch * args.batch_size : (batch + 1) * args.batch_size]
        input_seq = np.array(copy.deepcopy(input_seq))
        pre_score = model.calc_score_for_prematch(input_seq, interacted_input, pre_target_items).cpu().detach().numpy()
        score_of_target += np.sum(pre_score, axis=0)

    max2min_idx = np.argsort(score_of_target)[::-1][: args.n_items]
    target_items = [pre_target_items[idx] for idx in max2min_idx]
    print(f"target items in pre-match manner: {target_items}")
    return target_items


def main():
    args = parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"
    if not args.user_emb_path:
        args.user_emb_path = os.path.join(args.data_dir, "user_emb-v2-64.pt")
    if not args.item_emb_path:
        args.item_emb_path = os.path.join(args.data_dir, "item_emb-v2-64.pt")

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}"
    args.checkpoint_path = os.path.join(args.output_dir, args_str + ".pt")
    show_args_info(args)

    lir_config, reranker, feedback_simulator, replanner = build_lir(args)
    use_lir = args.lir_ablation != "baseline" and args.model_scheme == "proact"

    random.seed(2021)
    pre_target_items = None
    if args.target_mode == "PreMatch":
        target_file_name = os.path.join(args.data_dir, "target_items.txt")
        pre_target_items = []
        with open(target_file_name, "r", encoding="utf-8") as f:
            for line in f:
                pre_target_items.append(int(line.strip()) - 1)
        random.shuffle(pre_target_items)
        pre_target_items = pre_target_items[: args.n_prematch] if len(pre_target_items) >= args.n_prematch else pre_target_items
        print(f"pre matched target items: {pre_target_items}")

    [user_train, user_valid, user_test, usernum, itemnum] = load_partition(args)
    prematch_model = SASRecModel(args=args).to(args.device)
    prematch_model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    prematch_model.eval()
    prematch_interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()
    prematch_seq_guidance = []
    for u in range(1, args.num_users + 1):
        seq_u = user_train[u]
        prematch_interacted[u - 1][seq_u] = 1
        seq_u = seq_u[-args.max_seq_length :] if len(seq_u) > args.max_seq_length else seq_u + [0] * (args.max_seq_length - len(seq_u))
        prematch_seq_guidance.append(seq_u)
    target_items = choose_target_items(args, prematch_model, user_train, prematch_interacted, prematch_seq_guidance, pre_target_items)

    all_hit_ratios = []
    all_ratings_avg = []
    all_ranking_increase = []
    all_acceptability_avg = []
    all_bridge_avg = []
    feedback_counts = {"click": 0, "skip": 0, "reject": 0}
    reject_recovery_hits = 0
    reject_recovery_total = 0
    bridge_pair_counts = Counter()

    for count, target_item in enumerate(tqdm(target_items)):
        [user_train, user_valid, user_test, usernum, itemnum] = load_partition(args)
        model = SASRecModel(args=args).to(args.device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
        model.eval()

        click_logs_all = torch.zeros(args.num_users, args.episode_length)
        user_sr_all = torch.zeros(args.num_users, args.episode_length)
        env = RecSim(device="cpu", num_users=args.num_users, num_items=args.num_items, env_omega=args.env_omega, env_slope=1, env_offset=args.env_offset, boredom_decay=0.8)
        rand_array = env._dynamics_random.rand(args.n_items)
        env.reset_new()
        env.load_user_item_embeddings(args.user_emb_path, args.item_emb_path)

        ratings_logs = torch.zeros(args.num_users, args.episode_length)
        ranking_logs = torch.zeros(args.num_users, args.episode_length)
        acceptability_logs = torch.zeros(args.num_users, args.episode_length)
        bridge_logs = torch.zeros(args.num_users, args.episode_length)
        interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()
        cluster_cen = model.intention_cluster.weight
        seq_guidance = []
        previous_feedback = [""] * args.num_users

        for u in range(1, args.num_users + 1):
            seq_u = user_train[u]
            if target_item + 1 in seq_u:
                user_sr_all[u - 1] = 1
            interacted[u - 1][seq_u] = 1
            seq_u = seq_u[-args.max_seq_length :] if len(seq_u) > args.max_seq_length else seq_u + [0] * (args.max_seq_length - len(seq_u))
            seq_guidance.append(seq_u)

        for i in range(args.episode_length):
            recommendations = []
            coeffs = []
            meta_by_user = [{"acceptability": 0.0, "bridge": 0.0, "transition": 0.0, "risk": 0.0} for _ in range(args.num_users)]

            for start in range(0, args.num_users, args.batch_size):
                end = min(start + args.batch_size, args.num_users)
                input_seq = np.array(copy.deepcopy(seq_guidance[start:end]))
                interacted_input = interacted[start:end]

                if args.model_scheme == "PIPG":
                    recommendation, coeff = model.next_item_prediction_with_IPG3(input_seq, interacted_input, target_item, k=args.k)
                    recommendations.extend(recommendation.cpu().numpy().tolist())
                    coeffs.extend(np.around(coeff.cpu().numpy(), 3))
                elif args.model_scheme == "Coarse":
                    recommendation = model.next_item_prediction_with_IPG4(input_seq, interacted_input, target_item, cluster_cen, lam=args.lam)
                    recommendations.extend(recommendation.cpu().numpy().tolist())
                elif use_lir:
                    top_items, top_scores, coeff, user_idx = model.topk_prediction_with_ProRec(
                        input_seq,
                        interacted_input,
                        target_item,
                        cluster_cen,
                        lam=args.lam,
                        k=args.k,
                        candidate_k=lir_config.candidate_k,
                    )
                    top_items = top_items.cpu().numpy().tolist()
                    top_scores = top_scores.cpu().detach().numpy().tolist()
                    coeff = np.around(coeff.cpu().numpy(), 3)
                    for row_idx, candidates in enumerate(top_items):
                        user_id = start + row_idx + 1
                        if args.lir_dump_bridge_pairs_path:
                            for candidate in candidates:
                                bridge_pair_counts[(int(target_item) + 1, int(candidate) + 1)] += 1
                        selected, meta = reranker.rerank(user_id, user_train[user_id], target_item, candidates, top_scores[row_idx])
                        recommendations.append(selected)
                        coeffs.append(coeff[row_idx])
                        meta_by_user[user_id - 1] = meta
                else:
                    recommendation, coeff, _ = model.next_item_prediction_with_ProRec(input_seq, interacted_input, target_item, cluster_cen, lam=args.lam, k=args.k)
                    recommendations.extend(recommendation.cpu().numpy().tolist())
                    coeffs.extend(np.around(coeff.cpu().numpy(), 3))

            recommendations = torch.LongTensor(recommendations).cpu()
            recommendations[recommendations < 0] = 0

            r_target = env.get_avg_rating_new(target_item, reduce=False)
            initial_already_clicked_mask = torch.where(user_sr_all[:, 0] == 1, 0, 1)
            user_item_score = torch.matmul(env.user_embedd, env.item_embedd.T)
            idx_list = []
            for uu in range(args.num_users):
                temp_interacted = [x - 1 for x in user_train[uu + 1]]
                user_item_score[uu, temp_interacted] = float("-inf")
                _, candidate_items = torch.topk(user_item_score[uu, :], 200)
                idx_ = torch.nonzero(candidate_items == target_item).squeeze()
                try:
                    if idx_.shape[0] == 0:
                        idx_ = 199
                except IndexError:
                    idx_ = idx_.item()
                idx_list.append(idx_)

            ratings_logs[:, i] = r_target
            ranking_logs[:, i] = torch.Tensor(np.array(idx_list)) * initial_already_clicked_mask
            obs = env.step_new(recommendations=recommendations.flatten(), user_mask=user_sr_all[:, i], coeffs=coeffs, noise=rand_array[count])

            for u in range(args.num_users):
                meta = meta_by_user[u]
                acceptability_logs[u, i] = float(meta.get("acceptability", 0.0))
                bridge_logs[u, i] = float(meta.get("bridge", 0.0))
                clicked = bool(obs["clicks"][u].item())
                if lir_config.use_feedback:
                    feedback = feedback_simulator.observe(
                        clicked,
                        float(meta.get("acceptability", 0.0)),
                        float(meta.get("bridge", 0.0)),
                        float(meta.get("transition", 0.0)),
                        float(meta.get("risk", 0.0)),
                    )
                    feedback_counts[feedback.action] += 1
                    if previous_feedback[u] == "reject":
                        reject_recovery_total += 1
                        if feedback.action == "click":
                            reject_recovery_hits += 1
                    previous_feedback[u] = feedback.action
                    if lir_config.use_replanning:
                        replanner.update(u + 1, target_item, recommendations[u].item(), feedback.action)

                interacted[u, recommendations[u] + 1] = 1
                if clicked:
                    seq_guidance[u] = [seq_guidance[u][j + 1] for j in range(len(seq_guidance[u]) - 1)] + [recommendations[u].item() + 1]
                if clicked and recommendations[u].item() == target_item:
                    user_sr_all[u, i + 1 :] = 1
            click_logs_all[:, i] = obs["clicks"]

        for r in range(1, args.episode_length):
            ratings_logs[:, r] = ratings_logs[:, r] - ratings_logs[:, 0]

        all_hit_ratios.append(click_logs_all.mean().item())
        all_ratings_avg.append(ratings_logs.mean(dim=0))
        all_ranking_increase.append(ranking_logs.mean(dim=0))
        all_acceptability_avg.append(acceptability_logs.mean(dim=0))
        all_bridge_avg.append(bridge_logs.mean(dim=0))

    all_ratings_avg = torch.stack(all_ratings_avg)
    all_ranking_increase = torch.stack(all_ranking_increase)
    all_acceptability_avg = torch.stack(all_acceptability_avg)
    all_bridge_avg = torch.stack(all_bridge_avg)
    if args.lir_dump_bridge_pairs_path:
        grouped = defaultdict(list)
        for (target_item, candidate_item), count_value in bridge_pair_counts.items():
            grouped[target_item].append((candidate_item, count_value))
        dump_dir = os.path.dirname(os.path.abspath(args.lir_dump_bridge_pairs_path))
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(args.lir_dump_bridge_pairs_path, "w", encoding="utf-8") as f:
            for target_item in sorted(grouped):
                rows = sorted(grouped[target_item], key=lambda x: (-x[1], x[0]))[: args.lir_dump_pairs_per_target]
                for candidate_item, count_value in rows:
                    f.write(json.dumps({
                        "target_item": int(target_item),
                        "candidate_item": int(candidate_item),
                        "count": int(count_value),
                    }, ensure_ascii=False) + "\n")
        print(f"LIR bridge pairs dumped to {args.lir_dump_bridge_pairs_path}, targets={len(grouped)}, pairs={sum(min(len(rows), args.lir_dump_pairs_per_target) for rows in grouped.values())}")
    print(f"LIR, k=5, hit_ratio={torch.tensor(all_hit_ratios)[:5].mean():.4f}, IoI={(all_ratings_avg[:, 4]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 4]).mean():.4f}")
    print(f"LIR, k=10, hit_ratio={torch.tensor(all_hit_ratios)[:10].mean():.4f}, IoI={(all_ratings_avg[:, 9]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 9]).mean():.4f}")
    print(f"LIR, k=15, hit_ratio={torch.tensor(all_hit_ratios)[:15].mean():.4f}, IoI={(all_ratings_avg[:, 14]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 14]).mean():.4f}")
    print(f"LIR, k=20, hit_ratio={torch.tensor(all_hit_ratios)[:20].mean():.4f}, IoI={(all_ratings_avg[:, 19]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 19]).mean():.4f}")
    if use_lir:
        print(f"LIR semantic acceptability@20={(all_acceptability_avg[:, 19]).mean():.4f}, bridge_coherence@20={(all_bridge_avg[:, 19]).mean():.4f}")
        recovery = reject_recovery_hits / reject_recovery_total if reject_recovery_total else 0.0
        print(f"LIR feedback counts={feedback_counts}, reject_recovery={recovery:.4f}")


if __name__ == "__main__":
    main()
