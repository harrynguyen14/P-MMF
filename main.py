import os
import argparse
import numpy as np
import pandas as pd

from reranking import P_MMF_CPU, P_MMF_GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulator")
    parser.add_argument('--base_model', default='bpr')
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument('--dataset-dir', type=str, default='datasets')
    parser.add_argument('--Time', type=int, default=256,
                        help='fair time sep.')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--gpu', type=str, default="False")


    args = parser.parse_args()
    print("Using GPU:",args.gpu)
    if args.gpu == "True":
        P_MMF_GPU(lambd=1e-1,args=args)
    else:
        P_MMF_CPU(lambd=1e-1,args=args)

    dataset_dir = dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset {args.dataset_name} not found in {args.dataset_dir}."
        )

    if not os.path.exists(
        os.path.join(dataset_dir, ".npy")
    ):
        raise FileNotFoundError(
            f"Dataset {args.dataset_name} is not formatted. Please run formatter.py first."
        )
    
    for model_name in ["ccfcrec", "clcrec"]:
        if model_name == "clcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result_formated.npy"))
            S = preprocess_clcrec_result(S)
        elif model_name == "ccfcrec":
            S = np.load(os.path.join(dataset_dir, f"{model_name}_result.npy"))
            S = preprocess_ccfcrec_result(S)
        else:
            raise ValueError("Invalid model name.")

        print(model_name.upper())
        print("Precision:", Metrics.precision_score(R, S, k=top_k))
        print("Recall:", Metrics.recall_score(R, S, k=top_k))
        print("NDCG:", Metrics.ndcg_score(R, S, k=top_k))

        B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k=top_k)
        print("MDG_min_10:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.1))
        print("MDG_min_20:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.2))
        print("MDG_min_30:", Metrics.mdg_score(S=S, B=B, k=top_k, p=0.3))
        print("MDG_max_10:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.1))
        print("MDG_max_20:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.2))
        print("MDG_max_30:", Metrics.mdg_score(S=S, B=B, k=top_k, p=-0.3))


        if not args.not_reranking:
            # group_items = divide_group(B, group_p=0.7)

            reranking = ReRanking(WorstOffNumberOfItemReRanking())
            # W = reranking.optimize(S, k=top_k, i_epsilon=args.epsilon, group_items=group_items)
            W = reranking.optimize(S, k=top_k, epsilon=args.epsilon)
            S_reranked = reranking.apply_reranking_matrix(S, W)

            print("Precision (reranked):", Metrics.precision_score(R, S_reranked, k=top_k))
            print("Recall (reranked):", Metrics.recall_score(R, S_reranked, k=top_k))
            print("NDCG (reranked):", Metrics.ndcg_score(R, S_reranked, k=top_k))

            print("MDG_min_10 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.1))
            print("MDG_min_20 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.2))
            print("MDG_min_30 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=0.3))
            print("MDG_max_10 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.1))
            print("MDG_max_20 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.2))
            print("MDG_max_30 (reranked):", Metrics.mdg_score(S=S_reranked, B=W, k=top_k, p=-0.3))
            print()