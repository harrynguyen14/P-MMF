import numpy as np
from typing import List
import sklearn.metrics as metrics

from data.converter import DataConverter


EPSILON = 1e-10


class Metrics:
    """
    A class to calculate the metrics for evaluating the recommendation system.

    Note: This class only supports the binary relevance score.
    """

    def __init__(self):
        pass

    @staticmethod
    def dcg_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the discounted cumulative gain (DCG) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCG score.
        """
        return metrics.dcg_score(R, S, k=k)

    @staticmethod
    def ndcg_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the normalized discounted cumulative gain (NDCG) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The NDCG score.
        """
        return metrics.ndcg_score(R, S, k=k)

    @staticmethod
    def map_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean average precision (MAP) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MAP score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k
        return np.mean(np.sum(S * R, axis=1) / k)

    @staticmethod
    def lrap_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the label-ranking average precision (LRAP) score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The LRAP score.
        """
        return metrics.label_ranking_average_precision_score(R, S)

    @staticmethod
    def dcf_score(S: np.ndarray, groups: List[np.array], k: int = 30) -> float:
        """
        Calculate the deviation from producer fairness (DCF) score.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        groups (List[np.array]): The list of group indices.
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The DCF score.
        """
        assert len(groups) == 2, "The number of groups must be 2."

        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        cnt = np.sum(B, axis=0) / B.shape[0]
        cnt_g = [np.sum(cnt[g]) / len(g) for g in groups]

        return cnt_g[0] - cnt_g[1]

    @staticmethod
    def mdg_score_each_item(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the mean discounted gain (MDG) score for each item.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The MDG score for each item.
        """
        matrix_rank = DataConverter.convert_score_matrix_to_rank_matrix(S)
        matrix_rank = matrix_rank * (matrix_rank < k)

        return np.mean(R / np.log2(matrix_rank + 2), axis=0)

    @staticmethod
    def mdg_score(p: float, S: np.ndarray = None, B: np.ndarray = None, k: int = 30,  items_mdg: np.array = None) -> float:
        """
        Calculate the mean discounted gain (MDG) score.

        Parameters:
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.
        items_mdg (np.array): The MDG score for each item.
        p (float): The proportion of items to consider.

        Returns:
        float: The MDG score.
        """
        if items_mdg is None:
            if S is None:
                raise ValueError("S or items_mdg must be provided.")
            if B is None:
                B = DataConverter.convert_score_matrix_to_relevance_matrix(S, k)
            items_mdg = Metrics.mdg_score_each_item(B, S, k)

        n_item = items_mdg.shape[0]
        k = int(n_item * p)
        if k > 0:
            partition_idx = np.argpartition(items_mdg, k)[:k]
        else:
            partition_idx = np.argpartition(items_mdg, k)[k:]

        return np.mean(items_mdg[partition_idx])

    @staticmethod
    def precision_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the precision score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The precision score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        groundtruth = np.sum(R, axis=1)
        groundtruth = np.where(groundtruth < k, groundtruth, k)
        true_positive = np.sum(B * R, axis=1)

        divider = np.where(groundtruth == 0, 1, groundtruth)
        return np.sum(true_positive / divider) / np.count_nonzero(groundtruth)

    @staticmethod
    def recall_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the recall score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The recall score.
        """
        B = DataConverter.convert_score_matrix_to_rank_matrix(S) < k

        groundtruth = np.sum(R, axis=1)
        true_positive = np.sum(B * R, axis=1)

        divider = np.where(groundtruth == 0, 1, groundtruth)
        return np.sum(true_positive / divider) / np.count_nonzero(groundtruth)

    @staticmethod
    def f1_score(R: np.ndarray, S: np.ndarray, k: int = 30) -> float:
        """
        Calculate the F1 score.

        Parameters:
        R (np.ndarray): The binary relevance score matrix of shape (n_users, n_items).
        S (np.ndarray): The predicted score matrix of shape (n_users, n_items).
        k (int): The number of items to consider for each user. Default is 30.

        Returns:
        float: The F1 score.
        """
        precision = Metrics.precision_score(R, S, k)
        recall = Metrics.recall_score(R, S, k)

        return 2 * (precision * recall) / (precision + recall)




if __name__ == "__main__":
    R = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    S = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

    print("Test Metrics Strategy Pattern Design.")
    print(Metrics.dcg_score(R, S))
    print(Metrics.ndcg_score(R, S))

    print(Metrics.map_score(R, S))
    print(Metrics.lrap_score(R, S))

    print(Metrics.dcf_score(S, [[0], [1]]))

    print(Metrics.mdg_score_each_item(R, S))
    print(Metrics.mdg_score(Metrics.mdg_score_each_item(R, S), 0.1))
    print(Metrics.mdg_score(Metrics.mdg_score_each_item(R, S), -0.1))

    print(Metrics.precision_score(R, S, k=2))
    print(Metrics.recall_score(R, S, k=2))
    print(Metrics.f1_score(R, S, k=2))