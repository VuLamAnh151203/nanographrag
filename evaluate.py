def average_precision_at_k(actual, predicted, k):
    """
    Calculate Average Precision at K (AP@K) for a single query.

    Parameters:
    - actual (list): List of relevant items.
    - predicted (list): List of predicted items.
    - k (int): Number of top elements to consider.

    Returns:
    - float: AP@K score.
    """
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    list_precision_k_at_relevant_docs = []
    true_positives = 0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            true_positives += 1
            precision_at_k = true_positives / (i+1)
            list_precision_k_at_relevant_docs.append(precision_at_k)
    
    return sum(list_precision_k_at_relevant_docs) / true_positives if true_positives != 0 else 0.0

        


def mean_average_precision_at_k(actual_list, predicted_list, k):
    """
    Calculate Mean Average Precision at K (MAP@K).

    Parameters:
    - actual_list (list of lists): Each sublist contains the relevant items for a query.
    - predicted_list (list of lists): Each sublist contains the predicted items for a query.
    - k (int): Number of top elements to consider.

    Returns:
    - float: MAP@K score.
    """
    return sum(average_precision_at_k(a, p, k) for a, p in zip(actual_list, predicted_list)) / len(actual_list)



def mrr_at_k(actual_list, predicted_list, k):
    """
    Calculate Mean Reciprocal Rank at K (MRR@K).

    Parameters:
    - actual_list (list of lists): Each sublist contains the relevant items for a query.
    - predicted_list (list of lists): Each sublist contains the predicted items for a query.
    - k (int): The number of top elements to consider in the ranking.

    Returns:
    - float: The MRR@K score.
    """
    reciprocal_ranks = []
    for actual, predicted in zip(actual_list, predicted_list):
        try:
            # Find the rank of the first relevant item within top K
            # rank = next(i + 1 for i, p in enumerate(predicted[:k]) if p in actual)
            rank = 0
            for i, p in enumerate(predicted[:k]):
                if p in actual:
                    rank = i + 1
                    break
            if rank != 0:
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        except StopIteration:
            # No relevant item found in top K
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)



def recall_at_k(actual_list, predicted_list, k):
    """
    Calculate mean Recall@K for multiple queries.

    Parameters:
    - actual_list (list of lists): Each sublist contains the relevant items for a query.
    - predicted_list (list of lists): Each sublist contains the predicted items for a query.
    - k (int): Number of top elements to consider.

    Returns:
    - float: Mean Recall@K score across all queries.
    """
    recall_scores = []
    for actual, predicted in zip(actual_list, predicted_list):
        actual_set = set(actual)
        predicted_top_k = set(predicted[:k])
        relevant_retrieved = actual_set & predicted_top_k
        recall = len(relevant_retrieved) / len(actual_set) if actual_set else 0.0
        recall_scores.append(recall)
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    return mean_recall

def average_four_metrics(actual_list, predicted_list, k):
    recall_at_k_score = recall_at_k(actual_list, predicted_list, k)
    mrr_at_k_score = mrr_at_k(actual_list, predicted_list, k)
    mean_precision_at_k_score = mean_average_precision_at_k(actual_list, predicted_list, k)
    # nDCG_at_k_score = mean_ndcg_at_k(actual_list, predicted_list,k)
    results = ""
    results += f"Recall@{k}:{recall_at_k_score}" + "\n"
    results += f"MRR@{k}:{mrr_at_k_score}" +"\n"
    results += f"MAP@{k}:{mean_precision_at_k_score}" +"\n"
    results += f"Average Score: {(recall_at_k_score + mrr_at_k_score + mean_precision_at_k_score) / 3}" +"\n"

    return results