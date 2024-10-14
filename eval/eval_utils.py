import pandas as pd


def get_dept_score_by_top_k_preds(predictions_df, assignments_df, k):
    #top_scores_idx = predictions_df.groupby(['SP_ID'])['Skill Score'].idxmax()
    #top_preds_df = predictions_df.loc[top_scores_idx]  
    top_k_preds_df = predictions_df.groupby('SP_ID').apply(lambda x: x.nlargest(k, 'Skill Score')).reset_index(drop=True)
    pred_assigns_join_df = pd.merge(top_k_preds_df, assignments_df, left_on='SP_ID', right_on='SP ID', how='inner')
    pred_assigns_truth_df = pred_assigns_join_df.copy() 
    pred_assigns_truth_k_grouped_df = pred_assigns_truth_df.groupby(['SP_ID', 'Seva Dept']).agg({'Department': list, 'Skill Score': list}).reset_index()
    pred_assigns_truth_k_grouped_df['InTopK'] = pred_assigns_truth_k_grouped_df.apply(lambda row: row['Seva Dept'] in row['Department'], axis=1)
    accuracy = pred_assigns_truth_k_grouped_df['InTopK'].mean()
    return pred_assigns_truth_k_grouped_df, accuracy
