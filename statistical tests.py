import pandas as pd
from scipy import stats

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'caffe'
path1 = f'./results/{project}_NB_detailed_results.csv'
path2 = f'./results/{project}_SVM_Feature_Fusion_detailed_results.csv'

baseline_method_df = pd.read_csv(path1)
experiment_method_df = pd.read_csv(path2)


# 1. save data
data = {
    "Experiment_Method_Accuracy": experiment_method_df["Accuracy"],
    "Experiment_Method_Precision": experiment_method_df["Precision"],
    "Experiment_Method_Recall": experiment_method_df["Recall"],
    "Experiment_Method_F1": experiment_method_df["F1"],
    "Experiment_Method_AUC": experiment_method_df["AUC"],
    "Baseline_Accuracy": baseline_method_df["Accuracy"],
    "Baseline_Precision": baseline_method_df["Precision"],
    "Baseline_Recall": baseline_method_df["Recall"],
    "Baseline_F1": baseline_method_df["F1"],
    "Baseline_AUC": baseline_method_df["AUC"]
}


# 2. statistical tests
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]  # performance indicators to be tested

for metric in metrics:
    # Extract data for both methods
    experiment_method_scores = data[f"Experiment_Method_{metric}"]
    baseline_scores = data[f"Baseline_{metric}"]

    # Normality test (Shapiro-Wilk test)
    _, p_normal_experiment = stats.shapiro(experiment_method_scores)
    _, p_normal_baseline = stats.shapiro(baseline_scores)

    # Decide whether to use paired t-test or Wilcoxon test
    if p_normal_experiment > 0.05 and p_normal_baseline > 0.05:
        # If the data follow a normal distribution, use the paired t-test
        test_stat, p_value = stats.ttest_rel(experiment_method_scores, baseline_scores)
        test_name = "Paired t-test"
    else:
        # If the data do not follow a normal distribution, use the Wilcoxon test
        test_stat, p_value = stats.wilcoxon(experiment_method_scores, baseline_scores)
        test_name = "Wilcoxon Signed-Rank Test"

    # output
    print(f"Metric: {metric}")
    print(f"Test used: {test_name}")
    print(f"Test statistic: {test_stat}")
    print(f"p-value: {p_value}")
    if p_value < 0.05:
        print("Result: Significant difference!")
    else:
        print("Result: No significant difference.")
    print("------")