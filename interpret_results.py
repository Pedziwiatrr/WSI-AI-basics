from sklearn.metrics import f1_score, roc_auc_score


def compare_diagnoses(true_diagnosis, suspicious_diagnosis):
    correct = 0
    wrong = 0
    for i in range(len(true_diagnosis)):
        true_verdict = true_diagnosis.iloc[i]
        suspicious_verdict = suspicious_diagnosis[i]
        print(f"{true_verdict} vs {suspicious_verdict}")
        if true_verdict == suspicious_verdict:
            correct += 1
        else:
            wrong += 1
    return [correct, wrong]

def get_accuracy(verdicts):
    return verdicts[0] / (verdicts[0] + verdicts[1])

def print_results(verdicts, accuracy, f1, auroc):
    print("\n" + "=" * 100)
    print(f"Correct diagnoses: {verdicts[0]}")
    print(f"Wrong diagnoses: {verdicts[1]}")
    print(f"Algorithm accuracy: {get_accuracy(verdicts)*100:.2f}%")
    print(f"F1 score: {f1:.2f}")
    print(f"AUROC: {auroc:.2f}")
    print("="*100 + "\n")

def interpret_results(true_results, algorithm_results, predicted_probabilities):
    compared_diagnoses = compare_diagnoses(true_results, algorithm_results)
    accuracy = get_accuracy(compared_diagnoses)
    f1 = f1_score(true_results, algorithm_results)
    auroc = roc_auc_score(true_results, predicted_probabilities)
    return [compared_diagnoses, accuracy, f1, auroc]
