import numpy as np
import pandas as pd

import find_transform
import quality_of_fit
import model

def cv_split_by_file(data_meta, data_x):
    files = np.unique(data_meta.file_number)
    np.random.shuffle(files)

    n_train = int(round(len(files)*0.5))
    train_files = files[:n_train]
    print(train_files)

    train_cols = data_meta.file_number.isin(train_files)
    test_cols = ~train_cols
    return data_meta[train_cols], data_x[train_cols,:], data_meta[test_cols], data_x[test_cols,:]

def validate_speech_detection(result_meta):
    print('---- speech detection accuracy ----')

    print(result_meta.groupby('file_number').agg('mean'))
    from sklearn.metrics import roc_auc_score
    print('AUC-ROC:', roc_auc_score(result_meta.label, result_meta.predicted_score))

def test_correct_sync(result_meta):
    print('---- synchronization accuracy ----')

    results = []
    for file_number in np.unique(result_meta.file_number):
        part = result_meta[result_meta.file_number == file_number]
        skew, shift, quality = find_transform.find_transform_parameters(part.label, part.predicted_score)
        skew_error = skew != 1.0
        results.append([skew_error, shift, quality])

    sync_results = pd.DataFrame(np.array(results), columns=['skew_error', 'shift_error', 'quality'])
    print(sync_results)

    print('skew errors:', sync_results.skew_error.sum())
    print('shift MAE:', np.mean(np.abs(sync_results.shift_error)))

    return np.array(list(sync_results.quality))


def test_quality_of_fit_mismatch(result_meta):

    all_files = np.unique(result_meta.file_number)
    pairs = [(n1, n2) for n1 in all_files for n2 in all_files if n2 != n1]

    print('---- quality of fit (computing for %d mismatches) ----' % len(pairs))

    qualities = []

    for fn1, fn2 in pairs:
        labels0 = result_meta.label[result_meta.file_number == fn1]
        probs0 = result_meta.predicted_score[result_meta.file_number == fn2]
        l = min(len(labels0), len(probs0))

        labels = probs0*0
        labels[:l] = labels0[:l]

        for flip in [False, True]:
            if flip: probs = probs0[::-1]
            else: probs = probs0[::]

            skew, shift, quality = find_transform.find_transform_parameters(labels, probs)
            quality_error = quality >= quality_of_fit.threshold

            qualities.append(quality)
            print(quality)

    return np.array(qualities)

if __name__ == '__main__':

    data_x = np.load('training-data/features.npy')
    print('loaded training features of size', data_x.shape)
    data_meta = pd.read_csv('training-data/meta.csv', index_col=0)
    n_folds = 4
    np.random.seed(1)

    correct_qualities = []

    for i in range(n_folds):
        print('### Cross-validation fold %d/%d' % (i+1, n_folds))
        train_meta, train_x, test_meta, test_x = cv_split_by_file(data_meta, data_x)

        print('Training...', train_x.shape)
        trained_model = model.train(train_x, train_meta.label, train_meta)

        print('Validating...')
        predicted_score = model.predict(trained_model, test_x, test_meta.file_number)
        result_meta = test_meta.assign(predicted_score=predicted_score)
        result_meta = result_meta.assign(predicted_label=np.round(predicted_score))
        result_meta = result_meta.assign(label=np.round(result_meta.label))
        result_meta = result_meta.assign(correct=result_meta.predicted_label==result_meta.label)

        validate_speech_detection(result_meta)
        correct_qualities.extend(test_correct_sync(result_meta))

    correct_qualities = np.array(correct_qualities)

    print('### Quailty of fit mismatch test (with last fold)')
    mismatch_qualities = test_quality_of_fit_mismatch(result_meta)

    min_correct = np.min(correct_qualities)
    max_incorrect = np.max(mismatch_qualities)

    quality_margin = min_correct - max_incorrect

    print('estimated threshold:', (min_correct + max_incorrect)*0.5)
    print('current threshold:', quality_of_fit.threshold)
    print('quality margin:', quality_margin)

    print('false negative quality errors:', np.sum(correct_qualities < quality_of_fit.threshold))
    print('false positive quality errors:', np.sum(mismatch_qualities > quality_of_fit.threshold))
