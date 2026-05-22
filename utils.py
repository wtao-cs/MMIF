import numpy as np


def summayResults(test_datalabel, y_pred, coef_):
    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import cohen_kappa_score
    # Kappa系数是基于混淆矩阵的计算得到的模型评价参数(越接近 1 越好)
    kappa = cohen_kappa_score(np.array(y_pred), np.array(test_datalabel))

    cm = confusion_matrix(test_datalabel, y_pred)
    TP = cm[1][1]  # abnoraml
    TN = cm[0][0]  # nomral
    FP = cm[0][1]
    FN = cm[1][0]
    # assert TP + FN == 126
    # assert TN + FP == 150
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    import math
    def cutFloat(val, precision=10000):
        """
        小数点整段截取,返回 单位为 %
        :param val:
        :param precision:
        :return:
        """
        return int(val * precision) / precision

    testMetrics = {}
    testMetrics['accuracy'] = accuracy
    testMetrics['g_mean'] = cutFloat(math.sqrt(sensitivity * specificity))
    testMetrics['f1_score'] = f1_score
    testMetrics['sensitivity'] = sensitivity
    testMetrics['specificity'] = specificity
    testMetrics['precision'] = precision
    testMetrics['kappa'] = kappa
    testMetrics['confusion_matrix'] = cm


    mask = []
    for index in abs(coef_):
        if index > 1e-6:
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    selectVariableNum = np.sum(mask != False)
    testMetrics['selectVariableNum'] = selectVariableNum
    testMetrics['coef_'] = coef_

    testMetrics['test_datalabel'] = test_datalabel
    testMetrics['y_pred'] = y_pred

    testMetrics['FN'] = FN

    testMetrics['label'] = 'test metrics'
    testMetrics['TP'] = TP
    testMetrics['FN'] = FN
    testMetrics['FP'] = FP
    testMetrics['TN'] = TN
    return testMetrics



