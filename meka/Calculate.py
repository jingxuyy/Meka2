import warnings
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *


warnings.filterwarnings("ignore")



class Performance(object):
    def __init__(self, filename, label_num, zeros=False):
        self.data = self.format_conversion(filename, label_num)
        self.number = len(self.data)
        self.label_num = label_num
        self.data_zero = pd.DataFrame(columns=['a', 'b'], index=[i for i in range(self.data.shape[0])])
        if zeros:
            for i in range(self.number):
                if sum(self.data.iloc[i, 0]) == 0:
                    if sum(self.data.iloc[i, 1]) == 0:
                        self.data.iloc[i, 0] = [1 for _ in range(self.label_num)]
                        self.data.iloc[i, 1] = [1 for _ in range(self.label_num)]
                    else:
                        self.data.iloc[i, 0] = [0 if num == 0 else 1 for num in self.data.iloc[i, 1]]
                        self.data.iloc[i, 1] = [0 for _ in range(self.label_num)]


    def format_conversion(self, filename, label):
        start_index = 2 * label + 1

        data = pd.read_csv(filename, sep='\\n', header=None)
        index = data.iloc[1, 0]

        number = list(filter(str.isdigit, index))

        number = int(''.join(number[:-1]))

        data = data.iloc[2: number + 2, :]
        for i in range(data.shape[0]):
            data.iloc[i, 0] = data.iloc[i, 0][7:]

        new_data = pd.DataFrame(data=None, columns=['a', 'b'], index=range(number))

        for i in range(data.shape[0]):
            values = data.iloc[i, 0]
            data.iloc[i, 0] = values.replace(' ', ',')
            data.iloc[i, 0] = '[' + data.iloc[i, 0][2:start_index] + ']' + ' ' + '[' + data.iloc[i, 0][
                                                                                       start_index + 5:-2] + ']'

        for j in range(data.shape[0]):
            a, b = data.iloc[j, 0].split()
            new_data.iloc[j, 0] = a
            new_data.iloc[j, 1] = b

        return self.float2int(new_data)

    def float2int(self, new_data):
        for i in range(new_data.shape[0]):
            part_data = list(eval(new_data.iloc[i, 1]))
            actual_data = list(eval(new_data.iloc[i, 0]))
            for j in range(len(part_data)):
                part_data[j] = int(part_data[j])

            new_data.iloc[i, 1] = part_data
            new_data.iloc[i, 0] = actual_data
        return new_data

    def threshold_conversion(self, threshold=0.5):
        data_actual_list = []
        data_predict_list = []
        for i in range(self.data.shape[0]):
            data_actual_list.append(self.data.iloc[i, 0])
            data_predict_list.append(self.data.iloc[i, 1])

        data_actual_matrix = pd.DataFrame(data_actual_list).values
        data_predict_matrix_a = pd.DataFrame(data_predict_list).values

        data_predict_matrix = np.where(data_predict_matrix_a >= threshold, 1, 0)

        return data_actual_matrix, data_predict_matrix, data_predict_matrix_a


    def Accuracy_score(self, single_actual, single_predict):
        return round(accuracy_score(single_actual, single_predict), 4)

    def Precision_score(self, single_actual, single_predict, average="binary"):
        return round(precision_score(single_actual, single_predict, average=average), 4)

    def Recall_score(self, single_actual, single_predict, average="binary"):
        return round(recall_score(single_actual, single_predict, average=average), 4)

    def F1_score(self, single_actual, single_predict):
        return round(f1_score(single_actual, single_predict), 4)

    def Single_roc_estimated(self, single_actual, single_predict):
        return round(roc_auc_score(single_actual, single_predict), 4)

    def Single_roc_exact(self, single_actual, single_predict):
        fpr, tpr, thresholds = roc_curve(single_actual, single_predict)
        return round(auc(fpr, tpr), 4)

    def Single_prc_estimated(self, single_actual, single_predict):
        return round(average_precision_score(single_actual, single_predict), 4)

    def Single_prc_exact(self, single_actual, single_predict):
        precision, recall, thresholds = precision_recall_curve(single_actual, single_predict)
        return round(auc(recall, precision), 4)

    def Zero_one_loss(self, single_actual, single_predict):
        return round(zero_one_loss(single_actual, single_predict), 4)

    def Hamming_loss(self, single_actual, single_predict):
        return round(hamming_loss(single_actual, single_predict), 4)

    def Jaccard_score(self, single_actual, single_predict, average="binary"):
        return round(jaccard_score(single_actual, single_predict, average=average), 4)

    def Specificity(self, single_actual, single_predict):
        cfmetric = confusion_matrix(single_actual, single_predict)
        tn, fp, fn, tp = cfmetric.ravel()
        # sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return round(specificity, 4)


    def Single_indicator(self, flag=False, estimated=True):
        data_actual_matrix, data_predict_matrix, data_predict_matrix_a = self.threshold_conversion()
        arr = np.zeros((self.label_num, 7))
        file = pd.DataFrame(arr, columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'specificity', 'AUROC', 'AUPRC'], index=['L'+str(j+1) for j in range(self.label_num)])

        for i in range(self.label_num):
            single_actual = data_actual_matrix[:, i]
            single_predict = data_predict_matrix[:, i]
            single_predict_a = data_predict_matrix_a[:, i]
            accuracy = self.Accuracy_score(single_actual, single_predict)
            precision = self.Precision_score(single_actual, single_predict)
            recall = self.Recall_score(single_actual, single_predict)
            f1_score = self.F1_score(single_actual, single_predict)

            specificity = self.Specificity(single_actual, single_predict)

            if estimated:
                roc_auc = self.Single_roc_estimated(single_actual, single_predict_a)
                pr_auc = self.Single_prc_estimated(single_actual, single_predict_a)
            else:
                roc_auc = self.Single_roc_exact(single_actual, single_predict_a)
                pr_auc = self.Single_prc_exact(single_actual, single_predict_a)

            file.iloc[i, :] = [accuracy, precision, recall, f1_score, specificity, roc_auc, pr_auc]
        if flag:
            file.to_csv('Single label performance indicators.csv')
        else:
            print(file)

    def Multi_indicator(self, flag=False, zeros=False):
        arr = np.zeros((1, 5))
        file = pd.DataFrame(arr, columns=['Aiming', 'Coverage', 'Accuracy', 'Absolute_true', 'Absolute_false'])
        data_actual_matrix, data_predict_matrix, data_predict_matrix_a = self.threshold_conversion()
        Absolute_true = self.Accuracy_score(data_actual_matrix, data_predict_matrix)
        Absolute_false = self.Hamming_loss(data_actual_matrix, data_predict_matrix)
        Accuracy = self.Jaccard_score(data_actual_matrix, data_predict_matrix, average='samples')
        Coverage = self.Recall_score(data_actual_matrix, data_predict_matrix, average='samples')
        Aiming = self.Precision_score(data_actual_matrix, data_predict_matrix, average='samples')

        file.iloc[0, :] = [Aiming, Coverage, Accuracy, Absolute_true, Absolute_false]

        if flag:
            file.to_csv('Multi-label performance indicators.csv', index=False)
        else:
            print(file)


    def Plot_ROC(self, label_name, save=False, estimated=True):
        if len(label_name) < self.label_num:
            print('error! label_name num != label_num num')
            return
        label_value = pd.read_csv('Single label performance indicators.csv')[['AUROC']]
        plt.flag()
        plt.rc('font', family='Times New Roman')

        ax1 = plt.gca()

        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('False Positive Rate', fontsize=15, color='k')
        ax1.set_ylabel('True Positive Rate', fontsize=15, color='k')
        plt.tick_params(labelsize=15)

        data_actual_matrix, data_predict_matrix, data_predict_matrix_a = self.threshold_conversion()
        for i in range(self.label_num):
            single_actual = data_actual_matrix[:, i]
            single_predict_a = data_predict_matrix_a[:, i]
            fpr, tpr, thresholds2 = roc_curve(single_actual, single_predict_a)
            if not estimated:
                ax1.plot(fpr, tpr, label=label_name[i] + ' (area=' + format(label_value.iloc[i, 0], '.4f') + ')')
            else:
                ax1.step(fpr, tpr, label=label_name[i] + ' (area=' + format(label_value.iloc[i, 0], '.4f') + ')')
        ax1.plot([0, 1], [0, 1], '--')
        plt.title('ROC Curve')
        plt.rc('legend', fontsize=14)

        legend = ax1.legend(loc='lower right', frameon=False)
        ax1.legend(loc=2, bbox_to_anchor=(1.02, 1.0), borderaxespad=0., prop={'family': 'monospace'})
        frame = legend.get_frame()
        frame.set_alpha(1)
        frame.set_facecolor('none')  # 设置图例legend背景透明
        plt.subplots_adjust(right=0.7)

        # ax1.legend()
        if save:
            plt.savefig('all_ROC.pdf')
            plt.close()
        else:
            plt.show()


    def Plot_PRC(self, label_name, save=False, estimated=True):
        if len(label_name) < self.label_num:
            print('error! label_name num != label_num num')
            return
        label_value = pd.read_csv('Single label performance indicators.csv')[['AUPRC']]
        plt.flag()
        plt.rc('font', family='Times New Roman')

        ax = plt.gca()

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Recall', fontsize=15, color='k')
        ax.set_ylabel('Precision', fontsize=15, color='k')
        plt.tick_params(labelsize=15)

        data_actual_matrix, data_predict_matrix, data_predict_matrix_a = self.threshold_conversion()
        for i in range(self.label_num):
            single_actual = data_actual_matrix[:, i]
            single_predict_a = data_predict_matrix_a[:, i]
            precision, recall, thresholds = precision_recall_curve(single_actual, single_predict_a)
            precision = np.append(np.array([0]), precision)
            recall = np.append(np.array([1]), recall)
            if not estimated:
                ax.plot(recall, precision, label=label_name[i] + ' (area=' + format(label_value.iloc[i, 0], '.4f') + ')')
            else:
                ax.step(recall, precision,
                        label=label_name[i] + ' (area=' + format(label_value.iloc[i, 0], '.4f') + ')')
        plt.title('PR Curve')
        plt.rc('legend', fontsize=14)

        ax.legend()
        if save:
            plt.savefig('all_PRC.pdf')
            plt.close()
        else:
            plt.show()


class Parameters(object):
    def __init__(self, path, same=True, norm='Accuracy'):
        self.path = path
        self.number = self.samples_num()
        self.Accuracy = 21
        self.Exact_match = 24
        self.norm = norm
        self.line_num = 0
        self.same = same
        if self.norm == 'Accuracy':
            self.line_num = self.number + self.Accuracy
        elif self.norm == 'Exact match':
            self.line_num = self.number + self.Exact_match

    def maxparameters(self):
        file_list = os.listdir(self.path)
        self.get_dict(self.path, file_list)


    def get_dict(self, path, file_list):
        i = 0
        indicator = {}
        while i < len(file_list):
            flag = True
            while flag:
                if not self.same:
                    if self.norm == 'Accuracy':
                        self.line_num = self.openfile(file_list[i]) + self.Accuracy
                    elif self.norm == 'Exact match':
                        self.line_num = self.openfile(file_list[i]) + self.Exact_match
                with open(path + file_list[i], "r") as f:
                    for num, line in enumerate(f):
                        if num == self.line_num:
                            acc = float(re.findall(r"\d+\.?\d*", line)[0])
                            indicator[file_list[i]] = acc
                            flag = False
                            break
            i += 1

        sort_indicator = sorted(indicator.items(), key=lambda x: x[1], reverse=True)
        fileObject = open('result.txt', "w")
        for indicate in sort_indicator:
            indicate = str(indicate).replace('(', '').replace(')', '').replace('\'', '')
            fileObject.write(indicate)
            fileObject.write('\n')
        fileObject.close()

    def samples_num(self):
        file = os.listdir(self.path)[0]
        return self.openfile(file)

    def openfile(self, file):
        with open(self.path + file, "r") as f:
            for num, line in enumerate(f):
                if num == 2:
                    number = list(filter(str.isdigit, line))
                    number = int(''.join(number[:-1]))
                    return number


class File_merge(Parameters):
    def __init__(self, path):
        super().__init__(path, same=True)
        self.all_number = 0
        self.file_list = os.listdir(self.path)
        self.samples_num_list = []

    def get_samples(self):
        for file in self.file_list:
            self.samples_num_list.append(super().openfile(file))
        self.all_number = sum(self.samples_num_list)

    def file_merge(self):
        self.get_samples()
        new_file = open('merge_samples.txt', 'w')
        for index, file in enumerate(self.file_list):
            with open(self.path + file, "r") as f:
                for num, line in enumerate(f):
                    if num == self.samples_num_list[index]+3:
                        break
                    if index == 0:
                        if num == 2:
                            new_file.writelines(f'|==== PREDICTIONS(N={self.all_number}.0) =====>\n')
                            continue
                        new_file.writelines(line)
                    else:
                        if num < 3:
                            continue
                        new_file.writelines(line)
        new_file.close()


class Average_single_indicator(object):
    def __init__(self, path, label_num, zeros=False):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.label_num = label_num
        self.zeros = zeros

    def get_data(self):
        data_list = []
        for filename in self.file_list:
            obj = Performance(self.path+filename, self.label_num, zeros=self.zeros)
            data_actual_matrix, data_predict_matrix, data_predict_matrix_a = obj.threshold_conversion()
            data_list.append([data_actual_matrix, data_predict_matrix_a])
        return data_list

    def Average_Plot_ROC(self, label_name, save=False):
        if len(label_name) < self.label_num:
            print('error! label_name num != label_num num')
            return

        plt.flag()
        plt.rc('font', family='Times New Roman')

        ax = plt.gca()

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('False Positive Rate', fontsize=15, color='k')
        ax.set_ylabel('True Positive Rate', fontsize=15, color='k')
        plt.tick_params(labelsize=15)

        all_data = self.get_data()
        for i in range(self.label_num):
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            for data in all_data:
                data_actual_matrix = data[0]
                data_predict_matrix_a = data[1]

                single_actual = data_actual_matrix[:, i]
                single_predict_a = data_predict_matrix_a[:, i]
                fpr, tpr, thresholds = roc_curve(single_actual, single_predict_a)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, label=label_name[i] + r' (area=%0.4f)' % mean_auc)

        ax.plot([0, 1], [0, 1], '--')
        plt.title('ROC Curve')
        plt.rc('legend', fontsize=14)

        ax.legend()
        if save:
            plt.savefig('all_ROC.pdf')
        else:
            plt.show()


    def Average_Plot_PRC(self, label_name, save=False):
        if len(label_name) < self.label_num:
            print('error! label_name num != label_num num')
            return

        all_data = self.get_data()
        fig, ax = plt.subplots()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Recall', fontsize=15, color='k')
        ax.set_ylabel('Precision', fontsize=15, color='k')
        plt.tick_params(labelsize=15)

        for i in range(self.label_num):
            precisions = []
            mean_recall = np.linspace(0, 1, 100)
            for data in all_data:
                data_actual_matrix = data[0]
                data_predict_matrix_a = data[1]

                single_actual = data_actual_matrix[:, i]
                single_predict_a = data_predict_matrix_a[:, i]

                precision, recall, _ = precision_recall_curve(single_actual, single_predict_a)
                interp_precision = np.interp(mean_recall, precision, recall)
                interp_precision[0] = 1.0
                precisions.append(interp_precision)
            mean_precision = np.mean(precisions, axis=0)
            mean_precision[-1] = 0.0
            mean_auc = auc(mean_recall, mean_precision)

            ax.plot(mean_recall, mean_precision, label=label_name[i] + r' (area=%0.4f)' % mean_auc)

        plt.title('PR Curve')
        plt.rc('legend', fontsize=14)
        plt.legend()
        plt.show()

        if save:
            plt.savefig('all_PRC.pdf')
        else:
            plt.show()












