import string
import tempfile

import fasttext
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import sklearn
import sklearn.metrics
from unidecode import unidecode


class QuestionClassifier:
    def __init__(self, *args, **kwargs):
        """
        Arguments (optional and mutually exclusive):
        path (str): path for saved model
        model (Model): previously fitted model
        """
        self.path = kwargs.get('path')

        if self.path:
            self.model = self.load(self.path)
            if '__' in str(self.model.labels[0]):
                self.model.labels_ = [int(lab.replace('__label__', '')) for lab in self.model.labels]
            else:
                self.model.labels_ = self.model.labels

        else:
            self.model = kwargs.get('model')

    def train(self, train_questions, train_labels, dev_questions, dev_labels):
        assert len(train_questions) == len(train_labels)
        assert len(dev_questions) == len(dev_labels)

    def save(self, path):
        assert self.model, "Run train() beforehand."

    def load(self, path):
        pass

    def predict(self, questions):
        assert self.model, "Run train() beforehand."

    def evaluate(self, true, pred):
        assert len(true) == len(pred)
        assert self.model, "Run train() beforehand."

        return sklearn.metrics.classification_report(true, pred, labels=self.model.labels_, digits=3, output_dict=True)

    def plot_confusion(self, true, pred, size=(11, 5), fmt='.1f'):
        assert len(true) == len(pred)
        assert self.model, "Run train() beforehand."

        matrix = sklearn.metrics.confusion_matrix(true, pred,
                                                  labels=self.model.labels_,
                                                  normalize='true') * 100
        idx = np.argsort(np.diag(-matrix))
        matrix = matrix[idx, :][:, idx]

        plt.figure(figsize=size)
        sns.heatmap(matrix,
                    cmap='viridis',
                    annot=True,
                    xticklabels=self.model.labels_,
                    yticklabels=self.model.labels_,
                    fmt=fmt)


class FastTextClassifier(QuestionClassifier):
    def __init__(self, multilabel=False, *args, **kwargs):
        QuestionClassifier.__init__(self, *args, **kwargs)
        self._multilabel = multilabel

    def train(self, train_questions, train_labels, dev_questions, dev_labels):
        super(FastTextClassifier, self).train(train_questions, train_labels, dev_questions, dev_labels)

        train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="fttrain")
        dev_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="ftdev")

        if self._multilabel:
            for labels, question in zip(train_labels, train_questions):
                label_string = ' '.join(f'__label__{label}' for label in labels.split())

                question_tokenized = ' '.join(
                    [tok for tok in nltk.word_tokenize(question.lower()) if tok not in string.punctuation])
                train_file.write(label_string + ' ' + question_tokenized + '\n')

            for labels, questions in zip(dev_labels, dev_questions):
                if type(labels) != str:
                    labels = str(labels)

                label_string = ' '.join(f'__label__{label}' for label in labels.split())
                question_tokenized = ' '.join(
                    [tok for tok in nltk.word_tokenize(question.lower()) if tok not in string.punctuation])
                dev_file.write(label_string + ' ' + question_tokenized + '\n')

        else:
            for label, question in zip(train_labels, train_questions):
                label = str(label)
                question_tokenized = ' '.join(
                    [tok for tok in nltk.word_tokenize(question.lower()) if tok not in string.punctuation])
                train_file.write(f'__label__{label}' + ' ' + question_tokenized + '\n')

            for label, question in zip(dev_labels, dev_questions):
                label = str(label)
                question_tokenized = ' '.join(
                    [tok for tok in nltk.word_tokenize(question.lower()) if tok not in string.punctuation])
                dev_file.write(f'__label__{label}' + ' ' + question_tokenized + '\n')

        self.model = fasttext.train_supervised(input=train_file.name,
                                               autotuneValidationFile=dev_file.name,
                                               # autotuneDuration=30,
                                               minCountLabel=0,)
        print(self.model.test(dev_file.name, k=3))
        train_file.close()
        dev_file.close()

        self.model.labels_ = [int(lab.replace('__label__', '')) for lab in self.model.labels]

    def save(self, path):
        super(FastTextClassifier, self).save(path)

        self.model.save_model(path)

    def load(self, path):
        return fasttext.load_model(path)

    def predict(self, questions, multiple_predictions=False, k=5):
        super(FastTextClassifier, self).predict(questions)

        labels, probas = [], []

        for question in questions:
            question_tokenized = ' '.join(
                [tok for tok in nltk.word_tokenize(unidecode(question.lower())) if tok not in string.punctuation]).strip()
            label, proba = self.model.predict(question_tokenized, k=k)

            if multiple_predictions:
                labels.append([int(lab.replace('__label__', '')) for lab in label])
                probas.append(proba)

            else:
                labels.append(int(label[0].replace('__label__', '')))
                probas.append(proba[0])

        return labels, probas

    def evaluate(self, true, pred):
        return super(FastTextClassifier, self).evaluate(true=true, pred=pred)
