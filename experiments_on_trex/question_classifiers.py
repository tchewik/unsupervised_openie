import pickle
import tempfile

import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


class QuestionClassifier:
    def __init__(self, *args, **kwargs):
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

        print(sklearn.metrics.classification_report(true, pred, labels=self.model.labels_, digits=4))

    def plot_confusion(self, true, pred):
        assert len(true) == len(pred)
        assert self.model, "Run train() beforehand."

        matrix = sklearn.metrics.confusion_matrix(true, pred,
                                                  labels=self.model.labels_,
                                                  normalize='true')

        plt.figure(figsize=(11, 5))
        sns.heatmap(matrix, cmap='viridis', annot=True, xticklabels=self.model.labels_, yticklabels=self.model.labels_)


class FastTextClassifier(QuestionClassifier):
    def __init__(self, *args, **kwargs):
        QuestionClassifier.__init__(self, *args, **kwargs)

    def train(self, train_questions, train_labels, dev_questions, dev_labels):
        super(FastTextClassifier, self).train(train_questions, train_labels, dev_questions, dev_labels)

        train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="fttrain")
        dev_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="ftdev")

        for label, question in zip(train_labels, train_questions):
            train_file.write(f'__label__{label} {question.lower()}\n')

        for label, question in zip(dev_labels, dev_questions):
            dev_file.write(f'__label__{label} {question.lower()}\n')

        self.model = fasttext.train_supervised(input=train_file.name, autotuneValidationFile=dev_file.name)

        train_file.close()
        dev_file.close()

        self.model.labels_ = [int(lab.replace('__label__', '')) for lab in self.model.labels]

    def save(self, path):
        super(FastTextClassifier, self).save(path)

        self.model.save_model(path)

    def load(self, path):
        return fasttext.load_model(path)

    def predict(self, questions):
        super(FastTextClassifier, self).predict(questions)

        labels, probas = [], []

        for question in questions:
            label, proba = self.model.predict(question)
            labels.append(int(label[0].replace('__label__', '')))

            ## multiple predictions
            # labels.append([int(lab.replace('__label__', '')) for lab in label])
            # probas.append(proba)

            probas.append(proba[0])

        return labels, probas

    def evaluate(self, true, pred):
        super(FastTextClassifier, self).evaluate(true=true, pred=pred)

