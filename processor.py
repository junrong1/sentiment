# encoding utf-8
"""Processor for different experiment"""

import csv
import pandas as pd
import os
import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Semeval_QA_EXPT_Processor(DataProcessor):
    """Processor for semeval dataset"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_QA_EXPT.csv"), header=None, sep="\t").values
        return self._create_example(train_data, "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_QA_EXPT.csv"), header=None, sep="\t").values
        return self._create_example(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_QA_EXPT.csv"), header=None, sep="\t").values
        return self._create_example(test_data, "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = None
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Semeval_QA_T_Processor(DataProcessor):
    """Processor for semeval dataset"""

    def get_train_examples(self, data_dir):
        train_data = pd.read_csv(os.path.join(data_dir, "train_QA_T.csv"), header=None, sep="\t").values
        return self._create_example(train_data, "train")

    def get_dev_examples(self, data_dir):
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_QA_T.csv"), header=None, sep="\t").values
        return self._create_example(dev_data, "dev")

    def get_test_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "test_QA_T.csv"), header=None, sep="\t").values
        return self._create_example(test_data, "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Travel_exp_data(DataProcessor):
    """Processor for travel experience dataset"""

    def get_train_examples(self, data_dir):
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t", encoding="iso8859_15").values
        return self._create_example(train_data, "train")

    def get_dev_examples(self, data_dir):
        dev_data = pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None, sep="\t", encoding="iso8859_15").values
        return self._create_example(dev_data, "dev")

    def get_test_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t", encoding="iso8859_15").values
        return self._create_example(test_data, "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["very satisfied", "fairly satisfied", "neither satisfied nor dissatisfied", "fairly dissatisfied",
                "very dissatisfied"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            _, label, comment = line[0].split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(comment.strip(",")))
            text_b = None
            label = tokenization.convert_to_unicode(str(label))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Semeval_single_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
