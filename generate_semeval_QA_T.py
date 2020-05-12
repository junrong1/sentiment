import os
import xml.etree.ElementTree as ET

data_dir = '../data/semeval2014/'

dir_path = data_dir+'bert-pair/'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

labels = ['positive', 'neutral', 'negative', 'conflict', 'none']

with open(dir_path + "train_QA_T.csv", "w", encoding="utf-8") as g,\
    open(dir_path + "train_QA_EXPT.csv", "r", encoding="utf-8") as f:
    s = f.readline().strip()
    while s:
        tmp = s.split("\t")
        for label in labels:
            if label == tmp[1]:
                g.write(tmp[0] + "\t" +
                        "1" + "\t" +  # match or not
                        "The description of " + tmp[2] + " shows " + label + " polarity." + "\t" +
                        tmp[-1] + "\n")
            else:
                g.write(tmp[0] + "\t" +
                        "0" + "\t" +  # match or not
                        "The description of " + tmp[2] + " shows " + label + " polarity." + "\t" +
                        tmp[-1] + "\n")
        s = f.readline().strip()


with open(dir_path + "test_QA_T.csv", "w", encoding="utf-8") as g,\
    open(dir_path + "test_QA_EXPT.csv", "r", encoding="utf-8") as f:
    s = f.readline().strip()
    while s:
        tmp = s.split("\t")
        for label in labels:
            if label == tmp[1]:
                g.write(tmp[0] + "\t" +
                        "1" + "\t" +  # match or not
                        "The description of " + tmp[2] + " shows " + label + " polarity." + "\t" +
                        tmp[-1] + "\n")
            else:
                g.write(tmp[0] + "\t" +
                        "0" + "\t" +  # match or not
                        "The description of " + tmp[2] + " shows " + label + " polarity." + "\t" +
                        tmp[-1] + "\n")
        s = f.readline().strip()