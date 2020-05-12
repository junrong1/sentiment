import os
import xml.etree.ElementTree as ET

data_dir = '../data/semeval2014/'
dir_path = data_dir+'bert-pair/'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

data_tree_test = ET.parse(data_dir + "Restaurants_Test_Gold.xml")
data_tree_test_root = data_tree_test.getroot()

with open(dir_path+"test_QA_EXPT.csv", "w", encoding="utf-8") as g:
    for i in range(len(data_tree_test_root)):
        if "term" in data_tree_test_root[i][1][0].attrib.keys():
            for j in range(len(data_tree_test_root[i][1])):
                g.write(data_tree_test_root[i].attrib["id"] + "\t" +
                        data_tree_test_root[i][1][j].attrib["polarity"] + "\t" +
                        data_tree_test_root[i][1][j].attrib["term"] + "\t" +
                        data_tree_test_root[i][0].text + "\n")
        else:
            g.write(data_tree_test_root[i].attrib["id"] + "\t"
                    + "none" + "\t" +
                    "none" + "\t" +
                    data_tree_test_root[i][0].text + "\n")


data_tree_train = ET.parse(data_dir + "Restaurants_Train.xml")
data_tree_train_root = data_tree_train.getroot()

with open(dir_path+"train_QA_EXPT.csv", "w", encoding="utf-8") as f:
    for i in range(len(data_tree_train_root)):
        if "term" in data_tree_train_root[i][1][0].attrib.keys():
            for j in range(len(data_tree_train_root[i][1])):
                f.write(data_tree_train_root[i].attrib["id"] + "\t" +
                        data_tree_train_root[i][1][j].attrib["polarity"] + "\t" +
                        data_tree_train_root[i][1][j].attrib["term"] + "\t" +
                        data_tree_train_root[i][0].text + "\n")
        else:
            f.write(data_tree_train_root[i].attrib["id"] + "\t"
                    + "none" + "\t" +
                    "none" + "\t" +
                    data_tree_train_root[i][0].text + "\n")