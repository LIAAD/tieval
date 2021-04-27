from text2timeline import read_xml as rxml

from sklearn.metrics import accuracy_score

TRAIN_PATH = r'C:\Users\hmoso\OneDrive\Data\TimeBankPT\train'
TEST_PATH = r'C:\Users\hmoso\OneDrive\Data\TimeBankPT\test'


"""Read datasets."""
train = rxml.get_tags(TRAIN_PATH, rxml.get_tlinks)
test = rxml.get_tags(TEST_PATH, rxml.get_tlinks)


"""Majority class baseline."""
# Train.
task_A_class = train.loc[train.task == 'A', 'relType'].value_counts().index[0]
task_B_class = train.loc[train.task == 'B', 'relType'].value_counts().index[0]
task_C_class = train.loc[train.task == 'C', 'relType'].value_counts().index[0]

# Predict.
test.loc[test.task == 'A', 'pred'] = task_A_class
test.loc[test.task == 'B', 'pred'] = task_B_class
test.loc[test.task == 'C', 'pred'] = task_C_class

# Evaluate.
tasks = ['A', 'B', 'C']
t_acc = [accuracy_score(test[test.task == task].relType, test[test.task == task].pred) for task in tasks]

for task, acc in zip(tasks, t_acc):
    print(f"Accuracy in task {task}: {acc * 100:.3} %")

