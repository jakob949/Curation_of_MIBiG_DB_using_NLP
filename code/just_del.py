import pickle
from pybliometrics.scopus import AbstractRetrieval

with open('creation_of_dataset/abstracts_bad.pickle', 'rb') as f:
    abstracts_bad = pickle.load(f)


with open('hard_dataset.txt', 'w') as f:
    for i, item in enumerate(abstracts_bad):
        try:
            ID = abstracts_bad[item][0]
            abstract = abstracts_bad[item][1].replace('\n', ' ').replace('\t', ' ')
            ab = AbstractRetrieval(ID)
            title = ab.title.replace('\n', ' ').replace('\t', ' ')
            f.write(f"Title: {title} Abstract: {abstract}\t2\n")
        except:
            print('something went wrong with the title or ID')


print(i)