import pdb
import ftfy
f = open("./lexicon.txt.small", "r")
list_words = []
for line in f:
    word = line.strip().split(" ")[0]
    list_words.append(ftfy.fix_text(word.strip()))

with open('./new_lexicon.txt', 'w') as f:
    for item in list_words:
        f.write("%s\n" % item)

