nltk.download('punkt')
import nltk
from nltk.corpus import treebank
from nltk import PCFG, ViterbiParser

# Load the treebank dataset
nltk.download('treebank')
corpus = treebank.parsed_sents()

# Train a PCFG parser
productions = []
for tree in corpus:
    productions += tree.productions()
S = nltk.Nonterminal('S')
grammar = nltk.induce_pcfg(S, productions)

# Initialize the parser with the trained grammar
parser = ViterbiParser(grammar)

# Tokenize and parse a sentence
sentences = ["this is a beautiful"]

# Prepare gold standard parse trees for the sentences
gold_standard_trees = list(treebank.parsed_sents()[:len(sentences)])

# Initialize counters
true_positives = 0
false_positives = 0
false_negatives = 0

# Evaluate each sentence
for sentence, gold_tree in zip(sentences, gold_standard_trees):
    tokens = nltk.word_tokenize(sentence)
    parsed_trees = list(parser.parse(tokens))
    if parsed_trees:
        # If the parser produced a parse tree, consider the first one
        parsed_tree = parsed_trees[0]
        # Compare each production in the parsed tree with the gold standard tree
        for production in gold_tree.productions():
            if production in parsed_tree.productions():
                true_positives += 1
            else:
                false_negatives += 1
        for production in parsed_tree.productions():
            if production not in gold_tree.productions():
                false_positives += 1

# Compute precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Compute accuracy
accuracy = true_positives / (true_positives + false_positives + false_negatives)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)
