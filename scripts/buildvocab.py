# buildvocab.py

import argparse
import cPickle
import operator
from collections import Counter


def processline(line):
    return line.strip().split()


def countword(name):
    fd = open(name, "r")
    vocab = {}

    for line in fd:
        wordlist = processline(line)
        for word in wordlist:
            vocab[word] = 1 if word not in vocab else vocab[word] + 1

    fd.close()

    return vocab


def countchar(name):
    fd = open(name, "r")
    vocab = {}

    for line in fd:
        wordlist = processline(line)
        for word in wordlist:
            word = word.decode("utf-8")
            for char in word:
                char = char.encode("utf-8")
                vocab[char] = 1 if word not in vocab else vocab[char] + 1

    fd.close()

    return vocab


def sortbyfreq(vocab):
    tup = [(item[0], item[1]) for item in vocab.items()]
    tup = sorted(tup, key=operator.itemgetter(0))
    tup = sorted(tup, key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in tup]


def sortbyalpha(vocab):
    tup = sorted(vocab)
    return tup


def save(name, voc):
    newvoc = {}
    for i, v in enumerate(voc):
        newvoc[v] = i

    fd = open(name, "wb")
    cPickle.dump(newvoc, fd, cPickle.HIGHEST_PROTOCOL)
    fd.close()


def parsetokens(s):
    tlist = s.split(";")
    return tlist


def removespecial(vocab, tokens):
    for tok in tokens:
        if tok in vocab:
            del vocab[tok]

    return vocab


def inserttokens(vocab, tokens):
    tokens = tokens[::-1]

    for tok in tokens:
        vocab.insert(0, tok)

    return vocab


def coverage(voc, counts):
    n = 0
    total = sum(counts.itervalues())

    for key in voc:
        if key in counts:
            n += counts[key]

    return float(n) / float(total)


def create_dictionary(name, lim=0):
    global_counter = Counter()
    fd = open(name)

    for line in fd:
        words = line.strip().split()
        global_counter.update(words)

    combined_counter = global_counter

    if lim <= 4:
        lim = len(combined_counter) + 4

    vocab_count = combined_counter.most_common(lim - 4)
    total_counts = sum(combined_counter.values())
    print 100.0 * sum([count for word, count in vocab_count]) / total_counts

    vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}

    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 4

    return vocab


def parseargs():
    msg = "build vocabulary"
    parser = argparse.ArgumentParser(description=msg)

    msg = "corpus"
    parser.add_argument("--corpus", required=True, help=msg)
    msg = "output"
    parser.add_argument("--output", required=True, help=msg)
    msg = "limit"
    parser.add_argument("--limit", default=0, type=int, help=msg)
    msg = "character mode"
    parser.add_argument("--char", action="store_true", help=msg)
    msg = "sort by alphabet"
    parser.add_argument("--alpha", action="store_true", help=msg)
    msg = "add token"
    parser.add_argument("--token", type=str, help=msg)
    msg = "compatible with groundhog"
    parser.add_argument("--groundhog", action="store_true", help=msg)

    return parser.parse_args()


def buildvocab(args):
    if args.char:
        counts = countchar(args.corpus)
    else:
        counts = countword(args.corpus)

    if args.token != None:
        tokens = parsetokens(args.token)
    else:
        tokens = []

    vocab = removespecial(counts, tokens)
    vocab = sortbyfreq(vocab)
    vocab = inserttokens(vocab, tokens)

    if args.limit != 0:
        vocab = vocab[:args.limit]
        print "coverage: ", coverage(vocab, counts) * 100, "%"

    if args.alpha:
        n = len(tokens)
        vocab = vocab[:n] + sorted(vocab[n:])

    save(args.output, vocab)


if __name__ == "__main__":
    args = parseargs()

    if args.groundhog:
        vocab = create_dictionary(args.corpus, args.limit)
        fd = open(args.output, "wb")
        cPickle.dump(vocab, fd, cPickle.HIGHEST_PROTOCOL)
        fd.close()
    else:
        buildvocab(args)
