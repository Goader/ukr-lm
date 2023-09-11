from argparse import ArgumentParser

import conllu


def extract_ids(conllu):
    document_ids = set()
    paragraph_ids = set()
    sentence_ids = set()

    document_id = None
    paragraph_id = None
    for sentence in conllu:
        document_id = sentence.metadata.get("newdoc id", document_id)
        paragraph_id = sentence.metadata.get("newpar id", paragraph_id)

        document_ids.add(document_id)
        paragraph_ids.add(paragraph_id)
        sentence_ids.add(sentence.metadata['sent_id'])
    return document_ids, paragraph_ids, sentence_ids


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train', help='input CONLLU file')
    parser.add_argument('dev', help='input CONLLU file')
    parser.add_argument('test', help='input CONLLU file')
    args = parser.parse_args()

    with open(args.train) as train, open(args.dev) as dev, open(args.test) as test:
        train = conllu.parse(train.read())
        dev = conllu.parse(dev.read())
        test = conllu.parse(test.read())

    train_doc_ids, train_par_ids, train_sent_ids = extract_ids(train)
    dev_doc_ids, dev_par_ids, dev_sent_ids = extract_ids(dev)
    test_doc_ids, test_par_ids, test_sent_ids = extract_ids(test)

    # print overlap if it exists
    print('train_doc_ids & dev_doc_ids', train_doc_ids & dev_doc_ids)
    print('train_doc_ids & test_doc_ids', train_doc_ids & test_doc_ids)
    print('dev_doc_ids & test_doc_ids', dev_doc_ids & test_doc_ids)
    print()
    print('train_par_ids & dev_par_ids', train_par_ids & dev_par_ids)
    print('train_par_ids & test_par_ids', train_par_ids & test_par_ids)
    print('dev_par_ids & test_par_ids', dev_par_ids & test_par_ids)
    print()
    print('train_sent_ids & dev_sent_ids', train_sent_ids & dev_sent_ids)
    print('train_sent_ids & test_sent_ids', train_sent_ids & test_sent_ids)
    print('dev_sent_ids & test_sent_ids', dev_sent_ids & test_sent_ids)

