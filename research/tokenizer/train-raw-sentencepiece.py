from argparse import ArgumentParser
import time

import sentencepiece as spm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='text file to train tokenizer')
    parser.add_argument('--output', type=str, required=True, help='output directory path')
    parser.add_argument('--vocab-size', type=int, default=32000, help='vocab size')
    parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram'], help='model type')
    args = parser.parse_args()
    
    t1 = time.time()
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.output,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        train_extremely_large_corpus=True
    )
    print(f"Training time: {time.time() - t1:.2f} sec")
