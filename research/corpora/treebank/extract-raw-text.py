from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input', nargs='+', help='input CONLLU file')
    parser.add_argument('output', help='output TXT file')
    args = parser.parse_args()

    with open(args.output, 'w') as out:
        for path in args.input:
            with open(path) as f:
                for line in f:
                    if line.startswith('# text = '):
                        out.write(line.removeprefix('# text = '))
            out.write('\n')
