from start import *


def main():
    aligner = Start('data.csv')
    aligner.read_df()
    aligner.get_alignments()


if __name__ == '__main__':
    main()
