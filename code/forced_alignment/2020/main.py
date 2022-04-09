from start import *


def main():
    aligner = Start('2020data.pickle')
    aligner.read_df()
    aligner.get_alignments()


if __name__ == '__main__':
    main()
