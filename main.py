from classifier import WordsClassifier

classifier = WordsClassifier(r'C:\Users\albin\PycharmProjects\sql_hangman_words\english_dictionary.js')


def main():
    result = classifier.run(total_definitions=1000)
    print(result[2])
    print(len(result[2]))
    # print(help(classifier))


if __name__ == '__main__':
    main()
