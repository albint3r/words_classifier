from classifier import WordsClassifier

classifier = WordsClassifier(r'C:\Users\albin\PycharmProjects\sql_hangman_words\english_dictionary.js')


def main():
    result = classifier.run()
    print(result[1])


if __name__ == '__main__':
    main()
