# Import
from collections import Counter
from typing import List
import json
# Words Manager
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel


class WordsClassifier:
    counter_definitions: List[Counter] = []
    dictionary: Dictionary = None
    tfidf: TfidfModel = None

    def __init__(self, file_path: str):
        self.words_dict: dict[str, str] = WordsClassifier.get_words(file_path)

    def run(self, *, total_definitions):
        """Run all the project"""
        # Get the tokens definitions list of list
        tokenize_definitions = self.get_tokenize_definitions(total_definitions=total_definitions)
        # Create the Dictionary Instances
        self.set_dictionary(tokens_definitions=tokenize_definitions)
        # Create the Counter
        self.set_counter_tokens(tokens_definitions=tokenize_definitions)
        corpus = self.get_corpus(tokens_definitions=tokenize_definitions)
        self.set_tf_idf_model(corpus=corpus)
        most_common_words = self.get_most_common()
        tf_idf = self.get_tf_idf(corpus=corpus)
        return corpus, most_common_words, tf_idf

    def set_dictionary(self, tokens_definitions: list[list]):
        """Create Dictionary Instances

        Parameters:
        -----------
        tokens_definitions: list[list]:
            This is a list of list that contain the definitions tokenized.

        """
        self.dictionary = Dictionary(tokens_definitions)

    def set_tf_idf_model(self, corpus: list[list[tuple[int, int]]]):
        """Create TfidfModel Instances"""
        self.tfidf = TfidfModel(corpus)

    def get_token_id(self) -> dict[str, int]:
        """Return the ID Words

        Returns:
        -----------
        dict[str, int]

        Example Results:
        -----------
        {'affected': 0, 'also': 1, 'angle': 2, 'anopheles': 3 ... 'sport': 190, 'turn': 191}
        """
        if self.dictionary:
            return self.dictionary.token2id
        else:
            raise 'Please set the dictionary to use this method.'

    def get_corpus(self, tokens_definitions: list[list]) -> list[list[tuple[int, int]]]:
        """Return the gensim corpus of the token list.

        Parameters:
        -----------
        tokens_definitions: list[list]:
            This is a list of list that contain the definitions tokenized.

        Returns:
        -----------
            list[list[tuple[int, int]]]

        Example Results:
        -----------
        [[(0, 1), (1, 1), (2, 2), (3, 1), (4, 1), (5, 3), (6, 1), (7, 1), (8, 1), (9, 1)...
        """
        if self.dictionary:
            return [self.dictionary.doc2bow(doc) for doc in tokens_definitions]
        else:
            raise 'Please set the dictionary to use this method.'

    def get_tf_idf(self, corpus: list[list[tuple[int, int]]], target: int = None):
        """Return the token frequencies by ID.

        corpus: list[list[tuple[int, int]]
            Is the gensim corpus. Is comprehended by a list of tuples,
             inside this are the (ID, COUNTER) -> [(12, 3), (13, 1) ...]

        target: int
            Is the index of the definition.

        """
        if target:
            return [self.tfidf[corpus[1]]]
        return [self.tfidf[tokens] for tokens in corpus]

    @staticmethod
    def get_words(file_path: str) -> dict[str, str]:
        """Get the Json Words English dictionary file as Dictionary

        Parameters:
        -----------
        file_path: str:
            Is the file location system path.

        Returns:
        -----------
        dict[str, str]
        """
        with open(file_path) as file:
            return json.load(file)

    def get_definition_list(self) -> list[str]:
        """Convert all the values [definitions] to a list of sentences"""
        return list(self.words_dict.values())

    def get_tokenize_definitions(self, *, total_definitions: int = 5000, language: str = 'english') -> list[list]:
        """Tokenize all the definitions of the [words_dict] attribute.

        Parameters:
        -----------
        total_definitions: int
            Is the total number of definitions results expected.
            (Default = 5000)
        language: str
            Is the language or idiom of the stop words dictionary. This helps to avoid using
            not important words.
            Default( = 'english')

        Returns:
        list[list]
        """
        # 1) Extract all the definition list
        words_definition: list = self.get_definition_list()
        # 2) Tokenize the definitions words
        tokenize_definitions: list[list] = [word_tokenize(definition.lower()) for definition in
                                            words_definition[:total_definitions+1]]
        cleaned_tokenize_definitions = []
        # Clean all the definitions: No number and stopwords (are, the, etc.)
        for definition in tokenize_definitions:
            clean_words: list[str] = [word for word in definition
                                      if word.isalpha() and word not in stopwords.words(language)]
            cleaned_tokenize_definitions.append(clean_words)
        return cleaned_tokenize_definitions

    def set_counter_tokens(self, tokens_definitions: list[list]) -> None:
        """Return a list of Counter tokens

        Parameters:
        -----------
        tokens_definitions: list[list]:
            This is a list of list that contain the definitions tokenized.

        """
        self.counter_definitions = [Counter(definition) for definition in tokens_definitions]

    def get_most_common(self, *, target: int = None, total_results: int = 10) -> list[list]:
        """Return a list of tuples with the count of the  most common word used in each definition.

        Parameters:
        -----------
        target: int:
            Is the index in the list of [counter_definitions] to subscript. By Default is [None] to count the
            most common words in all the tokens of each definition.
            (Default = None)

        total_results: int:
            This is the number of result should return by definition.
            (Default = 10)

        Returns:
        -----------
        list[list]
        """
        # Check if exist counter definition
        if self.counter_definitions:
            if not target:
                return [common.most_common(total_results) for common in self.counter_definitions]
            else:
                # Check if is a valid integer to select the index list.
                if isinstance(target, int) and target >= -1:
                    # The return is wrap [] just to make all the returns of equal type.
                    return [self.counter_definitions[target].most_common(total_results)]
        else:
            raise 'Pleas, setup the [counter_definitions] attribute using the method -> set_counter_tokens'
