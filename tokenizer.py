from typing import Iterator

import hazm
import emoji

_normalizer = hazm.Normalizer()
_lemmatizer = hazm.Lemmatizer()

_stopwords = hazm.stopwords_list()
_punctuations = [')', '(', '>', '<', "؛",
                 "،", '{', '}', "؟", ':',
                 "–", '»', '"', '«', '[',
                 ']', '"', '+', '=', '?',
                 '%', '&', '*', '$', '#',
                 '؟', '*', '.', '_', '!',
                 '/', '\\', '|', '^', '-',
                 "'", '@', ',', '~', '`']
_numbers = [
    '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',  # Arabic
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',  # Urdu
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Latin
]


def _filter_stopwords(token: str):
    """
    Check if a token is a stopword.

    Args:
        token (str): The token to check.

    Returns:
        bool: True if the token is not a stopword, False otherwise.
    """
    return token not in _stopwords


def _translate(token, filter_list):
    """
    Translate characters in the token based on a filter list.

    Args:
        token (str): The token to translate.
        filter_list (list): A list of characters to remove from the token.

    Returns:
        str: The translated token with specified characters removed.
    """
    d = {ord(c): None for c in filter_list}
    return token.translate(d)


def _strip_punctuations(token):
    """
    Remove punctuation characters from a token.

    Args:
        token (str): The token to process.

    Returns:
        str: The token without punctuation characters.
    """
    return _translate(token, _punctuations)


def _strip_numbers(token):
    """
    Remove numeric characters from a token.

    Args:
        token (str): The token to process.

    Returns:
        str: The token without numeric characters.
    """
    return _translate(token, _numbers)


def _strip_emoji(text):
    """
    Remove emoji characters from a text.

    Args:
        text (str): The text to process.

    Returns:
        str: The text without emoji characters.
    """
    return emoji.replace_emoji(text)


def _lemmatize(token):
    """
    Lemmatize a token using hazm.

    Args:
        token (str): The token to lemmatize.

    Returns:
        str: The lemmatized token.
    """
    return _lemmatizer.lemmatize(token).strip()


def tokenize(
        content,
        normalize=True,
        lemmatize=True,
        filter_stopwords=True,
        strip_punctuations=True,
        strip_emoji=True,
        strip_numbers=False,
) -> Iterator[str]:
    """
    Tokenize content with various preprocessing options.

    Args:
        content (str): The text content to tokenize.
        normalize (bool): Whether to normalize the content (default: True).
        lemmatize (bool): Whether to lemmatize the tokens (default: True).
        filter_stopwords (bool): Whether to filter out stopwords (default: True).
        strip_punctuations (bool): Whether to remove punctuation characters (default: True).
        strip_emoji (bool): Whether to remove emoji characters (default: True).
        strip_numbers (bool): Whether to remove numeric characters (default: False).

    Returns:
        Iterator[str]: An iterator over the processed tokens.

    Example:
        >>> list(tokenize("این یک متن آزمایشی است."))
        ['این', 'یک', 'متن', 'آزمایشی', 'است']
    """

    if normalize:
        content = _normalizer.normalize(content)
    tokens = hazm.word_tokenize(content)

    if strip_punctuations:
        tokens = map(_strip_punctuations, tokens)
    if strip_emoji:
        tokens = map(_strip_emoji, tokens)
    if strip_numbers:
        tokens = map(_strip_numbers, tokens)
    if filter_stopwords:
        tokens = filter(_filter_stopwords, tokens)
    if lemmatize:
        tokens = map(_lemmatize, tokens)

    tokens = filter(lambda t: len(t) > 0, tokens)
    return tokens
