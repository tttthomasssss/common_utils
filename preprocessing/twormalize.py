__author__ = 'thomas'
from . import twokenize


def normalise_urls(tokens): # Twitter URLs should all be of the form http://t.co/<BLAHBLAH>, so this should be safe
	return [t if not t.startswith('http:') else 'HTTPLINK' for t in tokens]


def normalise_numbers(tokens): # Only actual numbers such as 3.223 or 59 at the moment
	return [t if not (isinstance(t, float) or isinstance(t, int)) else 'NUM' for t in tokens]


def tokenise_and_normalise(text):
	return normalise_numbers(normalise_urls(twokenize.tokenize(text)))