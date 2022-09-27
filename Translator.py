import translators as ts

"""
More info here:
https://pypi.org/project/translate-api/
https://towardsdatascience.com/language-translation-using-python-bd8020772ccc
"""

HEBREW_CODE = 'iw'
ENGLISH_CODE = 'en'


class GoogleTranslator:
    def __init__(self):
        self._translator = ts.google

    def translate(self, text_input: str):
        return self._translator(text_input, from_language=HEBREW_CODE, to_language=ENGLISH_CODE)
