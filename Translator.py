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


if __name__ == '__main__':
    T = GoogleTranslator()
    txt = """  דירת 4 חדרים מפרטי בשכונת תל גנים בגבעתיים. קומה 3 עם מעלית וחניה + 
    הרבה חניה ברחוב. מזגן מיני מרכזי ומזגן נפרד בכל חדר. 3 כיווני אויר עם נוף ירוק מכל חלון. שכ״ד 8500 ש״ח
כניסה בתחילת נובמבר. פרטים נוספים בפרטי    
    """
    print(T.translate(txt))
