from Translator import GoogleTranslator
from GPT3 import merge_text_with_two_shot, invoke_model


def main(hebrew_text):
    T = GoogleTranslator()
    english_text = T.translate(hebrew_text)
    prompt = merge_text_with_two_shot(english_text)
    _, out = invoke_model(prompt)
    print(f"Hebrew Text:\n {hebrew_text}.\n"
          f"#################################################\n"
          f"English Translation:\n {english_text}.\n"
          f"#################################################\n"
          f"Model Input:\n {prompt}\n"
          f"#################################################\n"
          f"Model Output:\n {out}.")


if __name__ == '__main__':
    txt = """
להשכרה חדר גדול ושקט בדירת 4 שותפים .בחדר חלון גדול,מיטה עם מזרן , שידה , ארון קיר ומזגן . רחוב פינקס 3 ,קומה 1, באזור הצפון הישן ליד אבן גבירול האזור הכי מבוקש בת"א ! קרוב לים לפארק הירקון , לאזור בילויים , לתחבורה ציבורית . 
2 שירותים ומקלחות ומטבח חדש ומאובזר .
החדר הכי שקט בבית !
2950 שח ותשלומים זולים
מתפנה ב 23.9 
שווה לראות ! מפרטי
0508667848 אורנה 
0505230404 רינה
            """
    main(txt)
