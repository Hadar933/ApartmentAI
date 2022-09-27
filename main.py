from Translator import GoogleTranslator
from LanguageModel import merge_text_with_two_shot, invoke_model


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
        -מראה מחר- 
דירת 2 חדרים , כ-35 מ' . קומה 2 . 
ברחוב הרצל במיקום מדהים שקרוב להכל! 
מחיר 3700 כולל ארנונה,מים וועד בית. 
תאריך כניסה 27.10 -28 
דירה שמורה ומתוחזקת 
לתיאום בפרטי !
            """
    main(txt)
