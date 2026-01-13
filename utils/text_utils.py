import re


def split_cn_en(text: str):
    pattern = r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+"
    return re.findall(pattern, text)


def check_en(text: str):
    if not text:
        return False

    symbol_pattern = re.compile(
        r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
        r"\u2000-\u206F"
        r"\u3000-\u303F"
        r"\uFF00-\uFFEF]"
    )

    for char in reversed(text):
        if char.isdigit() or symbol_pattern.match(char):
            continue
        if char >= "\u4e00" and char <= "\u9fff":  # is chinese
            return False
        else:
            return True

    return True
