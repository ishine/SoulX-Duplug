def check_backchannel(s: str) -> bool:
    """
    判断文本是否为空、无意义，或属于中英文 backchannel （无需回复）
    """

    s = s.strip().replace(",", "").replace(".", "").replace("，", "").replace("。", "")

    # 完全空白
    if not s:
        return True

    # === 中英文 backchannel ===
    BACKCHANNEL = {
        # 中文
        "嗯",
        "嗯嗯",
        "啊",
        "啊啊",
        "哦",
        "哦哦",
        "噢",
        "噢噢",
        "哎",
        "好",
        "好的",
        "好啊",
        "好吧",
        "好嘞",
        "对",
        "对对",
        "是",
        "是的",
        "行",
        "可以",
        "嗯哼",
        "哼",
        "嘿",
        # 英文
        "ok",
        "okay",
        "yeah",
        "yep",
        "yup",
        "right",
        "sure",
        "uh-huh",
        "uh huh",
        "uhhuh",
        "hmm",
        "hmmm",
        "mm",
        "mmm",
        "alright",
        "got it",
        "i see",
        "roger",
        "k",
        "kk",
        "y",
        "yep",
        "yes",
    }

    # 统一为小写判断英文
    if s.lower() in BACKCHANNEL:
        return True

    # 有些 backchannel 在 ASR 会被识别成长词，这里处理短句模式
    if len(s) <= 3 and any(
        key in s.lower() for key in ["ok", "mm", "hmm", "uh", "yes", "yeah"]
    ):
        return True
    if len(s) <= 2 and any(ch in s for ch in ["嗯", "啊", "哦"]):
        return True

    # 纯标点
    if all(ch in ".,!?！？。；;…" for ch in s):
        return True

    return False


def remove_leading_backchannel(text: str) -> str:
    """
    从前向后扫描字符串，移除开头的语气词（如“嗯”、“啊”）及标点符号，直到遇到第一个非语气词字符。
    """
    # 常见的中文语气词单字
    backchannel_chars = {"嗯", "啊", "哦", "噢", "呃", "哎", "哼", "嘿"}
    # 常见的标点符号和空白
    punctuation_chars = {
        " ",
        ",",
        ".",
        "?",
        "!",
        "，",
        "。",
        "？",
        "！",
        "、",
        "；",
        ";",
        "…",
        ":",
        "：",
    }

    skip_chars = backchannel_chars.union(punctuation_chars)

    for i, char in enumerate(text):
        if char not in skip_chars:
            return text[i:]

    return ""
