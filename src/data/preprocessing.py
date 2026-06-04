"""Text preprocessing for tweets."""
import re
from typing import List


# Keywords that may be masked during training (Model C)
ANTISEMITISM_KEYWORDS = [
    "jews", "jewish", "jew", "israel", "israeli", "israelis",
    "zionazi", "zionazis", "zionist", "zionists", "zionism",
    "kike", "kikes",
]


def preprocess_tweet(
    text: str,
    mask_usernames: bool = True,
    remove_urls: bool = True,
    remove_hashtag_symbol: bool = True,
    preserve_emojis: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Clean a tweet for model input.

    Design choices:
    - Usernames → @USER (privacy + reduces vocabulary)
    - URLs → [URL] (not informative for classification)
    - Hashtag # removed but text kept (e.g., #FreePalestine → FreePalestine)
    - Casing preserved (ALL CAPS carries emphasis signal)
    - Emojis preserved (carry sentiment/intent)
    - Retweet markers (RT) preserved (may indicate counterspeech/quoting)
    """
    if not isinstance(text, str):
        return ""

    # Replace usernames
    if mask_usernames:
        text = re.sub(r"@\w+", "@USER", text)

    # Replace URLs
    if remove_urls:
        text = re.sub(r"https?://\S+", "[URL]", text)
        text = re.sub(r"www\.\S+", "[URL]", text)

    # Remove hashtag symbol but keep text
    if remove_hashtag_symbol:
        text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Optional lowercase
    if lowercase:
        text = text.lower()

    return text


def mask_keywords(text: str, keywords: List[str] = None, mask_token: str = "[MASK]") -> str:
    """
    Replace antisemitism-related keywords with mask token.
    Used as training augmentation for Model C to prevent keyword shortcuts.

    Applied at the word level to preserve surrounding context.
    """
    if keywords is None:
        keywords = ANTISEMITISM_KEYWORDS

    keywords_set = set(keywords)
    tokens = text.split()
    masked_tokens = []

    for token in tokens:
        # Strip punctuation for matching, but preserve it in output
        clean = re.sub(r"[^\w]", "", token).lower()
        if clean in keywords_set:
            # Preserve punctuation around the mask
            prefix = re.match(r"^([^\w]*)", token).group(1)
            suffix = re.search(r"([^\w]*)$", token).group(1)
            masked_tokens.append(f"{prefix}{mask_token}{suffix}")
        else:
            masked_tokens.append(token)

    return " ".join(masked_tokens)
