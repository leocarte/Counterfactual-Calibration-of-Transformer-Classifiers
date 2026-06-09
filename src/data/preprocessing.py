"""Text preprocessing for tweets used throughout CCI v2."""
from __future__ import annotations

import re


# Keywords that may be masked during training (Model C).
ANTISEMITISM_KEYWORDS: tuple[str, ...] = (
    "jews", "jewish", "jew", "israel", "israeli", "israelis",
    "zionazi", "zionazis", "zionist", "zionists", "zionism",
    "kike", "kikes",
)


def preprocess_tweet(
    text: str,
    mask_usernames: bool = True,
    remove_urls: bool = True,
    remove_hashtag_symbol: bool = True,
    preserve_emojis: bool = True,
    lowercase: bool = False,
) -> str:
    """Clean a tweet for model input.

    Parameters
    ----------
    text : str
        Raw tweet text.
    mask_usernames : bool, default True
        Replace ``@handle`` occurrences with ``@USER`` for privacy and
        vocabulary reduction.
    remove_urls : bool, default True
        Replace ``http(s)://...`` and ``www.\\S+`` with ``[URL]``.
    remove_hashtag_symbol : bool, default True
        Strip the leading ``#`` from hashtags while keeping the body text
        (e.g. ``#FreePalestine`` → ``FreePalestine``).
    preserve_emojis : bool, default True
        Reserved for future emoji-specific handling; currently a no-op
        because emojis are preserved by default.
    lowercase : bool, default False
        Lowercase the result. Disabled by default so that
        ALL-CAPS emphasis carries through to the model.

    Returns
    -------
    str
        Preprocessed text. Returns ``""`` for non-string inputs.
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


def mask_keywords(
    text: str,
    keywords: list[str] | tuple[str, ...] | None = None,
    mask_token: str = "[MASK]",
) -> str:
    """Replace identity keywords with ``mask_token`` at word boundaries.

    Used as a training augmentation for Model C to suppress the keyword
    shortcut documented by Dixon et al. (2018). Replacement is applied
    case-insensitively but only at full-word matches, so surrounding
    context is preserved.

    Parameters
    ----------
    text : str
        Input text.
    keywords : sequence of str or None, default None
        Tokens to mask. ``None`` uses :data:`ANTISEMITISM_KEYWORDS`.
    mask_token : str, default ``"[MASK]"``
        Replacement string.

    Returns
    -------
    str
        Masked text.
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
