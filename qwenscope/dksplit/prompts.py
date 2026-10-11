"""System prompts for the DKSplit segmentation eval.

Only ONE prompt is published on the model card -- the "advanced" one in its
usage snippet, which is also the prompt the r128 adapter was trained with
(`r128_adv` in the card's cross-prompt table). The card's zero-shot row
(63.82 / 67.16) was measured with a *detailed* prompt whose text is not
published, so that exact number cannot be reproduced.

This matters asymmetrically. The card reports that after training, prompt
choice moves Lenient EM by <1pp ("behaviour is baked into the weights"), so the
fine-tuned arm is insensitive to the choice. An untrained model is not: zero-
shot instruction-following on a format-constrained task can swing a lot. So the
headline comparison uses ADVANCED for both arms -- the controlled choice, and
the one that favours the adapter's own training condition -- and the zero-shot
arm is additionally run across all three prompts to bracket that sensitivity.

`detailed` here is a reconstruction, not the original. It is labelled as such
wherever it appears in the results.
"""

# Verbatim from the model card's usage example.
ADVANCED = (
    "You are a domain name segmentation tool. Given a concatenated string "
    "that might be in any language, split it into separate words in the most "
    "accurate way. Do not add or remove any characters. Output ONLY the "
    "segmented result, nothing else."
)

SIMPLE = "Split the following concatenated string into words."

# RECONSTRUCTION of the card's unpublished "detailed_prompt".
DETAILED = (
    "You are an expert multilingual domain name segmentation tool.\n"
    "Given a concatenated domain prefix, split it into its constituent words.\n"
    "\n"
    "Rules:\n"
    "1. Preserve every character exactly. Do not add, remove, or change any "
    "character. The output with all spaces removed must equal the input.\n"
    "2. Insert only spaces. Do not add punctuation, capitalisation, or "
    "explanation.\n"
    "3. The string may be in any language (English, French, German, Spanish, "
    "Portuguese, Turkish, Vietnamese, Chinese, ...). Segment according to the "
    "language you detect.\n"
    "4. Keep brand names, proper nouns, and well-known compounds together "
    "when that is the natural reading.\n"
    "5. If the string is already a single word, return it unchanged.\n"
    "\n"
    "Examples:\n"
    "chatgptlogin -> chatgpt login\n"
    "spotifywrapped -> spotify wrapped\n"
    "mercibeaucoup -> merci beaucoup\n"
    "robertdeniro -> robert de niro\n"
    "brondsy -> brondsy\n"
    "\n"
    "Output ONLY the segmented result, nothing else."
)

PROMPTS = {"advanced": ADVANCED, "simple": SIMPLE, "detailed": DETAILED}
RECONSTRUCTED = {"detailed"}       # not the card's original text
DEFAULT_PROMPT = "advanced"
