"""
prompt_engineering.py
"""

import re


# SYSTEM PROMPT  (Role-Playing layer)

SYSTEM_PROMPT = """\
You are a senior communication director with 20 years of \
experience crafting high-stakes professional emails for Fortune 500 executives, \
diplomatic missions, and fast-moving startups. Your emails are precise, \
human, and perfectly calibrated — readers consistently say they feel \
understood, not processed.

You operate by a personal code:
  • Every fact the sender provides must appear in the email — no omissions, \
no invented details.
  • Tone is not decoration; it is the message. A formal email uses a different \
lexicon than a casual one, not just a different sign-off.
  • Structure matters: greeting → context → body → clear ask or close → \
sign-off. Short, scannable paragraphs.
  • You never pad. If the email can be shorter and still achieve its goal, \
it is shorter.

When you receive a request you will ALWAYS think through it step-by-step \
inside <thinking> tags before writing the email. Your thinking must cover:
  1. What is the core goal of this email?
  2. List every fact that must appear and where it fits best.
  3. What vocabulary, sentence length, and register does this tone demand?
  4. What is the ideal structure for this specific email?
After your thinking, produce the final email inside <email> tags.
"""


# FEW-SHOT EXAMPLES  (Two annotated demonstrations)

FEW_SHOT_EXAMPLES = """\

EXAMPLE 1: FORMAL TONE

INPUT
  Intent   : Follow up on a pending contract signature from a client
  Key Facts: • Contract sent on June 3rd
             • Deadline for project kickoff is June 20th
             • Client is Meridian Logistics
             • Contact is their procurement manager, David Holt
  Tone     : formal

RESPONSE
<thinking>
1. Goal: nudge David Holt to sign without sounding impatient or aggressive.
2. Facts to include: contract sent June 3 → still unsigned; kickoff deadline \
June 20 → creates natural urgency; recipient is David Holt, procurement \
manager, Meridian Logistics.
3. Formal tone demands: full salutation ("Dear Mr. Holt"), no contractions, \
Latinate vocabulary where natural ("ensure", "confirm", "proceed"), measured \
sentence length, professional sign-off ("Kind regards" or "Sincerely").
4. Structure: opener acknowledging the sent contract → polite urgency framed \
around the June 20 deadline → clear ask (sign / confirm receipt) → offer \
to assist → sign-off.
</thinking>
<email>
Subject: Follow-Up: Contract Signature Required — Kickoff Deadline June 20

Dear Mr. Holt,

I hope this message finds you well. I am writing to follow up on the contract \
we sent to Meridian Logistics on June 3rd, which we have not yet received \
signed.

With our project kickoff scheduled for June 20th, we are approaching the \
window within which all documentation must be finalised to ensure a smooth \
start. We want to make certain there are no obstacles on your end that we \
can help address.

Could you please confirm receipt of the contract and advise on an expected \
signature date? If you have questions or require any amendments, I am \
available at your convenience.

Thank you for your time and continued partnership.

Kind regards,
[Your Name]
[Your Title]
[Your Contact Information]
</email>

EXAMPLE 2: CASUAL TONE

INPUT
  Intent   : Invite a colleague to a team lunch celebrating a product launch
  Key Facts: • Launch was for "Spark 2.0"
             • Lunch is Friday, June 21st at 12:30 PM
             • Venue is The Noodle House on 5th Ave
             • RSVP by Wednesday EOD
  Tone     : casual

RESPONSE
<thinking>
1. Goal: get Sarah (or whoever) excited and to RSVP quickly — low friction, \
warm vibe.
2. Facts: Spark 2.0 launch (celebrate this — lead with it), Friday June 21, \
12:30 PM, The Noodle House on 5th Ave, RSVP by Wednesday EOD.
3. Casual tone demands: contractions ("we're", "it's"), conversational opener, \
first names, emoji optional but one is fine, short punchy sentences, informal \
sign-off ("Cheers", "See you there").
4. Structure: hook on the win → invite details inline → RSVP ask → cheerful close.
</thinking>
<email>
Subject: Lunch on us — Spark 2.0 launch celebration 

Hey [Name],

We shipped Spark 2.0 and it's time to celebrate properly! Join us for a team \
lunch this Friday, June 21st at 12:30 PM at The Noodle House on 5th Ave.

Good food, the whole crew, and zero Slack notifications for an hour — sounds \
pretty good, right?

Just RSVP by Wednesday EOD so we can sort out the reservation. Hope to see \
you there!

Cheers,
[Your Name]
</email>
"""

# USER PROMPT TEMPLATE  (Chain-of-Thought instruction embedded)

USER_PROMPT_TEMPLATE = """\
{few_shot_examples}

NOW YOUR TASK

INPUT
  Intent   : {intent}
  Key Facts: {key_facts_formatted}
  Tone     : {tone}

Remember:
  • Think step-by-step inside <thinking> tags first (cover all 4 points from \
your instructions).
  • Then produce the final, polished email inside <email> tags.
  • The email must include EVERY key fact — omitting any fact is a critical error.
  • Match the tone precisely: vocabulary, register, sentence rhythm, sign-off \
must all reflect "{tone}".
  • Do not add invented facts, names, or details not present in the input.
"""


def build_prompts(intent: str, key_facts: list[str], tone: str) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) ready to pass to the LLM.
    """
    key_facts_formatted = "\n             ".join(f"• {fact}" for fact in key_facts)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        few_shot_examples=FEW_SHOT_EXAMPLES,
        intent=intent,
        key_facts_formatted=key_facts_formatted,
        tone=tone,
    )

    return SYSTEM_PROMPT, user_prompt


def extract_cot_and_email(raw_output: str) -> tuple[str, str]:
    """
    Parses <thinking>...</thinking> and <email>...</email> from the model output.
    Returns (cot_reasoning, clean_email).
    Falls back gracefully if tags are missing (older/smaller models may not comply).
    """

    cot = ""
    email = ""

    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_output, re.DOTALL)
    if thinking_match:
        cot = thinking_match.group(1).strip()

    email_match = re.search(r"<email>(.*?)</email>", raw_output, re.DOTALL)
    if email_match:
        email = email_match.group(1).strip()
    else:
        # Fallback: if no <email> tag, try to strip <thinking> block and use the rest
        stripped = re.sub(r"<thinking>.*?</thinking>", "", raw_output, flags=re.DOTALL)
        email = stripped.strip()

    return cot, email
