---
name: plain-report-writing
description: Revise or write report prose (Word reports, READMEs, slides text) so it avoids the patterns listed in Wikipedia's "Signs of AI writing" page. Use when writing or editing the classification or regression project reports, or any write-up in this repo.
---

# Plain report writing

Source: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing (WikiProject AI Cleanup advice page).
The page is descriptive. Its main point: the signs are symptoms, and the real problems are unverifiable,
inaccurate or vague claims. Fix the claim, not just the wording. Never change a number to make a sentence read better.

## What to check, in order

1. **Markdown leaked into the target format.** In a .docx, literal `*word*`, `**word**`, backticks, `#` headings
   or `---` dividers are the strongest single sign. Convert them to real italic, bold, or a monospace font (Consolas).
2. **Leftover chatbot text and placeholders.** "I hope this helps", "Let me know", "Here is a…", `[Your Name]`,
   `2025-XX-XX`, example.com URLs. Fill them in or delete them. (Cover-page fields the author fills in by hand are fine, but list them in the hand-off.)
3. **Inline-header lists.** Bullets that each start with a bold label and a colon or period ("**Fairness.** I have not…").
   Rewrite each item as a full sentence with no bold label, or turn the list into a table or paragraph.
4. **Negative parallelisms and corrections nobody asked for.** "not just X but Y", "X, not Y", "Y rather than X",
   "is not magic", "is not the same as". State what the thing is.
5. **Significance padding.** "plays a crucial/pivotal role", "highlights the importance", "underscores", "a testament to",
   "setting the stage", trailing "-ing" clauses ("…, highlighting its robustness"). Replace with the result and its number, or delete.
6. **Overused vocabulary, in clusters.** Additionally (opening a sentence), align with, boasts, bolster, crucial, delve,
   emphasize, enduring, enhance, foster, garner, highlight (verb), interplay, intricate, landscape (abstract), meticulous,
   pivotal, robust, showcase, tapestry, testament, underscore, valuable, vibrant; newer: leverage, multifaceted, prioritize,
   seamless, dependable, quietly, steady. One or two are fine in technical writing. Many together are the sign.
7. **Avoiding "is/are/has".** "serves as", "stands as", "functions as", "boasts", "features". Write "is" or "has".
8. **Vague attribution.** "Experts argue", "studies show", "several sources". Name the source (author, year) or drop the claim.
9. **Rule of three.** Lists of exactly three adjectives or phrases added for rhythm. Keep only the items the evidence supports.
10. **Formula endings.** "Despite these challenges…", "Future outlook", "In conclusion/Overall/In summary" restating the section.
    End on the finding or the next concrete step.
11. **Elegant variation.** Swapping synonyms for the same thing. Use one name per metric, model, dataset and feature throughout.
12. **Punctuation.** Few or no spaced em dashes ( — ). Use commas, colons or parentheses. Do not swap every dash for a period either.
13. **Headings.** Sentence case, one title, no emoji, no "X and Y" filler headings like "Challenges and Recognition".
14. **Citations.** Every DOI resolves to the cited paper, no `utm_source=` parameters, every reference is cited in the text,
    page numbers given for books.

## What NOT to change

The page warns that these appear just as often in human writing and are not evidence of AI use:
- Correct grammar, formal register, transition words on their own ("However", "Therefore").
- Plain "is/has" sentences, "very", "perhaps", "in order to", superlatives backed by numbers.
- First-person explanations of the author's own decisions. These are the most human part of a report. Keep them.
- Curly quotes in Word (that is the editor's default).

## Workflow for .docx reports

1. Dump paragraphs and runs with python-docx (`pip install python-docx`), including bold/italic per run and table cells.
2. Make a backup copy of the original next to it before writing.
3. Fix Markdown leaks by splitting runs and setting `italic` / font `Consolas` on the code spans. Copy the run's `rPr` so the size and colour stay the same.
4. Rewrite flagged sentences one at a time. Re-check every number against the original sentence.
5. Leave headings that appear in a static table of contents alone, or update both.
6. Re-dump the result and grep for `*`, backticks, ` — `, and the vocabulary list above.
7. Report what changed, and list the placeholders the author still has to fill.
