# Evaluation results

- Date (UTC): 2026-10-09 23:29
- Document: `evaluation/handbook.pdf` (fictional, 17 answerable and 4 unanswerable questions, French and English)
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- LLM: `openai/gpt-oss-120b`
- Chunks: size 500, overlap 50, top_k 3

| Metric | Result |
|---|---|
| Retrieval hit rate (a top-3 chunk contains the evidence) | 88% |
| Page hit rate (a top-3 chunk is on the expected page; looser) | 94% |
| Answer accuracy | 88% |
| False refusals (answerable questions refused) | 12% |
| Correct refusals (unanswerable questions refused) | 100% |
| Source precision (cited pages that are the right ones) | 100% |

Answer accuracy is checked with regular expressions on the expected facts, so a correct answer worded unexpectedly counts as a miss. The sample is small: treat the numbers as indicative, not as a benchmark.

| Id | Question | Retrieval | Page | Answer | Refusal | Cited pages |
|---|---|---|---|---|---|---|
| q01 | In which city is Bluefern Robotics headquartered? | ✓ | ✓ | ✓ | - | 1 |
| q02 | Combien d'employés compte l'entreprise ? | ✗ | ✓ | ✗ | - | none |
| q03 | What are the core working hours? | ✓ | ✓ | ✓ | - | 2 |
| q04 | Combien de jours de télétravail par semaine sont autorisés ? | ✓ | ✓ | ✓ | - | 2 |
| q05 | In which months do Fridays end earlier, and at what time? | ✓ | ✓ | ✓ | - | 2 |
| q06 | How many days of paid annual leave do employees get? | ✓ | ✓ | ✓ | - | 3 |
| q07 | Après combien d'années de service obtient-on des jours de congé supplémentaires ? | ✓ | ✓ | ✓ | - | 3 |
| q08 | After how many consecutive days of sick leave is a medical certificate required? | ✓ | ✓ | ✓ | - | 3 |
| q09 | Quel est le plafond de remboursement d'une nuit d'hôtel en Europe ? | ✓ | ✓ | ✓ | - | 4 |
| q10 | What is the daily meal allowance on business trips? | ✓ | ✓ | ✓ | - | 4 |
| q11 | How long do employees have to submit an expense report? | ✗ | ✗ | ✗ | - | none |
| q12 | Combien de caractères un mot de passe doit-il contenir au minimum ? | ✓ | ✓ | ✓ | - | 5 |
| q13 | Within what time must a phishing email be reported, and to which address? | ✓ | ✓ | ✓ | - | 5 |
| q14 | What is the annual learning budget per employee? | ✓ | ✓ | ✓ | - | 6 |
| q15 | Combien de temps dure le programme de mentorat ? | ✓ | ✓ | ✓ | - | 6 |
| q16 | From what value must gifts be declared? | ✓ | ✓ | ✓ | - | 7 |
| q17 | Quel est le délai de préavis d'un ingénieur après la période d'essai ? | ✓ | ✓ | ✓ | - | 8 |
| q18 | What is the CEO's salary? | - | - | - | ✓ | none |
| q19 | Quelle est la politique de l'entreprise sur les animaux de compagnie au bureau ? | - | - | - | ✓ | none |
| q20 | Does the company offer stock options to its employees? | - | - | - | ✓ | none |
| q21 | Quelle est la capitale du Japon ? | - | - | - | ✓ | none |
