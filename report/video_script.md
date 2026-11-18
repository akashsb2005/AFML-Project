# Presentation video script (target 6:30, hard limit 7:00)

Record as a screen share of `AFML_Final_Presentation.pptx` in slideshow mode, using Zoom, Google Meet or QuickTime ("New Screen Recording"), with each speaker on their own slides.
The lines are suggestions. Say them in your own words, and don't read them word for word.
Timings add up to about 6:30, which leaves 30 seconds of slack.

| Time | Slides | Speaker |
|---|---|---|
| 0:00–0:50 | 1–3 | Samvith |
| 0:50–1:50 | 4–5 | Haneesh |
| 1:50–3:00 | 6–7 | Akash |
| 3:00–4:20 | 8 + demo | Haneesh |
| 4:20–5:20 | 9–11 | Akash |
| 5:20–6:30 | 12–14 | Samvith |

---

**Slide 1, title (Samvith, 15 s)**
"Hi, we're Akash, Haneesh and Samvith. Our project is Efficient Training and Instruction Fine-Tuning of Small Language Models."

**Slide 2, problem (Samvith, 20 s)**
"We asked three questions. Can a 5-million-parameter Transformer, trained from random weights on one free GPU, learn English?
Can instruction fine-tuning make it follow instructions? And does linear attention, which is O(T) instead of O(T²), actually
make training cheaper at this size?"

**Slide 3, uniqueness (Samvith, 15 s)**
"We built every stage ourselves, with no pre-trained weights. The two attention types differ in one config line, so the comparison
is controlled. We also evaluated beyond loss, because lower loss turned out not to mean better answers."

**Slide 4, dataset (Haneesh, 30 s)**
"Pre-training uses a 50,000-story sample of TinyStories, simple children's stories. Fine-tuning uses Databricks Dolly 15K,
fifteen thousand human-written instruction and response pairs. We removed 203 duplicate prompts so no prompt appears in two splits,
then split 90/5/5. We trained our own 8,000-token BPE tokenizer on the training stories only."

**Slide 5, design (Haneesh, 30 s)**
"Here's the pipeline. Stories go through the tokenizer into pre-training. Dolly is formatted as instruction, context, response,
and during fine-tuning the loss is computed only on the response tokens. Each model is then evaluated four ways:
perplexity, response loss with a blind rubric, speed and memory, and failure analysis."

**Slide 6, results (Akash, 40 s)**
"The model is a 4-layer decoder with 4 heads and 5.27 million parameters. After 10,000 steps on a T4, test perplexity is 8.5,
against 8,000 for random guessing, and validation loss was still falling, so there's no overfitting.
Linear attention reached 8.78, slightly worse, and took 2.3 times as long to train."

**Slide 7, efficiency (Akash, 30 s)**
"Linear attention was slower and used more memory at every length we tried: about 0.4 times the throughput and over twice
the memory. But the curve behaves as theory says. The gap shrinks from 2.5× at 128 tokens to 1.08× at 2,048, because
linear attention keeps a 64×64 state per token, which is more data than softmax until about 4,000 tokens."

**Slide 8, before vs after fine-tuning (Haneesh, 30 s)**
"Fine-tuning cut response loss on the Dolly test set from 6.38 to 3.93. Before, the model answered almost every question
with a children's story. After, it answers on topic and in the right format, but the content is usually wrong, and
with greedy decoding it often repeats itself."

**Demo (Haneesh, 50 s)**
Show the Colab demo cell running the three commands from `report/presentation_demo_plan.md`, or the saved outputs.
"Here's the pre-trained model continuing 'Once upon a time', which is fluent. Now the same question to both models:
'Identify which animal is a bird or a fish: sparrow, salmon.' The pre-trained model tells a story. The fine-tuned one says 'Redalo is a fish'.
Right format, wrong fact, which is exactly what our numbers show."

**Slide 9, done vs remaining (Akash, 15 s)**
"Everything we planned is done: data, both models, pre-training, fine-tuning, evaluation and benchmarks. Longer training and
a human second rater are future work."

**Slide 10, contributions (Akash, 25 s)**
"I built the model, both attention types and the training engine, and ran pre-training. Haneesh built the data pipeline,
tokenizer and fine-tuning. Samvith built the evaluation, benchmarks and report."

**Slide 11, quantity and quality (Akash, 20 s)**
"Every component runs end to end from one script in about 1.5 hours on a free Colab GPU, and all 67 automated tests pass."

**Slide 12, learnings (Samvith, 30 s)**
"Three lessons. Lower loss doesn't mean better answers; decoding matters. O(T) isn't automatically faster, because constants and
memory traffic dominate at small sizes. And fine-tuning teaches format, not knowledge."

**Slide 13, unresolved (Samvith, 25 s)**
"The fine-tuned model still loops under greedy decoding, most answers are factually wrong, and linear attention would need a fused
kernel to win. Our rubric was scored by an automated LLM judge, not human raters, so it's a single, partly blind rating."

**Slide 14, references (Samvith, 15 s)**
"These are the papers we built on: the Transformer, linear attention, TinyStories and Dolly. Thanks for watching."

---

## Recording checklist
- [ ] One rehearsal with a timer. If it runs over 7:00, shorten the demo first.
- [ ] Export as MP4 and upload to Google Drive with "Anyone with the link can view", or to YouTube as unlisted.
- [ ] Open the link in a private window to check it plays before you submit the form.
