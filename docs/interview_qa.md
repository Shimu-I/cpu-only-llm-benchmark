# Interview Q&A: `cpu-only-llm-benchmark`

66 questions an interviewer could ask about this project, with sample answers based on **your real results**. Rewrite every answer in your own words. An answer you have memorized will sound worse than an honest one you understand.

Rules for good answers:
1. Answer first, explain second.
2. Use a number when you have one.
3. When you don't know, say so, then say how you would find out.
4. Be honest about limits. Interviewers trust people who name their own weaknesses.

---

## A. Project overview

**1. Give me a one-minute summary of your project.**
I compared four open-source models on a support-ticket routing task, on a laptop with no GPU. The question was which model a small company should deploy on CPU-only servers. I tested on two samples of 200 messages: an easy one with clearly different categories and a hard one with look-alike categories. A small Hugging Face embedding model plus a simple classifier won: 88% accuracy on the hard sample, 8 ms per message, and about 640 MB of RAM. The best LLM got 76%, and was about 190 times slower. I wrote a decision memo and built a small demo app.

**2. What problem does it solve?**
A company gets thousands of customer messages and needs each one sent to the right team automatically. It has no GPU servers, so it needs a model that is accurate enough and fast enough on ordinary CPUs. The project gives evidence for that choice.

**3. Why did you pick this project?**
I wanted a project that shows I can use Hugging Face open-source models, but with a real business decision behind it. Most beginner projects just run one model. Comparing several under a real constraint (CPU only, limited memory) is closer to what a data scientist does at work.

**4. What was your main finding?**
On easy categories, almost everything looks good (96-99% for the top three). On look-alike categories the differences appear. The small embedding model dropped 11 points, the 3B LLM dropped 33, and the 7B LLM dropped 21. So an easy test can hide big differences.

**5. What do you recommend, and to whom?**
To a support-operations manager: deploy MiniLM plus logistic regression if labeled tickets exist. If there is no labeled data yet, use the 7B LLM as a stop-gap while collecting labels. I would not use the zero-shot DistilBERT model.

**6. Why CPU-only?**
Many small companies can't pay for GPU servers, and my own laptop has no GPU, so it was also a real constraint. It forced me to think about latency, memory and cost, not just accuracy.

**7. What did you build end to end?**
A reproducible repo: data preparation scripts, benchmark scripts for each model type, a shared config, a failure analysis script, a comparison table with charts, a decision memo, a Gradio demo app, and an activity log of every command, with one git commit per task.

---

## B. Business thinking

**8. Who is the "customer" of your analysis?**
The person deciding which model to deploy: a support or operations manager, with an engineer who has to run it.

**9. How did you decide what "best" means?**
Four things: accuracy (how often it is right), speed (latency per message), memory (RAM), and whether it needs labeled data. I did not treat accuracy as the only measure, because a model that is slightly more accurate but 190 times slower could be unusable.

**10. What does the speed difference mean in practice?**
From the measured latency, 100,000 messages would take about 13 minutes with MiniLM and about 42 hours with the 7B model, on one message at a time. That is my calculation, not a separate measurement.

**11. What does a wrong answer cost?**
It depends on the mistake. Sending "my card hasn't arrived" to the fraud team wastes an escalation. Sending a real fraud report to the delivery team could be worse. I only counted errors equally, and a real deployment should weight them by cost.

**12. When would you choose the LLM instead?**
When there is no labeled data, when categories change often, or when the messages need reasoning beyond matching. The LLM needs no training examples, which is a real advantage that the accuracy numbers alone don't show.

**13. What would you track in production?**
The share of messages sent to humans, the misroute rate from spot checks, latency, and how the mix of categories drifts over time.

---

## C. Data

**14. Why did you use banking77?**
It is a public dataset of real-style customer banking questions with 77 labeled intents, small enough to work with on a laptop, and it looks like a real routing problem.

**15. How did you split the data? Could there be leakage?**
The dataset has a train split (9,993 messages) and a test split (3,076). I trained only on the train split and drew both 200-message test samples only from the test split. So the models never saw the test messages during training.

**16. Why only 200 messages per sample?**
The 7B model takes about 1.4 seconds per message on my CPU, so more messages meant much longer runs. 200 was a compromise. The downside is that small differences can be noise.

**17. How did you choose the 10 intents? Isn't that cherry-picking?**
For the easy sample I chose clearly different intents, on purpose, to see the best case. For the hard sample I chose 5 look-alike pairs, on purpose, to see a harder case. It is a designed test, not a random one, and I say so in the memo. Both choices affect the results, which is why I report both.

**18. What is the hard sample, and why build it?**
Ten intents in five confusable pairs, such as `lost_or_stolen_card` vs `compromised_card`. After the easy sample showed models bunched at 96-99%, I needed a test where they would separate.

**19. Was the data balanced?**
Yes, 20 messages per intent in both samples. That makes accuracy and macro F1 nearly the same number, which you can see in the results.

**20. What data problem did you hit?**
`load_dataset("PolyAI/banking77")` failed because the newest `datasets` library no longer runs datasets that ship with a loading script. My first workaround, a parquet revision, returned a 404. I searched for another copy and used `mteb/banking77`, which stores plain data files.

---

## D. Hugging Face and models

**21. What is Hugging Face?**
A hub that hosts open-source models and datasets, plus Python libraries (`transformers`, `datasets`) to download and run them. The first use downloads to a local cache and later runs load from disk.

**22. What is a pipeline, and how is it different from AutoModel?**
A pipeline is a ready-made wrapper that does tokenizing, the model call and answer formatting in one line. `AutoTokenizer` and `AutoModel` are the lower-level parts, which give me control. I used a pipeline for zero-shot and the lower-level classes for MiniLM, so I could do pooling myself.

**23. What is a tokenizer?**
It splits text into small pieces (tokens) and turns them into numbers the model can read. Each model has its own tokenizer, so you must load the matching one.

**24. Explain zero-shot classification.**
The model sorts text into categories it was never trained on. With an NLI model, each category name is turned into a sentence like "This example is card arrival", and the model scores how well the message supports that sentence. The highest score wins.

**25. Why did the DistilBERT zero-shot model do so badly?**
It is small, it only sees category names, and on the hard sample many names overlap in meaning. It got 64.5% on the easy sample and 38.5% on the hard one. It also made scattered mistakes, such as sending card-arrival messages to `change_pin`.

**26. What is an embedding?**
A list of numbers that captures the meaning of a sentence, so similar sentences get similar numbers.

**27. Explain mean pooling and the attention mask.**
The model outputs one vector per token. To get one vector per sentence, I average them. The attention mask marks real tokens versus padding, so the average ignores the padding.

**28. Why normalize the vectors?**
So each vector has length 1 and only its direction matters. It makes vectors comparable and works well with the classifier.

**29. Why logistic regression on top?**
It is simple, fast, needs little data, and gives probabilities I could use as confidence. It trained in seconds on about 1,300 messages. A more complex model would need a reason, and I had none.

**30. Why didn't you fine-tune the model?**
Fine-tuning a whole model on CPU takes much longer, and the frozen embeddings plus a small classifier already reached 99% on easy and 88% on hard. Fine-tuning is a sensible next step if I needed to close the remaining gap.

**31. What is the difference between an encoder model and an LLM?**
An encoder like MiniLM reads text and produces vectors. It does not write. An LLM like Qwen generates text one token at a time. For a classification task, the encoder is much cheaper.

**32. Why did frozen embeddings plus a classifier work so well?**
The embedding model was already trained to put similar sentences close together. The classifier only had to learn where my categories' boundaries lie, using labeled examples. The LLMs had no way to learn those boundaries and only guessed from category names.

---

## E. Ollama and LLMs

**33. How do Ollama and Hugging Face relate here?**
Both give access to open models. I used Hugging Face for the small encoder models through Python, and Ollama to run the LLMs locally, because it handles the large model files and CPU inference well.

**34. How did you prompt the LLMs?**
The prompt told the model it was a bank support routing assistant, listed the 10 categories, asked for only the category name, and then gave the message. I did not include examples or category descriptions.

**35. Why temperature 0?**
For repeatable answers. With a higher temperature the same message could be routed differently on different runs, which makes a benchmark unreliable.

**36. What is an unparsed answer?**
When the LLM writes something that doesn't match any valid category. I counted those as wrong and tracked them. Both Qwen models had 0, so they followed the format.

**37. Why did the 7B beat the 3B only on the hard sample?**
On the easy sample both were around 96-97%, and 5 vs 7 errors is within noise. On the hard sample the 7B got 76% and the 3B got 64%. A bigger model seems to handle fine differences between similar categories better, but I only tested one model family.

**38. Is the LLM comparison fair?**
Not completely. MiniLM saw about 1,300 labeled examples. The LLMs saw none, only the category names. A fairer LLM setup would add category descriptions or a few examples in the prompt (few-shot). I list that as a next step, and I say the result is "zero-shot LLM vs trained classifier".

**39. What is quantization?**
Storing model numbers with fewer bits to shrink the model and speed it up, with a small quality loss. Ollama models are commonly around 4-bit, which is why a 3B model is only about 1.9 GB on disk.

---

## F. Evaluation and results

**40. Accuracy vs macro F1?**
Accuracy is the share of correct answers. Macro F1 calculates F1 for each category and averages them, so every category counts equally. With balanced classes they were almost equal (0.880 and 0.881 for MiniLM on the hard sample).

**41. How did you measure latency and RAM?**
Latency: time for each message with `time.perf_counter`, averaged over 200 messages, after a warm-up call. RAM: the resident memory of the Python process for Hugging Face models, and the summed memory of the `ollama` processes for LLMs. The Ollama number is an approximation.

**42. What are the limits of your speed measurement?**
One laptop, one message at a time, no batching, and other programs may have been running. Batching would make the small models even faster. For MiniLM the timing covers embedding plus classification but not model loading.

**43. How reliable are the results with 200 messages?**
Not very precise. For 88% on 200 messages, the 95% interval is roughly plus or minus 4.5 points, and for 76% roughly plus or minus 6. So I trust the big gaps (88% vs 38.5%) and treat small gaps (5 vs 7 errors) as noise.

**44. Is the 12-point gap between MiniLM and the 7B real?**
Probably, but I would not call it proven. The intervals nearly touch, and both models were tested on the same messages, which strengthens the comparison. I would confirm it with a larger sample before making a strong claim.

**45. Explain how you got the "100,000 messages" figures.**
Latency times 100,000: 8 ms gives about 13 minutes, 1,510 ms gives about 42 hours. It is a derived estimate, single-threaded, and not a measured run.

**46. What did comparing easy and hard samples teach you?**
That the test set matters as much as the model. On the easy sample the 3B model looked almost as good as MiniLM (97.5% vs 99%). On the hard sample it was 24 points behind (64% vs 88%). Reporting only the easy result would have led to a wrong decision.

---

## G. Failure analysis, honesty and limits

**47. What did your failure analysis show?**
DistilBERT made 71 errors on the easy sample, and they were scattered and illogical. The strong models made 2 to 7 errors, and those were understandable, such as a message saying "Is it lost?" labeled `card_arrival`. Three messages were missed by two strong models, suggesting ambiguous wording or labels.

**48. What systematic mistake did the 7B model make, and how would you fix it?**
4 of its 7 easy-sample errors were vague "transaction" messages sent to `top_up_failed` instead of `transfer_not_received_by_recipient`. I would add a one-line description of each category to the prompt and test whether it fixes them.

**49. Are some labels wrong or ambiguous?**
Some are ambiguous. "How long does it take for transfers to finish?" is labeled `transfer_not_received_by_recipient` but could be `pending_transfer`. So part of the remaining error comes from the labels, not the model.

**50. You said the zero-shot model's confidence was unreliable. How sure are you?**
Not very. I noticed it in a 4-message peek, where the wrong answers had high scores and the right ones had low scores. It was an observation that made me cautious, not a statistical result. I did not measure calibration.

**51. What does the weather-message demo show?**
"What's the weather today?" got only 21% confidence, so the app sent it to a human. The classifier only knows 15 intents and must pick one, so the confidence threshold is the safety net for messages outside its scope.

**52. How would you set the threshold properly?**
I set 0.4 by hand for the demo. Properly, I would use a validation set, plot the share of messages auto-routed against accuracy at different thresholds, and choose the point that matches what a human review costs.

**53. Are logistic regression probabilities reliable?**
Not necessarily. They can be over- or under-confident, which is called poor calibration. I would check with a calibration curve and, if needed, calibrate the probabilities before using a threshold in production.

**54. What is the biggest weakness of the project?**
Small samples and one dataset. Also, I designed the hard sample myself, and the LLMs did not get examples. The conclusions are evidence, not proof, and I would re-test on real company tickets.

---

## H. Engineering practice

**55. How did you make it reproducible?**
A fixed random seed, a pinned `requirements.txt`, scripts instead of manual steps, all code in git, and an activity log of every command.

**56. Why a virtual environment, and why the CPU build of PyTorch?**
A virtual environment keeps this project's packages separate from the system. The CPU build avoids several GB of GPU libraries I can't use, which mattered with limited disk space.

**57. How did you use git?**
One commit per task, with clear messages like `feat:`, `docs:`, `chore:`. Large files (data, models, the environment) are in `.gitignore`, and the scripts can recreate them.

**58. Tell me about something that went wrong and how you fixed it.**
Two examples. My first virtual environment broke because I pressed Ctrl+C during setup, so I deleted it and recreated it. And the dataset failed to load because of the loading-script change, so I tried a parquet revision, got a 404, then found a script-free copy. Each time I read the error message, formed a hypothesis and tested it.

**59. How would you deploy this?**
Save the trained classifier and embedding model, wrap them in a small API (for example FastAPI), add the confidence threshold and a human-review queue, and run it in a container on a CPU server.

**60. How would you monitor it?**
Log predictions and confidence, sample some for human review each week, track the misroute rate and the share sent to humans, and retrain when the categories or the message style change.

**61. What if the company adds a new category?**
With MiniLM I add labeled examples for it and retrain the classifier, which takes seconds because the embedding model stays frozen. With an LLM I would edit the prompt, but I would have less control over the boundaries.

**62. Is there a privacy angle?**
Yes. Local models keep customer messages on the company's own machines, and no data goes to an outside API. That can matter for banking. Real tickets should also have personal data removed before being logged or used for training.

**63. How would you scale to millions of messages?**
Batch the messages, which speeds up the encoder a lot, and run several workers. If needed, I would convert the model to a faster format such as ONNX or use quantization.

---

## I. Growth and next steps

**64. What would you do with two more weeks?**
1. Test on a larger sample and more intents.
2. Give the LLMs a fair chance with category descriptions and few-shot examples.
3. Try fine-tuning a small model.
4. Check confidence calibration and tune the threshold.
5. Host the demo as a Hugging Face Space.

**65. What surprised you?**
That a tiny 90 MB model plus a simple classifier beat 7-billion-parameter LLMs, and that the easy test hid it. It taught me to distrust a result until I have tried a harder test.

**66. Did you use AI tools to build this?**
*Answer truthfully.* An honest version: "Yes, I used an AI assistant as a guide and tutor. I ran every command myself, read every output, fixed the errors that came up, and kept a log of what each step does so I could learn it. I can explain any part of the code and every number in the results." Only say this if it is true, and be ready to explain any file in the repo.
