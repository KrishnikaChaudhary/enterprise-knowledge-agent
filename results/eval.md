Questions: 60 (synthetic company corpus, 39 chunks). Embeddings: hash-512. LLM: mock.

**Offline run:** hash embeddings are lexical (no synonyms) and the mock LLM is extractive, so these numbers test the pipeline, not a language model. Latency is simulated and prices are placeholders; do not quote cost or latency.

| config | answer acc | hit@1 | hit@4 | MRR | refuse OK | ACL OK | ACL leaks | inject OK | cited | tokens in | $/req | latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_only | 73% | 92% | 96% | 0.929 | 100% | 100% | 0 | 100% | 100% | 408.8 | 0.00078 | 0.82387 |
| bm25_only | 75% | 94% | 96% | 0.948 | 100% | 100% | 0 | 100% | 100% | 400.6 | 0.0008 | 0.83621 |
| hybrid | 75% | 88% | 96% | 0.917 | 100% | 100% | 0 | 100% | 100% | 210.6 | 0.00081 | 0.54152 |
| hybrid+rewrite | 75% | 94% | 96% | 0.948 | 100% | 100% | 0 | 100% | 100% | 218.8 | 0.00086 | 0.65424 |
| hybrid+rewrite+verify | 75% | 94% | 96% | 0.948 | 100% | 100% | 0 | 100% | 100% | 375.6 | 0.00135 | 0.94212 |
| full (routing) | 75% | 94% | 96% | 0.948 | 100% | 100% | 0 | 100% | 100% | 423.1 | 0.00084 | 0.846 |

### Failing questions for `full (routing)`

- Q09 (answerable, role=employee): Above what amount do I need a receipt? -> status=answered (), first gold rank=1
- Q33 (answerable, role=employee): How much time off do staff get each year? -> status=refused (no_relevant_context), first gold rank=1
- Q34 (answerable, role=employee): What happens to my API calls if I exceed my plan's rate limit? -> status=refused (model_not_found), first gold rank=1
- Q35 (answerable, role=employee): Can I take work abroad for a month? -> status=refused (model_not_found), first gold rank=1
- Q36 (answerable, role=employee): What is the most I can spend on a hotel room for one night? -> status=refused (model_not_found), first gold rank=1
- Q37 (answerable, role=employee): How soon after buying something must I file my claim? -> status=refused (no_relevant_context), first gold rank=1
- Q38 (answerable, role=employee): Is it okay to plug in a thumb drive? -> status=refused (no_relevant_context), first gold rank=None
- Q39 (answerable, role=employee): What should I do if my computer gets stolen? -> status=refused (no_relevant_context), first gold rank=2
- Q42 (answerable, role=employee): What is the monthly fee for the middle tier? -> status=refused (model_not_found), first gold rank=None
- Q44 (answerable, role=employee): How fast must a manager answer my time-off request? -> status=refused (model_not_found), first gold rank=1
- Q45 (answerable, role=employee): Who signs off on very large purchases? -> status=refused (no_relevant_context), first gold rank=1
- Q50 (answerable, role=legal): How long should a misconduct investigation take? -> status=refused (model_not_found), first gold rank=1
