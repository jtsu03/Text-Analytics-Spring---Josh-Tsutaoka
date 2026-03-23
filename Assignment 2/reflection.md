# Reflection — Model Generalization

| Category | Correct | Total |
|---|---|---|
| Easy Real | 3 | 5 |
| Easy Fake | 5 | 5 |
| Tricky | 2 | 5 |
| Different Context | 3 | 5 |
| **Total** | **13** | **20** |

The model classified 13 out of 20 examples. It got all the obvious fake examples right but struggled with the real news articles that did not sound like Reuters news. On the tricky examples it only flagged 2 out of 5, this seemed to be because that articles that were using uncertain or dramatic language were then flagged as fake. For the out-of-domain examples, it went 3 out of 5. This was not bad for international news that still followed formal writing styles but failing on real articels that were written in a more casual writing style. This tells me that the model learned to recognize writing style rather than actual factual information. Which is why it scored so high on the test set but dropped on fresh examples. I would not trust this to be put into production in its current form. It relies too heavily on pattern instead of understanding what actually makes something real news or fake. 
