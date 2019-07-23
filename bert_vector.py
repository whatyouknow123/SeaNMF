from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient

bc = BertClient()
values = bc.encode(['美丽', '漂亮', '呵呵'])
print(cosine_similarity(values))
