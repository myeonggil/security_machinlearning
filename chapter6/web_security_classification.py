from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from random import random
import matplotlib.pyplot as plt


# 샘플링을 위해 각 행에 랜덤 엔드리를 추가한다.
R = [random() for i in range(len(ngram_cluster_features))]
ngram_cluster_features['rand'] = R

# 2/3를 훈련 데이터셋으로, 1/3을 테스트 데이터셋으로 분할한다.
train, test = train_test_split(ngram_cluster_features.fillna(value=0), test_size=0.33)
sample_factor = 0.2
sampled_train = train[(train.label == 1) | (train.label == 0) & (train.rand < sample_factor)]

# 적합하고 예측한다.
features = sampled_train[sampled_train.columns.difference(['label', 'rand', 'score'])]
labels = sampled_train.label
clf = RandomForestClassifier(n_estimators=20)
clf.fit(features, labels)
probs = clf.predict_proba(test[train.columns.difference('label', 'rand', 'score')])

# P-R 커브를 계싼하고 시각화한다.
precision, recall, thresholds = precision_recall_curve(test.label, probs[:,1])
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve for 7-gram groupings')
plt.show()

# 95% 정밀도를 갖는 임계치를 찾는다.
m = min([i for i in range(len(precision)) if precision[i] > 0.95])
p, r, t = precision[m], recall[m], thresholds[m]
print(p, r, t)

# 아이템 수준 정밀도/재현율 계산
pos = (test.score * test['count'])
neg = (1 - test.score) * (test['count'])
tp = sum(pos[test.label >= t])
fp = sum(neg[test.label >= t])
tn = sum(neg[test.label < t])
fn = sum(pos[test.label < t])
item_precision = 1.0 * tp / (tp + fp)
item_recall = 1.0 * tp / (tp + fn)