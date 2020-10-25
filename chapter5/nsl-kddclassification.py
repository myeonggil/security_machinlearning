from collections import defaultdict
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')

# 모든 관련 데이터 파일이 들어 있는 디렉터리
dataset_root = 'datasets/nsl-kdd/'

category = defaultdict(list)
category['benign'].append('normal')

with open(dataset_root + 'training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v, k) for k in category for v in category[k])

train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')

# header_names는 데이터와 같은 순서로 정렬된 속성 이름의 리스트
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'attack_type', 'success_pred']
# 레이블(CSV 끝에서 두 개) 이름은 attack_type이고, CSV의 마지막 값은 success_pred다.

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# 훈련 데이터 로드
train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)

# 테스트 데이터 로드
test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()
# train_attack_types.plot(kind='barh')
# train_attack_cats.plot(kind='barh')
#
# plt.show()

# kddcup.names에 추가된 값인데 drop한 이유....
# train_df.drop('num_outbound_cmds', axis=1, inplace=True)
# test_df.drop('num_outbound_cmds', axis=1, inplace=True)
# numeric_cols.remove('num_outbound_cmds')

train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)

test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)

feature_names = defaultdict(list)

with open(dataset_root + 'kddcup.names', 'r') as f:
    for line in f.readlines()[1:]:
        name, nature = line.strip()[:-1].split(': ')
        feature_names[nature].append(name)

# 데이터프레임 결합
combined_df_raw = pd.concat([train_x_raw, test_x_raw])

# 연속형, 이진, 명목형 변수를 추적
continuous_features = feature_names['continuous']
continuous_features.remove('root_shell')

binary_features = ['land', 'logged_in', 'root_shell',
                   'su_attempted', 'is_host_login',
                   'is_guest_login']
nominal_features = list(
    set(feature_names['symbolic']) - set(binary_features)
)

# 더비 변수 생성
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

# 훈련 데이터셋과 테스트 셋을 다시 분할
train_x = combined_df[: len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# 더미 변수를 추적
dummy_variables = list(set(train_x) - set(combined_df_raw))

# print(train_x.describe())
# print(train_x['duration'].describe())

# 시그널을 스케일러로 변환하고 이 결과가 한 속성이 된다.
durations = train_x['duration'].values.reshape(-1, 1)

standard_scaler = StandardScaler().fit(durations)
standard_scaled_durations = standard_scaler.transform(durations)
# print(pd.Series(standard_scaled_durations.flatten()).describe())
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
# print(pd.Series(min_max_scaled_durations.flatten()).describe())

# 훈련 데이터를 StandardSclaer 에 적합
standard_scaler = StandardScaler().fit(train_x[continuous_features])

# 훈련 데이터 표준화
train_x[continuous_features] = \
standard_scaler.transform(train_x[continuous_features])

# 훈련 데이터에 적합된 스케일러에 테스트 데이터 표준화
test_x[continuous_features] = \
standard_scaler.transform(test_x[continuous_features])

train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)

# 의사 결정 트리
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, zero_one_loss
#
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(train_x, train_Y)
#
# pred_y = classifier.predict(test_x)
# results = confusion_matrix(test_Y, pred_y)
# error = zero_one_loss(test_Y, pred_y)

# print(error)

# print(test_Y.value_counts().apply(lambda x: x/float(len(test_Y))))

# k-means
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
#
# classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
# classifier.fit(train_x, train_Y)
# pred_y = classifier.predict(test_x)
# results = confusion_matrix(test_Y, pred_y)
# error = zero_one_loss(test_Y, pred_y)
#
# print(error)

# 선형 서포트 벡터 분류기
# from sklearn.svm import LinearSVC
# from sklearn.metrics import confusion_matrix, zero_one_loss
#
# classifier = LinearSVC()
# classifier.fit(train_x, train_Y)
# pred_y = classifier.predict(test_x)
#
# results = confusion_matrix(test_Y, pred_y)
# error = zero_one_loss(test_Y, pred_y)
#
# print(error)

# oversampling
# print(pd.Series(train_Y).value_counts())

# SMOTE 오버샘플링
# from imblearn.over_sampling import SMOTE
#
# sm = SMOTE(random_state=0)
# train_x_sm, train_Y_sm = sm.fit_sample(train_x, train_Y)
# print(pd.Series(train_Y_sm).value_counts())

from imblearn.under_sampling import RandomUnderSampler

mean_class_size = int(pd.Series(train_Y).value_counts().sum() / 5)
# 샘플링 대상의 최솟값 초과의 값은 먹히지 않음... 이유는?
ratio = {
    'benign': 52,
    'dos': 52,
    'probe': 52,
    'r2l': 52,
    'u2r': 52
}
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
train_x_rus, train_Y_rus = rus.fit_sample(train_x, train_Y)
# print(pd.Series(train_Y_rus).value_counts())
#
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=17)
classifier.fit(train_x_rus, train_Y_rus)
pred_y = classifier.predict(test_x)
results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
# print(error)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5).fit(train_x)
pred_y = kmeans.predict(test_x)

# 군집화 결과 확인
# print(pd.Series(pred_y).value_counts())

from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

# print('Completeness: {}'.format(completeness_score(test_Y, pred_y)))
# print('Homogeneity: {}'.format(homogeneity_score(test_Y, pred_y)))
# print('V-measure: {}'.format(v_measure_score(test_Y, pred_y)))

# 데이터 분산의 많은 부분을 포착하라면 충분한 수의 주성분 요소를 선택
from sklearn.decomposition import PCA
#
pca = PCA(n_components=2)
train_x_pca = pca.fit_transform(train_x)
#
# plt.figure()
# colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple']
#
# for color, cat in zip(colors, category.keys()):
#     plt.scatter(train_x_pca[train_Y==cat, 0],
#                 train_x_pca[train_Y==cat, 1],
#                 color=color, alpha=.8, lw=2, label=cat)
#
# # u2r의 샘플 데이터가 너무 적기 때문에 비적합
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.show()

# 점들 시각화
# 훈련 데이터를 k-평균 군집화 estimator 모델이 적합시킨다.
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=17).fit(train_x)
# 각 훈련 샘플에 할당된 레이블을 불러온다.
kmeans_y = kmeans.labels_

# train_x_pca_count를 2d로 시각화한다.
plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple']

# for color, cat in zip(colors, range(5)):
#     plt.scatter(train_x_pca[kmeans_y==cat, 0],
#                 train_x_pca[kmeans_y==cat, 1],
#                 color=color, alpha=.8, lw=2, label=cat)
#
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.show()

averages = train_df.loc[:, numeric_cols].mean()
averages_per_class = train_df[numeric_cols + ['attack_category']].groupby('attack_category').mean()

AR = {}
for col in numeric_cols:
    AR[col] = max(averages_per_class[col]) / averages[col]

def binary_AR(df, col):
    series_zero = df[df[col] == 0].groupby('attack_category').size()
    series_one = df[df[col] == 1].groupby('attack_category').size()

    return max(series_one / series_zero)

labels2 = ['normal', 'attack']
labels5 = ['normal', 'dos', 'probe', 'r2l', 'u2r']

train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)

test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()
test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.drop('num_outbound_cmds', axis=1, inplace=True)
test_df.drop('num_outbound_cmds', axis=1, inplace=True)

train_df['labels2'] = train_df.apply(lambda x: 'normal' if 'normal' in x['attack_type'] else 'attack', axis=1)
test_df['label2'] = test_df.apply(lambda x: 'normal' if 'normal' in x['attack_type'] else 'attack', axis=1)

combined_df = pd.concat([train_df, test_df])
original_cols = combined_df.columns

combined_df = pd.get_dummies(combined_df, columns=nominal_cols, drop_first=True)

added_cols = set(combined_df.columns) - set(original_cols)
added_cols= list(added_cols)

combined_df.attack_category = pd.Categorical(combined_df.attack_category)
combined_df.labels2 = pd.Categorical(combined_df.labels2)

combined_df['labels5'] = combined_df['attack_category'].cat.codes
combined_df['labels2'] = combined_df['labels2'].cat.codes

train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

for col in binary_cols + dummy_variables:
    cur_AR = binary_AR(train_df, col)
    if cur_AR:
        AR[col] = cur_AR

print(train_df[train_df.service_Z39_50 == 1].groupby('attack_category').size())
print(len(binary_cols + added_cols))

import operator

AR = dict((k, v) for k, v in AR.items() if not np.isnan(v))
sorted_AR = sorted(AR.items(), key=lambda x: x[1], reverse=True)
print(sorted_AR)

features_to_use = []
for x, y in sorted_AR:
    if y >= 0.01:
        features_to_use.append(x)

print(features_to_use)
print(len(features_to_use))
print(len(sorted_AR) - len(features_to_use))

train_df_trimmed = train_df[features_to_use]
test_df_trimmed = test_df[features_to_use]

numeric_cols_to_use = list(set(numeric_cols).intersection(features_to_use))

standard_scaler = StandardScaler()

train_df_trimmed[numeric_cols_to_use] = standard_scaler.fit_transform(train_df_trimmed[numeric_cols_to_use])
test_df_trimmed[numeric_cols_to_use] = standard_scaler.transform(test_df_trimmed[numeric_cols_to_use])

kmeans = KMeans(n_clusters=8, random_state=17)
kmeans.fit(train_df_trimmed[numeric_cols_to_use])
kmeans_train_y = kmeans.labels_

print(pd.crosstab(kmeans_train_y, train_Y_bin))

train_df['kmeans_y'] = kmeans_train_y
train_df_trimmed['kmeans_y'] = kmeans_train_y
kmeans_test_y = kmeans.predict(test_df_trimmed[numeric_cols_to_use])
test_df['kmeans_y'] = kmeans_test_y

train_y4 = train_df[train_df.kmeans_y == 4]
test_y4 = test_df[test_df.kmeans_y == 4]

dtc4 = DecisionTreeClassifier(random_state=17).fit(train_y4.drop(['kmeans_y'], axis=1), train_y4['labels2'])

dtc4_pred_y = dtc4.predict(test_y4.drop(['kmeans_y'], axis=1))
print(dtc4_pred_y)