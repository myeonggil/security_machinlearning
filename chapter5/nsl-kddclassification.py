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
print(pd.Series(train_Y_rus).value_counts())

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=17)
classifier.fit(train_x_rus, train_Y_rus)
pred_y = classifier.predict(test_x)
results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
print(error)