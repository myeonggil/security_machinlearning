import string
import email
import nltk

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

# 이메일의 여러 부분을 하나의 문자열로 합한다.
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()

    return ret

# 이메일로부터 제목과 내용 텍스트를 추출한다.
def extract_email_text(path):
    # 입력 파일로부터 하나의 이메일을 불러온다.
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)

    if not msg:
        return ""

    # 이메일 제목을 불러온다.
    subject = msg['Subject']
    if not subject:
        subject = ""

    # 이메일 내용을 불러온다.
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)

    if not body:
        body = ""

    return subject + ' ' + body

# 이메일을 형태소 분석한다.
def load(path):
    email_text = extract_email_text(path)
    if not email_text:
        return []

    # 메시지를 토큰화한다.
    tokens = nltk.word_tokenize(email_text)

    # 토큰에서 마침표를 제거한다.
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]

    # 자주 사용하지 않는 단어를 제거한다.
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]

    return []