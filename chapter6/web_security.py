import lsh


def compute_hashes(domains, n, num_perms=32, max_items=100, hash_function=lsh.md5hash):
    # domains는 도메인 객체로 사전 형태이다.
    # 도메인 이름을 키로 지정한다.

    # LSH 인덱스를 생성한다.
    hashes = lsh.lsh(num_perms, hash_function)

    # minHashes를 계산한다.
    for dom in domains:
        dg = hashes.digest(domains[dom].ngrams[n])
        domains[dom].digest = dg
        hashes.insert(dom, dg)

    return hashes

def compute_lsh_clusters(domains, hashes, min_size=10, threshold=0.5):
    # domains는 도메인 객체로 사전 형태다.
    # 도메인 이름을 키로 지정한다.
    # hashes는 compute_hashes에서 생성한 lsh 객체다.

    clusters = []
    for dom in domains:
        # 주어진 다이제스트와 일치하는 모든 도메인을 얻는다.
        # result는 {domain: score} 형태의 사전이다.
        result = hashes.query(domains[dom].digest)
        result_domains = {domains[d]: result[d] for d in result if result[d] >= threshold}
        if len(result_domains) >= min_size:
            # 결과 데이터를 갖고 클러스터 객체를 생성한다.
            clusters.append(cluster(dom, result_domains))
        return clusters

hashes = compute_hashes(data, n, 32, 100)
clusters = compute_lsh_clusters(data, hashes, 10, threshold)