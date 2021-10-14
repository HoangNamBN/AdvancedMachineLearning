import random as rd
import re
import math
import string


def pre_process_tweets(url):

    f = open(url, "r", encoding="utf8")
    tweets = list(f)
    list_of_tweets = []

    for i in range(len(tweets)):

        # xóa \ n khỏi cuối sau mỗi câu
        tweets[i] = tweets[i].strip('\n')

        # Xóa id tweet và dấu thời gian
        tweets[i] = tweets[i][50:]

        # Xóa bất kỳ từ nào bắt đầu bằng ký hiệu @
        tweets[i] = " ".join(filter(lambda x: x[0] != '@', tweets[i].split()))

        # Xóa bất kỳ URL nào
        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        tweets[i] = re.sub(r"www\S+", "", tweets[i])

        # xóa dấu hai chấm ở cuối câu (nếu có) sau khi xóa url
        tweets[i] = tweets[i].strip()
        tweet_len = len(tweets[i])
        if tweet_len > 0:
            if tweets[i][len(tweets[i]) - 1] == ':':
                tweets[i] = tweets[i][:len(tweets[i]) - 1]

        # Xóa bất kỳ ký hiệu thẻ băm nào
        tweets[i] = tweets[i].replace('#', '')

        # Chuyển mọi từ thành chữ thường
        tweets[i] = tweets[i].lower()

        # xóa dấu chấm câu
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))

        # cắt bớt khoảng trống
        tweets[i] = " ".join(tweets[i].split())

        # chuyển đổi từng tweet từ loại chuỗi thành dạng danh sách <string> bằng cách sử dụng "" làm dấu phân cách
        list_of_tweets.append(tweets[i].split(' '))

    f.close()

    return list_of_tweets


def k_means(tweets, k=4, max_iterations=50):

    centroids = []

    # khởi tạo, gán các tweet ngẫu nhiên làm trung tâm
    count = 0
    hash_map = dict()
    while count < k:
        random_tweet_idx = rd.randint(0, len(tweets) - 1)
        if random_tweet_idx not in hash_map:
            count += 1
            hash_map[random_tweet_idx] = True
            centroids.append(tweets[random_tweet_idx])

    iter_count = 0
    prev_centroids = []

    # chạy các lần lặp cho đến khi không hội tụ hoặc cho đến khi không đạt đến lần lặp tối đa
    while (is_converged(prev_centroids, centroids)) == False and (iter_count < max_iterations):

        print("running iteration " + str(iter_count))

        # phân công, gán các tweet cho các trung tâm gần nhất
        clusters = assign_cluster(tweets, centroids)

        # để kiểm tra xem k-means có hội tụ hay không, hãy theo dõi các pres_centroid
        prev_centroids = centroids

        # cập nhật, cập nhật centroid dựa trên các cụm được hình thành
        centroids = update_centroids(clusters)
        iter_count = iter_count + 1

    if (iter_count == max_iterations):
        print("đạt đến lần lặp tối đa, k mean không hội tụ ")
    else:
        print("Hội tụ ")

    sse = compute_SSE(clusters)

    return clusters, sse


def is_converged(prev_centroid, new_centroids):

    # sai nếu độ dài không bằng nhau
    if len(prev_centroid) != len(new_centroids):
        return False

    # lặp lại từng mục nhập của các cụm và kiểm tra xem chúng có giống nhau không
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False

    return True


def assign_cluster(tweets, centroids):

    clusters = dict()

    # đối với mỗi tweet, hãy lặp lại từng centroid và gán centroid gần nhất cho nó
    for t in range(len(tweets)):
        min_dis = math.inf
        cluster_idx = -1;
        for c in range(len(centroids)):
            dis = getDistance(centroids[c], tweets[t])

            # tìm kiếm một trung tâm gần nhất cho một tweet

            if centroids[c] == tweets[t]:
                # print("tweet và centroid ngang nhau với c: " + str(c) + ", t" + str(t))
                cluster_idx = c
                min_dis = 0
                break

            if dis < min_dis:
                cluster_idx = c
                min_dis = dis

        # ngẫu nhiên hóa việc gán centroid cho một tweet nếu không có gì phổ biến
        if min_dis == 1:
            cluster_idx = rd.randint(0, len(centroids) - 1)

        # chỉ định centroid gần nhất cho một tweet
        clusters.setdefault(cluster_idx, []).append([tweets[t]])
        # print("tweet t: " + str(t) + " is assigned to cluster c: " + str(cluster_idx))
        # thêm khoảng cách tweet từ trung tâm gần nhất của nó để tính sse cuối cùng
        last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)

    # print("clusters:", clusters)

    return clusters


def update_centroids(clusters):

    centroids = []

    # lặp lại từng cụm và kiểm tra một tweet có tổng khoảng cách gần nhất với tất cả các tweet khác trong cùng một cụm
    # chọn tweet đó làm trung tâm cho cụm
    for c in range(len(clusters)):
        min_dis_sum = math.inf
        centroid_idx = -1

        # để tránh tính toán thừa
        min_dis_dp = []

        for t1 in range(len(clusters[c])):
            min_dis_dp.append([])
            dis_sum = 0
            # nhận tổng khoảng cách cho mỗi tweet t1 với mọi tweet t2 trong cùng một cụm
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = getDistance(clusters[c][t1][0], clusters[c][t2][0])

                    min_dis_dp[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_dp[t1].append(0)

            # chọn tweet có tổng khoảng cách tối thiểu làm trọng tâm cho cụm
            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1

        # nối tweet đã chọn vào danh sách centroid
        centroids.append(clusters[c][centroid_idx][0])

    return centroids


def getDistance(tweet1, tweet2):

    # nhận được giao lộ
    intersection = set(tweet1).intersection(tweet2)

    # get the union
    union = set().union(tweet1, tweet2)

    
    # trả lại khoảng cách jaccard
    return 1 - (len(intersection) / len(union))


def compute_SSE(clusters):

    sse = 0
    # lặp lại mọi cụm 'c', tính SSE là tổng bình phương khoảng cách của tweet từ centroid của nó
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse


if __name__ == '__main__':

    data_url = 'Health_Tweets/bbchealth.txt'

    tweets = pre_process_tweets(data_url)

    # số lượng thử nghiệm mặc định được thực hiện
    experiments = 5

    # default value of K for K-means
    k = 3

    # for every experiment 'e', run K-means
    for e in range(experiments):

        print("------ Running K means for experiment no. " + str((e + 1)) + " for k = " + str(k))

        clusters, sse = k_means(tweets, k)

        # for every cluster 'c', print size of each cluster
        for c in range(len(clusters)):
            print(str(c+1) + ": ", str(len(clusters[c])) + " tweets")
            # # to print tweets in a cluster
            # for t in range(len(clusters[c])):
            #     print("t" + str(t) + ", " + (" ".join(clusters[c][t][0])))

        print("--> SSE : " + str(sse))
        print('\n')

        # increment k after every experiment
        k = k + 1
