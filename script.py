import subprocess
import os

bikes = 0
no_bikes = 0

FNULL = open(os.devnull, 'w')

while True:
    #subprocess.run(["python", "./download_data.py"])
    subprocess.run(["mkdir", "./networks", "./networks/classifiers", "./meta_datasets"], stdout=FNULL, stderr=FNULL)
    subprocess.run(["python", "./create_net.py", "--path", "./networks/target_classifier.pt"], stdout=FNULL, stderr=FNULL)
    for i in range(1, 11):
        subprocess.run(["python", "./create_net.py", "--path", "./networks/classifiers/classifier_{}.pt".format(i)], stdout=FNULL, stderr=FNULL)
    subprocess.run(["python", "./train_net.py", "--path", "./networks/target_classifier.pt", "--dataset", "0", "--epochs", "10"], stdout=FNULL, stderr=FNULL)
    for i in range(1, 11):
        subprocess.run(["python", "./train_net.py", "--path", "./networks/classifiers/classifier_{}.pt".format(i), "--dataset", "{}".format(i), "--epochs", "10"], stdout=FNULL, stderr=FNULL)
    subprocess.run(["python", "./extract_weights.py", "--dir", "./networks/classifiers","--out", "./meta_datasets"], stdout=FNULL, stderr=FNULL)
    subprocess.run(["python", "./create_meta_net.py", "--path", "./networks/meta_classifier.pt", "--input-size-vector", "3200"], stdout=FNULL, stderr=FNULL)
    subprocess.run(["python", "./train_meta_net.py", "--path", "./networks/meta_classifier.pt", "--dataset", "./meta_datasets"], stdout=FNULL, stderr=FNULL)
    completed_process = subprocess.run(["python", "./bikes_or_not_bikes_that_is_the_question.py", "--target", "./networks/target_classifier.pt", "--meta", "./networks/meta_classifier.pt"], stdout=subprocess.PIPE)

    answer = int(str(completed_process.stdout)[2])

    if answer == 0:
        no_bikes += 1
    else:
        bikes += 1

    print('bikes: {}; no bikes: {}; total: {}'.format(bikes, no_bikes, bikes + no_bikes))


