import subprocess

subprocess.run(["python", "./download_data.py"])
subprocess.run(["mkdir", "./networks", "./networks/classifiers", "./meta_datasets"])
subprocess.run(["python", "./create_net.py", "--path", "./networks/target_classifier.pt"])
for i in range(1, 11):
    subprocess.run(["python", "./create_net.py", "--path", "./networks/classifiers/classifier_{}.pt".format(i)])
subprocess.run(["python", "./train_net.py", "--path", "./networks/target_classifier.pt", "--dataset", "0"])
for i in range(1, 11):
    print("Training classifier {}".format(i))
    subprocess.run(["python", "./train_net.py", "--path", "./networks/classifiers/classifier_{}.pt".format(i), "--dataset", "{}".format(i)])
subprocess.run(["python", "./extract_weights.py", "--dir", "./networks/classifiers","--out", "./meta_datasets"])
subprocess.run(["python", "./create_meta_net.py", "--path", "./networks/meta_classifier.pt", "--input-size-vector", "1600"])
subprocess.run(["python", "./train_meta_net.py", "--path", "./networks/meta_classifier.pt", "--dataset", "./meta_datasets"])
subprocess.run(["python", "./bikes_or_not_bikes_that_is_the_question.py", "--target", "./networks/target_classifier.pt", "--meta", "./networks/meta_classifier.pt"])
