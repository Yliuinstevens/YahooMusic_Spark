from __future__ import print_function

with open("Data/trainData.txt","w") as trainData:
    with open("RawData/trainIdx2.txt") as trainFile:
        for line in trainFile:
            if "|" in line:
                cur_user = line.split("|")[0]
                print(cur_user)
            else:
                trainData.write(cur_user+"\t"+line)

with open("Data/testData.txt","w") as testData:
    with open("RawData/testIdx2.txt") as testFile:
        for line in testFile:
            if "|" in line:
                cur_user = line.split("|")[0]
                print(cur_user)
            else:
                testData.write(cur_user+"\t"+line)
