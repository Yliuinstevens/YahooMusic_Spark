import numpy as np
'''
prediction = []
trueResult = []

with open("prediction.txt") as predFile:
    for line in predFile:
        temp_list = line.split("|").strip("\n")
        temp_list = [int(x) for x in temp_list]
        prediction.append(temp_list)

with open("test_valid_new.txt") as trueFile:
    for line in trueFile:
        temp_list = line.split("|").strip("\n")
        temp_list = [int(x) for x in temp_list]
        trueResult.append(temp_list)
'''

prediction = np.genfromtxt("prediction.txt", delimiter = "|")
trueResult = np.genfromtxt("test_valid_new.txt", delimiter = "|")
'''
print(len(prediction))

for i in range(int(len(prediction)/6)):
	temp_list = prediction[i*6:i*6+6]
	rating_list = temp_list[:,2]
	if "None" in rating_list:
		mean = np.mean([x for x in rating_list if x != "None"])
		median = np.median([x for x in rating_list if x != "None"])
		for k in range(6):
			if prediction[i*6+k][2]=="None":
				prediction[i*6+k][2] = mean>median
			else:
				prediction[i*6+k][2]=prediction[i*6+k][2]>=median
	else:
		mean = np.mean(rating_list)
		median = np.median(rating_list)
		for j in range(6):
			prediction[i*6+j][2]=prediction[i*6+j][2]>=median
'''
prediction[prediction[:,0]]
prediction = np.array(prediction, dtype = [("User",int),("Item",int),("Rating",int)])
prediction.sort(order = ["User","Item"])
trueResult = np.array(trueResult, dtype = [("User",int),("Item",int),("Rating",int)])
trueResult.sort(order = ["User","Item"])
match_user = prediction[:,0]==trueResult[:,0]
match_item = prediction[:,1]==trueResult[:,1]
print(match_user.sum(),match_item.sum())