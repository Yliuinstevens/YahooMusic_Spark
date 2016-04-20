from __future__ import print_function
import numpy as np

def read_lines(file, num):
	lines = []
	line = file.readline()
	if line:
		lines.append(line.strip().split("|"))
		for i in range(1,num):
			lines.append(file.readline().strip().split("|"))
		return lines
	else:
		return line

temp_dic = {}
with open("Results/prediction.txt","w") as predFile:
	with open("Results/raw_prediction.txt") as rawFile:
		with open("RawData/testIdx2.txt") as testFile:
			raw_lines = read_lines(rawFile,6)
			while raw_lines:
				temp_dic.clear()
				user_median = np.median([int(x[2]) for x in raw_lines])
				for row in raw_lines:
					if int(row[2])>=user_median:
						temp_dic[str(row[1])]="1"
					else:
						temp_dic[str(row[1])]="0"
				test_line = testFile.readline()
				for i in range(6):
					test_line = testFile.readline().strip()
					predFile.write(temp_dic[test_line]+"\n")
				raw_lines = read_lines(rawFile,6)
