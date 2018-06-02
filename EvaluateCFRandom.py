import sys
import numpy as np
from random import randint
import statistics as s

#i use the transpose matrix for item based
def meanUser(userScores, itemsRatedPerUser, itemsRatedPerItem, user, item):
	tot = 0
	count = 0
	actualScore = userScores[user][item]
	userScores[user][item] = 99
	for row in userScores:
		if row[item] != 99:
			count += 1
			tot += row[item]
	userScores[user][item] = actualScore
	return tot/count


def main():
	#reading the data in
	method = sys.argv[1]
	size = int(sys.argv[2])
	repeats = int(sys.argv[3])

	f = open("jester-data-1.csv", "r")

	userScores = np.zeros((24983, 100))
	itemsRatedPerUser = [0 for i in range(24983)]
	itemsRatedPerItem = [0 for i in range(100)]

	i = 0
	for line in f:
		line = line.split(",")
		itemsRatedPerUser[i] = line[0]
		j = 0
		for item in line[1:]:
			score = float(item)
			userScores[i][j] = item
			if score != 99:
				itemsRatedPerItem[j] += 1
			j += 1
		i += 1 

	itemScores = np.transpose(userScores)
	#generate random  user item tuples
	#then perform method and find MAE
	MAEList = []
	for i in range(repeats):
		MAE = 0
		count = 0
		tupSet = set()
		while len(tupSet) < size:
			user = randint(0, 24982)
			item = randint(0, 99)
			if userScores[user][item] != 99:
				tupSet.add((user, item))
		if method == "meanUser":
			for tup in tupSet:
				actual = userScores[tup[0]][tup[1]]
				pred = meanUser(userScores, itemsRatedPerUser, itemsRatedPerItem, tup[0], tup[1])
				MAE += abs(actual - pred)
				count += 1
		if method == "meanItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = meanUser(itemScores, itemsRatedPerUser, itemsRatedPerItem, tup[1], tup[0])
				MAE += abs(actual - pred)
				count += 1		
		MAEList.append(MAE/count)
		print("Iteration ",i + 1," MAE: ",MAE/count)
	print("Mean of all MAE's is ",s.mean(MAEList))
	print("Standard deviation of all MAE's is ",s.stdev(MAEList))



if __name__ == "__main__":
	main()
