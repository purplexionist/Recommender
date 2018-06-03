import sys
import numpy as np
from random import randint
import statistics as s
import time

def pearson(row1, row2, row1Avg, row2Avg, finSet):
	top = 0
	bottomRow1 = 0
	bottomRow2 = 0
	for i in list(finSet)
		top += (row1[i] - row1Avg) * (row2[i] - row2Avg)
		bottomRow1 += (row1[i] - row1Avg) ** 2
		bottomRow2 += (row2[i] - row2Avg) ** 2
	return top/((bottomRow1 ** .5) * (bottomRow2 ** .5))

#i use the transpose matrix for item based
def meanUser(userScores, avg, numRated, first, second):
	temp = avg * numRated
	temp -= userScores[first][second]
	return temp/(numRated - 1)

def weightedUser(userScores, avgForIndividual, countForIndividual, first, second, allAvg, allCount, allSets):
	tot = 0
	kTot = 0
	actualScore = userScores[first][second]
	userScores[first][second] = 99
	userRow = userScores[first]
	userAvg = ((avgForIndividual * countForIndividual) - actualScore) / (countForIndividual - 1)
	for i in range(len(userScores)):
		row = userScores[i]
		if row[second] != 99:
			theirAvg = allAvg[i]
			finSet = allSets[first].intersection(allSets[i])
			finSet.remove(second)
			sim = pearson(userRow, row, userAvg, theirAvg, finSet)
			kTot += abs(sim)
			tot += sim * row[second]
	userScores[first][second] = actualScore
	return tot/kTot

def main():
	#reading the data in
	method = sys.argv[1]
	size = int(sys.argv[2])
	repeats = int(sys.argv[3])

	f = open("jester-data-1.csv", "r")

	userScores = np.zeros((24983, 100))
	itemsRatedPerUser = [0 for i in range(24983)]
	itemsRatedPerItem = [0 for i in range(100)]
	avgPerUser = [0 for i in range(24983)]
	avgPerItem = [0 for i in range(100)]
	userSet = [set() for i in range(24983)]
	itemSet = [set() for i in range(100)]

	i = 0
	for line in f:
		line = line.split(",")
		itemsRatedPerUser[i] = int(line[0])
		j = 0
		for item in line[1:]:
			score = float(item)
			userScores[i][j] = item
			if score != 99:
				itemsRatedPerItem[j] += 1
				avgPerUser[i] += score
				avgPerItem[j] += score
				userSet[i].add(j)
				itemSet[j].add(i)
			j += 1
		i += 1 


	for i in range(24983):
		avgPerUser[i] /= itemsRatedPerUser[i]

	for i in range(100):
		avgPerItem[i] /= itemsRatedPerItem[i]

	itemScores = np.transpose(userScores)

	#generate random  user item tuples
	#then perform method and find MAE
	MAEList = []
	start = time.clock()
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
				pred = meanUser(userScores, avgPerItem[tup[1]], itemsRatedPerItem[tup[1]], tup[0], tup[1])
				MAE += abs(actual - pred)
				count += 1
		if method == "meanItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = meanUser(itemScores, avgPerUser[tup[0]], itemsRatedPerUser[tup[0]], tup[1], tup[0])
				MAE += abs(actual - pred)
				count += 1	
		if method == "weightedUser":
			for tup in tupSet:
				actual = userScores[tup[0]][tup[1]]
				pred = weightedUser(userScores, avgPerUser[tup[0]], itemsRatedPerUser[tup[0]], tup[0], tup[1],
					avgPerUser, itemsRatedPerUser, userSet)
				MAE += abs(actual - pred)
				count += 1
		if method == "weightedItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = weightedUser(itemScores, avgPerItem[tup[1]], itemsRatedPerItem[tup[1]], tup[1], tup[0],
					avgPerItem, itemsRatedPerItem, itemSet)
				MAE += abs(actual - pred)
				count += 1

		MAEList.append(MAE/count)
		print("Iteration ",i + 1," MAE: ",MAE/count)
	print("Mean of all MAE's is ",s.mean(MAEList))
	if repeats > 1:
		print("Standard deviation of all MAE's is ",s.stdev(MAEList))
	print("final is ",time.clock() - start)



if __name__ == "__main__":
	main()
