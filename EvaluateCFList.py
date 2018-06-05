import sys
import numpy as np
from random import randint
import statistics as s
import time

def pearson(row1, row2, row1Avg, row2Avg, finSet):
	top = 0
	bottomRow1 = 0
	bottomRow2 = 0
	for i in list(finSet):
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

def knnUser(userScores, avgForIndividual, countForIndividual, first, second, allAvg, allCount, allSets, knnNum):
	tot = 0
	kTot = 0
	simList = []
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
			simList.append((sim, row[second]))
	thisSort = sorted(simList, key=lambda tup: abs(tup[0]), reverse = True)
	for i in range(knnNum):
		kTot += abs(thisSort[i][0])
		tot += thisSort[i][0] * thisSort[i][1]
	userScores[first][second] = actualScore
	return tot/kTot


def main():
	#reading the data in
	if len(sys.argv) < 3:
		print("Usage: python EvaluateCFList.py method filename [knnNeighbors]")
		print("Methods: meanUser meanItem weightedUser weightedItem knnUser knnItem")
		print("knnUser and knnItem must have the additional knnNeighbors flag to indicate the number of neighbors to use")
		return
	method = sys.argv[1]
	filename = sys.argv[2]
	knnNum = -1
	if(method == "knnUser" or method == "knnItem"):
		knnNum = int(sys.argv[3])

	f = open("jester-data-1.csv", "r")
	f2 = open("output.csv", "w")
	f3 = open(filename, "r")

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
	repeats = 1
	for i in range(repeats):
		MAE = 0
		count = 0
		tupSet = set()
		for line in f3:
			line = line.split(",")
			curFirst = int(line[0])
			curSecond = int(line[1])
			#assume that the users and items start at 1 and 1
			if userScores[curFirst - 1][curSecond - 1] != 99:
				tupSet.add((curFirst - 1, curSecond - 1))
			else:
				f2.write("Line " + str(line) + " is not valid\n")
		if method == "meanUser":
			for tup in tupSet:
				actual = userScores[tup[0]][tup[1]]
				pred = meanUser(userScores, avgPerItem[tup[1]], itemsRatedPerItem[tup[1]], tup[0], tup[1])
				MAE += abs(actual - pred)
				count += 1
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
		if method == "meanItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = meanUser(itemScores, avgPerUser[tup[0]], itemsRatedPerUser[tup[0]], tup[1], tup[0])
				MAE += abs(actual - pred)
				count += 1
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
		if method == "weightedUser":
			for tup in tupSet:
				actual = userScores[tup[0]][tup[1]]
				pred = weightedUser(userScores, avgPerUser[tup[0]], itemsRatedPerUser[tup[0]], tup[0], tup[1],
					avgPerUser, itemsRatedPerUser, userSet)
				MAE += abs(actual - pred)
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
				count += 1
		if method == "weightedItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = weightedUser(itemScores, avgPerItem[tup[1]], itemsRatedPerItem[tup[1]], tup[1], tup[0],
					avgPerItem, itemsRatedPerItem, itemSet)
				MAE += abs(actual - pred)
				count += 1
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
		if method == "knnUser":
			for tup in tupSet:
				actual = userScores[tup[0]][tup[1]]
				pred = knnUser(userScores, avgPerUser[tup[0]], itemsRatedPerUser[tup[0]], tup[0], tup[1],
					avgPerUser, itemsRatedPerUser, userSet, knnNum)
				MAE += abs(actual - pred)
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
				count += 1
		if method == "knnItem":
			for tup in tupSet:
				actual = itemScores[tup[1]][tup[0]]
				pred = knnUser(itemScores, avgPerItem[tup[1]], itemsRatedPerItem[tup[1]], tup[1], tup[0],
					avgPerItem, itemsRatedPerItem, itemSet, knnNum)
				MAE += abs(actual - pred)
				f2.write(str(tup[0] + 1) + ","  + str(tup[1] + 1) + "," + str(actual) + ","  + str(pred) + "," + str(actual - pred) +"\n")
				count += 1
		f2.write("MAE is: " + str(MAE/count))
	print("final time is ",time.clock() - start)



if __name__ == "__main__":
	main()
