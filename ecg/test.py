f = open('log_save.txt')

newlog = []

for line in f:

	val = line.strip().replace(",","").split(': ')
	val = float(val[-1])

	print(val)