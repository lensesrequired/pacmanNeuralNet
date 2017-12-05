def trivial():
	lower, upper = input().strip().split(" ")
	s = 0
	for i in range(int(lower), int(upper)+1):
		s += i
	print(s)

trivial()