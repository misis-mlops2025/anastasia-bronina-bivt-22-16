def sum_numbers(*args):
	return sum(args)

def main():
	numbers = map(int, input().split())
	print(sum_numbers(*numbers))

if __name__ == '__main__':
	main()