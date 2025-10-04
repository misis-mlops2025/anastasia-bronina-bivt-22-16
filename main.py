def sum_two_numbers(a: int, b: int):
	return a + b;

def main():
	a, b = map(int, input().split())
	print(sum_two_numbers(a, b))

if __name__ == '__main__':
	main()