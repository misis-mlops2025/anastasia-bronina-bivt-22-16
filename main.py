def sum_three_numbers(a: int, b: int, c: int):
	return a + b + c;

def main():
	a, b, c = map(int, input().split())
	print(sum_three_numbers(a, b, c))

if __name__ == '__main__':
	main()