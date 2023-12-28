import coverage

program = ""

cov = coverage.Coverage()
cov.start()

def get_row(lst, x):
    coords = [(i, j) for i in range(len(lst)) for j in range(len(lst[i])) if lst[i][j] == x]
    return sorted(sorted(coords, key=lambda x: x[1], reverse=True), key=lambda x: x[0])
assert get_row([[1, 2], [3, 4], [5, 6]], 0) == []
assert get_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0) == []
assert get_row([[1, 2], [3, 4], [5, 6]], 7) == []
assert get_row([[1,2,3],[4,5,6]], 0) == []
assert get_row([[1,2,3],[4,5,6]], 7) == []
assert get_row([], 2) == []
assert get_row([[1, 2], [3, 4]], 0) == []

cov.stop()
cov.save()

cov.html_report()