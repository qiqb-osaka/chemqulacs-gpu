def distribute(Na, Np, rank):
    if Np > Na:
        if rank + 1 <= Na:
            return rank, 1
        else:
            return 1, 0
    base, leftover = divmod(Na, Np)
    if rank + 1 <= leftover:
        size = base + 1
        start = rank * (base + 1)
    else:
        size = base
        start = (base + 1) * leftover + (rank - leftover) * base
    return start, size


if __name__ == "__main__":
    print(distribute(5, 10, 8))
    print(distribute(100, 10, 2))
    for i in range(0, 5):
        print(i)
