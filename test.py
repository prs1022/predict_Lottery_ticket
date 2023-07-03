from itertools import permutations

def count_sequences(arr):
    count = 0
    for perm in permutations(arr):
        up = 0
        for i in range(len(perm)-2):
            if perm[i+1] > perm[i] and perm[i+2] > perm[i+1]:
                up+=1
        if up==1:
            count += 1
    return count  # 减去原始数列本身的数量

arr = [1, 2, 3, 4]  # 数列，可以替换为任意长度为n的数列
result = count_sequences(arr)
print(f"恰好只存在三个连续上升数字的排列有 {result} 个")