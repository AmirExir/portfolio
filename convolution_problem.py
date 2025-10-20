"""
Convolution Problem Solver

Question: Consider the basic image example given above. Denote the matrix 
representation of this image as A; note that there is no padding. Suppose we 
convolute this image using a stride of 1, no bias, and the following 3×3 filter:

Image Matrix A (5×5):
1  1  1  0  0
0  1  1  1  0
0  0  1  1  1
0  0  1  1  0
0  1  1  0  0

Filter B (3×3):
1  0  1
0  1  0
1  0  1

The result of this convolution operation is a 3x3 matrix Ã, whose entry in 
the i-th row and j-th column we denote by (ã_i,j).

Multiple Choice Options:
1. The sum of the first row in Ã equals 10.
2. The sum of the third column in Ã equals the sum of the first row in Ã.
3. The sum of all elements in Ã equals 30.
4. ã_3,3 = 3
5. ã_2,2 > 5
"""

import numpy as np

# Define the image matrix A
A = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Define the filter B
B = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

print("Image Matrix A:")
print(A)
print("\nFilter B:")
print(B)

# Perform convolution manually with stride=1, no padding
# Output will be (5-3+1) x (5-3+1) = 3x3
output_size = A.shape[0] - B.shape[0] + 1
A_tilde = np.zeros((output_size, output_size))

print("\n" + "="*60)
print("COMPUTING CONVOLUTION (stride=1, no padding, no bias)")
print("="*60)

for i in range(output_size):
    for j in range(output_size):
        # Extract the patch from A
        patch = A[i:i+3, j:j+3]
        
        # Perform element-wise multiplication and sum
        conv_value = np.sum(patch * B)
        A_tilde[i, j] = conv_value
        
        print(f"\nã_{i+1},{j+1} (position [{i},{j}]):")
        print(f"Patch from A:")
        print(patch)
        print(f"Element-wise multiplication with B:")
        print(patch * B)
        print(f"Sum = {conv_value}")

print("\n" + "="*60)
print("RESULT: Output Matrix Ã")
print("="*60)
print(A_tilde.astype(int))

# Now check each statement
print("\n" + "="*60)
print("CHECKING EACH STATEMENT")
print("="*60)

# Statement 1: The sum of the first row in Ã equals 10
first_row_sum = np.sum(A_tilde[0, :])
print(f"\n1. Sum of first row in Ã: {first_row_sum}")
print(f"   Equals 10? {first_row_sum == 10} {'✓' if first_row_sum == 10 else '✗'}")

# Statement 2: The sum of the third column in Ã equals the sum of the first row in Ã
third_col_sum = np.sum(A_tilde[:, 2])
print(f"\n2. Sum of third column in Ã: {third_col_sum}")
print(f"   Sum of first row in Ã: {first_row_sum}")
print(f"   Are they equal? {third_col_sum == first_row_sum} {'✓' if third_col_sum == first_row_sum else '✗'}")

# Statement 3: The sum of all elements in Ã equals 30
total_sum = np.sum(A_tilde)
print(f"\n3. Sum of all elements in Ã: {total_sum}")
print(f"   Equals 30? {total_sum == 30} {'✓' if total_sum == 30 else '✗'}")

# Statement 4: ã_3,3 = 3
a_3_3 = A_tilde[2, 2]
print(f"\n4. ã_3,3 = {a_3_3}")
print(f"   Equals 3? {a_3_3 == 3} {'✓' if a_3_3 == 3 else '✗'}")

# Statement 5: ã_2,2 > 5
a_2_2 = A_tilde[1, 1]
print(f"\n5. ã_2,2 = {a_2_2}")
print(f"   Greater than 5? {a_2_2 > 5} {'✓' if a_2_2 > 5 else '✗'}")

print("\n" + "="*60)
print("ANSWER")
print("="*60)

statements = [
    (first_row_sum == 10, "The sum of the first row in Ã equals 10"),
    (third_col_sum == first_row_sum, "The sum of the third column in Ã equals the sum of the first row in Ã"),
    (total_sum == 30, "The sum of all elements in Ã equals 30"),
    (a_3_3 == 3, "ã_3,3 = 3"),
    (a_2_2 > 5, "ã_2,2 > 5")
]

for idx, (is_true, statement) in enumerate(statements, 1):
    if is_true:
        print(f"\n✓ CORRECT STATEMENT #{idx}: {statement}")

# Additional verification with detailed breakdown
print("\n" + "="*60)
print("DETAILED MATRIX Ã BREAKDOWN")
print("="*60)
print(f"\nÃ = ")
print(f"    [{int(A_tilde[0,0])}  {int(A_tilde[0,1])}  {int(A_tilde[0,2])}]")
print(f"    [{int(A_tilde[1,0])}  {int(A_tilde[1,1])}  {int(A_tilde[1,2])}]")
print(f"    [{int(A_tilde[2,0])}  {int(A_tilde[2,1])}  {int(A_tilde[2,2])}]")

print(f"\nRow sums: [{int(np.sum(A_tilde[0,:]))}  {int(np.sum(A_tilde[1,:]))}  {int(np.sum(A_tilde[2,:]))}]")
print(f"Column sums: [{int(np.sum(A_tilde[:,0]))}  {int(np.sum(A_tilde[:,1]))}  {int(np.sum(A_tilde[:,2]))}]")
print(f"Total sum: {int(total_sum)}")
