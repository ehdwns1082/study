
import numpy as np

a = np.arange(12)
print('a = ', a)
print(' ')
print('a.shape = ', a.shape)
print(' ')

b = a.reshape(4,3)
print('b = ', b)
print(' ')

c = np.arange(16).reshape(4,4)
print('c = ', c)
print(' ')

# np.hsplit(array, indices_or_sections) : horizontal_split

# ex1) Section 활용
print('====== np.hsplit() Section 활용 ======')
print('np.hsplit(c, 2) = ', np.hsplit(c, 2))
print(' ')
print('np.split(c, 2, axis=1) = ', np.split(c, 2, axis=1))
print(' ')

print('np.hsplit(a, 3)  = ', np.hsplit(a, 3))
print(' ')
print('np.split(a, 3, axis=0) = ', np.split(a, 3, axis=0))
print(' ')
'''
axis = 0 : shape[0] => 행
axis = 1 : shape[1] => 열
* np.split(a, 3, axis=1) 에서 axis = 1 일 때 오류가 나는 이유는 a.shape = (12,) 으로 a 의 열에 해당하는 값이 없다. tuple index out of range
'''
# ex2) Indices 활용
print('====== np.hsplit() Indices 활용 ======')
print('np.hsplit(a, [3,6]) = ', np.hsplit(a, [3,6]))
print(' ')
# [3,6] 은 인덱스로 [0:3], [3:6], [6:] 과 같은 의미이다

print('np.hsplit(a,[0,3]) = ', np.hsplit(a,[0,3]))
print(' ')
# [0,3] 은 인덱스로 [], [0:3], [3:]

print('np.split(b, [0], 1) = ', np.split(b, [0], 1))
# [0] 은 인덱스로 [:0]. [0:] 에 해당한다.
print(' ')

print('np.split(b, [1], 1) = ', np.split(b, [1], 1))
# [1] => [:1], [:1]
print(' ')

print('np.split(b, [2], 1) = ', np.split(b, [2], 1) )
print(' ')

print('np.split(b, [3], 1) = ', np.split(b, [3], 1) )
print(' ')

print('np.split(b, [4], 1) = ', np.split(b, [4], 1) )
# 열의 개수를 넘는 인덱스의 경우, 열의 개수를 기준으로 결과를 출력한다.
print(' ')



# np.vsplit(array, indices_or_sections) : vertical_split
# ex1) Section 활용
print('====== np.vsplit() Section 활용 ======')
print('np.vsplit(b,2) = ', np.vsplit(b,2))
print(' ')

print('np.split(b, 2, axis=0) = ', np.split(b, 2, axis=0))
print(' ')
'''
np.split(b, 2, axis=1) 에서 오류가 나는 이유는 b.shape(4,3) 에서 3열을 2부분으로 분리할 수 없기 때문이다.
'''

# ex2) indices 활용
print('====== np.vsplit() Indices 활용 ======')
print('np.split(b, [0], 0) = ', np.split(b, [0], 0))
# [0] 은 인덱스로 [:0]. [0:] 에 해당한다.
print(' ')

print('np.split(b, [1], 0) = ', np.split(b, [1], 0))
# [1] => [:1], [:1]
print(' ')

print('np.split(b, [2], 0) = ', np.split(b, [2], 0) )
print(' ')

print('np.split(b, [3], 0) = ', np.split(b, [3], 0) )
print(' ')

print('np.split(b, [4], 0) = ', np.split(b, [4], 0) )
print(' ')

print('np.split(b, [4], 0) = ', np.split(b, [5], 0) )
# 행의 개수를 넘는 인덱스의 경우, 행의 개수를 기준으로 결과를 출력한다.
print(' ')

'''
b =  [[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]
      [ 9 10 11]]

'''