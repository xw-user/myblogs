[b站视频](https://www.bilibili.com/video/BV1Ex411L7oT)

## numpy安装

```python
python -m pip install --upgrade pip

pip install numpy
```



## numpy属性


```python
import numpy as np

array = np.array([[1,2,3],[2,3,4]])
print(array)
# 维度
print('number of dimenson:', array.ndim)

# 形状
print('shape:', array.shape)

# 元素个数
print('size:', array.size)
```

    [[1 2 3]
     [2 3 4]]
    number of dimenson: 2
    shape: (2, 3)
    size: 6


## 创建数组


```python
# 指定数据类型
a = np.array([1, 2, 3], dtype=np.float)
print(a.dtype)
print(a)
```

    float64
    [1. 2. 3.]



```python
# 全0矩阵
a = np.zeros((3, 4), dtype=np.int)
print(a)
```

    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]



```python
# 全1矩阵
a = np.ones((3, 4))
print(a)
```

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]



```python
# 未初始化的矩阵
a = np.empty((3, 4))
print(a)
```

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]



```python
# 指定元素范围，[10， 20) 步长为2
a = np.arange(10, 20, 2)
print(a)
```

    [10 12 14 16 18]



```python
# 指定矩阵形状
a = np.arange(10).reshape((5, 2))
print(a)
```

    [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]



```python
# 等差数列，[1,10]分成3份
a = np.linspace(1, 10, 6).reshape((2, 3))
print(a)
```

    [[ 1.   2.8  4.6]
     [ 6.4  8.2 10. ]]


## 基础运算


```python
# 数组的加减乘除
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
print('a:', a)
print('b:', b)

c = a + b
print('a+b:', c)

c = a - b
print('a-b:', c)

c = a * b
print('a*b:', c)

c = a / b
print('a/b:', c)

c = a**2
print('a^3:', c)
```

    a: [10 20 30 40]
    b: [1 2 3 4]
    a+b: [11 22 33 44]
    a-b: [ 9 18 27 36]
    a*b: [ 10  40  90 160]
    a/b: [10. 10. 10. 10.]
    a^3: [ 100  400  900 1600]



```python
# 三角函数
a = np.array([30, 45, 60, 90]) # 默认弧度制
pi = np.arcsin(1)*2
print(np.pi)

a = np.multiply(a, pi/180)

print('a:', a)

c = np.sin(a)
print(c)

c = np.cos(a)
print(c)

c = np.tan(a)
print(c)
```

    3.141592653589793
    a: [0.52359878 0.78539816 1.04719755 1.57079633]
    [0.5        0.70710678 0.8660254  1.        ]
    [8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]
    [5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]



```python
# 随机生成矩阵
a = np.random.random((2, 4))
print(a)
print('min:', np.min(a))
print('max:', np.max(a))
print('sum:', np.sum(a))

# 1: 行 0: 列
print('min in each col:', np.min(a, axis=0))
print('min in each row:', np.min(a, axis=1))
```

    [[0.33028884 0.20715399 0.67513249 0.7976253 ]
     [0.09172116 0.65765596 0.04538128 0.36034413]]
    min: 0.045381281632210335
    max: 0.7976253001665536
    sum: 3.165303150763041
    min in each col: [0.09172116 0.20715399 0.04538128 0.36034413]
    min in each row: [0.20715399 0.04538128]



```python
# 数值运算
a = np.arange(1, 13).reshape(3, 4)
print(a)

# 最小值索引
print(np.argmin(a))

# 最大值索引
print(np.argmax(a))
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    0
    11



```python
# 平均值
print(np.average(a))
print(np.mean(a))
print(a.mean())
```

    6.5
    6.5
    6.5



```python
# 中位数
print(np.median(a))
```

    6.5



```python
# 前缀和
print(np.cumsum(a))
```

    [ 1  3  6 10 15 21 28 36 45 55 66 78]



```python
# 一阶差
print(np.diff(a))
```

    [[1 1 1]
     [1 1 1]
     [1 1 1]]



```python
# 非零的坐标
print(np.nonzero(a))
```

    (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))



```python
# 逐行排序
a = np.arange(14, 2, -1).reshape(3,4)
print(np.sort(a))
```

    [[11 12 13 14]
     [ 7  8  9 10]
     [ 3  4  5  6]]



```python
# clip
print(a)
print(np.clip(a, 5, 10))
```

    [[14 13 12 11]
     [10  9  8  7]
     [ 6  5  4  3]]
    [[10 10 10 10]
     [10  9  8  7]
     [ 6  5  5  5]]


## 矩阵运算


```python
# 矩阵相乘
a = np.array([[1,0], [1,2]])
b = np.arange(4).reshape(2,2)
print(a)
print(b)

c = np.dot(a, b)
# c = a.dot(b)
print(c)
```

    [[1 0]
     [1 2]]
    [[0 1]
     [2 3]]
    [[0 1]
     [4 7]]



```python
# 矩阵转置
print(a)
print(np.transpose(a))
print(a.T)
```

    [[1 0]
     [1 2]]
    [[1 1]
     [0 2]]
    [[1 1]
     [0 2]]


## 索引


```python
a = np.arange(1, 13).reshape(3, 4)
print(a)

# 第1行
print(a[1])
print(a[1, :])
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    [5 6 7 8]
    [5 6 7 8]



```python
# 第1列
print(a[:,1])
```

    [ 2  6 10]



```python
# 第1行的[1,3)
print(a[1, 1:3])
```

    [6 7]



```python
# 第（2，3）
print(a[2][3])
print(a[2, 3])
```

    12
    12



```python
# 迭代行
for row in a:
    print(row)
```

    [1 2 3 4]
    [5 6 7 8]
    [ 9 10 11 12]



```python
# 迭代列
for row in a.T:
    print(row)
```

    [1 5 9]
    [ 2  6 10]
    [ 3  7 11]
    [ 4  8 12]



```python
# 迭代每一个元素
print(a.flatten())

for item in a.flat:
    print(item)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12


## array合并


```python
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])

# 垂直合并
C = np.vstack((A, B))
print(C)

# 水平合并
D = np.hstack((A, B))
print(D)
```

    [[1 1 1]
     [2 2 2]]
    [1 1 1 2 2 2]



```python
# 数组转为行向量
print(A)
A = A[np.newaxis, :]
print(A)
```

    [1 1 1]
    [[1 1 1]]



```python
# 数组转为列向量
print(B)
B = B[:, np.newaxis]
print(B)
```

    [2 2 2]
    [[2]
     [2]
     [2]]



```python
# concatenate合并
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]
print(A)
print(B)

# 垂直合并
C = np.concatenate((A,B,B,A), axis=0)

# 水平合并
D = np.concatenate((A,B,B,A), axis=1)
print(C)
print(D)
```

    [[1]
     [1]
     [1]]
    [[2]
     [2]
     [2]]
    [[1]
     [1]
     [1]
     [2]
     [2]
     [2]
     [2]
     [2]
     [2]
     [1]
     [1]
     [1]]
    [[1 2 2 1]
     [1 2 2 1]
     [1 2 2 1]]


## array分割


```python
# 等量分割
A = np.arange(12).reshape((3, 4))
print(A)

# 上下分割成3份
print(np.split(A, 3, axis=0))
print(np.vsplit(A, 3))

# 左右分割2份
print(np.split(A, 2, axis=1))
print(np.hsplit(A, 2))
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]



```python
# 不等量分割
print(A)
t = np.array_split(A, 3, axis=1)
print(np.array_split(A, 3, axis=1))
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2],
           [ 6],
           [10]]), array([[ 3],
           [ 7],
           [11]])]


## copy


```python
# 默认浅拷贝
a = np.arange(4)
b = a
a[1] = 5

print(a)
print(b)
```

    [0 5 2 3]
    [0 5 2 3]



```python
# 深拷贝
a = np.arange(4)
b = a.copy()
a[1] = 5

print(a)
print(b)
```

    [0 5 2 3]
    [0 1 2 3]

## 01-Numpy初识

```python
import numpy as np
#  创建ndarray的方式
# 1.1使用array函数，传入一个列表初始化一个ndarray
n1 = np.arange(5)
print(n1)
print(type(n1))
# [0 1 2 3 4]
# <class 'numpy.ndarray'>

# 1.2使用arange（）函数
nd2=np.arrange(5)
nd2 = np.arange(1,11) #[ 1  2  3  4  5  6  7  8  9 10]

# 1.3使用array函数，传入range对象
nd2 = np.array(range(1,10)) #[1 2 3 4 5 6 7 8 9]
print(nd2)

# ------------以上创建的ndarray是一维向量
#  2创建n维ndarray
# 2.1 创建二维ndrray
nd3 = np.array([1,2,3,4],ndmin=2) #[[1 2 3 4]]
print(nd3)

# 2.2 查看ndarray的维度（秩）
print(f"nd3的维度，{nd3.ndim}")
# 2.3 查看ndarray的形状
print(f"nd3的形状，{nd3.shape}") #nd3的形状，(1, 4)
# 2.4 取矩阵的行跟列
print("nd3矩阵的行，{nd3.sahpe[0]}")
print("nd3矩阵的列，{nd3.sahpe[1]}")
# nd3矩阵的行，{nd3.sahpe[0]}
# nd3矩阵的列，{nd3.sahpe[1]}
# 2.5 根据ndarray返回的shape元祖的长度就是nadarry的秩
# 创建三维nadarray
nd4=np.array([1,2,3,4],ndmin=3)
print(nd4)
print(f"三维nd4的形状：{nd4.shape}")
#[[[1 2 3 4]]]

nd5 = np.array([1,2,3])
print(f"一维nd5的形状是：{nd5.shape}") #(3, )

#2.2 以2.1中的nd3为例
#矩阵的形状可以改变
#2.2.1 改变矩阵的形状
nd3 = np.array([1,2,3,4],ndmin=2)
print(f"nd3的原形状，{nd3.shape}") #（1,4）
# 2.2.1 改变矩阵的形状 直接通过shape属性来进行形状调整
nd3.shape = (2,2)
print(f"nd3改变之后的形状，{nd3.shape}")#[
#     [1 2]
#     [3 4]
# ]

#2.2.2 使用reshape（）方法来对矩阵进行改变形状， 但该方法不会影响到原矩阵， 而是返回一个新的矩阵
new_nd = nd3.reshape(4,1)
print(f"new_nd的矩阵元素是：{new_nd}")
print(f"nd3中矩阵的元素是：\n{nd3}")

#----------- 3 创建三维矩阵
nd_3 = np.arange(24).reshape(2,3,4)
print(f"三维nd_3矩阵元素：\n{nd_3}")
print(f"三维nd_3矩阵的形状：\n{nd_3.shape}")
# 三维 两个深度的三行四列


#练习：创建一个包含1-10之间所有偶数的向量
import numpy as np
n_1 = np.array(range(2,11,2))
print (n_1)
#练习：创建一个包含两个深度，第一个深度为1-8之间
import numpy as np
j_num = [x for x in range(1,8) if x%2 == 1]
o_num = [x for x in range(1,9) if x%2 == 0]
n_2 = np.array([[j_num,o_num]]).reshape(2,2,2)
print(n_2)

```

## 02-Numpy创建ndarray的其他方式

```python
import numpy as np

#zeros
# 创建的全0向量
nd_zero_1 = np.zeros(5)
print(nd_zero_1) 
# 创建的全0矩阵
nd_zero_1= np.zeros((3,2))
print(nd_zero_1)

#zeros_like
nd = np.arange(9).reshape(3,3)
nd_zero = np.zeros_like(nd)
print(nd_zero)
# zeros_likes是int类型，zeros是float类型

#ones
#创建的全1的
np_one  = np.ones(5)
nd_one = np.ones((3,3))
print(nd_one)

#ps：ones_like用法和zeros——like一样，根据传入的矩阵返回相同形状和数据类型的全1矩阵

#创建方阵
np.eye(5)

# linspace
print(np.linspace(1,3,6)) #[1.  1.4 1.8 2.2 2.6 3. ]
print(np.linspace(1,3,6).reshape(2,3))

# 对角矩阵
print(np.diag([1,2,3,4]))

#随机数矩阵
# 包含low，不包含high（前闭后开）
np.random.randint(100,301,size=(3,4))
nd = np.random.randint(1,3,size=10) #[2 2 2 2 1 2 2 1 1 2]
nd = np.random.randint(1,3,size=(10,))
print(f"100~301之间随机抽取数字组成3*4的矩阵：\n{nd}")

# 生成0-1区间的随机array
print(np.random.rand(5)) #[0.08693665 0.53849503 0.0657666  0.20866307 0.9438665 ]
print(np.random.rand(8).reshape(2,4))
```



## 03-Numpy维度总结

```python
import numpy as np

# 1维 --> 向量 只包含一个0轴
print(np.array([1,2,3,4])) #[1 2 3 4]

# 2维 -->面（矩阵） 包含一个0（行）轴一个1（列）轴
print(np.array([1,2,3,4,5,6]).reshape(2,3))
'''
[[1 2 3]
 [4 5 6]]
'''

# 3维 --> 体 （带有深度的矩阵） 包含0深度轴，1（行）轴，2（列）轴
nd_3 = np.arange(12).reshape(2,2,3)
print(nd_3)
'''
[[[ 0  1  2]
  [ 3  4  5]]

 [[ 6  7  8]
  [ 9 10 11]]]
'''
print("-" * 50)
# 关于ndarray的属性
# 维度（秩）
print(nd_3.ndim)
# 形状（元组）
print(nd_3.shape)
# size (矩阵的元素个数)
print(nd_3.size)
# 数据类型dtype
print(f"原数据类型：{nd_3.dtype}")
# 将int32 转换成float32
# 修改nd_3数据类型为float
# astype（）该方法修改数据类型属于非原地操作，返回一个新的ndarray
nd_3 = nd_3.astype('f2')

print(f"数据类型：{nd_3.dtype}")


# 数据类型转换
# int --> bool
# int转bool的转换机制  非0即true
n1 = np.array([1,2,0,4,5,6], dtype=np.bool).reshape(2,3)
print(f"n1矩阵：\n{n1}")

# bool转int
n2 = np.array(n1, dtype='i4')
print(n2)

```

## 04-Numpy数据类型转换

```python
import numpy as np

# 创建矩阵
# int32 ->float 32 低精度 -》高精度
nd = np.arange(12).reshape((2,2,3)).astype('f4')
print(nd)
'''
[[[ 0.  1.  2.]
  [ 3.  4.  5.]]

 [[ 6.  7.  8.]
  [ 9. 10. 11.]]]
  '''
print(nd.dtype) # float32

# float -> int 高精度-》低精度（丢失精度）
nd = np.random.rand(12).reshape(2,6).astype('i4')
print(nd)

# int -> bool 转换机制非0转True
nd = np.arange(12).reshape((2,2,3))
print(nd)
print(nd.dtype)
print('*'*50)
nd = nd.astype(np.bool)
print(nd)
print(nd.dtype)

# bool -> int False-》0 True-》1
nd = nd.astype(np.int16)
print(nd)

#字符串矩阵
str_nd = np.array(['hello','world','python','data']).reshape(2,2)
print(str_nd)
print(str_nd.dtype) #字符串最长的长度

str_nd_num = np.array(['2.2','2.3','2.4','2.5']).reshape(2,2)

# 将字符串数字矩阵转换成int矩阵
str_nd_num = str_nd_num.astype('f4').astype('i4')
print(str_nd_num)

# 将int矩阵转换成str矩阵
nd = np.arange(9).reshape(3,3).astype('<U1')
print(nd)
print(nd.dtype)

#  -------------------自定义数据类型（基于内置的数据类型int，float，bool，<U)
# 1.创建dtype对象
mydtype = np.dtype([('age',np.int32)])
# 2.使用自定义dtype创建矩阵
age_nd = np.array([18,19,20],dtype=mydtype)
print(age_nd)
print(age_nd.dtype)

# 练习：样本数据信息矩阵
stu_dtype = np.dtype([('name','<U10'),('age','i4'),('score','f4'),('id','<U10')])
# 创建样本矩阵
stu = np.array(
  [
    ('徐卫中',21,100,'180827147')
    ('王建国',22,99,'180827137')
    ('王永辉',21,98,'180827140')
  ],dtype = stu_dtype).reshape(3,1)
print(stu)
print(stu.dtype)
```



## 05-Numpy保存与读取外部文件

```python
import numpy as np
import csv

# 保存文件（了解）
data = np.arange(12).reshape(3, 4)
print(data)
# 定义保存的数据文件路径
data_path = './data/savedata.txt'
# 将data（ndarray）保存到文件中, 默认数据的格式是按照%.18e保存
np.savetxt(data_path, data)

# 通过指定fmt参数，来决定保存的文件数据格式
np.savetxt(data_path, data, fmt='%d')

# 通过指定delimiter, 决定保存文件数据之间的连接符
np.savetxt(data_path, data, fmt='%.2f', delimiter=',')


# 读取文件（必须掌握）
data_from_txt = np.loadtxt(data_path, delimiter=',')
print(f"从./data/savedata.txt读取到的数据为：\n{data_from_txt}")
print(f"从./data/savedata.txt读取到的数据类型为：\n{data_from_txt.dtype}")

# 从csv中读取数据(必须掌握)
# csv是一种数据集，经常用于大数据分析。
# 绝大部分情况我们的在机器学习中使用的数据都来自于某些大型科学数据统计网站提供，比如sklearn，kaggle
# 这些网站的大部分数据都是以csv文件来进行存储
# 定义数据路径
file_path  = './data/data2/US_video_data_numbers.csv'
# 读取
youtube_us_data = np.loadtxt(file_path,delimiter=',',dtype='i4')
print(youtube_us_data)
print(youtube_us_data.shape)

# 读取Iris数据集
file_path = './data/data1/Iris.csv'
# 读取
iris_data = np.loadtxt(file_path,delimiter=',')
print(iris_data)

# 当你的csv数据集中如果包含特征值，那么此时我们只能通过python中的io进行读取
# 需要注意的是，由于io会堵塞线程，所以该读取方法只适合小数据量
iris_data = []
with open(file_path,'r')as f:
    # 1. 根据文件流对象得到csv读取对象
    csv_reader = csv.reader(f)
    # 2. 启动生成器对象读取每一行
    feature = next(csv_reader)
    # ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    # print(feature)
    # 遍历
    for row in csv_reader:
        # 清理特征值，第一列id不要的
        iris_data.append(tuple(row[1:]))

# 自定义数据类型
data_custom_dtype = np.dtype(
    [ ('Sepal.Length', '<U40'),  # 花萼的长度
    ('Speal.Width', '<U40'),   # 宽
    ('petal.Length', '<U40'),  # 花瓣长
    ('petal.Width', '<U40'),   # 宽
    ('Species', '<U40') ]      # 种类
)
# 使用自定义数据类型创建矩阵
iris = np.array(iris_data,dtype=data_custom_dtype)
print(iris.shape)
```

## 06-Numpy的索引与切片

```python
import numpy as np

# 1维向量的索引与切片
nd = np.arange(10)
print(f'原矩阵元素：\n{nd}')

# 通过索引来取值
print(nd[7])
# 直接赋值
# bool -> int
nd[7] = True
# float -> int
nd[7] = 12.33
print(nd)
#通过切片来取值(取出子向量)
print(nd[7:8])
# 通过切片进行赋值
nd[3:6]=[9,9,9] #[0 1 2 9 9 9 6 7 8 9]
print(nd)

# ndarray通过切片取出的子向量进行赋值的时候属于原地操作
sub = nd[3:6].copy()
sub[0] = 99
print(sub)
print(nd)

# -----------------二维矩阵的索引与切片
nd = np.arange(12).reshape(3,4)
print(nd)

# 索引取值
print(nd[0][1])   
print(nd[0,1])

# 赋值
nd[2,2] = 99
print(f"修改后：\n{nd}")

# 切片取值（子矩阵）
print(f"取前两行前两列：\n{nd[:2, :2]}")
# 再取子矩阵的值
print(nd[:2, :2][0, 1])
# 练习： 存在以下矩阵，根据以下矩阵截取子矩阵并组成全新矩阵
# 45 46
# 54 53
ex = np.arange(64).reshape(8,8)

# 先拆后组
sub_1 = ex[5:6,5:7]
sub_2 = ex[6:7,5:7][:,-1::-1]

print(np.array([sub_1,sub_2]))

#------------------------------------3维的索引与切片
nd = np.arange(18).reshape((2,3,3))
print(f"原矩阵：\n{nd}")

# 取值
print(nd[0][0][0])
print(nd[0, 0, 0])
# 索引数 < 轴数
# 取行
print(nd[0, 0])
# 取子矩阵（第0个深度的矩阵）
print(nd[0])

# 切片
print(nd[:, 1:2, 1:])
print(nd[:, 1:2, 1:])
print(nd[:1, -1:0:-1, -1:1:-1])

# 赋值子矩阵
nd[:1, :2, :3] = 999
print(f"赋值后的矩阵：\n{nd}")

# 考虑形状问题
# 子矩阵进行赋值的时候，先看其形状， 该子矩阵比较特殊虽然为三维矩阵但是只有一个深度，所以在赋值时可以直接不考虑深度
# （1,2,3）  -----  （2,3）
nd[:1, :2, :3] = np.array([[11, 22, 33], [44, 55, 66]])
print(f"赋值后的矩阵：\n{nd}")
print(nd[:, :2, :3])
print("-" * 50)
nd[:, :2, :3] = np.array([[11, 22, 33], [44, 55, 66]])
print(nd)
# 特别注意： 子矩阵赋值的时候， 不仅仅是考虑元素的个数，还要保证赋值的矩阵和被赋值的子矩阵行列一致



```

## 07-Numpy的花式索引

```python
import numpy as np


# 花式索引
# 可以指定传入的整数数组来进行取值

# 切片
# 切片只能以相同的步长或者切片规则来进行取值（连续的或者间隔相同的）

# 花式索引就是为了解决切片不能做的问题

# 二维 花式
nd = np.arange(64).reshape(8, 8)
print(f"原矩阵：\n{nd}")
# 取单一的值
print(nd[5, 5]) #45
# 输入的索引<轴数 取的是向量
print(nd[5]) #[40 41 42 43 44 45 46 47]
# 切片取连续的行或者间隔相同的行、列
print(nd[1:6:2, :])#取第一行到第六行之间每间隔2行

# 取0， 3， 2， 1行
# 取不连续行利用花式索引
print(nd[[0, 3, 1, 2]][[0, 1]])# 取0,3,1,2的行，再取0~1行

# 取多个值
# 0,0 2,1 1,3
print(nd[[0, 2, 1], [0, 1, 3]])

# 取区域（子矩阵）
# 19 18
# 27 26
print(nd[[2, 3]][:, [3, 2]])

print(nd[[1, 4, 3, 2]][:2, [3, 1, 2]])
#[[11  9 10]
# [35 33 34]]

# 练习
arr = np.arange(32).reshape(8, 4)
print(f"原矩阵：\n{arr}")
print(arr[[0, 4, 5], [0, 2, 1]])   # 0,0 4,2 5,1
print(arr[::1, [3, 1]])            
print(arr[3:4, 2:])       
print(arr[[0, 4, 5]][::2, [0, 3, 1, 2]])
print(arr[3:4, 2:3].shape)


# 三维的花式索引
nd = np.arange(18).reshape((2, 3, 3))
print(nd)

# 取值
print(nd[0, 0, 0])

# 当索引数《轴数
print(nd[0, 1]) # [3 4 5]

# 通过切片取值
print(nd[:1, 1:, 1:])# 空间，行，列

# 0,1,2  1,1,2
print(nd[0, 1, 2])
print(nd[1, 1, 2])
print(nd[[0, 1], [1, 1], [2, 2]])
print(nd[:, 1:2, 2:3])

# 通过花式索引取值
print(nd[[1, 0]][:, [1, 2], [1, 2]])
print(nd[[1, 0]][:, :1, [2, 1]])
print(nd[[1, 0]][-1:-2:-1, :1, [2, 1]])

# 14 13
# 17 16
print(nd[[1, 0]][:1, 1:, -1:-3:-1])
print(nd[[1]][:, 1:3, [2, 1]])


# ---------------------------- > 练习
arr = np.arange(32).reshape(8, 4)
print(f"原矩阵：\n{arr}")

# 取0,4,5行按照0,3,1,2列排列
print(arr[[0, 4, 5]][:, [0, 3, 1, 2]])

# 15 14
# 31 30
print(arr[[3, 7]][:, [3, 2]])

# 15 14
a1 = arr[[3]][:, [3, 2]]
print(a1.shape)
# # 30 31
a2 = arr[[7]][:, [2, 3]]
print(a2.shape)

# 矩阵合并
# 行合并 & 列合并
print("a1和a2两个矩阵形状完全一致")
print(f"a1矩阵与a2矩阵行合并最终的结果是：\n{np.hstack((a1, a2))}")
print(f"a1矩阵与a2矩阵列合并最终的结果是：\n{np.vstack((a1, a2))}")


# 合并的矩阵形状不一致
n1 = np.arange(12).reshape(3, 4)
n2 = np.arange(12).reshape(2, 6)
print(np.hstack((n1, n2)))

n1 = np.arange(12).reshape(3, 4)
n2 = np.arange(12).reshape(4, 3)
print(np.hstack((n1, n2)))


n1 = np.arange(12).reshape(3, 4)
n2 = np.arange(8).reshape(2, 4)
print(np.vstack((n1, n2)))

# 结论：合并的矩阵可以形状不一致， 但是必须保证某一个轴（方向）是相同的

# hstack和vstack方法使用比较灵活，但是占用内容较大，不适合大矩阵合并适用于小矩阵数据量的合并处理
# 下面的这种方式， 不存在内存开销大的问题， 推荐使用
n1 = np.arange(12).reshape(4, 3)
n2 = np.arange(8).reshape(4, 2)
print(np.concatenate((n1, n2), axis=1))
```

## 08-Numpy的bool索引

```python
import numpy as np


# 假设有一个姓名向量
names = ['lucy', 'lily', 'bill', 'lucy', 'lucy', 'joe']
names_nd = np.array(names)

# 数据矩阵
datas = np.random.randn(len(names), 4)
print(f"原矩阵：\n{datas}")
print("-" * 50)

# 生成一个bool向量
print(names_nd == 'lucy')

# 取0行数据
print(datas[0])
print(datas[:2])
print(datas[:2, :2])

# 传入bool ndarray 0 3 4行为true 保留 不为true就舍弃
print(datas[names_nd=="lucy"])

# 保留下表为1的那行数据
print(datas[names_nd=="lily"])


# 索引， 花式索引， bool索引

# 在筛选的结果之上在取0,1列
print(datas[names_nd=="lucy", :2])

# 在筛选的结果之上在区0,1行
print(datas[names_nd=="lucy"][:2])

print(datas[names_nd=="lucy", :2][:2])

# 在筛选的结果之上在1,1和0，1
print(datas[names_nd=="lucy"][[1, 0], [1, 1]])

# 保留不符合lucy所在索引的行
print(datas[~(names_nd=="lucy")])
```

## 09-Numpy的bool索引应用

```python
import numpy as np


data = np.arange(24).reshape(4, 6)
print(f"原矩阵：\n{data}")

# 将data矩阵中的所有小于10的数变成99
data[data<10] = 99
print(f"修改后的矩阵：\n{data}")

# 将data矩阵中所有的偶数变成0
data[data%2==0] = 0
print(f"修改后的矩阵：\n{data}")

# 将data中所有小于10的数变成0， 大于10的变成10
print(np.where(data<=10, 0, 10))

# 小于4的全部变成4， 大于5的全部变成5
print(data.clip(4, 5))
```

## 10-Numpy中的Nan值

```python
import numpy as np


# Nan(非数 not a number): 表示未定义或不可表示的值
nd = np.arange(12).reshape(3, 4)
print(f"原矩阵：\n{nd}")

# np.where(nd<3, np.nan, 5)
# print(f"修改后的矩阵：\n{nd}")

nd = nd.astype('f4')
nd[2, 3] = np.nan
# print(f"修改后的矩阵：\n{nd}")

# nd[nd<10] = 99
# print(f"修改后的矩阵：\n{nd}")

# 处理数据中的nan值
# 数据清洗 --->  先处理nan
# 1，填充
# 1,1 ：一般采用nan所在的行或者列的中值（平均值）来填充 
# 2，删除
# 2,1 ：很少会直接删除nan所在行， 如果做的是非统计功能， 可以考虑删除（前提是数据样本多）

# nan值非常重要的特点：
# 两个nan不相等
print(np.nan != np.nan)

# 例子：
nd = np.arange(9, dtype='f4').reshape(3, 3)
# print(nd)
# 修改nan
nd[:, 1] = np.nan
nd[0, 0] = np.nan

# print(nd)

# 统计nd中非0的个数
# print(f"nd中非0的个数：{np.count_nonzero(nd)}")

# 方式一
new_nd = (nd != nd)
print(new_nd)
# print(f"new_nd中nan的个数：{np.count_nonzero(new_nd)}")

# 方式二
print(f"new_nd中nan的个数：{np.count_nonzero(np.isnan(nd))}")
```

## 11-Numpy填充nan值

```python
import numpy as np

class NanProcss:
    @classmethod
    def fill_nan(cls, ndarray, fill_method="mean"):
        """
        填充nan值得方法
        @ndarray  填充nan的矩阵
        @fill_method 填充方法， 默认均值填充， 可选项 fill_method = "median"
        """
        # 1. 找nan
        for i in range(ndarray.shape[1]):
            # 查找当前的列
            current_col = ndarray[:, i]
            # 判断当前列是否nan
            nan = np.count_nonzero(current_col != current_col)
            if nan != 0:
                # 就说明nan值存在
                nan_col_index = current_col[current_col == current_col]
                # 选中当前nan的位置
                current_col[np.isnan(current_col)] = eval(f'nan_col_index.{fill_method}()')
                


if __name__ == "__main__":
    # 假设存在矩阵
    nd = np.arange(12).reshape(3, 4).astype('f4')
    nd[1, 2:] = np.nan
    print(f"填充nan值前的矩阵：\n{nd}")
    # 调用工具类提供的填充方法填充nan值
    NanProcss.fill_nan(nd)
    print(f"填充后nan值矩阵为：\n{nd}")

```

## 12-Numpy常用函数

```python
import numpy as np


nd = np.array([3.5, np.nan, 2.2, 1.7, -2.5, -7.1])
print(f"原向量:{nd}")
# 一元函数

# abs 绝对值函数
print(f"绝对值函数:{np.abs(nd)}")

# square函数
print(f"平方函数:{np.square(nd)}")

# sign函数
# 1(正数) -1(负数)
print(f"符号函数:{np.sign(nd)}")
# 统计nd中有多少个正数
# nd[1] = -1.2
nd = nd.astype('i4')
print(nd[nd>0].size)
# ceil
print(f"ceil(大于该值的最小整数)):{np.ceil(nd)}")

# floor
print(f"floor函数:{np.floor(nd)}")

# rint
print(f"四舍五入函数rint:{np.rint(nd)}")

# round
print(f"四舍五入函数round:{np.round(nd)}")


# 二元函数
n1 = np.random.randint(1, 20, (4, 5))
n2 = np.random.randint(-1, 3, (4, 5))
n3 = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, 4.0, 1.0, 2.0, 3.0, 4.0]).reshape(4, 5)
print(n1)
print(n2)
# 过滤n2中的0值
n2 = np.where(n2==0, 11, n2)
print(n2)

# add函数 四则运算（形状完全一致） + - * /
print(np.add(n1, n2))
print(n1 + n2)

# 形状不一致不能相加
print(n2 + n3)

# np.subtract
# np.multiply
# np.divide

# np.maximum
# 比较两个矩阵中的元素， 返回较大的元素组成相同形状的全新矩阵
print(np.maximum(n1, n2))

# 忽略nan
print(np.fmax(n1, n3))

# nan值填充比较结果
print(np.maximum(n1, n3))

# np.minimum() 和上面相反

# 拷贝符号给前一个矩阵（将n2的符号拷贝给n1）
print(np.copysign(n1, n2))

# 比较级函数, 返回值是一个bool矩阵，该矩阵可以用来作为bool索引
print(np.greater(n1, n2))





```

## 13-Numpy常用的统计函数

```python
import numpy as np

import numpy as np


n1 = np.random.randint(1, 20, (4, 5))
print(f"原矩阵:\n{n1}")
# 求和
# print(f"求和函数：{np.sum(n1)}")
# 指定轴来进行求和
# print(f"求和函数：{np.sum(n1, axis=1)}")
# print(f"求和函数：{np.sum(n1, axis=0)}")

# argmax 返回最大元素的索引
# print(f"最大元素的索引位置：{np.argmax(n1)}")
# print(f"最大元素的索引位置：{np.argmax(n1, axis=1)}")

# mean 求平均值默认求整个ndarray的均值
print(f"所有元素均值:{np.mean(n1)}")
print(f"行均值:{np.mean(n1, axis=1)}")
```

## 14-Numpy练习题

```python
import numpy as np
import time


def ex_1():
    """
    创建一个长度为10的全0向量，并且市第五个元素为1
    """
    nd = np.zeros(10)
    nd[4] = 1
    print(nd)


def ex_2():
    """
    创建一个ndarray， 包含10-49元素
    """
    nd = np.random.randint(10, 50, (3, 4))
    print(nd)


def ex_3():
    """
    将ex_2中的所有元素位置反转
    """
    nd = np.random.randint(10, 50, 10)
    print(nd)
    nd = nd[::-1]
    print(nd)


def ex_4():
    """
    使用随机数创建10*10的矩阵并输出最小元素
    """
    nd = np.random.random(size=(10, 10))
    print(nd)
    print(f"nd中最小的元素是：{nd.min()}")


def ex_5():
    """
    创建一个10*10的矩阵，并且矩阵边界都是1， 内部都是0
    """
    nd = np.zeros(shape=(10, 10)).astype('i4')
    # 花式索引修改边界
    nd[[0, 9]] = 1
    # 修改列
    nd[:, [0, 9]] = 1
    print(nd)


def ex_6():
    """
    创建每一行都是0到4的 5*5矩阵
    """
    nd = np.array([x for x in range(5)] * 5).reshape(5, 5)
    print(nd)


def ex_7():
    """
    创建一个范围在0-1之间的长度为12的等差数列
    """
    nd = np.linspace(0, 1, 12)
    print(nd)


def ex_8():
    """
    创建一个长度为10的随机数组并排序
    """
    nd = np.random.random(10)
    # 排序
    print(np.sort(nd))
    print(nd.argsort())


def ex_8_2():
    """
    排序
    """
    nd = np.random.randint(1, 20, 10)
    print(f"原向量：{nd}")
    # 排序
    # nd.sort()
    # print(f"排序后的向量:{nd}")
    print(f"arg排序：{nd.argsort()}")
    nd.sort()
    print(nd)

def ex_9():
    """
    创建10*10矩阵，并将最大值替换成0
    """
    nd = np.array([1, 2, 3, 8, 5, 6, 7, 4]).reshape(2, 4)
    print(f"原矩阵：\n{nd}")
    nd = nd.reshape((nd.size, ))
    print(f"重置矩阵形状后矩阵为：{nd}")
    # 对于向量来说，我们使用哪个argmax很容易就能得到最大值所在的index，进行修改
    # 但是对于二唯或以上唯独， 我们得到的index不好直接修改（index是一个值）， 所以我们想到
    # 先将矩阵展开，变成向量， 修改后在变成矩阵
    print(f"最大值所在的索引为:{nd.argmax()}")
    nd[nd.argmax()] = 0
    nd = nd.reshape(2, 4)
    print(nd)


def ex_10():
    """
    如何根据第3列来对一个5 * 5矩阵排序
    """
    nd = np.random.randint(1, 100, (5, 5))
    print(f"随即矩阵:{nd}")
    print(f"第3列：{nd[:, 2]}")
    print(f"根据第3列排序：{np.argsort(nd[:, 2])}")


def ex_11():
    """
    给定一个4维矩阵， 得到最后两个维度的和
    """
    # 创建4维矩阵
    nd = np.random.randint(1, 101, (2, 2, 2, 2))
    print(nd)
    print(np.sum(nd))
    print("-" * 50)
    # print(np.sum(nd, axis=1))
    print(np.sum(nd, axis=(2, 3)))
    print(np.sum(nd, axis=(-1, -2)))


def ex_12():
    """
    给定数组[1, 2, 3, 4]，如何得到这个数组的每个元素之间插入3个0后的新数组
    """
    pass


def ex_13():
    """
    给定一个二维矩阵， 如何交换其中两行的元素
    """
    nd = np.random.randint(1, 100, (4, 4))
    print(nd)
    print(nd[[1, 0, 2]])


def ex_14():
    """
    创建一个100000长度的随机数组，使用两种方法对其求3次方， 并比较所用的时间
    """
    # 第一种方法
    nd = np.random.randint(1, 3, (100000, ))
    now = time.time()
    print(nd ** 3)
    end = time.time()
    print(f"消耗的时间:{end - now}")

    # np.power(nd, 3)


def ex_15():
    """
    矩阵的每一行的元素都减去该行的平均值
    :return:
    """
    nd = np.random.randint(1, 10, (3, 3))

    nd_avg = nd.mean(axis=1).reshape(3, 1)
    print(f"每一行的平均值组成的向量：\n{nd_avg}")

    print(nd - nd_avg)


def ex_16():
    """
    打印出以下形状
    :return:
    """
    nd = np.ones((8, 8), dtype='i4')
    # 先变偶数行
    nd[::2, ::2] = 0
    # 奇数行
    nd[1::2, 1::2] = 0
    print(nd)


def ex_17():
    """
    正则化
    :return:
    """
    nd = np.random.randint(1, 101, (5, 6))
    nd_min = nd.min()
    nd_max = nd.max()
    result = (nd - nd_min) / (nd_max - nd_min)
    print(result)


if __name__ == "__main__":
    # 调用函数
    ex_17()

```

## 15-Numpy其他函数

```python
import numpy as np


# all和any
# all逻辑与
# any逻辑或
# n1 = np.arange(5)
# print(np.all(n1))
#
# n2 = np.array([0, 3, 0, 0, 0])
# print(np.all(n2))
#
# n3 = np.array([1, 1, 1, 3])
# print(np.all(n3))
#
# n4 = np.array([1, 1, 0, 2])
# print(np.all((n3, n4)))
#
# # 根据传入的ndarray形状创建向量或矩阵
# n5 = np.full_like(n2, 2)
# print(n5)


# ------------------- append函数
# ndim=1
# n1 = np.arange(4)
# append: 非原地函数， 向1维末尾追加元素（返回的是一个全新的向量）
# print(np.append(n1, [1, 2, 3]))
# print(np.append(n1, 11))

# ndim=2
# n2 = np.arange(9).reshape(3, 3)
# print(n2)
# # n2.reshape((n2.size,)).append(n2, 11)
# print(np.append(n2, 11))
# print(np.append(n2, 11).reshape(2, 5))

# ------------------- concatnate
# n3 = np.arange(12).reshape(3, 4)
# n4 = np.random.randint(1, 10, (4, ))
# n5 = np.random.randint(1, 10, (5, ))
# n6 = np.random.randint(1, 10, (3, 4))
# print(n3)
# print(n4)
# print(n5)
# print(n6)

# ndim=1的合并（向量和向量的合并）
# print(f"向量n4和n5合并:{np.concatenate((n4, n5))}")
# print(f"二维矩阵n3和n6合并（行增加列不变）:\n{np.concatenate((n3, n6))}")
# print(f"二维矩阵n3和n6合并（列增加行不变）:\n{np.concatenate((n3, n6), axis=1)}")

# 上述演示的都是形状相同矩阵之间的合并
# 当形状不一致时
# n7 = np.random.randint(1, 10, (3, 3))
# print(n7)
# print(f"二维矩阵n3和n7合并(行不变列增加):{np.concatenate((n3, n7), axis=1)}")

# 只要在某一个方向上保证一致就可以合并
# 如果两个矩阵的行一致， 那么合并之后就是行不变但是列增加， 即指定axis=1
# 如果两个矩阵的列一致， 那么合并之后就是列不变但是行增加， 即指定axis=0

# --------------- 转置矩阵
# nd = np.arange(12).reshape(3, 4)
# print(nd)
# print(nd.T)
# print(nd.transpose((1, 0)))
# print(nd.transpose((-1, -2)))
# print(nd.transpose((-1, 0)))

# --------------- 轴对换
# nd = np.arange(12).reshape(3, 4)
# print(nd)
# print(nd.T)
# print(nd.swapaxes(1, 0))

# --------------- delete 删除
nd = np.arange(20).reshape(4, 5)
print(f"原矩阵：\n{nd}")
# print(f"删除0行后得到的新矩阵是：\n{np.delete(nd, 0, axis=0)}")
# print("注意当不指定axis的时候，那么删除的就是元素！！！")
# print(f"删除0行2行得到的新矩阵是：\n{np.delete(nd, [0, 2], axis=0)}")
# print(f"删除0, 1, 2列得到的新矩阵是：\n{np.delete(nd, [0, 1, 2], axis=1)}")
print(f"使用切片对象np.s_[]来删除0, 1, 2列得到的新矩阵是：\n{np.delete(nd, np.s_[:3], axis=1)}")
```

