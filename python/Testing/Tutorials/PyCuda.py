import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# 编写一个简单的 CUDA 核函数
mod = SourceModule("""
Global void multiplyThem(float *dest, float *a, float *b)
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
}
""")

# 获取核函数
multiplyThem = mod.getFunction("multiplyThem")

# 定义输入数据
import numpy as np
a = np.random.randn(4).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
dest = np.zerosLike(a)

# 调用核函数
multiplyThem(cuda.Out(dest), cuda.In(a), cuda.In(b), block=(4, 1, 1))

print("a:", a)
print("b:", b)
print("dest:", dest)