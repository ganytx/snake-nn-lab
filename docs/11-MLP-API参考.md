# MLP API 参考 — `src/nn/`

> 速查用。想理解"为什么这么设计"，看 `docs/10-神经网络教程.md`。

## 头文件

```cpp
#include "nn/matrix.h"       // 矩阵 / 向量 / 线性代数
#include "nn/activation.h"   // 激活函数
#include "nn/loss.h"         // 损失函数 / softmax
#include "nn/network.h"      // MLP 本体（已包含前三个）
```

`CMakeLists.txt` 已把 `src/` 加入头文件搜索路径，所以写 `"nn/network.h"` 即可。
所有内容都在 `namespace nn` 里。

---

## `matrix.h` — 编译期定长线性代数

```cpp
template <typename T, int R, int C> class matrix;   // T a[R][C]，a[行][列]
template <typename T, int N>        class vec;      // T a[N]
```

两者都提供 `static constexpr int ROWS / COLS`（`vec` 的 `COLS == 1`，另有 `SIZE`），
以及 `void print(const char* name = "") const`。构造函数把元素全部置 0。

### 运算符一览

| 表达式 | 结果 | 维度 |
|--------|------|------|
| `M + M` / `M - M` | `matrix<R,C>` | 同形 |
| `v + v` / `v - v` | `vec<N>` | 同长 |
| `M += M` / `M -= M` / `v += v` / `v -= v` | 引用 | 就地 |
| `k * M`、`M * k`、`k * v`、`v * k` | 同形 | 标量乘 |
| `M * v` | `vec<R>` | (R×C)·(C) |
| **`v * M`** | `vec<C>` | **vᵀ·M，(R)·(R×C) ← 神经网络用这个** |
| `M * M` | `matrix<R,C>` | (R×K)·(K×C) |

### 自由函数

| 函数 | 说明 |
|------|------|
| `transpose(M)` | (R×C) → (C×R) |
| `hadamard(A, B)` | 逐元素相乘（矩阵和向量都有重载） |
| `outer(u, v)` | 外积 `u·vᵀ`，(R)×(C) → (R×C)。**反向传播里算 `dW` 用它** |
| `dot(u, v)` | 点积，返回标量 |
| `sum(v)` | 求和 |
| `apply(f, v)` / `apply(f, M)` | 逐元素应用函数 `f`，`f` 可以是 lambda |

---

## `activation.h` — 激活函数

### ★ 核心约定：所有导数的输入都是**激活前的 `z`**

| 函数 | 表达式 | 导数 | 备注 |
|------|--------|------|------|
| `relu(z)` | `max(0,z)` | `drelu(z)` = `z>0 ? 1 : 0` | 负区间导数为 0，可能"死神经元" |
| `sigmoid(z)` | `1/(1+e^-z)` | `dsigmoid(z)` = `σ(z)(1-σ(z))` | 内部会重算一次 σ(z) |
| `tanhAct(z)` | `tanh(z)` | `dtanhAct(z)` = `1-tanh²(z)` | 零中心，小网络首选 |
| `identity(z)` | `z` | `didentity(z)` = `1` | 输出 logits 时用 |

> 你原来的 `dsigmoid(x)` 期望传入的是**激活后的 `σ(z)`**，与本库约定相反。
> 混用会静默算错梯度，务必注意。

---

## `loss.h` — 损失函数

```cpp
// 均方误差：L = (1/N)·Σ(y-t)²
template <typename T, int N> T        mseLoss(const vec<T,N>& y, const vec<T,N>& t);
template <typename T, int N> vec<T,N> mseGrad(const vec<T,N>& y, const vec<T,N>& t);

// softmax（内含减最大值的数值稳定处理）
template <typename T, int N> vec<T,N> softmax(const vec<T,N>& z);

// 交叉熵：L = -log(p[action])
template <typename T, int N> T crossEntropyLoss(const vec<T,N>& p, int action);

// ★ softmax + 交叉熵 的梯度化简：∂L/∂z = p - onehot(action)
template <typename T, int N> vec<T,N> softmaxCrossEntropyGrad(const vec<T,N>& p, int action);

// 最大元素下标
template <typename T, int N> int argmax(const vec<T,N>& v);
```

---

## `network.h` — MLP

### 类型

```cpp
template <typename T, int IN, int OUT, int NHID, int HW,
          T (*AF)(T) = relu, T (*dAF)(T) = drelu>
class MLP;
```

| 参数 | 含义 |
|------|------|
| `T` | 元素类型，用 `double` |
| `IN` | 输入维度 |
| `OUT` | 输出维度（logits 个数） |
| `NHID` | 隐藏层**个数**，≥ 1 |
| `HW` | 隐藏层宽度（所有隐藏层等宽） |
| `AF` / `dAF` | 隐藏层激活函数及其导数。**输出层是线性的，不受它影响** |

```cpp
using Net = nn::MLP<double, 20, 3, 2, 32, nn::tanhAct, nn::dtanhAct>;
//            输入20  输出3  2个隐藏层  每层32宽
```

### 静态常量与类型

```cpp
Net::InputVec   // = vec<double, IN>
Net::OutputVec  // = vec<double, OUT>
Net::IN_SIZE  Net::OUT_SIZE  Net::HIDDEN_COUNT  Net::HIDDEN_WIDTH
```

### 成员函数

| 方法 | 说明 |
|------|------|
| `void init(Init mode = Init::He, uint32_t seed = 42)` | 初始化权重。`Init::{He, Xavier, Uniform01}` |
| `const OutputVec& forward(const InputVec& x)` | 前向传播，返回输出 logits，**同时缓存每层的 z 和 a** |
| `void backward(const OutputVec& dLogits)` | 反向传播，把梯度**累加**到 `dw/db`。参数是 `∂L/∂z_out` |
| `void zeroGrad()` | 梯度清零（每个 batch 开始前必须调） |
| `void scaleGrad(T k)` | 梯度整体乘 `k`（累加 N 条样本后调 `scaleGrad(1.0/N)`） |
| `void step(T lr, T momentum = 0)` | 更新权重。`momentum` 典型 0.9 |
| `bool save(const std::string& path) const` | 存权重（纯文本） |
| `bool load(const std::string& path)` | 读权重。结构不匹配或数量不符会返回 `false` |
| `int  paramCount() const` | 参数总数 |
| `void forEachParam(F f)` | 遍历所有参数，`f(double* 参数, double* 梯度)` |
| `const vec<T,HW>& hiddenActivation(int layer) const` | 读第 layer 个隐藏层的激活值（调试/教学用） |
| `const vec<T,HW>& hiddenPreActivation(int layer) const` | 读第 layer 个隐藏层的 `z` |

### 公开数据成员

```cpp
// 参数
w_in (IN×HW)   b_in (HW)
w_hid[NHID-1] (HW×HW)   b_hid[NHID-1] (HW)
w_out (HW×OUT)  b_out (OUT)
// 梯度：dw_in db_in dw_hid[] db_hid[] dw_out db_out
// 动量缓存：vw_in vb_in vw_hid[] vb_hid[] vw_out vb_out
// 前向缓存：a_in  z_hid[NHID]  a_hid[NHID]  z_out
```

> 全部公开是为了教学和调试方便——你可以随时 `net.print()` 看内部状态。
> 代价是没有封装保护，改坏了编译器不会拦你。

---

## 标准训练循环模板

```cpp
using Net = nn::MLP<double, 20, 3, 2, 32, nn::tanhAct, nn::dtanhAct>;
Net net;
net.init(nn::Init::Xavier, /*seed=*/42);

const int BATCH = 32;
for (int epoch = 0; epoch < 1000; ++epoch) {
    net.zeroGrad();                        // 1. 清梯度
    for (int i = 0; i < BATCH; ++i) {
        auto logits = net.forward(x[i]);   // 2. 前向
        net.backward(gradOf(logits));      // 3. 反向（累加）
    }
    net.scaleGrad(1.0 / BATCH);            // 4. ★ 变成平均梯度
    net.step(0.01, 0.9);                   // 5. 更新
}
```

**四步顺序不能乱，第 4 步不能忘。**

---

## 权重文件格式

纯文本，人可以读、可以用 diff 比较：

```
# snake-nn-lab MLP v1
IN 20
OUT 3
NHID 2
HW 32
# w_in 20x32
0.123  -0.456  ... （每行 32 个数）
...
# b_in 32
...
# w_hid 32x32
...
# b_hid 32
...
# w_out 32x3
...
# b_out 3
...
```

`load()` 会校验 `IN/OUT/NHID/HW` 和数值总个数；
不匹配直接返回 `false`，**不会**把权重硬塞进错误的形状。

---

## 常见错误速查

| 错误 | 后果 | 正确做法 |
|------|------|---------|
| 忘记 `zeroGrad()` | 梯度无限累加，迅速发散 | 每个 batch 开头调用 |
| 忘记 `scaleGrad(1/N)` | 等效学习率放大 N 倍 | 累加完立刻调用 |
| `dAF` 传了激活后的 `a` | 静默算错梯度 | 本库统一传 `z` |
| 先 `backward` 再 `forward` | 缓存与梯度不匹配 | 严格 `forward → 算损失 → backward` |
| 用 `M * v` 而不是 `v * M` | 维度不匹配（编译期报错） | 本库用行向量约定 `v * M` |
| 中途 `setSegmentCount` 之类改结构 | 缓存张量尺寸对不上 | 重新 `init()` |

---

## 自检程序

`tools/nn_selftest.cpp` → `build\nn_selftest.exe`，6 项检查：

| 项 | 内容 | 通过判据 |
|----|------|---------|
| 1 | **数值梯度检查**（反向传播正确性） | 最大相对误差 < `1e-6` |
| 2 | XOR 训练 | 4 个样本全对，损失 < 0.02 |
| 3 | 拟合 `sin(x)` | MSE < `1e-3` |
| 4 | `softmax + 交叉熵` 梯度公式 | 相对误差 < `1e-6` |
| 5 | `save()` → `load()` 往返一致 | 输出差异 == 0 |
| 6 | 初始化方式对照（第 7 章那组实验） | Xavier 明显优于 `U(0,1)` |

**改动 `src/nn/` 任何一行之后，都跑一次它。**
