#pragma once

// ============================================================================
//  matrix.h — 编译期定长的矩阵 / 向量库（神经网络的最小线性代数底座）
//
//  设计取舍：
//    · 行数列数都是【模板参数】，编译期确定 → 尺寸不匹配时直接编译报错，
//      而不是运行到一半才崩。这是这个库最大的价值。
//    · 所有运算都返回值语义（每次运算产生一个新对象）→ 好读、好教学，
//      代价是有拷贝开销。小网络（几千个参数）完全无所谓。
//    · 下标一律 0-based：matrix 是 a[行][列]，vec 是 a[下标]。
//
//  维度约定（非常重要，搞反了一定会算错）：
//    矩阵乘向量  M * v   →  (R×C)·(C)   = (R)     “矩阵作用在向量上”
//    向量乘矩阵  v * M   →  (R)·(R×C)   = (C)     “行向量 vᵀ 左乘矩阵”
//    本项目的神经网络统一用**行向量**写法：z = a_prev * W + b
//    （即把 a_prev 看作 1×fan_in 的行向量）
// ============================================================================

#include <iostream>
#include <cstddef>

namespace nn {

// ---------------------------------------------------------------------------
// matrix<T, R, C>：R 行 C 列
// 元素存在栈上（T a[R][C]），所以别用它开几千×几千的大矩阵。
// ---------------------------------------------------------------------------
template <typename T, int R, int C>
class matrix {
public:
    static constexpr int ROWS = R;
    static constexpr int COLS = C;

    T a[R][C];

    matrix() {
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < C; ++j)
                a[i][j] = T();
    }

    void print(const char* name = "") const {
        if (name && *name) std::cout << name << " (" << R << "x" << C << "):\n";
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) std::cout << a[i][j] << ' ';
            std::cout << '\n';
        }
    }
};

// ---------------------------------------------------------------------------
// vec<T, N>：长度为 N 的向量
// ---------------------------------------------------------------------------
template <typename T, int N>
class vec {
public:
    static constexpr int ROWS = N;   // 让 vec 也能被泛型代码当成 (N×1) 看待
    static constexpr int COLS = 1;
    static constexpr int SIZE = N;

    T a[N];

    vec() {
        for (int i = 0; i < N; ++i) a[i] = T();
    }

    void print(const char* name = "") const {
        if (name && *name) std::cout << name << " (" << N << "):\n";
        for (int i = 0; i < N; ++i) std::cout << a[i] << ' ';
        std::cout << '\n';
    }
};

// ===========================================================================
//  逐元素算术
// ===========================================================================

template <typename T, int R, int C>
matrix<T, R, C> operator+(const matrix<T, R, C>& x, const matrix<T, R, C>& y) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = x.a[i][j] + y.a[i][j];
    return r;
}

template <typename T, int R, int C>
matrix<T, R, C> operator-(const matrix<T, R, C>& x, const matrix<T, R, C>& y) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = x.a[i][j] - y.a[i][j];
    return r;
}

template <typename T, int N>
vec<T, N> operator+(const vec<T, N>& x, const vec<T, N>& y) {
    vec<T, N> r;
    for (int i = 0; i < N; ++i) r.a[i] = x.a[i] + y.a[i];
    return r;
}

template <typename T, int N>
vec<T, N> operator-(const vec<T, N>& x, const vec<T, N>& y) {
    vec<T, N> r;
    for (int i = 0; i < N; ++i) r.a[i] = x.a[i] - y.a[i];
    return r;
}

// 就地累加/累减：反向传播里要反复往梯度上叠加，用 += 可以少一次拷贝
template <typename T, int R, int C>
matrix<T, R, C>& operator+=(matrix<T, R, C>& x, const matrix<T, R, C>& y) {
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            x.a[i][j] += y.a[i][j];
    return x;
}

template <typename T, int R, int C>
matrix<T, R, C>& operator-=(matrix<T, R, C>& x, const matrix<T, R, C>& y) {
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            x.a[i][j] -= y.a[i][j];
    return x;
}

template <typename T, int N>
vec<T, N>& operator+=(vec<T, N>& x, const vec<T, N>& y) {
    for (int i = 0; i < N; ++i) x.a[i] += y.a[i];
    return x;
}

template <typename T, int N>
vec<T, N>& operator-=(vec<T, N>& x, const vec<T, N>& y) {
    for (int i = 0; i < N; ++i) x.a[i] -= y.a[i];
    return x;
}

// ===========================================================================
//  标量乘法
// ===========================================================================

template <typename T, int R, int C>
matrix<T, R, C> operator*(T k, const matrix<T, R, C>& x) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = k * x.a[i][j];
    return r;
}

template <typename T, int R, int C>
matrix<T, R, C> operator*(const matrix<T, R, C>& x, T k) {
    return k * x;
}

template <typename T, int N>
vec<T, N> operator*(T k, const vec<T, N>& x) {
    vec<T, N> r;
    for (int i = 0; i < N; ++i) r.a[i] = k * x.a[i];
    return r;
}

template <typename T, int N>
vec<T, N> operator*(const vec<T, N>& x, T k) {
    return k * x;
}

// ===========================================================================
//  核心线性运算
// ===========================================================================

// 矩阵 · 向量：(R×C)·(C) → (R)
template <typename T, int R, int C>
vec<T, R> operator*(const matrix<T, R, C>& m, const vec<T, C>& v) {
    vec<T, R> r;
    for (int i = 0; i < R; ++i) {
        T acc = T();
        for (int j = 0; j < C; ++j) acc += m.a[i][j] * v.a[j];
        r.a[i] = acc;
    }
    return r;
}

// 向量 · 矩阵：vᵀ·M，(R)·(R×C) → (C)
// 神经网络里 z = a_prev * W 用的就是这一个
template <typename T, int R, int C>
vec<T, C> operator*(const vec<T, R>& v, const matrix<T, R, C>& m) {
    vec<T, C> r;
    for (int j = 0; j < C; ++j) {
        T acc = T();
        for (int i = 0; i < R; ++i) acc += v.a[i] * m.a[i][j];
        r.a[j] = acc;
    }
    return r;
}

// 矩阵 · 矩阵：(R×K)·(K×C) → (R×C)
// 循环顺序是 i-k-j（而不是 i-j-k），这样最内层连续访问 b.a[k][j]，缓存友好
template <typename T, int R, int K, int C>
matrix<T, R, C> operator*(const matrix<T, R, K>& x, const matrix<T, K, C>& y) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int k = 0; k < K; ++k) {
            T xik = x.a[i][k];
            if (xik == T()) continue;          // 稀疏时的小优化
            for (int j = 0; j < C; ++j)
                r.a[i][j] += xik * y.a[k][j];
        }
    return r;
}

// 转置：(R×C) → (C×R)
template <typename T, int R, int C>
matrix<T, C, R> transpose(const matrix<T, R, C>& m) {
    matrix<T, C, R> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[j][i] = m.a[i][j];
    return r;
}

// 哈达玛积（逐元素相乘）：反向传播里 "误差 ⊙ 激活导数" 用的就是它
template <typename T, int R, int C>
matrix<T, R, C> hadamard(const matrix<T, R, C>& x, const matrix<T, R, C>& y) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = x.a[i][j] * y.a[i][j];
    return r;
}

template <typename T, int N>
vec<T, N> hadamard(const vec<T, N>& x, const vec<T, N>& y) {
    vec<T, N> r;
    for (int i = 0; i < N; ++i) r.a[i] = x.a[i] * y.a[i];
    return r;
}

// 外积：outer(列向量 u, 列向量 v) = u·vᵀ → (R×C)
// 反向传播里 dW = a_prev ⊗ delta 用的就是它
template <typename T, int R, int C>
matrix<T, R, C> outer(const vec<T, R>& u, const vec<T, C>& v) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = u.a[i] * v.a[j];
    return r;
}

// 点积：两个等长向量逐元素相乘再求和
template <typename T, int N>
T dot(const vec<T, N>& x, const vec<T, N>& y) {
    T acc = T();
    for (int i = 0; i < N; ++i) acc += x.a[i] * y.a[i];
    return acc;
}

// 求和（算损失时会用到）
template <typename T, int N>
T sum(const vec<T, N>& x) {
    T acc = T();
    for (int i = 0; i < N; ++i) acc += x.a[i];
    return acc;
}

// ===========================================================================
//  逐元素应用一个函数
//  这里用「模板参数 F」而不是「函数指针」，好处是可以直接传 lambda，
//  例如 apply([](double z){ return z*z; }, v)
// ===========================================================================

template <typename F, typename T, int N>
vec<T, N> apply(F f, const vec<T, N>& v) {
    vec<T, N> r;
    for (int i = 0; i < N; ++i) r.a[i] = f(v.a[i]);
    return r;
}

template <typename F, typename T, int R, int C>
matrix<T, R, C> apply(F f, const matrix<T, R, C>& m) {
    matrix<T, R, C> r;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            r.a[i][j] = f(m.a[i][j]);
    return r;
}

}  // namespace nn
