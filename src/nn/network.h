#pragma once

// ============================================================================
//  network.h — 多层感知机 MLP（前向传播 + 反向传播 + 优化器 + 权重存取）
//
//  网络结构（固定形态，所有隐藏层等宽）：
//
//      输入 IN
//        │  W_in  (IN × HW)
//        ▼
//      隐藏层 1 (HW 个神经元, 激活 AF)
//        │  W_hid (HW × HW)      ← 共 NHID-1 组
//       ...
//      隐藏层 NHID (HW)
//        │  W_out (HW × OUT)
//        ▼
//      输出 OUT 个 logits（线性，不加激活）
//
//  ★ 与原始草稿的三个关键区别 ★
//   1. forward() 会把每一层的 z（激活前）和 a（激活后）**缓存下来**。
//      反向传播需要它们，算完就丢是没法求梯度的。
//   2. backward(dLogits) 接收「损失对输出 logits 的梯度」，自 Action 式地
//      往回推算每一层的梯度，并**累加**到 dw/db 上（支持 mini-batch）。
//   3. 输出层是线性的，激活函数只作用于隐藏层。
//      这样同一套网络既能接 MSE 做回归，也能接 softmax 做策略。
//
//  典型用法：
//      using Net = nn::MLP<double, 8, 3, 2, 32, nn::tanhAct, nn::dtanhAct>;
//      Net net;
//      net.init(nn::Init::Xavier, 42);
//      auto logits = net.forward(x);            // 前向
//      net.backward(nn::mseGrad(logits, target)); // 反向，梯度累加到 dw/db
//      net.step(0.01);                          // 更新权重
// ============================================================================

#include "matrix.h"
#include "activation.h"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <filesystem>   // ★ 权重的 UTF-8 路径（见 save/load 里 toFsPath 的说明）
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace nn {

// ---------------------------------------------------------------------------
// 权重初始化方式
//
//   Uniform01 是刻意保留的：它就是你原来那份代码的写法 U(0,1)。
//   留着是为了让你能亲眼看它有多难训（自检程序里有一组对照实验）。
// ---------------------------------------------------------------------------
enum class Init {
    He,         // 正态分布 N(0, 2/fan_in)，为 ReLU 设计
    Xavier,     // 均匀分布 U(-√(6/(fan_in+fan_out)), +...)，为 tanh/sigmoid 设计
    Uniform01   // 你原来的写法：U(0,1)，全是正数 —— 反面教材
};

template <typename T, int IN, int OUT, int NHID, int HW,
          T (*AF)(T) = relu, T (*dAF)(T) = drelu>
class MLP {
    static_assert(IN   > 0, "输入维度 IN 必须大于 0");
    static_assert(OUT  > 0, "输出维度 OUT 必须大于 0");
    static_assert(NHID >= 1, "至少要有 1 个隐藏层");
    static_assert(HW   > 0, "隐藏层宽度 HW 必须大于 0");

    // 隐藏层之间只有 NHID-1 组连接。
    // 当 NHID==1 时不存在这种连接，但数组长度不能为 0，所以兜底成 1（那一格不用）。
    static constexpr int NH = (NHID > 1) ? (NHID - 1) : 1;

public:
    using InputVec  = vec<T, IN>;
    using OutputVec = vec<T, OUT>;

    static constexpr int IN_SIZE       = IN;
    static constexpr int OUT_SIZE      = OUT;
    static constexpr int HIDDEN_COUNT  = NHID;
    static constexpr int HIDDEN_WIDTH  = HW;

    // ======================= 可训练参数 =======================
    matrix<T, IN, HW>  w_in;      vec<T, HW>  b_in;       // 输入 → 隐藏层1
    matrix<T, HW, HW>  w_hid[NH]; vec<T, HW>  b_hid[NH];  // 隐藏层 i → i+1
    matrix<T, HW, OUT> w_out;     vec<T, OUT> b_out;      // 隐藏层NHID → 输出

    // ======================= 梯度（累加器）=======================
    matrix<T, IN, HW>  dw_in;      vec<T, HW>  db_in;
    matrix<T, HW, HW>  dw_hid[NH]; vec<T, HW>  db_hid[NH];
    matrix<T, HW, OUT> dw_out;     vec<T, OUT> db_out;

    // ======================= 动量缓存（优化器用）=======================
    matrix<T, IN, HW>  vw_in;      vec<T, HW>  vb_in;
    matrix<T, HW, HW>  vw_hid[NH]; vec<T, HW>  vb_hid[NH];
    matrix<T, HW, OUT> vw_out;     vec<T, OUT> vb_out;

    // ======================= 前向缓存（反向传播的原料）=======================
    //
    //  这是整个网络最容易被忽略、却最不能省的部分。
    //  反向传播要用到每一层的：
    //      z（激活前的值）→ 算激活函数的导数 dAF(z)
    //      a（激活后的值）→ 算权重的梯度 dW = a_prev ⊗ delta
    //  所以 forward 时必须把它们留下来。
    vec<T, IN>  a_in;                // 输入（就是所谓的 a^(0)）
    vec<T, HW>  z_hid[NHID];         // 每个隐藏层的 z
    vec<T, HW>  a_hid[NHID];         // 每个隐藏层的 a
    vec<T, OUT> z_out;               // 输出层 logits（线性层，所以 a_out == z_out）

    // =======================================================================
    //  初始化
    // =======================================================================
    void init(Init mode = Init::He, std::uint32_t seed = 42u) {
        // 用带种子的 mt19937：每次运行结果完全一致，方便复现和对比实验。
        // （你原来的写法用的是全局 std::random_device + 全局 gen，
        //   既无法复现，也会和其它 .cpp 里的同名符号冲突。）
        std::mt19937 rng(seed);
        std::normal_distribution<double>       normal(0.0, 1.0);
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        std::uniform_real_distribution<double> unit01(0.0, 1.0);

        auto fillW = [&](auto& W, int fanIn, int fanOut) {
            using M = typename std::decay<decltype(W)>::type;
            const double heStd  = std::sqrt(2.0 / static_cast<double>(fanIn));
            const double xavLim = std::sqrt(6.0 / static_cast<double>(fanIn + fanOut));
            for (int i = 0; i < M::ROWS; ++i) {
                for (int j = 0; j < M::COLS; ++j) {
                    double value = 0.0;
                    switch (mode) {
                        case Init::He:        value = normal(rng) * heStd;  break;
                        case Init::Xavier:    value = unit(rng)   * xavLim; break;
                        case Init::Uniform01: value = unit01(rng);          break;
                        default:              value = unit(rng)   * xavLim; break;
                    }
                    W.a[i][j] = static_cast<T>(value);
                }
            }
        };

        fillW(w_in, IN, HW);
        for (int i = 0; i + 1 < NHID; ++i) fillW(w_hid[i], HW, HW);
        fillW(w_out, HW, OUT);

        // 偏置初始化为 0（业界标准）。
        // 打破神经元之间的对称性靠的是权重的随机性，偏置没必要随机。
        zeroAll(b_in);
        for (int i = 0; i + 1 < NHID; ++i) zeroAll(b_hid[i]);
        zeroAll(b_out);

        clearCache();
        zeroGrad();
        zeroAll(vw_in);  zeroAll(vb_in);
        for (int i = 0; i + 1 < NHID; ++i) { zeroAll(vw_hid[i]); zeroAll(vb_hid[i]); }
        zeroAll(vw_out); zeroAll(vb_out);
    }

    // =======================================================================
    //  前向传播
    //  返回输出层的 logits（线性输出，未经过任何激活函数）
    // =======================================================================
    const vec<T, OUT>& forward(const vec<T, IN>& x) {
        a_in = x;

        // 第 1 个隐藏层：z = x·W_in + b_in，a = AF(z)
        z_hid[0] = x * w_in + b_in;
        a_hid[0] = apply(AF, z_hid[0]);

        // 其余隐藏层
        for (int i = 1; i < NHID; ++i) {
            z_hid[i] = a_hid[i - 1] * w_hid[i - 1] + b_hid[i - 1];
            a_hid[i] = apply(AF, z_hid[i]);
        }

        // 输出层：线性，不加激活
        z_out = a_hid[NHID - 1] * w_out + b_out;
        return z_out;
    }

    // =======================================================================
    //  反向传播
    //
    //  参数 dLogits = ∂L/∂z_out，即「损失对输出层 logits 的梯度」。
    //  它的来历取决于你用什么损失：
    //      · MSE          → mseGrad(y, target)
    //      · softmax+交叉熵 → softmaxCrossEntropyGrad(p, action)
    //      · 策略梯度      → 见 rl/ 目录
    //
    //  本函数只负责把梯度**累加**到 dw/db 上，不做任何权重更新。
    //  这样同一次更新里可以先跑一个小批量的多条样本，再统一 step()。
    //
    //  四个公式（务必对照推导看一遍）：
    //      dW_out  = a_hid_last ⊗ delta_out
    //      db_out  = delta_out
    //      delta_h = (W_next · delta_next) ⊙ dAF(z_h)
    //      dW_in   = a_in ⊗ delta_h0
    // =======================================================================
    void backward(const vec<T, OUT>& dLogits) {
        // ---- 输出层 ----
        dw_out += outer(a_hid[NHID - 1], dLogits);
        db_out += dLogits;

        // 误差从输出层回传到最后一个隐藏层：
        //   ∂L/∂a_hid_last = W_out · delta_out
        //   再乘上激活函数的导数（注意 dAF 吃的是 z，不是 a！）
        vec<T, HW> delta = hadamard(w_out * dLogits, apply(dAF, z_hid[NHID - 1]));

        // ---- 逐个隐藏层往回走 ----
        for (int i = NHID - 1; i >= 1; --i) {
            dw_hid[i - 1] += outer(a_hid[i - 1], delta);
            db_hid[i - 1] += delta;
            delta = hadamard(w_hid[i - 1] * delta, apply(dAF, z_hid[i - 1]));
        }

        // ---- 输入到第 1 个隐藏层 ----
        // （不需要再往输入方向传了，输入不是可训练参数）
        dw_in += outer(a_in, delta);
        db_in += delta;
    }

    // =======================================================================
    //  梯度清零 —— 每个 batch 开始前必须调用，否则梯度会一直累加
    // =======================================================================
    void zeroGrad() {
        zeroAll(dw_in);  zeroAll(db_in);
        for (int i = 0; i + 1 < NHID; ++i) { zeroAll(dw_hid[i]); zeroAll(db_hid[i]); }
        zeroAll(dw_out); zeroAll(db_out);
    }

    // =======================================================================
    //  梯度整体缩放
    //
    //  backward() 是「累加」语义，方便一个 batch 里塞多条样本。
    //  但累加 N 条之后梯度是「和」而不是「平均」，直接丢给 step() 等于把学习率
    //  偷偷放大了 N 倍 —— 配上动量几乎必然发散成 nan。
    //
    //  正确姿势：
    //      net.zeroGrad();
    //      for (每条样本) net.backward(...);     // 累加
    //      net.scaleGrad(1.0 / batchSize);       // ★ 变成平均梯度
    //      net.step(lr, momentum);
    // =======================================================================
    void scaleGrad(T k) {
        forEachParam([k](T* /*参数*/, T* grad) { *grad *= k; });
    }

    // =======================================================================
    //  参数更新
    //
    //  momentum = 0 时为最朴素的 SGD：      w ← w - lr · dw
    //  momentum > 0 时为带动量的 SGD：
    //      v ← momentum · v - lr · dw
    //      w ← w + v
    //  动量把历史梯度累积成一个"速度"，好处是：
    //      · 方向一致的维度越走越快（加速收敛）
    //      · 来回震荡的维度互相抵消（抑制抖动）
    //  典型取值 0.9。
    //
    //  lr 是学习率，一般取 1e-3 ~ 1e-1。太大 → 损失震荡发散；
    //  太小 → 收敛慢得像没在学。它是训练时最需要调的参数。
    // =======================================================================
    void step(T lr, T momentum = T()) {
        auto updMat = [&](auto& P, const auto& G, auto& V) {
            using M = typename std::decay<decltype(P)>::type;
            for (int i = 0; i < M::ROWS; ++i) {
                for (int j = 0; j < M::COLS; ++j) {
                    if (momentum > T()) {
                        V.a[i][j] = momentum * V.a[i][j] - lr * G.a[i][j];
                        P.a[i][j] += V.a[i][j];
                    } else {
                        P.a[i][j] -= lr * G.a[i][j];
                    }
                }
            }
        };
        auto updVec = [&](auto& P, const auto& G, auto& V) {
            using Vt = typename std::decay<decltype(P)>::type;
            for (int i = 0; i < Vt::SIZE; ++i) {
                if (momentum > T()) {
                    V.a[i] = momentum * V.a[i] - lr * G.a[i];
                    P.a[i] += V.a[i];
                } else {
                    P.a[i] -= lr * G.a[i];
                }
            }
        };

        updMat(w_in, dw_in, vw_in);
        updVec(b_in, db_in, vb_in);
        for (int i = 0; i + 1 < NHID; ++i) {
            updMat(w_hid[i], dw_hid[i], vw_hid[i]);
            updVec(b_hid[i], db_hid[i], vb_hid[i]);
        }
        updMat(w_out, dw_out, vw_out);
        updVec(b_out, db_out, vb_out);
    }

    // =======================================================================
    //  遍历所有可训练参数
    //  回调签名：f(T* 参数指针, T* 对应梯度指针)
    //  自检程序里的"数值梯度检查"就靠它自动覆盖全部参数
    //
    //  ⚠️ 这个接口被 tools/nn_selftest.cpp 依赖，**不能删、不能改签名**。
    //     （2026-09-25 差点在加只读版本时把它覆盖掉，注意。）
    // =======================================================================
    template <typename F>
    void forEachParam(F f) {
        forEachElem(w_in, dw_in, f);
        forEachElem(b_in, db_in, f);
        for (int i = 0; i + 1 < NHID; ++i) {
            forEachElem(w_hid[i], dw_hid[i], f);
            forEachElem(b_hid[i], db_hid[i], f);
        }
        forEachElem(w_out, dw_out, f);
        forEachElem(b_out, db_out, f);
    }

    // =======================================================================
    //  ★ 梯度范数与"实际走了多大一步"（2026-09-25 新增，只读诊断）
    //
    //  ★ 为什么需要它 —— 「越训越差」这件事必须能看见参数被推了多远
    //
    //  REINFORCE 的更新量 = lr × 平均梯度，而平均梯度又正比于整局回报的量级。
    //  于是**同一套超参在不同局之间，实际步长能差几十倍** ——
    //  这种情况从"损失曲线"上是完全看不出来的（本项目也没有损失曲线），
    //  只能直接量：
    //      gradNorm      → 这一步的梯度有多大
    //      updateNorm    → 乘上 lr 之后，参数实际被移动了多远
    //  两者之比就是有效步长。如果 updateNorm 经常远大于权重量级，
    //  那"越训越差"就是**步子太大把已经学好的东西踩坏了**，
    //  而不是"奖励设计不对" —— 两者的修法完全不同。
    //
    //  ★ 只读，不改任何行为：不参与 step()，纯粹是给外面看的数。
    // =======================================================================

    // 当前累积梯度的 L2 范数（step 之前调用才有意义）
    T gradNorm() const {
        T acc = T();
        forEachConstParam([&](const T* /*p*/, const T* g) { acc += (*g) * (*g); });
        return std::sqrt(acc);
    }

    // 参数本身的 L2 范数（用来判断 updateNorm 是不是"太大"）
    T paramNorm() const {
        T acc = T();
        forEachConstParam([&](const T* p, const T* /*g*/) { acc += (*p) * (*p); });
        return std::sqrt(acc);
    }

    // 遍历所有可训练参数（只读版）
    template <typename F>
    void forEachConstParam(F f) const {
        forEachElemC(w_in, dw_in, f);
        forEachElemC(b_in, db_in, f);
        for (int i = 0; i + 1 < NHID; ++i) {
            forEachElemC(w_hid[i], dw_hid[i], f);
            forEachElemC(b_hid[i], db_hid[i], f);
        }
        forEachElemC(w_out, dw_out, f);
        forEachElemC(b_out, db_out, f);
    }

    // =======================================================================
    //  梯度裁剪（按全局 L2 范数）
    //
    //  ★ 为什么需要它
    //
    //  策略梯度的方差本来就大，而"整局回报"偶尔会异常大（本项目实测单局回报
    //  从 −167 到 +499，差 30 倍）。那种局会一次性产生巨大的梯度，
    //  把已经学好的参数一脚踹飞 —— 现象就是"训练好好的，突然就废了"。
    //
    //  裁剪把"这一步最多走多远"钉死，于是**异常大的局只影响方向、不影响幅度**。
    //  这是 PPO/A2C 里的标准做法，对 REINFORCE 同样必要。
    //
    //  ★ maxNorm <= 0 表示不裁剪 —— 保证默认行为逐位不变。
    //  ★ 返回是否真的裁剪了（给外面统计"有多少步撞到上限"）。
    // =======================================================================
    bool clipGradNorm(T maxNorm) {
        if (maxNorm <= T()) return false;
        const T n = gradNorm();
        if (n <= maxNorm || n <= T()) return false;
        scaleGrad(maxNorm / n);
        return true;
    }

    // 参数总个数（网络容量）
    int paramCount() const {
        int c = IN * HW + HW;                       // w_in + b_in
        c += (NHID - 1) * (HW * HW + HW);           // 隐藏层之间
        c += HW * OUT + OUT;                        // w_out + b_out
        return c;
    }

    // 读取第 layer 个隐藏层（0-based）刚刚前向算出的激活值
    // 调试和教学用：可以直观看到信号在网络里是"保持"还是"爆炸/衰减"
    const vec<T, HW>& hiddenActivation(int layer) const { return a_hid[layer]; }
    const vec<T, HW>& hiddenPreActivation(int layer) const { return z_hid[layer]; }

    // =======================================================================
    //  权重存取（纯文本，方便肉眼检查和版本对比）
    // =======================================================================
    // ⚠️⚠️ 路径必须过 std::filesystem::path，【不能】把 std::string 直接喂给流！
    //
    //   2026-09-24 修：中文存档名在文件管理器里显示成乱码。
    //
    //   根因：`std::ofstream f(std::string)` 会把字符串按【ANSI 本地代码页】
    //   （中文 Windows = GBK）解释，而我们给的是 **UTF-8** 字节。
    //   于是"学会觅食"这几个字的 UTF-8 字节被当成 GBK 解码 → 乱码文件名，
    //   而且只有中文名才会乱（ASCII 名两种编码一致，看不出问题）。
    //
    //   std::filesystem::path 在 Windows 上内部是宽字符（UTF-16），
    //   从 UTF-8 构造时【明确按 UTF-8 解释】，再由流走宽字符 API 落到 NTFS
    //   （NTFS 文件名本来就是 UTF-16）—— 这才是唯一正确的路径。
    //
    //   ★ ASCII 路径（如 "train_probe_weights.txt"）行为与以前完全一致。
    //
    //   ⚠️⚠️ **本文件绝对不能 #include <windows.h>**（2026-09-24 实测踩到）
    //
    //      windows.h 会定义 `IN` / `OUT` / `min` / `max` 这些**宏**，
    //      而本文件的模板参数**就叫** IN 和 OUT（`MLP<T, IN, OUT, NHID, HW, ...>`）
    //      → 宏替换把声明撕碎，报一片 "expected primary-expression before '>' token"，
    //      而且错误行号指向几百行之前，完全看不出真正原因。
    //
    //      所以 UTF-8 → 宽字符的转换放在 **ckpt**（`src/rl/Checkpoint.cpp`）里做，
    //      那边本来就包含 windows.h。NN 层只收一个已经转好的 path。
    static std::filesystem::path toFsPath(const std::filesystem::path& p) {
        return p;   // 已经是 path，直接透传
    }

    bool save(const std::filesystem::path& path) const {
        std::ofstream f(path);
        if (!f) return false;
        f << "# snake-nn-lab MLP v1\n";
        f << "IN "   << IN   << '\n';
        f << "OUT "  << OUT  << '\n';
        f << "NHID " << NHID << '\n';
        f << "HW "   << HW   << '\n';
        f.precision(17);                 // 双精度全精度输出，存回来不会有精度损失
        dumpMat(f, w_in, "w_in");
        dumpVec(f, b_in, "b_in");
        for (int i = 0; i + 1 < NHID; ++i) {
            dumpMat(f, w_hid[i], "w_hid");
            dumpVec(f, b_hid[i], "b_hid");
        }
        dumpMat(f, w_out, "w_out");
        dumpVec(f, b_out, "b_out");
        return true;
    }

    bool load(const std::filesystem::path& path) {
        std::ifstream f(path);
        if (!f) return false;

        int in = 0, out = 0, nhid = 0, hw = 0;
        std::vector<double> nums;
        std::string line;

        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;      // 注释和空行直接跳过
            if (line.rfind("IN ",   0) == 0) { in   = std::atoi(line.c_str() + 3); continue; }
            if (line.rfind("OUT ",  0) == 0) { out  = std::atoi(line.c_str() + 4); continue; }
            if (line.rfind("NHID ", 0) == 0) { nhid = std::atoi(line.c_str() + 5); continue; }
            if (line.rfind("HW ",   0) == 0) { hw   = std::atoi(line.c_str() + 3); continue; }

            std::istringstream ss(line);
            double v;
            while (ss >> v) nums.push_back(v);
        }

        // 结构不匹配宁可失败，也不要把权重硬塞进错误的形状里
        if (in != IN || out != OUT || nhid != NHID || hw != HW) return false;
        if (static_cast<int>(nums.size()) != paramCount()) return false;

        std::size_t k = 0;
        loadMat(nums, k, w_in);
        loadVec(nums, k, b_in);
        for (int i = 0; i + 1 < NHID; ++i) {
            loadMat(nums, k, w_hid[i]);
            loadVec(nums, k, b_hid[i]);
        }
        loadMat(nums, k, w_out);
        loadVec(nums, k, b_out);
        return true;
    }

private:
    // -----------------------------------------------------------------------
    //  把 matrix / vec 统一置零（两种类型分开重载）
    // -----------------------------------------------------------------------
    template <typename M>
    static void zeroAll(M& m) { m = M(); }

    // 清空前向缓存
    void clearCache() {
        a_in = vec<T, IN>();
        for (int i = 0; i < NHID; ++i) { z_hid[i] = vec<T, HW>(); a_hid[i] = vec<T, HW>(); }
        z_out = vec<T, OUT>();
    }

    // -----------------------------------------------------------------------
    //  遍历一个参数张量 / 向量里的每个元素
    // -----------------------------------------------------------------------
    template <typename T2, int R, int C, typename F>
    static void forEachElem(matrix<T2, R, C>& p, matrix<T2, R, C>& g, F f) {
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < C; ++j)
                f(&p.a[i][j], &g.a[i][j]);
    }
    template <typename T2, int N, typename F>
    static void forEachElem(vec<T2, N>& p, vec<T2, N>& g, F f) {
        for (int i = 0; i < N; ++i) f(&p.a[i], &g.a[i]);
    }

    // ★ 只读版本（2026-09-25 新增）—— 给 gradNorm()/paramNorm() 用。
    //   和上面两个的唯一区别是参数是 const，不改任何行为。
    template <typename T2, int R, int C, typename F>
    static void forEachElemC(const matrix<T2, R, C>& p, const matrix<T2, R, C>& g, F f) {
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < C; ++j)
                f(&p.a[i][j], &g.a[i][j]);
    }
    template <typename T2, int N, typename F>
    static void forEachElemC(const vec<T2, N>& p, const vec<T2, N>& g, F f) {
        for (int i = 0; i < N; ++i) f(&p.a[i], &g.a[i]);
    }

    // -----------------------------------------------------------------------
    //  文本存取的读写细节
    // -----------------------------------------------------------------------
    template <typename M>
    static void dumpMat(std::ostream& os, const M& m, const char* label) {
        os << "# " << label << " " << M::ROWS << "x" << M::COLS << '\n';
        for (int i = 0; i < M::ROWS; ++i) {
            for (int j = 0; j < M::COLS; ++j) os << m.a[i][j] << ' ';
            os << '\n';
        }
    }
    template <typename V>
    static void dumpVec(std::ostream& os, const V& v, const char* label) {
        os << "# " << label << " " << V::SIZE << '\n';
        for (int i = 0; i < V::SIZE; ++i) os << v.a[i] << ' ';
        os << '\n';
    }
    template <typename M>
    static void loadMat(const std::vector<double>& nums, std::size_t& k, M& m) {
        for (int i = 0; i < M::ROWS; ++i)
            for (int j = 0; j < M::COLS; ++j)
                m.a[i][j] = static_cast<T>(nums[k++]);
    }
    template <typename V>
    static void loadVec(const std::vector<double>& nums, std::size_t& k, V& v) {
        for (int i = 0; i < V::SIZE; ++i) v.a[i] = static_cast<T>(nums[k++]);
    }
};

}  // namespace nn
