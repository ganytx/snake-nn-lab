// ============================================================================
//  nn_selftest.cpp — MLP 自检 / 教学演示
//
//  这个程序不依赖图形库，纯控制台，秒级跑完。它要回答一个问题：
//      「我写的反向传播，算出来的梯度到底对不对？」
//
//  答案不靠"训练能收敛就行"（那只能说明大方向没错），而是靠
//  【数值梯度检查】：用最笨但绝对可靠的中心差分
//        ∂L/∂w ≈ [ L(w+ε) - L(w-ε) ] / (2ε)
//  去逼近真实的偏导数，再和反向传播算出的解析梯度逐个比对。
//  两者吻合到 1e-8 量级，才能说反向传播写对了。
//
//  运行：build\nn_selftest.exe
// ============================================================================

#include "nn/network.h"
#include "nn/loss.h"
#include "utils/ConsoleUtf8.h"   // ★ 控制台 UTF-8 输出（否则中文乱码）

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace nn;

static int g_pass = 0;
static int g_fail = 0;

static void report(bool ok, const std::string& name, const std::string& detail = "") {
    std::cout << (ok ? "  [通过] " : "  [失败] ") << name;
    if (!detail.empty()) std::cout << "    " << detail;
    std::cout << '\n';
    if (ok) ++g_pass; else ++g_fail;
}

static void section(const std::string& title) {
    std::cout << "\n" << std::string(70, '-') << "\n" << title << "\n";
}

// 一维向量的标准差：用来看"信号在网络里是保持、爆炸还是衰减"
template <typename V>
static double stdev(const V& v) {
    double mean = 0.0;
    for (int i = 0; i < V::SIZE; ++i) mean += v.a[i];
    mean /= V::SIZE;
    double acc = 0.0;
    for (int i = 0; i < V::SIZE; ++i) acc += (v.a[i] - mean) * (v.a[i] - mean);
    return std::sqrt(acc / V::SIZE);
}

// ============================================================================
//  数值梯度检查
//
//  net        : 待检查的网络（检查完会保留参数原值）
//  x          : 一个固定输入
//  lossValue  : 给定网络输出，返回标量损失
//  lossGrad   : 给定网络输出，返回 ∂L/∂z_out（喂给 backward）
//  samples    : 抽查多少个参数（全查太慢，均匀抽样即可覆盖各层）
//  eps        : 差分步长
// ============================================================================
template <typename Net, typename ValueFn, typename GradFn>
bool gradientCheck(Net& net, const typename Net::InputVec& x,
                   ValueFn lossValue, GradFn lossGrad,
                   int samples, double eps,
                   double& worstRel, std::string& detail)
{
    // ---- 1) 解析梯度：反向传播算出来的 ----
    net.zeroGrad();
    net.backward(lossGrad(net.forward(x)));

    // ---- 2) 收集所有 (参数, 梯度) 指针对 ----
    std::vector<std::pair<double*, double*>> params;
    net.forEachParam([&](double* p, double* g) { params.push_back(std::make_pair(p, g)); });

    if (params.empty()) { detail = "网络没有任何参数"; return false; }

    // ---- 3) 均匀抽样，逐个做中心差分 ----
    const int total = static_cast<int>(params.size());
    const int n = (samples > 0 && samples < total) ? samples : total;

    worstRel = 0.0;
    int worstIdx = -1;
    double worstNum = 0.0, worstAna = 0.0;

    for (int k = 0; k < n; ++k) {
        const std::size_t idx =
            static_cast<std::size_t>(static_cast<long long>(k) * total / n);

        double* p = params[idx].first;
        double* g = params[idx].second;

        const double orig = *p;

        *p = orig + eps;
        const double lossPlus = lossValue(net.forward(x));
        *p = orig - eps;
        const double lossMinus = lossValue(net.forward(x));
        *p = orig;                                   // 还原，不能留下痕迹

        const double numeric  = (lossPlus - lossMinus) / (2.0 * eps);
        const double analytic = *g;

        // 相对误差：用 |数值|+|解析| 做分母，避免接近 0 时误差被放大
        const double denom = std::max(1e-8, std::abs(numeric) + std::abs(analytic));
        const double rel   = std::abs(numeric - analytic) / denom;

        if (rel > worstRel) {
            worstRel = rel;
            worstIdx = static_cast<int>(idx);
            worstNum = numeric;
            worstAna = analytic;
        }
    }

    std::ostringstream oss;
    oss << "抽查 " << n << "/" << total << " 个参数，最大相对误差 = "
        << std::scientific << std::setprecision(2) << worstRel;
    if (worstRel > 1e-6 && worstIdx >= 0) {
        oss << "\n           最差点 #" << worstIdx
            << "：数值=" << worstNum << "  解析=" << worstAna;
    }
    detail = oss.str();

    return worstRel < 1e-6;
}

// ============================================================================
//  XOR 训练（最经典的"线性不可分"问题）
//  两层线性模型永远学不会它，必须要有隐藏层 + 非线性激活
// ============================================================================
template <typename Net>
void trainXor(Net& net, int epochs, double lr, double momentum = 0.0) {
    const double xs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double ts[4]    = {0, 1, 1, 0};

    for (int e = 0; e < epochs; ++e) {
        net.zeroGrad();
        for (int i = 0; i < 4; ++i) {
            typename Net::InputVec x;
            x.a[0] = xs[i][0];
            x.a[1] = xs[i][1];
            typename Net::OutputVec t;
            t.a[0] = ts[i];
            // 注意顺序：先 forward 拿输出，再算梯度，最后 backward 累加
            net.backward(mseGrad(net.forward(x), t));
        }
        net.step(lr, momentum);
    }
}

template <typename Net>
double xorLoss(Net& net) {
    const double xs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double ts[4]    = {0, 1, 1, 0};
    double total = 0.0;
    for (int i = 0; i < 4; ++i) {
        typename Net::InputVec x;
        x.a[0] = xs[i][0];
        x.a[1] = xs[i][1];
        typename Net::OutputVec t;
        t.a[0] = ts[i];
        total += mseLoss(net.forward(x), t);
    }
    return total / 4.0;
}

// ============================================================================

int main() {
    // ★ 先把控制台输出代码页切成 UTF-8，否则下面的中文全是乱码。
    //   见 src/utils/ConsoleUtf8.h 的说明（默认代码页 936 会花屏）。
    textutil::enableUtf8Console();

    std::cout << "======================================================================\n";
    std::cout << " snake-nn-lab · MLP 自检\n";
    std::cout << " 目标：证明反向传播算出的梯度是【正确的】，而不只是【能收敛】\n";
    std::cout << "======================================================================\n";

    // ------------------------------------------------------------------------
    section("[1/6] 数值梯度检查 —— 反向传播正确性（最关键的一项）");
    // ------------------------------------------------------------------------
    {
        using Net = MLP<double, 3, 2, 2, 5, tanhAct, dtanhAct>;   // IN=3 OUT=2 隐藏层2个 宽5
        Net net;
        net.init(Init::Xavier, 1234);
        std::cout << "  网络：3 → [5, 5] → 2，激活 tanh，参数 " << net.paramCount() << " 个\n";

        Net::InputVec x;
        x.a[0] = 0.5; x.a[1] = -0.3; x.a[2] = 1.7;
        Net::OutputVec target;
        target.a[0] = 0.8; target.a[1] = -0.4;

        auto lossValue = [&](const Net::OutputVec& y) { return mseLoss(y, target); };
        auto lossGrad  = [&](const Net::OutputVec& y) { return mseGrad(y, target); };

        double worst = 0.0;
        std::string detail;
        const bool ok = gradientCheck(net, x, lossValue, lossGrad, 120, 1e-5, worst, detail);
        report(ok, "反向传播梯度 vs 数值梯度", detail);
        std::cout << "        判据：最大相对误差 < 1e-6（双精度中心差分的理论极限）\n";
    }

    // ------------------------------------------------------------------------
    section("[2/6] XOR 训练 —— 证明网络真的在学");
    // ------------------------------------------------------------------------
    {
        using Net = MLP<double, 2, 1, 1, 4, tanhAct, dtanhAct>;   // 2 → [4] → 1
        Net net;
        net.init(Init::Xavier, 7);

        const double before = xorLoss(net);
        trainXor(net, 4000, 0.10);
        const double after = xorLoss(net);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  损失：训练前 " << before << "  →  训练后 " << after << "\n";

        const double xs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const double ts[4]    = {0, 1, 1, 0};
        bool allRight = true;
        std::cout << "  逐个样本（输出 > 0.5 判为 1）：\n";
        for (int i = 0; i < 4; ++i) {
            Net::InputVec x;
            x.a[0] = xs[i][0]; x.a[1] = xs[i][1];
            const double y = net.forward(x).a[0];
            const int pred = (y > 0.5) ? 1 : 0;
            const bool right = (pred == static_cast<int>(ts[i]));
            if (!right) allRight = false;
            std::cout << "      " << static_cast<int>(xs[i][0]) << " XOR "
                      << static_cast<int>(xs[i][1]) << " = " << static_cast<int>(ts[i])
                      << "   网络输出 " << std::setprecision(4) << y
                      << (right ? "   √" : "   ×") << "\n";
        }
        report(allRight && after < 0.02, "XOR 全部 4 个样本都学对");
    }

    // ------------------------------------------------------------------------
    section("[3/6] 拟合 sin(x) —— 回归任务");
    // ------------------------------------------------------------------------
    {
        using Net = MLP<double, 1, 1, 1, 32, tanhAct, dtanhAct>;  // 1 → [32] → 1
        Net net;
        net.init(Init::Xavier, 2024);
        std::cout << "  网络：1 → [32] → 1，参数 " << net.paramCount() << " 个\n";

        const int N = 64;
        std::vector<double> xs(N), ts(N);
        for (int i = 0; i < N; ++i) {
            const double t = -3.14159265358979 + 2.0 * 3.14159265358979 * i / (N - 1);
            xs[i] = t / 3.14159265358979;      // 归一化到 [-1,1]，这一步很重要
            ts[i] = std::sin(t);
        }

        auto fullMse = [&]() {
            double acc = 0.0;
            for (int i = 0; i < N; ++i) {
                Net::InputVec x; x.a[0] = xs[i];
                Net::OutputVec t; t.a[0] = ts[i];
                acc += mseLoss(net.forward(x), t);
            }
            return acc / N;
        };

        const double before = fullMse();
        for (int e = 0; e < 6000; ++e) {
            net.zeroGrad();
            for (int i = 0; i < N; ++i) {
                Net::InputVec x; x.a[0] = xs[i];
                Net::OutputVec t; t.a[0] = ts[i];
                net.backward(mseGrad(net.forward(x), t));   // 累加
            }
            // ★ 关键：backward 是累加语义，这里攒了 N 条样本的梯度，
            //   必须先除以 N 变成平均梯度。少了这一步，等效学习率被放大 N 倍，
            //   配上动量会直接发散成 nan（这是新手最常踩的坑之一）。
            net.scaleGrad(1.0 / N);
            net.step(0.05, 0.6);
        }
        const double after = fullMse();

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "MSE " << before << " → " << after
            << "（RMSE " << std::sqrt(std::max(0.0, after)) << "）";
        const bool finite = std::isfinite(after);
        report(finite && after < 1e-3, "拟合 sin(x)", oss.str());
        if (!finite) std::cout << "        提示：出现 nan，说明学习率过大导致发散\n";
    }

    // ------------------------------------------------------------------------
    section("[4/6] softmax + 交叉熵 的梯度公式验证");
    // ------------------------------------------------------------------------
    {
        vec<double, 4> z;
        z.a[0] = 0.3; z.a[1] = -1.2; z.a[2] = 2.1; z.a[3] = 0.05;
        const int action = 2;

        const vec<double, 4> p = softmax(z);
        const vec<double, 4> analytic = softmaxCrossEntropyGrad(p, action);

        std::cout << "  logits      = " << z.a[0] << ", " << z.a[1] << ", " << z.a[2] << ", " << z.a[3] << "\n";
        std::cout << "  softmax 后  = " << std::setprecision(4)
                  << p.a[0] << ", " << p.a[1] << ", " << p.a[2] << ", " << p.a[3]
                  << "   （和 = " << (p.a[0]+p.a[1]+p.a[2]+p.a[3]) << "）\n";
        std::cout << "  正确动作下标 = " << action << "，交叉熵损失 = "
                  << crossEntropyLoss(p, action) << "\n";

        const double eps = 1e-6;
        double worst = 0.0;
        for (int i = 0; i < 4; ++i) {
            vec<double, 4> zp = z; zp.a[i] += eps;
            vec<double, 4> zm = z; zm.a[i] -= eps;
            const double numeric = (crossEntropyLoss(softmax(zp), action)
                                  - crossEntropyLoss(softmax(zm), action)) / (2.0 * eps);
            const double denom = std::max(1e-8, std::abs(numeric) + std::abs(analytic.a[i]));
            worst = std::max(worst, std::abs(numeric - analytic.a[i]) / denom);
        }
        std::cout << "  解析梯度 (p - onehot) = " << std::setprecision(4)
                  << analytic.a[0] << ", " << analytic.a[1] << ", "
                  << analytic.a[2] << ", " << analytic.a[3] << "\n";

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2) << "最大相对误差 = " << worst;
        report(worst < 1e-6, "dL/dz = p - onehot 化简公式", oss.str());
    }

    // ------------------------------------------------------------------------
    section("[5/6] 权重存 / 取 往返一致性");
    // ------------------------------------------------------------------------
    {
        using Net = MLP<double, 4, 3, 1, 6, tanhAct, dtanhAct>;
        Net a;
        a.init(Init::He, 99);

        // 先把参数随便扰动一下，确保存下来的不是"初始值"（不然存丢了大半也看不出来）
        int k = 0;
        a.forEachParam([&](double* p, double*) { *p += 0.01 * ((k++ % 7) - 3); });

        const std::string path = "nn_selftest_weights.txt";
        const bool saved = a.save(path);

        Net b;
        b.init(Init::Uniform01, 1);        // 故意用不同的初始化，确保 load 真的覆盖了
        const bool loaded = b.load(path);

        double worst = 0.0;
        for (int i = 0; i < 8; ++i) {
            Net::InputVec x;
            x.a[0] = std::sin(i * 1.1);
            x.a[1] = std::cos(i * 0.7);
            x.a[2] = i * 0.13 - 0.5;
            x.a[3] = (i % 2) ? 1.0 : -1.0;
            const Net::OutputVec ya = a.forward(x);
            const Net::OutputVec yb = b.forward(x);
            for (int j = 0; j < Net::OUT_SIZE; ++j)
                worst = std::max(worst, std::abs(ya.a[j] - yb.a[j]));
        }

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2)
            << "存取后输出最大差异 = " << worst << "，文件：" << path;
        report(saved && loaded && worst == 0.0, "save() → load() 完全一致", oss.str());
    }

    // ------------------------------------------------------------------------
    section("[6/6] 初始化方式对照：U(0,1) 到底错在哪");
    // ------------------------------------------------------------------------
    {
        // 用 4 个隐藏层而不是 1 个：初始化的影响会随层数被指数级放大。
        // 层数太浅时两种初始化都"碰巧能work"，对比就失去意义了。
        using Net = MLP<double, 1, 1, 4, 16, tanhAct, dtanhAct>;

        Net bad;  bad.init(Init::Uniform01, 7);
        Net good; good.init(Init::Xavier,   7);
        std::cout << "  网络：1 → [16, 16, 16, 16] → 1，参数 " << bad.paramCount() << " 个\n\n";

        // ---- (a) 前向信号统计：同一输入下，各隐藏层激活值的标准差 ----
        Net::InputVec probe; probe.a[0] = 0.5;
        bad.forward(probe);
        std::vector<double> sb;
        for (int i = 0; i < Net::HIDDEN_COUNT; ++i) sb.push_back(stdev(bad.hiddenActivation(i)));
        good.forward(probe);
        std::vector<double> sg;
        for (int i = 0; i < Net::HIDDEN_COUNT; ++i) sg.push_back(stdev(good.hiddenActivation(i)));

        std::cout << "  (a) 输入固定为 0.5，看信号能不能活着穿过每一层：\n";
        std::cout << "        隐藏层        U(0,1)        Xavier\n";
        for (int i = 0; i < Net::HIDDEN_COUNT; ++i) {
            std::cout << "          " << (i + 1) << "      "
                      << std::scientific << std::setprecision(3)
                      << std::setw(12) << sb[static_cast<std::size_t>(i)]
                      << std::setw(14) << sg[static_cast<std::size_t>(i)] << "\n";
        }
        std::cout << "        （标准差趋近 0 = 神经元全部被 tanh 压到 ±1 饱和，梯度随之消失）\n\n";

        // ---- (b) 训练速度对照：同样的任务、同样的轮数、同样的学习率 ----
        const int N = 64;
        std::vector<double> xs(N), ts(N);
        for (int i = 0; i < N; ++i) {
            const double t = -3.14159265358979 + 2.0 * 3.14159265358979 * i / (N - 1);
            xs[i] = t / 3.14159265358979;
            ts[i] = std::sin(t);
        }

        // epochs=0 时只评估不训练，方便复用同一个函数读当前 MSE
        auto fitSin = [&](Net& net, int epochs) {
            for (int e = 0; e < epochs; ++e) {
                net.zeroGrad();
                for (int i = 0; i < N; ++i) {
                    Net::InputVec xi; xi.a[0] = xs[i];
                    Net::OutputVec ti; ti.a[0] = ts[i];
                    net.backward(mseGrad(net.forward(xi), ti));
                }
                net.scaleGrad(1.0 / N);
                net.step(0.02, 0.5);
            }
            double acc = 0.0;
            for (int i = 0; i < N; ++i) {
                Net::InputVec xi; xi.a[0] = xs[i];
                Net::OutputVec ti; ti.a[0] = ts[i];
                acc += mseLoss(net.forward(xi), ti);
            }
            return acc / N;
        };

        const double badBefore  = fitSin(bad,  0);
        const double goodBefore = fitSin(good, 0);
        const double badAfter   = fitSin(bad,  600);
        const double goodAfter  = fitSin(good, 600);

        std::cout << "  (b) 拟合 sin(x)：同样 600 轮、同样学习率 0.02：\n";
        std::cout << "        U(0,1) ：" << std::scientific << std::setprecision(3)
                  << badBefore << "  →  " << badAfter << "\n";
        std::cout << "        Xavier ：" << goodBefore << "  →  " << goodAfter << "\n";

        std::ostringstream oss;
        oss << "600 轮后 Xavier=" << goodAfter << "，U(0,1)=" << badAfter;
        report(std::isfinite(badAfter) && std::isfinite(goodAfter) &&
               goodAfter * 2.0 < badAfter,
               "Xavier 明显优于 U(0,1)", oss.str());
        std::cout << "        根因：U(0,1) 的方差是固定的 1/12，与 fan_in 无关。\n";
        std::cout << "              一层 16 个输入相加，方差被放大 16 倍、均值被抬高 0.5×16=8，\n";
        std::cout << "              于是 tanh 立刻饱和、导数≈0，信号传两层就死了。\n";
        std::cout << "              Xavier 让方差按 1/fan_in 缩放，信号标准差稳定在 1 附近。\n";
    }

    // ------------------------------------------------------------------------
    std::cout << "\n======================================================================\n";
    std::cout << " 结果：" << g_pass << " 项通过，" << g_fail << " 项失败\n";
    std::cout << "======================================================================\n";
    return g_fail == 0 ? 0 : 1;
}
