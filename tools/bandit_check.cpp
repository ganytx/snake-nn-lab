// ============================================================================
//  bandit_check.cpp — 用【已知答案】的多臂老虎机检验 REINFORCE 更新是否正确
//
//  ★ 为什么必须做这个实验
//
//  实测：连"正前方放食物、蛇出生就朝着它"这个最简单任务，跑 600 局后
//  每局只吃 0.16 个，而且贪心动作是【左转】。
//  环境、奖励、观测全都简单到不可能出错 —— 所以怀疑点落在**学习器本身**。
//
//  但是"学不会"这个现象有两个完全不同的可能：
//      (a) 策略梯度【算错了】 —— 一个真 bug
//      (b) 策略梯度【算对了】但方差太大 / 信号太弱 —— 是量的问题
//  两者的修法完全不同。必须把它们分开。
//
//  ★ 怎么分开：换一个【已知最优解】的问题
//
//  单状态 k 臂老虎机是策略梯度的最小可验证问题：
//      · 状态恒定（网络输入固定），所以"没有状态信息"这个借口不成立
//      · 每臂奖励 N(μ_i, 1)，μ_best 明确知道
//      · 最优策略就是"永远选最优臂"，argmax 必须收敛到它
//      · 没有时序、没有信用分配问题、没有 γ、没有死亡
//
//  如果 REINFORCE 在这里能收敛 → 学习器是正确的，问题在环境/信用分配
//  如果在这里也收敛不了 → **学习器本身有 bug**，一切下游调参都是白费
//
//  ★ 用的是训练器里【同一个】网络类型和【同一套】梯度四步
//     （zeroGrad → 累加 backward → scaleGrad(1/T) → step）
//     只是把"环境"换成老虎机。这样结论才能迁移回去。
//
//  用法: bandit_check.exe [总步数] [臂数] [隐藏宽度]
// ============================================================================

#include "nn/loss.h"
#include "nn/network.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int TOTAL_STEPS = (argc > 1) ? std::atoi(argv[1]) : 60000;
    // ★ 臂数必须是编译期常量（它是网络模板参数），固定成 6
    //   —— 正好等于项目真实的 ACT_SIZE（3 转向 × 2 速度），
    //     这样这个实验的结论可以直接迁移回去。
    constexpr int NARM = 6;

    // 真实的臂均值：第 2 臂最好。故意让它们【很接近】，
    // 这样"能不能分辨"才是个真问题（差距太大就变成送分题）。
    std::vector<double> mu = { 0.0, 0.05, 1.00, 0.10, -0.05, 0.02 };
    mu.resize(static_cast<std::size_t>(NARM), 0.0);
    int bestArm = 0;
    for (int i = 1; i < NARM; ++i) if (mu[i] > mu[bestArm]) bestArm = i;

    std::printf("============================================================\n");
    std::printf("  REINFORCE 学习器隔离检验（单状态多臂老虎机）\n");
    std::printf("  臂数 %d | 总步数 %d | 最优臂 = %d (μ=%.3f)\n",
                NARM, TOTAL_STEPS, bestArm, mu[bestArm]);
    std::printf("------------------------------------------------------------\n");
    std::printf("  臂均值:");
    for (int i = 0; i < NARM; ++i) std::printf("  %d:%.2f", i, mu[i]);
    std::printf("\n============================================================\n\n");

    // ---- 网络：输入固定为常量（单状态），输出 = 各臂 logits ----
    //   输入维度故意设成 1（常量 1.0），这样"网络"退化成一个只有偏置和
    //   一层 tanh 的映射 —— 但它仍然走【完全相同的】前反向代码路径。
    constexpr int IN = 1, HW = 32, NHID = 1;
    using Net = nn::MLP<double, IN, NARM, NHID, HW, nn::tanhAct, nn::dtanhAct>;
    Net net;
    net.init(nn::Init::Xavier, 12345u);

    nn::vec<double, IN> x;
    x.a[0] = 1.0;

    // ---- REINFORCE 超参：和 Trainer.h 的默认值一致 ----
    const double lr       = 0.02;    // Trainer::lr 默认
    const double baseMom  = 0.95;    // Trainer::baselineMomentum 默认
    double baseline = 0.0;

    std::mt19937 rng(999u);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    long   nBest  = 0;               // 选到最优臂的次数
    double retSum = 0.0, retSumTail = 0.0;
    long   tailN  = 0;

    const int REPORT = TOTAL_STEPS / 10;

    for (int step = 1; step <= TOTAL_STEPS; ++step) {
        // ---- 前向 ----
        const auto& logits = net.forward(x);
        const auto  probs  = nn::softmax(logits);

        // ---- 采样动作（和 EpisodeRunner 的采样逻辑同一套写法）----
        const double u = uni(rng);
        int action = NARM - 1;
        double acc = 0.0;
        for (int i = 0; i < NARM; ++i) {
            acc += probs.a[i];
            if (u < acc) { action = i; break; }
        }

        // ---- 环境给奖励 ----
        const double r = mu[static_cast<std::size_t>(action)] + noise(rng);
        retSum += r;
        if (action == bestArm) ++nBest;
        if (step > TOTAL_STEPS / 2) { retSumTail += r; ++tailN; }

        // ---- 梯度四步（与 Trainer::updateBaseline 完全一致）----
        net.zeroGrad();                                    // ①
        auto g = nn::softmaxCrossEntropyGrad(probs, action);
        g = g * (r - baseline);                            //   减基线
        net.backward(g);                                   // ② 累加
        net.scaleGrad(1.0);                                // ③ T=1，平均
        net.step(lr, 0.0);                                 // ④

        baseline = baseMom * baseline + (1.0 - baseMom) * r;

        if (step % REPORT == 0) {
            const auto p2 = nn::softmax(net.forward(x));
            const int am = nn::argmax(net.forward(x));
            std::printf("  步 %7d | 选最优臂占比(累计) %5.1f%% | 平均回报 %+7.4f | "
                        "argmax=%d %s | p=[",
                        step, 100.0 * static_cast<double>(nBest) / step,
                        retSum / step, am, (am == bestArm ? "✅最优" : "  "));
            for (int i = 0; i < NARM; ++i) std::printf("%.3f%s", p2.a[i], i + 1 < NARM ? " " : "");
            std::printf("]\n");
        }
    }

    // ---- 判定 ----
    const auto pFinal = nn::softmax(net.forward(x));
    const int  amFinal = nn::argmax(net.forward(x));
    const double avgTail = (tailN > 0) ? retSumTail / static_cast<double>(tailN) : 0.0;
    double bestPossible = mu[static_cast<std::size_t>(bestArm)];
    double avgRandom = 0.0;
    for (int i = 0; i < NARM; ++i) avgRandom += mu[static_cast<std::size_t>(i)];
    avgRandom /= NARM;

    std::printf("\n------------------------------------------------------------\n");
    std::printf("  最终 argmax         : %d  (最优是 %d)  %s\n",
                amFinal, bestArm, (amFinal == bestArm) ? "✅" : "❌");
    std::printf("  最终 p(最优臂)      : %.4f\n", pFinal.a[bestArm]);
    std::printf("  后半程平均回报      : %+.4f\n", avgTail);
    std::printf("  全随机平均回报      : %+.4f\n", avgRandom);
    std::printf("  最优臂平均回报      : %+.4f\n", bestPossible);
    std::printf("  选最优臂占比(全程)  : %.1f%%   (全随机应为 %.1f%%)\n",
                100.0 * static_cast<double>(nBest) / TOTAL_STEPS, 100.0 / NARM);
    std::printf("------------------------------------------------------------\n");

    const bool converged = (amFinal == bestArm) &&
                           (avgTail > avgRandom + 0.5 * (bestPossible - avgRandom));
    if (converged) {
        std::printf("  判定: ✅ 学习器【正确】—— REINFORCE 在这个已知最优解的问题上收敛了。\n");
        std::printf("        所以「蛇学不会」不是梯度算错，而在环境/信用分配/方差。\n");
    } else {
        std::printf("  判定: ❌ 学习器【有问题】—— 连单状态老虎机都收敛不到最优臂。\n");
        std::printf("        这是一个真 bug，必须先修它，下游一切调参都是白费。\n");
    }
    std::printf("============================================================\n");
    return converged ? 0 : 1;
}
