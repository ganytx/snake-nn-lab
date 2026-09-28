// ============================================================================
//  statebandit_check.cpp — 关键对照：网络能不能学会【依赖输入的策略】？
//
//  ★ 为什么必须做这个实验（这是目前唯一没做过的关键对照）
//
//  已确证的三件事：
//    ① 单状态老虎机（输入恒定）—— REINFORCE 立刻收敛，99.7% 选最优臂
//    ② 训练前/后的蛇网络 —— 10 种极端不同的输入给出【同一个动作】
//    ③ 灵敏度不差（一维扰动 0.2 → logit 动 0.075），但决策差距 1.51，
//       比例 1:20，所以 argmax 在真实输入下永远翻不过来
//
//  ①②③ 合起来说明：**这个网络只学会了"设一个常量输出"，没学会
//  "输出随输入变"**。但单状态老虎机【本来就只需要常量输出】——
//  所以它证明不了"网络能不能学条件策略"。
//
//  ★ 这个实验就是补上那一环
//
//  做一个最小【上下文老虎机】：观测里有一个 bit 表示"现在哪种情况"，
//  两种情况下最优动作不同：
//      情况 A（ctx=0）→ 动作 2 最优
//      情况 B（ctx=1）→ 动作 4 最优
//  最优策略必须是【条件策略】：看到 ctx 决定动作。
//  如果 REINFORCE 在这里学不会 → **学习条件策略的能力本身有问题**，
//  那就找到了"蛇学不会"的真正根因，而且与蛇、与奖励设计全都无关。
//
//  ★ 与前一个实验的关键差别
//    单状态老虎机：最优是常量策略 → 只看偏置就够 → 必过
//    这个实验    ：最优是条件策略 → 必须用上权重 → 才是真正的检验
//
//  用法: statebandit_check.exe [总步数]
// ============================================================================

#include "nn/loss.h"
#include "nn/network.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int TOTAL = (argc > 1) ? std::atoi(argv[1]) : 120000;

    // ---- 网络：输入 2 维（ctx 位 + 常量 1），和蛇网络同为单隐藏层 tanh ----
    //   ★ 用 2 维而不是 33 维，是为了排除"维度太高"这个干扰因素。
    //     如果 2 维都学不会，33 维更不可能。
    constexpr int IN = 2, HW = 32, OUT = 6;
    using Net = nn::MLP<double, IN, OUT, 1, HW, nn::tanhAct, nn::dtanhAct>;
    Net net;
    net.init(nn::Init::Xavier, 20260921u);   // 和蛇一样用 Xavier

    // 两种情况下的臂均值：情况 A 里臂 2 最好；情况 B 里臂 4 最好
    //   ⚠️ 两个次优臂故意设得很近，这样"分辨"才有意义
    const double muA[OUT] = { 0.0, 0.1, 1.0, 0.1, 0.0, 0.1 };
    const double muB[OUT] = { 0.0, 0.1, 0.0, 0.1, 1.0, 0.1 };
    const int bestA = 2, bestB = 4;

    // ---- REINFORCE 超参：与 Trainer.h 默认一致 ----
    const double lr      = 0.02;
    const double baseMom = 0.95;

    // ★ 每个"上下文"各自一个基线 —— 因为两种情况的回报水平不同，
    //   共用一个基线会让 advantage 有偏。这也是对的做法。
    double baseline[2] = { 0.0, 0.0 };

    std::mt19937 rng(4242u);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    long correct = 0, total = 0;
    long correctA = 0, nA = 0, correctB = 0, nB = 0;

    std::printf("============================================================\n");
    std::printf("  上下文老虎机：网络能不能学会【条件策略】？\n");
    std::printf("  总步数 %d | 情况A(ctx=0)→最优臂%d | 情况B(ctx=1)→最优臂%d\n",
                TOTAL, bestA, bestB);
    std::printf("  最优策略必须是【条件策略】—— 看输入决定动作。\n");
    std::printf("============================================================\n\n");

    const int REPORT = TOTAL / 12;

    for (int step = 1; step <= TOTAL; ++step) {
        // ---- 随机选一种情况，构造输入 ----
        const int ctx = (uni(rng) < 0.5) ? 0 : 1;
        nn::vec<double, IN> x;
        x.a[0] = static_cast<double>(ctx);   // ★ 这个 bit 就是"该做什么"的全部信息
        x.a[1] = 1.0;                        // 常量偏置项

        // ---- 前向 + 采样 ----
        const auto& logits = net.forward(x);
        const auto  probs  = nn::softmax(logits);

        const double u = uni(rng);
        int action = OUT - 1;
        double acc = 0.0;
        for (int i = 0; i < OUT; ++i) { acc += probs.a[i]; if (u < acc) { action = i; break; } }

        // ---- 环境给奖励 ----
        const double r = (ctx == 0 ? muA[action] : muB[action]) + noise(rng);

        const int best = (ctx == 0) ? bestA : bestB;
        ++total;
        if (action == best) ++correct;
        if (ctx == 0) { ++nA; if (action == bestA) ++correctA; }
        else          { ++nB; if (action == bestB) ++correctB; }

        // ---- 梯度四步（与 Trainer::updateBaseline 逐字一致）----
        net.zeroGrad();
        auto g = nn::softmaxCrossEntropyGrad(probs, action);
        g = g * (r - baseline[ctx]);
        net.backward(g);
        net.scaleGrad(1.0);
        net.step(lr, 0.0);

        baseline[ctx] = baseMom * baseline[ctx] + (1.0 - baseMom) * r;

        if (step % REPORT == 0) {
            // 分别喂两种上下文，看网络给出什么
            nn::vec<double, IN> xa; xa.a[0] = 0.0; xa.a[1] = 1.0;
            nn::vec<double, IN> xb; xb.a[0] = 1.0; xb.a[1] = 1.0;
            const auto za = nn::softmax(net.forward(xa));
            const auto zb = nn::softmax(net.forward(xb));
            const int aa = nn::argmax(net.forward(xa));
            const int ab = nn::argmax(net.forward(xb));
            std::printf("  步 %7d | 选对率 %5.1f%% (A:%5.1f%% B:%5.1f%%) | "
                        "ctx=0→动作%d %s | ctx=1→动作%d %s\n",
                        step, 100.0 * correct / total,
                        100.0 * correctA / (nA ? nA : 1),
                        100.0 * correctB / (nB ? nB : 1),
                        aa, aa == bestA ? "✅" : "  ", ab, ab == bestB ? "✅" : "  ");
        }
    }

    // ---- 判定 ----
    nn::vec<double, IN> xa; xa.a[0] = 0.0; xa.a[1] = 1.0;
    nn::vec<double, IN> xb; xb.a[0] = 1.0; xb.a[1] = 1.0;
    const int aa = nn::argmax(net.forward(xa));
    const int ab = nn::argmax(net.forward(xb));

    // 后半程选对率
    const double accA = (nA > 0) ? static_cast<double>(correctA) / nA : 0.0;
    const double accB = (nB > 0) ? static_cast<double>(correctB) / nB : 0.0;

    std::printf("\n------------------------------------------------------------\n");
    std::printf("  最终 ctx=0 的 argmax : %d  (最优 %d)  %s\n", aa, bestA, aa == bestA ? "✅" : "❌");
    std::printf("  最终 ctx=1 的 argmax : %d  (最优 %d)  %s\n", ab, bestB, ab == bestB ? "✅" : "❌");
    std::printf("  全程选对率           : %.1f%%   (全随机 = %.1f%%)\n",
                100.0 * correct / total, 100.0 / OUT);
    std::printf("------------------------------------------------------------\n");

    const bool learned = (aa == bestA && ab == bestB);
    if (learned) {
        std::printf("  判定: ✅ 网络【能】学会条件策略（输入不同 → 动作不同）。\n");
        std::printf("        所以「蛇学不会」不在网络结构，而在蛇那边的\n");
        std::printf("        信用分配 / 奖励尺度 / 训练信号上。\n");
    } else if (aa == ab) {
        std::printf("  判定: ❌ 网络【学不会条件策略】—— 两种上下文给出同一个动作。\n");
        std::printf("        这和蛇网络的表现完全一致（10 种极端输入 → 同一个 argmax）。\n");
        std::printf("        ★ 这是根因：策略梯度在这个实现下只能学常量输出，\n");
        std::printf("          学不出「看输入决定动作」这种行为。\n");
    } else {
        std::printf("  判定: ⚠️ 部分学会 —— 两种上下文动作不同，但至少一个不对。\n");
        std::printf("        说明能学条件策略，但收敛不足或信号偏弱。\n");
    }
    std::printf("============================================================\n");
    return learned ? 0 : 1;
}
