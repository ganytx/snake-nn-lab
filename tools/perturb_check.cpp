// ============================================================================
//  perturb_check.cpp — 量化「网络的输出对输入有多敏感」
//
//  ★ 为什么需要它（而不是"看代码觉得对"）
//
//  实测事实：训练后的网络对 10 种【极端不同】的观测给出同一个 argmax，
//  而且【未训练】的网络也已经如此。这说明输出几乎不依赖输入。
//
//  但"几乎不依赖"是一个连续量，光看 argmax 相同分不出来：
//      · 是输出对输入完全没反应（灵敏度 = 0）？
//      · 还是输出有反应、只是响应幅度小于动作之间的差距？
//  这两种的根因和修法完全不同。
//
//  ★ 做法：逐个输入维度做扰动，量输出的移动
//
//  对每个输入维度 i：
//      x 和 x' 只在第 i 维差 δ（其它维完全相同）
//      记录 ‖f(x') − f(x)‖ 以及每个 logit 的变化
//  于是得到：
//      平均灵敏度   —— 整体上网络"听不听"输入
//      最大灵敏度   —— 有没有某一维被特别重视
//      逐维灵敏度表 —— 哪几维是"死"的（灵敏度 ≈ 0）
//
//  ★ 关键判据
//      观测各维的真实变化幅度：射线维在 0~1 之间、逐帧变化 ~0.001~0.2
//      动作之间的 logit 差距      ：约 0.5 ~ 3（决定了 argmax 要不要翻）
//      如果‖Δlogit‖ 对 0.2 的输入变化只有 0.01 量级
//        → 要把 argmax 翻过来需要输入变化 40 倍 → **策略实际上与输入无关**
//
//  用法: perturb_check.exe <weights.txt> [delta]
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int OBS = 29;
constexpr int ACT = 6;    // 诊断表长固定 6（回滚后只用到前 3 个）
// ★ 2026-09-28 随 29 维回滚更新
constexpr int OBSTACLE_BEGIN = 0;    // [0..7]   障碍射线 8 方向
constexpr int FOOD_BEGIN     = 8;    // [8..15]  食物 8 扇区
constexpr int SELF_BEGIN     = 16;   // [16..19] 自身标量
constexpr int ZONE_BEGIN     = 20;   // [20..25] 两个圈
constexpr int ALLY_BEGIN     = 26;   // [26..28] 队友

struct Net {
    int IN = 0, HW = 0, OUT = 0;
    std::vector<double> w_in, b_in, w_out, b_out;
    bool ok = false;
};

Net load(const std::string& path) {
    Net n;
    std::ifstream f{ std::filesystem::path(path) };
    if (!f) return n;
    std::string line, section;
    std::vector<double> cur;
    auto flush = [&]() {
        if (section == "w_in")  n.w_in  = cur;
        if (section == "b_in")  n.b_in  = cur;
        if (section == "w_out") n.w_out = cur;
        if (section == "b_out") n.b_out = cur;
        cur.clear();
    };
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::istringstream ss(line.substr(1));
            std::string nm; ss >> nm;
            if (!nm.empty()) { flush(); section = nm; }
            continue;
        }
        if (line.rfind("IN ",   0) == 0) { n.IN  = std::atoi(line.c_str() + 3); continue; }
        if (line.rfind("OUT ",  0) == 0) { n.OUT = std::atoi(line.c_str() + 4); continue; }
        if (line.rfind("NHID ", 0) == 0) continue;
        if (line.rfind("HW ",   0) == 0) { n.HW  = std::atoi(line.c_str() + 3); continue; }
        std::istringstream ss(line);
        double v; while (ss >> v) cur.push_back(v);
    }
    flush();
    n.ok = (n.IN > 0 && n.HW > 0 && n.OUT > 0 &&
            static_cast<int>(n.w_in.size())  == n.IN * n.HW &&
            static_cast<int>(n.b_in.size())  == n.HW &&
            static_cast<int>(n.w_out.size()) == n.HW * n.OUT &&
            static_cast<int>(n.b_out.size()) == n.OUT);
    return n;
}

std::vector<double> fwd(const Net& n, const std::vector<double>& x) {
    std::vector<double> a(static_cast<std::size_t>(n.HW), 0.0);
    for (int h = 0; h < n.HW; ++h) {
        double s = n.b_in[static_cast<std::size_t>(h)];
        for (int i = 0; i < n.IN; ++i)
            s += x[static_cast<std::size_t>(i)] * n.w_in[static_cast<std::size_t>(i * n.HW + h)];
        a[static_cast<std::size_t>(h)] = std::tanh(s);
    }
    std::vector<double> z(static_cast<std::size_t>(n.OUT), 0.0);
    for (int o = 0; o < n.OUT; ++o) {
        double s = n.b_out[static_cast<std::size_t>(o)];
        for (int h = 0; h < n.HW; ++h)
            s += a[static_cast<std::size_t>(h)] * n.w_out[static_cast<std::size_t>(h * n.OUT + o)];
        z[static_cast<std::size_t>(o)] = s;
    }
    return z;
}

double l2diff(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

// ★ 29 维回滚后障碍不再分"墙/敌身"两个通道，合并成一个 8 向障碍块
const char* dimName(int i) {
    if (i < FOOD_BEGIN)   return "障碍射线";
    if (i < SELF_BEGIN)   return "食物扇区";
    if (i < ZONE_BEGIN)   return "自身标量";
    if (i < ALLY_BEGIN)   return "圈(当前/下一个)";
    return "队友";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("用法: perturb_check.exe <weights.txt> [delta]\n"); return 1; }
    const double DELTA = (argc > 2) ? std::atof(argv[2]) : 0.2;

    const Net n = load(argv[1]);
    if (!n.ok) { std::printf("权重加载失败（需要单隐藏层 MLP）\n"); return 1; }

    std::printf("============================================================\n");
    std::printf("  输出对输入的灵敏度（逐维扰动 %.2f）\n", DELTA);
    std::printf("  权重: %s   IN=%d HW=%d OUT=%d\n", argv[1], n.IN, n.HW, n.OUT);
    std::printf("============================================================\n\n");

    // 基准输入：一个"平静"的观测（射线畅通、无食物、无敌人、圈不可见）
    //   ★ 用符号常量而不是写死下标 —— 原来写死的 20/21/23 是 33 维布局的下标，
    //     29 维回滚后会全部错位。
    std::vector<double> base(static_cast<std::size_t>(n.IN), 1.0);
    const int selfB = SELF_BEGIN;
    if (selfB + 3 < n.IN) {
        base[static_cast<std::size_t>(selfB + 0)] = 0.55;   // 速度/450
        base[static_cast<std::size_t>(selfB + 1)] = 0.2;    // 长度/60
        base[static_cast<std::size_t>(selfB + 2)] = 0.0;    // 朝向 sin
        base[static_cast<std::size_t>(selfB + 3)] = 1.0;    // 朝向 cos
    }
    // 圈与队友：按 (0,0,1) 哨兵铺（每组第 3 维 = 1）
    for (int i = ZONE_BEGIN; i < n.IN; ++i) {
        base[static_cast<std::size_t>(i)] = (i % 3 == 2) ? 1.0 : 0.0;
    }

    const auto z0 = fwd(n, base);
    double zmax0 = z0[0], zmin0 = z0[0];
    for (double v : z0) { if (v > zmax0) zmax0 = v; if (v < zmin0) zmin0 = v; }
    std::printf("  基准输入的 logits:");
    for (double v : z0) std::printf(" %+7.3f", v);
    std::printf("\n  基准 logit 幅度（max−min）= %.3f   ← argmax 要不要翻，看的就是它\n\n",
                zmax0 - zmin0);

    std::printf("  %-4s %-12s %12s %12s %s\n", "维", "属于", "‖Δlogit‖", "相对幅度", "说明");
    std::printf("  ------------------------------------------------------------------\n");

    double sumSens = 0.0, maxSens = 0.0;
    int    maxSensDim = -1, deadDims = 0;
    const double spread = (zmax0 - zmin0 > 1e-9) ? (zmax0 - zmin0) : 1.0;

    for (int i = 0; i < n.IN; ++i) {
        auto xp = base;
        double v = xp[static_cast<std::size_t>(i)] + DELTA;
        if (v > 1.0) v = xp[static_cast<std::size_t>(i)] - DELTA;   // 越界就反向扰动
        xp[static_cast<std::size_t>(i)] = v;

        const auto zp = fwd(n, xp);
        const double d = l2diff(z0, zp);
        sumSens += d;
        if (d > maxSens) { maxSens = d; maxSensDim = i; }
        if (d < 1e-9) ++deadDims;
    }

    // 只打印最灵敏的 8 维 + 最迟钝的 4 维，避免刷屏
    std::vector<std::pair<double,int>> sens;
    for (int i = 0; i < n.IN; ++i) {
        auto xp = base;
        double v = xp[static_cast<std::size_t>(i)] + DELTA;
        if (v > 1.0) v = xp[static_cast<std::size_t>(i)] - DELTA;
        xp[static_cast<std::size_t>(i)] = v;
        sens.push_back({ l2diff(z0, fwd(n, xp)), i });
    }
    std::vector<std::pair<double,int>> byHi = sens, byLo = sens;
    std::sort(byHi.begin(), byHi.end(), [](auto& a, auto& b){ return a.first > b.first; });
    std::sort(byLo.begin(), byLo.end(), [](auto& a, auto& b){ return a.first < b.first; });

    auto row = [&](const std::pair<double,int>& s) {
        std::printf("  %-4d %-12s %12.6f %11.4f%% %s\n", s.second, dimName(s.second), s.first,
                    100.0 * s.first / spread, s.second == maxSensDim ? "← 最灵敏" : "");
    };
    std::printf("  ---- 最灵敏的 8 维 ----\n");
    for (int k = 0; k < 8 && k < static_cast<int>(byHi.size()); ++k) row(byHi[static_cast<std::size_t>(k)]);
    std::printf("  ---- 最迟钝的 4 维 ----\n");
    for (int k = 0; k < 4 && k < static_cast<int>(byLo.size()); ++k) row(byLo[static_cast<std::size_t>(k)]);

    const double avgSens = sumSens / static_cast<double>(n.IN);
    std::printf("\n------------------------------------------------------------\n");
    std::printf("  平均灵敏度 ‖Δlogit‖        : %.6f   （输入每变 %.2f）\n", avgSens, DELTA);
    std::printf("  最大灵敏度 ‖Δlogit‖        : %.6f  （第 %d 维）\n", maxSens, maxSensDim);
    std::printf("  logit 幅度（决策差距）      : %.3f\n", spread);
    std::printf("  完全无反应的维数            : %d / %d\n", deadDims, n.IN);
    if (avgSens > 1e-12) {
        std::printf("  ★ 要把 argmax 翻过来，输入需要变化 : %.2f\n", DELTA * spread / avgSens);
        std::printf("    （观测各维的实际逐帧变化量级是 0.001 ~ 0.2）\n");
    }
    std::printf("------------------------------------------------------------\n");

    const double need = (avgSens > 1e-12) ? (DELTA * spread / avgSens) : 1e30;
    if (need > 2.0) {
        std::printf("  判定: ❌ 策略实际上【与输入无关】。\n");
        std::printf("        输入要在 0~1 的全量程上摆动 %.1f 次才够翻一次 argmax，\n", need);
        std::printf("        而真实观测每帧只变 0.001~0.2 → argmax 永远不动。\n");
    } else if (need > 0.7) {
        std::printf("  判定: ⚠️ 响应偏弱 —— 需要输入变化 %.2f 才翻一次，勉强够用。\n", need);
    } else {
        std::printf("  判定: ✅ 响应正常 —— 输入变 %.2f 就足以改变决策。\n", need);
    }
    std::printf("============================================================\n");
    return 0;
}
