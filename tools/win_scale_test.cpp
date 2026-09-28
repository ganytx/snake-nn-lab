// ============================================================================
//  win_scale_test.cpp — 检验「w_in 太小 / tanh 饱和」这个假设
//
//  ★ 假设（由实测推理得出，必须验证）
//
//  实测：训练 1000 局后
//      w_out |max| = 0.65   （Xavier 界 0.397 的 1.64 倍）
//      w_in  |max| = 0.375  （Xavier 界 0.304 的 1.24 倍）
//  → **增长几乎全在输出层**，输入层几乎没动。
//
//  机理假设：
//      观测各维大多 = 1.0（射线畅通），33 维求和 → 隐藏层 pre-activation
//      远大于 1 → tanh 深度饱和 → tanh'(z) ≈ 0
//      → 回传到 w_in 的梯度 ≈ 0  → w_in 学不动
//      → 网络只会"设常量输出"，学不出"看输入决定动作"
//
//  ★ 验证方法
//  把训练好的网络复制一份，只把 w_in 整体放大 k 倍，然后测：
//      ① 隐藏层 pre-activation 的分布（看饱和程度）
//      ② 逐维扰动的灵敏度（看响应幅度）
//      ③ 要把 argmax 翻过来需要多大的输入变化
//  如果放大 k 之后灵敏度显著上升、所需输入变化显著下降
//  → **假设成立**，修法是给 w_in 更快的有效学习率（或更好的初始化）
//
//  用法: win_scale_test.exe <weights.txt>
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

// 返回 {logits, 隐藏层 pre-activation 绝对值的平均}
std::pair<std::vector<double>, double>
fwd(const Net& n, const std::vector<double>& x, double winScale) {
    std::vector<double> a(static_cast<std::size_t>(n.HW), 0.0);
    double absZsum = 0.0;
    for (int h = 0; h < n.HW; ++h) {
        double s = n.b_in[static_cast<std::size_t>(h)];
        for (int i = 0; i < n.IN; ++i)
            s += x[static_cast<std::size_t>(i)]
               * n.w_in[static_cast<std::size_t>(i * n.HW + h)] * winScale;
        absZsum += std::fabs(s);
        a[static_cast<std::size_t>(h)] = std::tanh(s);
    }
    std::vector<double> z(static_cast<std::size_t>(n.OUT), 0.0);
    for (int o = 0; o < n.OUT; ++o) {
        double s = n.b_out[static_cast<std::size_t>(o)];
        for (int h = 0; h < n.HW; ++h)
            s += a[static_cast<std::size_t>(h)] * n.w_out[static_cast<std::size_t>(h * n.OUT + o)];
        z[static_cast<std::size_t>(o)] = s;
    }
    return { z, absZsum / static_cast<double>(n.HW) };
}

double l2(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("用法: win_scale_test.exe <weights.txt>\n"); return 1; }
    const Net n = load(argv[1]);
    if (!n.ok) { std::printf("权重加载失败\n"); return 1; }

    // 基准观测：和真实观测同量级（射线多为 1.0）
    //
    // ★ 下标全部由布局常量推出，不写死 —— 原来这里写死了 20/21/23/24，
    //   那是 33 维布局的位置；29 维回滚后它们全部错位。
    //   （本工具用 n.IN 而不是编译期 OBS，是因为它要能读任意维度的权重文件。）
    std::vector<double> base(static_cast<std::size_t>(n.IN), 1.0);
    // 自身标量块 = 障碍8 + 食物8 = 16 起，共 4 维：速度/长度/朝向sin/朝向cos
    const int selfB = 16;
    if (selfB + 3 < n.IN) {
        base[static_cast<std::size_t>(selfB + 0)] = 0.55;   // 速度/450
        base[static_cast<std::size_t>(selfB + 1)] = 0.2;    // 长度/60
        base[static_cast<std::size_t>(selfB + 2)] = 0.0;    // 朝向 sin
        base[static_cast<std::size_t>(selfB + 3)] = 1.0;    // 朝向 cos
    }
    // 圈与队友：按 (0,0,1) 哨兵铺（每组第 3 维 = 1）
    for (int i = 20; i < n.IN; ++i) base[static_cast<std::size_t>(i)] = (i % 3 == 2) ? 1.0 : 0.0;

    const double DELTA = 0.2;

    std::printf("============================================================\n");
    std::printf("  w_in 放大对「输入灵敏度」的影响\n");
    std::printf("  权重: %s\n", argv[1]);
    std::printf("============================================================\n\n");
    std::printf("  %-8s %-14s %-16s %-14s %s\n",
                "w_in×", "隐藏层|z|均值", "平均灵敏度", "所需输入变化", "判定");
    std::printf("  --------------------------------------------------------------------------\n");

    for (double k : { 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0 }) {
        const auto r0 = fwd(n, base, k);
        const auto& z0 = r0.first;
        double zmax = z0[0], zmin = z0[0];
        for (double v : z0) { if (v > zmax) zmax = v; if (v < zmin) zmin = v; }
        const double spread = zmax - zmin;

        double sum = 0.0;
        for (int i = 0; i < n.IN; ++i) {
            auto xp = base;
            double v = xp[static_cast<std::size_t>(i)] + DELTA;
            if (v > 1.0) v = xp[static_cast<std::size_t>(i)] - DELTA;
            xp[static_cast<std::size_t>(i)] = v;
            sum += l2(z0, fwd(n, xp, k).first);
        }
        const double avgSens = sum / static_cast<double>(n.IN);
        const double need = (avgSens > 1e-12) ? (DELTA * spread / avgSens) : 1e30;

        const char* verdict = (need > 2.0) ? "❌ 与输入无关"
                            : (need > 0.7) ? "⚠️ 偏弱" : "✅ 正常";
        std::printf("  %-8.1f %-14.4f %-16.6f %-14.2f %s\n",
                    k, r0.second, avgSens, need, verdict);
    }

    std::printf("\n------------------------------------------------------------\n");
    std::printf("  说明：\n");
    std::printf("    · 「隐藏层|z|均值」是 pre-activation 的绝对值平均。\n");
    std::printf("      |z| 越大 → tanh 越饱和 → tanh'(z) 越小 → 回传到 w_in 的梯度越小。\n");
    std::printf("    · 「所需输入变化」= 要让 argmax 翻一次，输入需要变多少。\n");
    std::printf("      真实观测各维的逐帧变化量级是 0.001 ~ 0.2。\n");
    std::printf("    · 若放大 w_in 能让「所需输入变化」从 >2 掉到 <1\n");
    std::printf("      → 假设成立：问题在 w_in 学不动（tanh 饱和吃掉梯度）。\n");
    std::printf("============================================================\n");
    return 0;
}
