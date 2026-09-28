// ============================================================================
//  analyze_metrics.cpp — 训练指标统计分析（只读，不参与训练）
//
//  ★ 为什么需要这个工具
//
//  排查"reward 越跑越低"时，需要反复做同一类计算：
//      单局回报 与 吃食数 / 存活帧 / 各梯度的【相关性】
//      前后半程的【趋势】
//      回报的【分布形状】（是不是双峰）
//      各项梯度的【相对大小】（谁在推策略）
//
//  用 PowerShell 做这些事既慢又错（实测：`Measure-Object -StandardDeviation`
//  在这个环境里根本不存在）。写成一个只读工具，一次编译，反复用。
//
//  用法：
//      analyze_metrics.exe <train_metrics.csv>
//
//  ⚠️ 它【只读 CSV】，不碰训练、不碰世界。删掉它不影响任何东西。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

struct Series {
    std::vector<double> v;
    void push(double x) { v.push_back(x); }
    bool empty() const { return v.empty(); }
    double mean() const {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
    }
    double sd() const {
        if (v.size() < 2) return 0.0;
        const double m = mean();
        double s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / static_cast<double>(v.size() - 1));
    }
    double pct(double q) const {
        if (v.empty()) return 0.0;
        std::vector<double> c = v;
        std::sort(c.begin(), c.end());
        const double idx = q * static_cast<double>(c.size() - 1);
        const std::size_t i0 = static_cast<std::size_t>(std::floor(idx));
        const std::size_t i1 = std::min(i0 + 1, c.size() - 1);
        const double f = idx - static_cast<double>(i0);
        return c[i0] * (1.0 - f) + c[i1] * f;
    }
};

double corr(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    if (n < 3) return 0.0;
    double ma = 0.0, mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= static_cast<double>(n); mb /= static_cast<double>(n);
    double sab = 0.0, saa = 0.0, sbb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0.0 || sbb <= 0.0) return 0.0;
    return sab / std::sqrt(saa * sbb);
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: analyze_metrics <train_metrics.csv>\n");
        return 1;
    }
    std::FILE* f = std::fopen(argv[1], "r");
    if (!f) { std::printf("打不开 %s\n", argv[1]); return 1; }

    // 读表：kind,episode,value,tag
    char line[4096];
    // 跳过表头
    if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); return 1; }

    // per-episode 序列（kind == "train"）
    std::map<std::string, std::vector<double>> perEp;   // tag -> 按局序的值
    std::map<std::string, std::vector<long>>   perEpIdx;
    std::map<long, std::map<std::string, double>> rowOf;

    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty()) continue;
        const auto c = split(s, ',');
        if (c.size() < 4) continue;
        if (c[0] != "train") continue;
        const long ep = std::atol(c[1].c_str());
        const double val = std::atof(c[2].c_str());
        const std::string& tag = c[3];
        rowOf[ep][tag] = val;
    }
    std::fclose(f);

    for (const auto& kv : rowOf) {
        const long ep = kv.first;
        for (const auto& t : kv.second) {
            perEp[t.first].push_back(t.second);
            perEpIdx[t.first].push_back(ep);
        }
    }

    if (rowOf.empty()) { std::printf("CSV 里没有 kind=train 的行\n"); return 1; }

    std::printf("============================================================\n");
    std::printf("  训练指标分析: %s\n", argv[1]);
    std::printf("  train 局数: %zu\n", rowOf.size());
    std::printf("============================================================\n\n");

    auto has = [&](const char* t) { return perEp.count(t) && !perEp[t].empty(); };
    auto S   = [&](const char* t) -> const std::vector<double>& { return perEp[t]; };

    // ---- 1. 回报分布 ----
    if (has("reward")) {
        const auto& R = S("reward");
        std::printf("【1】单局回报分布\n");
        std::printf("    均值 %9.3f   标准差 %9.3f\n", Series{R}.mean(), Series{R}.sd());
        std::printf("    最小 %9.3f   p10 %9.3f   p25 %9.3f   中位 %9.3f   p75 %9.3f   p90 %9.3f   最大 %9.3f\n",
                    Series{R}.pct(0.0), Series{R}.pct(0.10), Series{R}.pct(0.25),
                    Series{R}.pct(0.50), Series{R}.pct(0.75), Series{R}.pct(0.90),
                    Series{R}.pct(1.0));
        std::printf("    ★ 标准差/均值 = %.2f   (>1 = 噪声比信号大)\n\n",
                    std::abs(Series{R}.mean()) > 1e-9
                        ? Series{R}.sd() / std::abs(Series{R}.mean()) : 0.0);
    }

    // ---- 2. 回报与各量的相关性 ----
    std::printf("【2】单局回报 与 各量的相关性（同局内）\n");
    const char* targets[] = { "food", "survival", "died", "gradNorm",
                              "entGradNorm", "updateNorm", "paramNorm", "baseline" };
    for (const char* t : targets) {
        if (!has(t) || !has("reward")) continue;
        // 按 episode 对齐
        std::vector<double> a, b;
        for (const auto& kv : rowOf) {
            auto ia = kv.second.find("reward");
            auto ib = kv.second.find(t);
            if (ia != kv.second.end() && ib != kv.second.end()) {
                a.push_back(ia->second);
                b.push_back(ib->second);
            }
        }
        std::printf("    corr(reward, %-12s) = %+7.4f\n", t, corr(a, b));
    }
    std::printf("\n");

    // ---- 3. 五等分趋势 ----
    std::printf("【3】五等分趋势（看「越跑越低」到底发生在哪一段）\n");
    std::printf("    %-14s", "指标");
    for (int q = 1; q <= 5; ++q) std::printf("   Q%d", q);
    std::printf("     Q5-Q1\n");
    const char* trend[] = { "reward", "food", "survival", "gradNorm",
                            "entGradNorm", "updateNorm", "paramNorm" };
    for (const char* t : trend) {
        if (!has(t)) continue;
        const auto& v = S(t);
        const std::size_t n = v.size();
        const std::size_t q = n / 5;
        if (q == 0) continue;
        std::printf("    %-14s", t);
        double first = 0.0, last = 0.0;
        for (int i = 0; i < 5; ++i) {
            const std::size_t lo = static_cast<std::size_t>(i) * q;
            const std::size_t hi = (i == 4) ? n : lo + q;
            double s = 0.0;
            for (std::size_t k = lo; k < hi; ++k) s += v[k];
            const double m = s / static_cast<double>(hi - lo);
            if (i == 0) first = m;
            if (i == 4) last = m;
            std::printf(" %6.3f", m);
        }
        std::printf("   %+8.3f\n", last - first);
    }
    std::printf("\n");

    // ---- 4. 熵项 vs 任务梯度 ----
    if (has("entGradNorm") && has("gradNorm")) {
        const auto& E = S("entGradNorm");
        const auto& G = S("gradNorm");
        std::printf("【4】熵项梯度 / 总梯度（谁在推策略）\n");
        const std::size_t n = std::min(E.size(), G.size());
        const std::size_t q = n / 5;
        if (q > 0) {
            std::printf("    %-8s", "分段");
            for (int i = 1; i <= 5; ++i) std::printf("      Q%d", i);
            std::printf("\n    %-8s", "ent/total");
            for (int i = 0; i < 5; ++i) {
                const std::size_t lo = static_cast<std::size_t>(i) * q;
                const std::size_t hi = (i == 4) ? n : lo + q;
                double se = 0.0, sg = 0.0;
                for (std::size_t k = lo; k < hi; ++k) { se += E[k]; sg += G[k]; }
                std::printf("  %6.1f%%", sg > 1e-12 ? 100.0 * se / sg : 0.0);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }

    // ---- 5. 锁定局占比与动作统计（如果有） ----
    if (has("lockedFrac") || has("greedySwitchRate")) {
        std::printf("【5】策略确定性化指标\n");
        if (has("lockedFrac")) {
            const auto& L = S("lockedFrac");
            std::printf("    lockedFrac   :");
            for (double x : L) std::printf(" %5.1f%%", 100.0 * x);
            std::printf("\n");
        }
        if (has("greedySwitchRate")) {
            const auto& Gs = S("greedySwitchRate");
            std::printf("    switchRate   :");
            for (double x : Gs) std::printf(" %6.4f", x);
            std::printf("\n");
        }
        if (has("topGreedyFrac")) {
            const auto& T = S("topGreedyFrac");
            std::printf("    topGreedyFrac:");
            for (double x : T) std::printf(" %5.1f%%", 100.0 * x);
            std::printf("\n");
        }
        std::printf("\n");
    }

    // ---- 6. 奖励构成（如果有 reward_<name> 形式） ----
    std::printf("【6】可用的 tag 一览（便于核对上面引用的指标是否存在）\n    ");
    int col = 0;
    for (const auto& kv : perEp) {
        std::printf("%s(%zu)  ", kv.first.c_str(), kv.second.size());
        if (++col % 6 == 0) std::printf("\n    ");
    }
    std::printf("\n");
    return 0;
}
