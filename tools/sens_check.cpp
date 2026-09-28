// ============================================================================
//  sens_check.cpp — 检验训练出来的网络「到底有没有在用输入」
//
//  ★ 这个工具回答一个很基础、但我一直没直接测的问题
//
//    之前我量到"相邻帧 logits 只动 0.001~0.03，而决策差距是 ~3"，
//    于是推断"argmax 永远不变"。但相邻帧的输入本来就只差一点点 ——
//    那个测法**不能**证明网络无视输入，只能说它对这个量级的输入不敏感。
//
//    要判定"网络是不是根本没用输入"，必须喂它【极端不同】的观测：
//        · 食物在正前方 vs 食物在正后方
//        · 前面是墙（贴脸）vs 四周全空
//        · 全 0 vs 全 1
//    如果这些情况下 argmax 都不变 —— 那才是铁证。
//
//  ★ 为什么必须用【训练后】的权重
//    未训练的网络本来就没学到什么，测它没有意义。所以这个工具
//    只做一件事：加载权重文件 → 喂构造观测 → 报告输出。
//
//  用法:
//      sens_check.exe <weights.txt> [obs_size]
//      （obs_size 默认 33，与 rl::OBS_SIZE 一致）
//
//  ⚠️ 只读工具：不改训练、不改世界、不写任何文件。
// ============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int OBS  = 29;
constexpr int ACT  = 3;    // 左 / 直 / 右（回滚后没有加速档）

// 观测布局（见 rl/encoder.h）—— ★ 2026-09-28 随 29 维回滚更新
constexpr int OBSTACLE_BEGIN = 0;    // [0..7]   障碍射线 8 方向（墙|敌身|敌头|自己）
constexpr int FOOD_BEGIN     = 8;    // [8..15]  食物 8 扇区
constexpr int SELF_BEGIN     = 16;   // [16..19] 速度/长度/朝向sin/朝向cos
constexpr int ZONE_BEGIN     = 20;   // [20..22] 当前圈  [23..25] 下一个圈
constexpr int ALLY_BEGIN     = 26;   // [26..28] 队友

struct Weights {
    int IN = 0, OUT = 0, NHID = 0, HW = 0;
    std::vector<double> w_in, b_in, w_out, b_out;
    bool ok = false;
};

Weights loadWeights(const std::string& path) {
    Weights w;
    std::ifstream f{ std::filesystem::path(path) };
    if (!f) { std::printf("打不开权重文件: %s\n", path.c_str()); return w; }

    std::string line;
    std::vector<double> cur;
    std::string section;
    auto flush = [&]() {
        if (section == "w_in")  w.w_in  = cur;
        if (section == "b_in")  w.b_in  = cur;
        if (section == "w_out") w.w_out = cur;
        if (section == "b_out") w.b_out = cur;
        cur.clear();
    };
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            // "# w_in 33x32" 或 "# b_in 32"
            std::istringstream ss(line.substr(1));
            std::string name; ss >> name;
            if (!name.empty()) { flush(); section = name; }
            continue;
        }
        if (line.rfind("IN ",   0) == 0) { w.IN   = std::atoi(line.c_str() + 3); continue; }
        if (line.rfind("OUT ",  0) == 0) { w.OUT  = std::atoi(line.c_str() + 4); continue; }
        if (line.rfind("NHID ", 0) == 0) { w.NHID = std::atoi(line.c_str() + 5); continue; }
        if (line.rfind("HW ",   0) == 0) { w.HW   = std::atoi(line.c_str() + 3); continue; }
        std::istringstream ss(line);
        double v;
        while (ss >> v) cur.push_back(v);
    }
    flush();
    w.ok = (w.IN > 0 && w.OUT > 0 && w.NHID == 1 &&
            static_cast<int>(w.w_in.size())  == w.IN * w.HW &&
            static_cast<int>(w.b_in.size())  == w.HW &&
            static_cast<int>(w.w_out.size()) == w.HW * w.OUT &&
            static_cast<int>(w.b_out.size()) == w.OUT);
    return w;
}

// 单隐藏层 tanh：z_out = tanh(x·W_in + b_in)·W_out + b_out
std::vector<double> forward(const Weights& w, const std::vector<double>& x) {
    std::vector<double> a(static_cast<std::size_t>(w.HW), 0.0);
    for (int h = 0; h < w.HW; ++h) {
        double s = w.b_in[static_cast<std::size_t>(h)];
        for (int i = 0; i < w.IN; ++i) {
            s += x[static_cast<std::size_t>(i)] * w.w_in[static_cast<std::size_t>(i * w.HW + h)];
        }
        a[static_cast<std::size_t>(h)] = std::tanh(s);
    }
    std::vector<double> z(static_cast<std::size_t>(w.OUT), 0.0);
    for (int o = 0; o < w.OUT; ++o) {
        double s = w.b_out[static_cast<std::size_t>(o)];
        for (int h = 0; h < w.HW; ++h) {
            s += a[static_cast<std::size_t>(h)] * w.w_out[static_cast<std::size_t>(h * w.OUT + o)];
        }
        z[static_cast<std::size_t>(o)] = s;
    }
    return z;
}

int argmaxOf(const std::vector<double>& z) {
    int b = 0;
    for (std::size_t i = 1; i < z.size(); ++i) if (z[i] > z[static_cast<std::size_t>(b)]) b = static_cast<int>(i);
    return b;
}

// ★ 动作名表固定 6 项（表长与 ACT 解耦）：回滚后 ACT = 3，只用前三项。
//   写成 [ACT] 而给 6 个初始值会编译失败（too many initializers）。
const char* kActName[6] = { "左转", "直行", "右转", "左转+加速", "直行+加速", "右转+加速" };

// 造一个"什么都没看见"的基准观测：所有射线畅通(1.0)、无食物、无敌人
//
// ★ 全部用【符号常量】而不是写死下标 —— 这个函数原来写死了 20..32，
//   是 33 维布局的下标。29 维回滚后那些下标全错位了。
//   用常量之后，以后改布局这个函数自动跟着走（这正是它该有的写法）。
std::vector<double> baseObs() {
    std::vector<double> x(OBS, 1.0);          // 射线默认畅通
    // 自身标量：速度/长度/朝向 sin/朝向 cos
    for (int i = SELF_BEGIN; i < SELF_BEGIN + 4; ++i) x[static_cast<std::size_t>(i)] = 0.0;
    x[SELF_BEGIN + 0] = 0.55;                  // 速度/450（正常速）
    x[SELF_BEGIN + 1] = 0.2;                   // 长度/60
    x[SELF_BEGIN + 2] = 0.0;                   // 朝向 sin
    x[SELF_BEGIN + 3] = 1.0;                   // 朝向 cos
    // 圈：默认哨兵 (0, 0, 1) = "没有圈"
    for (int slot = 0; slot < 2; ++slot) {
        const int b = ZONE_BEGIN + slot * 3;
        x[static_cast<std::size_t>(b + 0)] = 0.0;
        x[static_cast<std::size_t>(b + 1)] = 0.0;
        x[static_cast<std::size_t>(b + 2)] = 1.0;
    }
    // 队友：无
    x[ALLY_BEGIN + 0] = 0.0;
    x[ALLY_BEGIN + 1] = 0.0;
    x[ALLY_BEGIN + 2] = 0.0;
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: sens_check.exe <weights.txt>\n");
        return 1;
    }
    const Weights w = loadWeights(argv[1]);
    if (!w.ok) { std::printf("权重加载失败或结构不匹配（需要单隐藏层 MLP）\n"); return 1; }

    std::printf("============================================================\n");
    std::printf("  网络「是否在使用输入」直接检验\n");
    std::printf("  权重: %s\n", argv[1]);
    std::printf("  结构: IN=%d  HW=%d  OUT=%d  (参数 %zu)\n",
                w.IN, w.HW, w.OUT, w.w_in.size() + w.b_in.size() + w.w_out.size() + w.b_out.size());
    std::printf("============================================================\n\n");

    struct Case { const char* name; std::vector<double> x; };
    std::vector<Case> cases;

    // ---- 1. 全 0 / 全 1 / 随机：极端不同的输入 ----
    { std::vector<double> z(OBS, 0.0);  cases.push_back({ "全部输入 = 0", z }); }
    { std::vector<double> o(OBS, 1.0);  cases.push_back({ "全部输入 = 1", o }); }

    // ---- 2. 食物在【正前方】(扇区 0)：这是最关键的一例 ----
    {
        auto x = baseObs();
        x[FOOD_BEGIN + 0] = 0.15;                  // 前方 120px 有食物
        cases.push_back({ "食物在【正前方】(120px)", x });
    }
    // ---- 3. 食物在【正后方】(扇区 4) ----
    {
        auto x = baseObs();
        x[FOOD_BEGIN + 4] = 0.15;
        cases.push_back({ "食物在【正后方】(120px)", x });
    }
    // ---- 4. 食物在【正左方】(扇区 6) ----
    {
        auto x = baseObs();
        x[FOOD_BEGIN + 6] = 0.15;
        cases.push_back({ "食物在【正左方】(120px)", x });
    }
    // ---- 5. 前面【贴墙】 ----
    {
        auto x = baseObs();
        // ★ 29 维回滚后障碍只有一个通道（不再分墙/敌身），所以这一维
        //   同时代表"墙或敌人"。名字保留是因为它测的仍然是"前方有障碍"。
        x[OBSTACLE_BEGIN + 0] = 0.03;              // 正前方 30px 有障碍
        cases.push_back({ "正前方 30px 是障碍", x });
    }
    // ---- 6. 四周全空、远处有食物 ----
    {
        auto x = baseObs();
        x[FOOD_BEGIN + 2] = 0.9;
        cases.push_back({ "右侧很远有食物(720px)", x });
    }
    // ---- 7. 敌人正前方贴脸 ----
    {
        auto x = baseObs();
        x[OBSTACLE_BEGIN + 0] = 0.05;              // 前方 50px 有障碍（= 敌人）
        cases.push_back({ "正前方 50px 是障碍(敌人)", x });
    }
    // ---- 8. 圈在正前方（引导信号） ----
    {
        auto x = baseObs();
        x[ZONE_BEGIN + 0] = 0.0; x[ZONE_BEGIN + 1] = 1.0; x[ZONE_BEGIN + 2] = 0.3;
        cases.push_back({ "当前圈在正前方(600px)", x });
    }
    // ---- 9. 圈在正后方 ----
    {
        auto x = baseObs();
        x[ZONE_BEGIN + 0] = 0.0; x[ZONE_BEGIN + 1] = -1.0; x[ZONE_BEGIN + 2] = 0.3;
        cases.push_back({ "当前圈在正后方(600px)", x });
    }
    // ---- 10. 随机观测（均值 0.5）----
    {
        std::vector<double> r(OBS, 0.0);
        unsigned s = 12345u;
        for (int i = 0; i < OBS; ++i) {
            s = s * 1103515245u + 12345u;
            r[static_cast<std::size_t>(i)] = static_cast<double>((s >> 16) & 0x7FFF) / 32767.0;
        }
        cases.push_back({ "随机观测(0~1)", r });
    }

    std::printf("  %-26s %-46s %-14s %s\n", "输入情形", "logits (6 个动作)", "argmax", "幅度");
    std::printf("  ------------------------------------------------------------------------------------------\n");

    int prevArg = -1;
    int distinctArg = 0;
    bool seen[ACT] = { false, false, false, false, false, false };
    double spreadMin = 1e30, spreadMax = -1e30;

    for (const auto& c : cases) {
        // 维度对齐（万一权重是别的观测维度，就截断/补零，保证不越界）
        std::vector<double> x(static_cast<std::size_t>(w.IN), 0.0);
        const std::size_t n = std::min(x.size(), c.x.size());
        for (std::size_t i = 0; i < n; ++i) x[i] = c.x[i];

        const auto z = forward(w, x);
        const int am = argmaxOf(z);
        double zmax = z[0], zmin = z[0];
        for (double v : z) { if (v > zmax) zmax = v; if (v < zmin) zmin = v; }
        const double sp = zmax - zmin;
        if (sp < spreadMin) spreadMin = sp;
        if (sp > spreadMax) spreadMax = sp;

        std::string zs;
        char buf[32];
        for (int i = 0; i < w.OUT; ++i) {
            std::snprintf(buf, sizeof(buf), "%+7.3f ", z[static_cast<std::size_t>(i)]);
            zs += buf;
        }
        std::printf("  %-26s %-46s %-14s %6.3f\n", c.name, zs.c_str(),
                    kActName[am % ACT], sp);

        if (!seen[am % ACT]) { seen[am % ACT] = true; ++distinctArg; }
        if (am != prevArg) { prevArg = am; }
    }

    std::printf("\n------------------------------------------------------------\n");
    std::printf("  这些输入的差别是【极端】的（食物从正前挪到正后、前面从空变成墙）。\n");
    std::printf("  出现的不同 argmax 个数 : %d / %d\n", distinctArg, ACT);
    std::printf("  logit 幅度范围         : %.3f ~ %.3f\n", spreadMin, spreadMax);
    std::printf("------------------------------------------------------------\n");
    if (distinctArg <= 1) {
        std::printf("  判定: ❌ 网络【完全无视输入】—— 10 种极端不同输入给出同一个动作。\n");
        std::printf("        这就是「学不会」的根因：策略根本不是输入的函数。\n");
    } else if (distinctArg == 2) {
        std::printf("  判定: ⚠️ 网络对输入【几乎无反应】—— 10 种极端输入只产生 2 种动作。\n");
        std::printf("        策略与输入的关系极弱，不足以支撑任何有智慧的行为。\n");
    } else {
        std::printf("  判定: ✅ 网络确实在使用输入（%d 种不同动作）。\n", distinctArg);
        std::printf("        那么「学不会」的原因在上面这些具体映射【对不对】上，\n");
        std::printf("        请逐行核对：食物在正前方时应选【直行】、正前方是墙时应选【转向】。\n");
    }
    std::printf("============================================================\n");
    return 0;
}
