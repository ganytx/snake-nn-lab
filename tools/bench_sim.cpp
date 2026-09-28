// ============================================================================
//  bench_sim.cpp — 仿真性能计时探针
//
//  目的：回答"这台机器一小时能跑多少帧"，以及"时间花在哪了"。
//
//  ★ 测量方法（上一版栽过跟头，这一版修正）
//
//  1. 同一个二进制连跑两次，帧率能差 11%~46%（机器负载、CPU 频率状态）。
//     所以"每次只测一遍"是不可信的。
//     → 本版每个用例跑 ROUNDS 轮，每轮都【重建全新场景】，取各环节的【最小值】。
//       最小值对偶发干扰最鲁棒：它代表"没有被打扰时"的真实速度。
//       顺带还省掉了单独的预热轮 —— 冷启动那一轮自然会被最小值淘汰。
//
//  2. 如果让 AI 操纵转向，长身体的蛇很快互相撞死；而 CollisionSystem
//     对死蛇直接 continue ——【死亡会悄悄降低负载】，测出来的不是满负载。
//     → 本版故意【不把 AI 的决策结果施加到蛇身上】，蛇直线飞行保证全部存活，
//       同时 AI 的计算成本照测不误。并加了存活数自检，一旦有蛇死了就报警。
//
//  3. 蛇身是【向后延伸】的，n 条蛇如果并排放在同一行，后一条的身体会
//     直接穿过前一条的头。→ 本版改成单列纵向排布，y 间隔 240px。
//
//  运行：build\bench_sim.exe
// ============================================================================

#include "entities/Snake.h"
#include "entities/FoodManager.h"
#include "systems/CollisionSystem.h"
#include "agent/PerceptionBuilder.h"
#include "brain/RuleBrain.h"
#include "utils/Math.h"
#include "utils/ConsoleUtf8.h"   // ★ 控制台 UTF-8 输出（否则中文乱码）

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

// 防止编译器把纯测量的循环优化掉
static double g_sink = 0.0;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
//  ★ 对拍测试：解析法标墙 vs 逐格朴素实现
//
//  PerceptionBuilder::markWalls 用的是解析法，它依赖
//  "地图是凸矩形 → 每一行里图内的格子必然连续" 这个性质，
//  还要处理旋转角接近 0°/90° 时约束退化的分支。
//  这属于"看起来对、容易错一格"的代码，所以合并前必须随机对拍：
//  位置 / 朝向 / 地图尺寸全随机，两份实现标出来的墙必须【逐格完全一致】。
// ---------------------------------------------------------------------------
static void markWallsNaive(float selfX, float selfY, float heading, float cell,
                           float mapW, float mapH, arena::VisionGrid& g) {
    const float c = std::cos(heading);
    const float s = std::sin(heading);
    for (int r = 0; r < arena::VISION_ROWS; ++r) {
        const float f = static_cast<float>(arena::VISION_CENTER_ROW - r) * cell;
        for (int col = 0; col < arena::VISION_COLS; ++col) {
            const float rr = static_cast<float>(col - arena::VISION_CENTER_COL) * cell;
            const float wx = selfX + f * c - rr * s;
            const float wy = selfY + f * s + rr * c;
            if (wx < 0.0f || wx > mapW || wy < 0.0f || wy > mapH) {
                g.cells[r][col] |= arena::CELL_WALL;
            }
        }
    }
}

static bool verifyWallMarking() {
    std::printf("\n=== 对拍：解析法标墙 vs 逐格朴素实现 ===\n");

    std::mt19937 rng(20260921u);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    agent::PerceptionBuilder builder;
    agent::PerceptionConfig cfg;
    cfg.cellSize = 32.0f;
    builder.setConfig(cfg);

    int badCases = 0, badCells = 0, totalCases = 0;

    for (int t = 0; t < 400; ++t) {
        // 交替换地图尺寸：小图（视野比图大）/ 大图 / 极小图 / 与视野同尺寸
        float mapW, mapH;
        switch (t % 4) {
            case 0: mapW = mapH = 2000.0f; break;   // 目标地图尺寸
            case 1: mapW = mapH = 8000.0f; break;   // 现在的尺寸
            case 2: mapW = mapH = 600.0f;  break;   // 视野比图还大
            default: mapW = mapH = 1024.0f; break;  // 与视野正好相等
        }
        // 允许跑到地图外一点（真实游戏里蛇身会拖出边界）
        const float sx = (uni(rng) * 1.4f - 0.2f) * mapW;
        const float sy = (uni(rng) * 1.4f - 0.2f) * mapH;
        float hd = (uni(rng) * 2.0f - 1.0f) * arena::limits::PI;
        if (t % 8 == 3) hd = 0.0f;                        // 专门打退化分支
        if (t % 8 == 7) hd = arena::limits::PI * 0.5f;

        arena::Perception pa;   // 解析法
        arena::Perception pn;   // 朴素法
        pa.mapWidth  = pn.mapWidth  = mapW;
        pa.mapHeight = pn.mapHeight = mapH;
        pa.grid.clear();
        pn.grid.clear();

        builder.markWalls(sx, sy, hd, pa);
        markWallsNaive(sx, sy, hd, cfg.cellSize, mapW, mapH, pn.grid);

        ++totalCases;
        int diff = 0;
        for (int r = 0; r < arena::VISION_ROWS; ++r)
            for (int col = 0; col < arena::VISION_COLS; ++col)
                if (pa.grid.cells[r][col] != pn.grid.cells[r][col]) ++diff;

        if (diff > 0) {
            ++badCases;
            badCells += diff;
            if (badCases <= 3) {
                std::printf("  [差异] 用例 %d：地图 %.0fx%.0f 位置 (%.1f, %.1f) "
                            "朝向 %.4f rad，%d 格不一致\n",
                            t, mapW, mapH, sx, sy, hd, diff);
            }
        }
    }

    if (badCases == 0) {
        std::printf("  [通过] %d 个随机用例，共 %lld 格全部一致\n",
                    totalCases,
                    static_cast<long long>(totalCases) * arena::VISION_ROWS * arena::VISION_COLS);
        return true;
    }
    std::printf("  [失败] %d/%d 个用例有差异，共 %d 格不一致 —— 解析法不能合并！\n",
                badCases, totalCases, badCells);
    return false;
}

// ---------------------------------------------------------------------------
//  各环节耗时（单位：毫秒/帧）
// ---------------------------------------------------------------------------
struct PhaseTimes {
    double move     = 0.0;   // 蛇移动 + 体节重算
    double ai       = 0.0;   // 规则 AI 决策
    double perceive = 0.0;   // ★ 新增：感知填充（光栅化视野网格）
    double collide  = 0.0;   // 碰撞检测（空间网格）
    double food     = 0.0;   // 食物更新 / 过期清理
    double total() const { return move + ai + perceive + collide + food; }
};

// ---------------------------------------------------------------------------
struct BenchScene {
    std::vector<Snake>                          snakes;
    // ★ 用 RuleBrain（对局里真正在用的算法），不再用已废弃的 AIController。
    //   之前测的是一个已经不参与对局的旧实现，量出来的"AI 决策成本"没有意义。
    std::vector<std::unique_ptr<arena::Brain>>  brains;
    std::vector<Snake*>             ptrs;
    std::vector<arena::Perception>  percepts;     // 每条蛇一份感知缓冲（复用）
    FoodManager                    food{8000.0f, 8000.0f};
    CollisionSystem                collision{8000.0f, 8000.0f};
    float mapW = 8000.0f;
    float mapH = 8000.0f;

    int aliveCount() const {
        int n = 0;
        for (const auto& s : snakes) if (s.isAlive()) ++n;
        return n;
    }
};

// ---------------------------------------------------------------------------
//  构造场景
//
//  蛇排成一列（同一个 x，不同的 y），全部朝 +x 前进：
//    · y 间隔 240px 远大于蛇身直径 32px → 不会互相撞死
//    · 300 帧最多前进 250 x 5 = 1250px → 从 x=1000 出发到 2250，不出界
// ---------------------------------------------------------------------------
static void buildScene(BenchScene& sc, int nSnakes, int segs, int nFoods) {
    sc = BenchScene();

    sc.food = FoodManager(sc.mapW, sc.mapH);
    if (nFoods > 0) {
        sc.food.applyConfig(nFoods * 2, 1, 1.0e9f, 1, 3, 2);
    } else {
        sc.food.applyConfig(1, 1, 1.0e9f, 1, 3, 2);
        sc.food.getFoods().clear();
    }

    sc.collision = CollisionSystem(sc.mapW, sc.mapH);
    sc.collision.applyConfig(5.0f, 10);

    const sf::Color colors[7] = {
        sf::Color(255, 107, 107), sf::Color(255, 180, 60), sf::Color(255, 230, 60),
        sf::Color(100, 220, 100), sf::Color(60, 180, 255), sf::Color(160, 100, 255),
        sf::Color(255, 130, 200),
    };

    const float ySpacing = std::min(240.0f, 7000.0f / std::max(1, nSnakes - 1));

    for (int i = 0; i < nSnakes; ++i) {
        Snake s(1000.0f, 400.0f + i * ySpacing, 0.0f, colors[i % 7], false);
        s.applyConfig(250.0f, 450.0f, 8.0f, 20.0f, 5, 2);
        s.setSegmentCount(segs);
        s.setTeamId(i / 3);            // 3 条一队，模拟真实队伍结构
        sc.snakes.push_back(std::move(s));
        // 每条蛇一份独立的 RuleBrain，并给不同的种子 ——
        // 否则所有蛇抽到同一个反应时间、同一条漫游轨迹，量出来的成本会偏。
        auto brain = std::make_unique<arena::RuleBrain>();
        brain->onEpisodeStart(static_cast<std::uint32_t>(1000 + i * 97));
        sc.brains.push_back(std::move(brain));
    }

    sc.ptrs.clear();
    for (auto& s : sc.snakes) sc.ptrs.push_back(&s);

    sc.percepts.resize(static_cast<std::size_t>(nSnakes));

    // 预热：让 trail / 体节缓冲达到稳定状态
    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 10; ++i) {
        for (auto& s : sc.snakes) s.update(dt);
    }
}

// ---------------------------------------------------------------------------
//  跑 frames 帧，返回【每帧平均】的各环节耗时
// ---------------------------------------------------------------------------
static PhaseTimes run(BenchScene& sc, int frames, bool aiUnthrottled) {
    const float dt    = 1.0f / 60.0f;

    agent::PerceptionBuilder builder;
    agent::PerceptionConfig  pcfg;
    pcfg.cellSize         = 32.0f;   // 32 格 x 32px = 视野 1024px
    pcfg.allyVisibleCount = 2;
    builder.setConfig(pcfg);

    agent::WorldView wv;
    wv.snakes    = &sc.ptrs;
    wv.foods     = &sc.food.getFoods();
    wv.mapWidth  = sc.mapW;
    wv.mapHeight = sc.mapH;
    wv.dt        = dt;

    PhaseTimes pt;
    const int n = static_cast<int>(sc.snakes.size());

    for (int f = 0; f < frames; ++f) {
        Clock::time_point t = Clock::now();

        // ---- 1. 感知填充（光栅化视野网格）----
        // ★ 必须排在 AI 决策【之前】：RuleBrain 只吃 Perception，
        //   先决策就等于让它看上一帧的（第一帧则是空的）感知。
        wv.tick = f;
        for (int i = 0; i < n; ++i) {
            builder.build(wv, static_cast<std::size_t>(i),
                          sc.percepts[static_cast<std::size_t>(i)]);
            g_sink += static_cast<double>(
                sc.percepts[static_cast<std::size_t>(i)].grid.cells[16][16]);
        }
        pt.perceive += msSince(t);
        t = Clock::now();

        // ---- 2. 规则 AI 决策 ----
        // 故意【不施加】决策结果，保证蛇直线飞行、全程存活
        for (int i = 0; i < n; ++i) {
            const std::size_t si = static_cast<std::size_t>(i);
            arena::Brain* b = sc.brains[si].get();
            if (aiUnthrottled) {
                // 上界参考：直接调 decide()，绕过 RuleBrain 内部的反应时间节流。
                // 注意——实际对局中大脑是【每帧都被调用】的（只是内部会跳过决策），
                // 所以这一档量的是"如果每帧都真的重算一次"的成本上界。
                // 场景里造的就是 RuleBrain，所以这个向下转型是安全的。
                const arena::Action act =
                    static_cast<arena::RuleBrain*>(b)->decide(sc.percepts[si]);
                g_sink += static_cast<double>(act.steer);
            } else {
                const arena::Action act = b->think(sc.percepts[si]);
                g_sink += static_cast<double>(act.steer);
            }
        }
        pt.ai += msSince(t);
        t = Clock::now();

        // ---- 3. 蛇移动 + 体节重算 ----
        for (auto& s : sc.snakes) s.update(dt);
        pt.move += msSince(t);
        t = Clock::now();

        // ---- 4. 碰撞（撞墙 / 撞蛇 / 吃食物）----
        sc.collision.checkAll(sc.ptrs, sc.food);
        pt.collide += msSince(t);
        t = Clock::now();

        // ---- 5. 食物更新 ----
        sc.food.update(dt);
        pt.food += msSince(t);
    }

    const double inv = 1.0 / static_cast<double>(frames);
    pt.move *= inv; pt.ai *= inv; pt.perceive *= inv;
    pt.collide *= inv; pt.food *= inv;
    return pt;
}

// ---------------------------------------------------------------------------
static void report(const char* title, int nSnakes, int segs, int nFoods,
                   const PhaseTimes& pt, int rounds, float cellSize, int visionPx) {
    const double total = pt.total();
    auto pct = [&](double v) { return total > 0.0 ? v / total * 100.0 : 0.0; };

    std::printf("\n=== %s ===\n", title);
    std::printf("  配置：蛇 %d 条 x %d 节, 食物 %d 个, 地图 8000x8000\n", nSnakes, segs, nFoods);
    std::printf("  感知：%d x %d 格 x %.0fpx = 视野 %dpx；统计取 %d 轮最小值\n",
                arena::VISION_ROWS, arena::VISION_COLS, cellSize, visionPx, rounds);
    std::printf("  --------------------------------------------------\n");
    std::printf("  蛇移动 + 体节重算 : %9.4f ms/帧  ( %4.1f%% )\n", pt.move,     pct(pt.move));
    std::printf("  AI 决策 (RuleBrain): %9.4f ms/帧  ( %4.1f%% )\n", pt.ai,     pct(pt.ai));
    std::printf("  ★ 感知填充        : %9.4f ms/帧  ( %4.1f%% )\n", pt.perceive, pct(pt.perceive));
    std::printf("  碰撞检测          : %9.4f ms/帧  ( %4.1f%% )\n", pt.collide,  pct(pt.collide));
    std::printf("  食物更新          : %9.4f ms/帧  ( %4.1f%% )\n", pt.food,     pct(pt.food));
    std::printf("  --------------------------------------------------\n");
    std::printf("  合计              : %9.4f ms/帧  ->  %7.0f 帧/秒 (单线程)\n",
                total, total > 0.0 ? 1000.0 / total : 0.0);
    std::printf("  6 线程并行估算     : %7.0f 帧/秒\n",
                total > 0.0 ? 6.0 * 1000.0 / total : 0.0);
    const double fps = total > 0.0 ? 1000.0 / total : 0.0;
    std::printf("  一局按 3000 帧算   : %.0f 局/小时 (单线程) / %.0f 局/小时 (6 线程)\n",
                fps * 3600.0 / 3000.0, fps * 6.0 * 3600.0 / 3000.0);
}

// ---------------------------------------------------------------------------
int main() {
    // ★ 先把控制台输出代码页切成 UTF-8，否则下面的中文全是乱码。
    //   见 src/utils/ConsoleUtf8.h 的说明（默认代码页 936 会花屏）。
    textutil::enableUtf8Console();
    // 关掉 stdout 缓冲：出问题（崩溃）时能立刻看到最后执行到哪一步
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("======================================================================\n");
    std::printf(" snake-nn-lab · 仿真性能计时探针\n");
    std::printf(" 目的：量出真实吞吐，判断训练可行性；不是猜\n");
    std::printf(" 方法：每个用例重建场景跑多轮，取各环节【最小值】以抗机器噪声\n");
    std::printf(" 注意：AI 决策结果不施加到蛇上 —— 否则蛇会撞死，死亡会降低负载\n");
    std::printf("======================================================================\n");

    // 先做对拍：解析法标墙必须和逐格朴素实现完全一致，否则后面的性能数字没意义
    const bool wallsOk = verifyWallMarking();

    const int FRAMES = 300;
    const int ROUNDS = 4;

    struct Case { const char* title; int snakes; int segs; int foods; bool aiUnthrottled; };
    const Case cases[] = {
        { "训练初期：6 条蛇（2 队），蛇还短",              6,   5,  400, false },
        { "训练标准：9 条蛇（3 队 x 3），长度中等",         9,  50,  600, false },
        { "对局变大：18 条蛇（6 队 x 3）",                18,  50, 1200, false },
        { "满负载：30 条蛇，较长",                        30, 100, 2000, false },
        { "上界参考：9 条蛇，AI 每帧都重算（绕过节流）",     9,  50,  600, true  },
    };

    for (const auto& c : cases) {
        PhaseTimes best;
        int aliveAtEnd = c.snakes;
        bool first = true;

        for (int r = 0; r < ROUNDS; ++r) {
            BenchScene sc;
            buildScene(sc, c.snakes, c.segs, c.foods);
            const PhaseTimes pt = run(sc, FRAMES, c.aiUnthrottled);
            if (first || pt.total() < best.total()) best = pt;
            first = false;
            aliveAtEnd = sc.aliveCount();
        }

        report(c.title, c.snakes, c.segs, c.foods, best, ROUNDS, 32.0f, 1024);

        if (aliveAtEnd != c.snakes) {
            std::printf("  [!!] 警告：结束时只剩 %d/%d 条蛇存活，本次测量偏乐观\n",
                        aliveAtEnd, c.snakes);
        }
    }

    std::printf("\n======================================================================\n");
    std::printf(" 说明\n");
    std::printf("  · 每轮都重建全新场景，取各环节最小值 —— 冷启动那轮自然被淘汰\n");
    std::printf("  · 单线程帧率 = 1000 / 合计耗时；多线程并行环境约可线性放大\n");
    std::printf("  · 感知成本 = 格数 x 每格约 10 次运算，与格边长【无关】\n");
    std::printf("  · 标墙对拍：%s\n", wallsOk ? "通过（解析法可信）" : "【失败】解析法有问题");
    std::printf("======================================================================\n");
    return wallsOk ? 0 : 1;
}
