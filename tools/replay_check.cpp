// ============================================================================
//  replay_check.cpp — 「世界指纹」回归工具
//
//  ★ 它要回答一个问题：
//
//      「我重构了世界推进的代码（比如把 GameScene 里的循环搬进 Arena），
//        游戏的【行为】到底有没有变？」
//
//  游戏本体是图形程序，没法在无头环境里自动比对"改动前后画面是否一致"。
//  所以这里把同一局跑一遍，输出一个【确定性的世界指纹】：
//
//      固定配置 + 固定种子 + 固定动作序列  →  每 N 帧一行状态 + 全剧哈希
//
//  用法：
//      build\replay_check.exe            → 打印指纹
//      build\replay_check.exe > before.txt     重构前
//      build\replay_check.exe > after.txt      重构后
//      fc before.txt after.txt                 应当逐字节相同
//
//  ★ 为什么它必须是确定性的
//
//  游戏里的随机数来自全局 rand()，用在两处：
//      · Snake 出生时抽的"转向起伏相位"（影响每条蛇的转头手感）
//      · FoodManager 生成食物的位置
//  所以这里用 srand(固定种子) 把随机序列钉死，同一份代码跑两次结果完全一样。
//
//  ⚠️ 局限（必须清楚）：
//  这不是"完整对局"的指纹 —— 它不包含限时、胜负、结算，
//  因为那些还在 GameScene 里（见 docs/45-Arena设计.md 第 2 节的职责边界）。
//  它覆盖的是【世界推进】这一层：移动 / 碰撞 / 饥饿 / 尸体 / 决策 / 食物。
//
//  运行：build\replay_check.exe（必须在 build\ 目录下运行）
// ============================================================================

#include "entities/Snake.h"
#include "entities/FoodManager.h"
#include "systems/CollisionSystem.h"
#include "agent/PerceptionBuilder.h"
#include "brain/RuleBrain.h"
#include "brain/BrainRegistry.h"
#include "brain/Action.h"
#include "arena/Arena.h"          // ★ 对照模式：用真正的 Arena 跑同一局
#include "utils/ConsoleUtf8.h"   // ★ 控制台 UTF-8 输出（否则中文乱码）

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
//  固定的复现配置
//
//  与 config/settings.cfg 的默认值一致（3 队 x 3 蛇）。
//  改这里等于换一个被测场景 —— 改完必须重新录基线。
// ---------------------------------------------------------------------------
constexpr float MAP_W = 2000.0f;
constexpr float MAP_H = 2000.0f;
constexpr int   TEAMS = 3;
constexpr int   PER_TEAM = 3;
constexpr int   FRAMES = 1800;          // 30 秒
constexpr int   PRINT_EVERY = 60;       // 每 60 帧（1 秒）打一行
constexpr unsigned SEED = 20260922u;

constexpr float DT = 1.0f / 60.0f;

// ---------------------------------------------------------------------------
//  FNV-1a 64 位哈希
//
//  为什么要哈希：1800 帧 x 9 条蛇的完整状态太多，不适合逐行 diff。
//  但它也不能替代逐值输出 —— 哈希只能告诉你"变了"，
//  逐值输出才能告诉你"哪里变了"。所以两者都留。
// ---------------------------------------------------------------------------
struct Hash {
    std::uint64_t h = 1469598103934665603ull;

    void addBits(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (i * 8)) & 0xFFull;
            h *= 1099511628211ull;
        }
    }
    // ★ 浮点必须按【位模式】入哈希，不能先转成十进制文本再哈希 ——
    //   否则 -0.0 与 0.0、以及不同精度的打印都会引入假差异。
    void addFloat(float v) {
        std::uint32_t bits;
        static_assert(sizeof(bits) == sizeof(v), "float 必须是 4 字节");
        std::memcpy(&bits, &v, sizeof(bits));
        addBits(bits);
    }
    void addInt(std::uint64_t v) { addBits(v); }
};

// ---------------------------------------------------------------------------
//  被测世界
// ---------------------------------------------------------------------------
struct World {
    std::vector<Snake>                         snakes;
    std::vector<std::unique_ptr<arena::Brain>> brains;
    std::vector<Snake*>                        ptrs;
    std::vector<arena::Perception>             percepts;

    FoodManager     food{MAP_W, MAP_H};
    CollisionSystem collision{MAP_W, MAP_H};

    agent::PerceptionBuilder builder;
    agent::WorldView         view;

    unsigned tick = 0;

    void build() {
        // ★ 种子必须在【任何 rand() 之前】设好。
        //   Snake 的构造函数里就会抽"转向起伏相位"。
        std::srand(SEED);

        snakes.clear();
        brains.clear();
        ptrs.clear();
        percepts.clear();
        tick = 0;

        food = FoodManager(MAP_W, MAP_H);
        // ★ 参数表：(每秒生成数, 吃几个长一节, 尸体间隔, 尸体分, 尸体节数,
        //             普通食物寿命, 尸体寿命)
        //
        //   ⚠️ 这几个值必须和 runWithArena() 里填的 ArenaConfig【逐个对齐】。
        //      差任何一个（比如 foodSegments）都会让指纹从第一行就不同，
        //      而现象很像是"世界推进逻辑不等价" —— 极难排查。
        //      已经踩过一次。
        //
        //   「每秒生成数 = 0」= 不自动生成，只靠下面 seedUniform 撒初始量。
        //   FoodManager 改成"速率 + 食物圈"之后，逐帧生成节奏已经和老行为
        //   不同了；要两种模式逐位一致，就得让生成【在两边都完全不发生】，
        //   只比较"世界推进 + 碰撞 + 饥饿 + 决策"。
        //   食物圈本身由将来的 zones 模式单独验证。
        food.applyConfig(0.0f, 1, 1, 3, 2, 1.0e9f, 1.0e9f);
        food.seedUniform(60);

        collision = CollisionSystem(MAP_W, MAP_H);
        collision.applyConfig(5.0f, 10);
        collision.setFriendlyFire(false);

        agent::PerceptionConfig pc;
        pc.cellSize         = 32.0f;
        pc.allyVisibleCount = 2;
        builder.setConfig(pc);

        // ---- 出生布局：与 GameScene::buildTeams 完全一致 ----
        const float cx = MAP_W * 0.5f;
        const float cy = MAP_H * 0.5f;
        const float spawnRadius = std::min(MAP_W, MAP_H) * 0.32f;

        for (int t = 0; t < TEAMS; ++t) {
            const float baseAngle = arena::limits::TWO_PI * static_cast<float>(t)
                                  / static_cast<float>(TEAMS);
            for (int k = 0; k < PER_TEAM; ++k) {
                const float a = baseAngle +
                                (static_cast<float>(k) - (PER_TEAM - 1) * 0.5f) * 0.22f;
                const float r = spawnRadius * (0.82f + 0.18f * static_cast<float>(k % 2));
                const float x = cx + std::cos(a) * r;
                const float y = cy + std::sin(a) * r;
                const float heading = std::atan2(cy - y, cx - x);

                snakes.emplace_back(x, y, heading, sf::Color::White, false);
                Snake& s = snakes.back();
                // ★ 参数表：(速度, 加速, 转向率, 体节间距, 吃几个长一节, 吃几个尸体长一节)
                //   ⚠️「吃几个长一节」必须和 ArenaConfig::foodSegments 一致。
                //      这里曾经是硬编的 5、而 Arena 那边是 1，导致两边
                //      "吃同样多的食物、长度却不同"，指纹从第一行就分叉。
                //      现象看起来像"世界推进逻辑不等价"，极难排查 ——
                //      最后靠打印「每次进食前后的体节数」才定位到。
                s.applyConfig(250.0f, 450.0f, 8.0f, 20.0f, /*foodNeeded*/ 1, 2);
                s.setSegmentCount(12);
                s.setTeamId(t);
                s.resetStarvation();

                auto brain = std::make_unique<arena::RuleBrain>();
                brain->onEpisodeStart(static_cast<std::uint32_t>(1000 + t * 97 + k));
                brains.push_back(std::move(brain));
            }
        }

        for (auto& s : snakes) ptrs.push_back(&s);
        percepts.resize(snakes.size());
        // 尸体是否已转成食物 —— 每局重置（对应 GameScene 的 Agent::deathHandled）
        deadHandled.assign(snakes.size(), 0);

        view.snakes    = &ptrs;
        view.foods     = &food.getFoods();
        view.mapWidth  = MAP_W;
        view.mapHeight = MAP_H;
    }

    // 一帧：与 GameScene::fixedUpdate 的 1~6 步一一对应
    void step() {
        ++tick;

        // 1. 移动
        for (auto& s : snakes) s.update(DT);

        // 2. 碰撞
        collision.checkAll(ptrs, food);

        // 3. 饥饿
        for (auto& s : snakes) {
            if (s.isAlive() && s.getTimeSinceEat() > 20.0f) s.die();
        }

        // 4. 死亡 → 尸体（GameScene 用 deathHandled 保证只做一次）
        for (std::size_t i = 0; i < snakes.size(); ++i) {
            if (deadHandled[i]) continue;
            Snake& s = snakes[i];
            if (s.isAlive()) continue;
            food.spawnCorpseFood(s);
            deadHandled[i] = true;
        }

        // 5. 感知 + 决策
        view.tick = static_cast<int>(tick);
        view.dt   = DT;
        for (std::size_t i = 0; i < snakes.size(); ++i) {
            if (!snakes[i].isAlive()) continue;
            builder.build(view, i, percepts[i]);
            percepts[i].self.hungerLeft = 20.0f - snakes[i].getTimeSinceEat();

            arena::Action act = brains[i]->think(percepts[i]);

            // ★ 动作量化 —— 必须和 Arena::step() 里那一步完全一致，
            //   否则这个工具测的就不是"游戏真正的行为"了。
            //   队号 >= 0 且不是玩家 → 量化到 Discrete5。
            if (snakes[i].getTeamId() >= 0 && !snakes[i].isPlayer()) {
                act.steer = arena::quantizeSteer(act.steer, arena::ActionSpace::Discrete5);
            }

            Snake& s = snakes[i];
            // ★ 必须用 getSteerLead，不能写 steer * turnRate * dt
            s.setTargetAngle(s.getAngle() + s.getSteerLead(act.steer, DT));
            s.setBoosting(act.boost);
        }

        // 6. 食物更新
        //
        // ⚠️ 当前 FoodManager 的 update() 会推进【圈计时器】和【食物寿命】。
        //    为了让 arena 对照模式逐位一致，两边的食物寿命都设成了 1e9（不过期），
        //    圈计时器则不参与任何决策 —— 所以 update() 在这里对世界状态
        //    是"无副作用"的。
        food.update(DT);
    }

    std::vector<char> deadHandled;
};

// ---------------------------------------------------------------------------
//  指纹采集
// ---------------------------------------------------------------------------
struct FramePrint {
    std::uint64_t hash;
    std::string   line;
};

FramePrint fingerprint(const World& w) {
    Hash h;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "t=" << std::setw(4) << w.tick << "  ";

    int aliveCount = 0;
    for (std::size_t i = 0; i < w.snakes.size(); ++i) {
        const Snake& s = w.snakes[i];
        const int segs = s.getSegmentCount();
        h.addFloat(s.getX());
        h.addFloat(s.getY());
        h.addFloat(s.getAngle());
        h.addFloat(s.getSpeed());
        h.addInt(static_cast<std::uint64_t>(segs));
        h.addInt(static_cast<std::uint64_t>(s.getScore()));
        h.addInt(s.isAlive() ? 1u : 0u);
        if (s.isAlive()) ++aliveCount;

        oss << (s.isAlive() ? 'A' : '.')
            << i << "(" << std::setw(9) << s.getX() << ","
            << std::setw(9) << s.getY() << ","
            << std::setw(8) << s.getAngle() << ",L" << std::setw(3) << segs
            << ",S" << std::setw(3) << s.getScore() << ") ";
        if ((i % 3) == 2) oss << "\n            ";
    }
    h.addInt(static_cast<std::uint64_t>(w.food.getFoods().size()));
    h.addInt(static_cast<std::uint64_t>(aliveCount));
    oss << "| food=" << w.food.getFoods().size() << " alive=" << aliveCount;

    FramePrint fp;
    fp.hash = h.h;
    fp.line = oss.str();
    return fp;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Arena 版指纹
//
//  与上面 World 版【逐字相同的算法】，只是数据来源换成 Arena 的
//  agents() / foods() / tick()。
//
//  ★ 刻意不复用同一个函数模板：两份写法保持独立，任何一边多算了一项，
//    比对就会失败 —— 这比"共用一个泛型函数、结果两边一起算错"安全。
// ---------------------------------------------------------------------------
namespace {

FramePrint fingerprintArena(const arena::Arena& a) {
    Hash h;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "t=" << std::setw(4) << a.tick() << "  ";

    const auto& agents = a.agents();
    const auto& foods  = a.foods();

    int aliveCount = 0;
    for (std::size_t i = 0; i < agents.size(); ++i) {
        const Snake& s = agents[i].body;
        const int segs = s.getSegmentCount();
        h.addFloat(s.getX());
        h.addFloat(s.getY());
        h.addFloat(s.getAngle());
        h.addFloat(s.getSpeed());
        h.addInt(static_cast<std::uint64_t>(segs));
        h.addInt(static_cast<std::uint64_t>(s.getScore()));
        h.addInt(s.isAlive() ? 1u : 0u);
        if (s.isAlive()) ++aliveCount;

        oss << (s.isAlive() ? 'A' : '.')
            << i << "(" << std::setw(9) << s.getX() << ","
            << std::setw(9) << s.getY() << ","
            << std::setw(8) << s.getAngle() << ",L" << std::setw(3) << segs
            << ",S" << std::setw(3) << s.getScore() << ") ";
        if ((i % 3) == 2) oss << "\n            ";
    }
    h.addInt(static_cast<std::uint64_t>(foods.size()));
    h.addInt(static_cast<std::uint64_t>(aliveCount));
    oss << "| food=" << foods.size() << " alive=" << aliveCount;

    FramePrint fp;
    fp.hash = h.h;
    fp.line = oss.str();
    return fp;
}

// ---------------------------------------------------------------------------
//  对照模式：用真正的 arena::Arena 跑同一局
//
//  ★ 这是"搬家有没有改变行为"的 A/B 对照。
//    上面的 World 是一份【手写的】世界推进，与 Arena::step() 应该逐位等价；
//    两者跑同一局、输出同一个格式的指纹，逐字节比对即可判定。
// ---------------------------------------------------------------------------
int runWithArena() {
    arena::ArenaConfig ac;
    ac.mapWidth  = MAP_W;
    ac.mapHeight = MAP_H;
    ac.teamCount     = TEAMS;
    ac.snakesPerTeam = PER_TEAM;
    ac.playerTeam    = -1;               // 没有玩家
    for (int i = 0; i < arena::ArenaConfig::MAX_TEAMS; ++i) ac.teamBrains[i] = "rule";

    ac.snakeSpeed = 250.0f;  ac.snakeBoostSpeed = 450.0f;
    ac.snakeTurnRate = 8.0f; ac.segmentSpacing = 20.0f;
    ac.snakeInitSegs = 12;   ac.corpseSegments = 2;
    // ★ foodSegments 必须和 manual 模式传给 applyConfig 的那个值【完全一致】。
    //   差一个值就会出现"吃同样多的食物、长度却不同"，
    //   指纹从第一行就开始不一样，而且很难一眼看出是配置问题。
    //   manual 那边传的是 1（见下面 food.applyConfig 的参数表）。
    ac.foodSegments = 1;
    ac.foodSpawnCount = 2; ac.foodSpawnInterval = 2.0f;
    ac.corpseScore = 3;
    ac.starvationTime = 20.0f; ac.collisionForgive = 5.0f; ac.friendlyFire = false;
    ac.visionCellSize = 32.0f; ac.allyVisibleCount = 2;

    // ★ 这个模式下【故意关掉食物圈】，保持和 manual 模式同一套老规则 ——
    //   它的用途是"证明 Arena 与手写循环等价"，不是测食物圈。
    //   食物圈由 zones 模式单独验证。
    ac.zoneLayout.count = 0;
    ac.foodLifetime   = 1.0e9f;   // 老行为：普通食物不过期
    ac.corpseLifetime = 1.0e9f;
    ac.foodPerSnakePerSecond = 0.0f;   // 不自动生成，只靠初始撒的一批
    ac.initialFoodCount      = 60;     // 与 manual 模式的 maxFood/2 对齐

    // ★ 种子必须在 Arena 构造【之前】设好 —— 蛇的构造函数里就会抽转向起伏相位。
    //   初始食物由 ArenaConfig::initialFoodCount 在 reset() 里撒，
    //   位置固定在"建蛇之前"，所以两种模式的随机序列能对上。
    //   （内置算法的注册在 main() 里统一做了，这里不用重复。）
    std::srand(SEED);
    arena::Arena arena(ac);

    Hash total;
    std::uint64_t lastHash = 0;
    std::printf(" 初始: food=%zu alive=%zu\n", arena.foods().size(),
                arena.agents().size());

    for (int f = 1; f <= FRAMES; ++f) {
        arena.step(DT);

        const FramePrint fp = fingerprintArena(arena);
        total.addBits(fp.hash);
        lastHash = fp.hash;

        if ((f % PRINT_EVERY) == 0 || f == FRAMES) {
            std::printf(" %s  h=%016llx\n", fp.line.c_str(),
                        static_cast<unsigned long long>(fp.hash));
        }
    }

    std::printf("----------------------------------------------------------------------\n");
    std::printf(" 最终世界哈希 : %016llx\n", static_cast<unsigned long long>(lastHash));
    std::printf(" 全程累积哈希 : %016llx\n", static_cast<unsigned long long>(total.h));
    std::printf("======================================================================\n");
    return 0;
}

// ---------------------------------------------------------------------------
//  食物圈模式：验证圈机制本身
//
//  ★ 这里不比对"和谁等价"，只回答三个问题：
//      1. 食物是不是只在【当前活跃圈】里生成？
//      2. 生成节奏对不对？（每秒 蛇数/9 个，持续 9 秒，然后 5 秒空窗）
//      3. 5 秒空窗之后是不是轮到【下一个圈】？
//
//    输出逐秒的"各圈食物数"，人一眼就能看出轮换是否正确。
// ---------------------------------------------------------------------------
int runWithZones() {
    arena::ArenaConfig ac;
    ac.mapWidth  = MAP_W;
    ac.mapHeight = MAP_H;
    ac.teamCount     = TEAMS;
    ac.snakesPerTeam = PER_TEAM;
    ac.playerTeam    = -1;
    for (int i = 0; i < arena::ArenaConfig::MAX_TEAMS; ++i) ac.teamBrains[i] = "rule";

    ac.snakeSpeed = 250.0f;  ac.snakeBoostSpeed = 450.0f;
    ac.snakeTurnRate = 8.0f; ac.segmentSpacing = 20.0f;
    ac.snakeInitSegs = 12;   ac.foodSegments = 1;  ac.corpseSegments = 2;
    ac.corpseScore = 3;
    ac.starvationTime = 20.0f; ac.collisionForgive = 5.0f;
    ac.visionCellSize = 32.0f; ac.allyVisibleCount = 2;

    // ---- 食物圈：与 docs/70 定稿一致 ----
    ac.zoneLayout.count           = 4;
    ac.zoneLayout.centerRatio     = 0.31f;
    ac.zoneLayout.radiusRatio     = 0.075f;
    ac.zoneSchedule.spawnDuration = 9.0f;
    ac.zoneSchedule.gapDuration   = 5.0f;
    ac.foodPerSnakePerSecond      = 0.111111f;   // 每秒 1 个（9 条蛇时）
    ac.foodLifetime               = 30.0f;
    ac.corpseLifetime             = 30.0f;

    std::srand(SEED);
    arena::Arena arena(ac);

    const std::vector<arena::FoodZone>& zones = arena.food().zones();
    std::printf(" 圈数 = %zu\n", zones.size());
    for (std::size_t i = 0; i < zones.size(); ++i) {
        std::printf("   圈%zu: 中心(%.0f, %.0f)  半径 %.1f\n",
                    i, zones[i].x, zones[i].y, zones[i].radius);
    }
    std::printf(" 时间表: 生成 %.0fs + 空窗 %.0fs = 每轮 %.0fs, 全部 %.1f 秒/圈\n\n",
                static_cast<double>(arena.zoneSchedule().spawnDuration),
                static_cast<double>(arena.zoneSchedule().gapDuration),
                static_cast<double>(arena.zoneSchedule().cycleDuration()),
                static_cast<double>(arena.zoneSchedule().cycleDuration()
                                    * static_cast<float>(zones.size())));

    std::printf("  秒   活跃  生成中   圈0  圈1  圈2  圈3   总数  存活\n");
    std::printf("  ---------------------------------------------------------\n");

    const int SECONDS = 70;   // 覆盖 5 个整轮（每轮 14 秒）
    for (int s = 1; s <= SECONDS; ++s) {
        for (int f = 0; f < 60; ++f) arena.step(DT);

        const float t = arena.food().zoneTimer();
        const int active = arena.zoneSchedule().activeZone(
            t, static_cast<int>(zones.size()));
        const bool sp = arena.zoneSchedule().spawning(t);

        int alive = 0;
        for (const auto& a : arena.agents()) if (a.isAlive()) ++alive;

        std::printf("  %3d    %2d     %s   ", s, active, sp ? "是    " : "否(空窗)");
        for (std::size_t i = 0; i < zones.size(); ++i) {
            std::printf("%5d", zones[i].foodCount);
        }
        std::printf("  %5d   %2d\n", static_cast<int>(arena.foods().size()), alive);
    }

    std::printf("======================================================================\n");
    return 0;
}

// ---------------------------------------------------------------------------
//  ★★ 位置随机模式的圈轮换诊断（模式名：randomzones）
//
//  ★ 为什么必须单独做这个模式（负责人报的缺陷）
//
//    负责人原话（大意）：
//      "正常应该是：出现一个活跃圈 → 产一段时间食物 → 下一个圈选好 →
//        一段空窗 → 活跃圈关闭 → 下一个圈变活跃并开始产食物。
//        然而现状是：下一个圈出现（假圈！！），活跃圈关闭，
//        又选了一个新的位置作为活跃圈。那个假圈从始至终未产生食物。"
//
//    `zones` 模式用的是【固定四方位】圈，测不到"位置随机"这条路径。
//    而"位置随机"恰恰是唯一会**搬动圈心**的模式 ——
//    搬动与调度一旦错位，画面上就会出现"预告的圈从不供食"。
//
//  ★ 打什么
//    逐秒打出：轮次、两个圈的**坐标**、调度认为的活跃下标、
//    以及每个圈【实际长出来的食物数】。
//
//  ★ 判据（怎么看出"假圈"）
//    盯住空窗期被预告的"下一个圈"坐标 P：
//      · 正确：下一轮开始后 **P 变成活跃，且 P 的食物数开始上涨**
//      · 缺陷：下一轮活跃的坐标【不是 P】，或者 P 的食物数恒为 0
// ---------------------------------------------------------------------------
int runWithRandomZones() {
    arena::ArenaConfig ac;
    ac.mapWidth  = MAP_W;
    ac.mapHeight = MAP_H;

    // 位置随机模式恒为 2 个圈（当前 + 下一个）
    ac.zoneLayout.count          = 2;
    ac.zoneLayout.randomPosition = true;
    ac.zoneLayout.centerRatio    = 0.31f;
    ac.zoneLayout.radiusRatio    = 0.075f;
    ac.zoneLayout.minDistRatio   = 0.40f;
    ac.zoneLayout.maxDistRatio   = 0.60f;
    ac.zoneLayout.posSeed        = 20260924u;
    ac.zoneLayout.posSeedOffset  = 0u;      // 0 = 完全可复现

    ac.zoneSchedule.spawnDuration = 9.0f;
    ac.zoneSchedule.gapDuration   = 5.0f;
    ac.zoneSchedule.randomOrder   = false;  // 顺序模式：活跃下标可预测，便于判读

    // 食物只由圈产生，去掉一切其它来源，读数才干净
    ac.foodPerSnakePerSecond = 1.0f;
    ac.initialFoodCount      = 0;
    ac.foodLifetime          = 1.0e9f;      // 不过期（否则食物数被超时干扰）
    ac.corpseLifetime        = 1.0e9f;

    arena::Arena arena(ac);

    const arena::ZoneSchedule& sch = arena.zoneSchedule();
    std::printf("----------------------------------------------------------------------\n");
    std::printf(" 位置随机模式 · 圈轮换诊断\n");
    std::printf(" 时间表: 生成 %.0fs + 空窗 %.0fs = 每轮 %.0fs\n",
                static_cast<double>(sch.spawnDuration),
                static_cast<double>(sch.gapDuration),
                static_cast<double>(sch.cycleDuration()));
    std::printf(" 判据: 空窗期被预告的【下一个圈】坐标，应当在下一轮变成活跃圈，\n");
    std::printf("       并且它的食物数开始上涨。若活跃坐标换成了别处 = 假圈缺陷。\n");
    std::printf("----------------------------------------------------------------------\n");

    const std::size_t nz = arena.food().zones().size();
    std::printf(" 初始圈数 = %zu\n", nz);
    for (std::size_t i = 0; i < nz; ++i) {
        const auto& z = arena.food().zones()[i];
        std::printf("   槽位 %zu: (%.0f, %.0f) 半径 %.1f\n", i, z.x, z.y, z.radius);
    }
    std::printf("----------------------------------------------------------------------\n");

    constexpr float DT = 1.0f / 60.0f;
    const int SECONDS = 70;                      // 5 轮够看出规律
    const int STEPS = static_cast<int>(SECONDS / DT);

    std::printf(" 秒  轮次 活跃 生成中     槽位0 (x,y)      食物    槽位1 (x,y)      食物\n");
    for (int f = 1; f <= STEPS; ++f) {
        arena.step(DT);

        if (f % 60 != 0) continue;               // 每秒一行

        const float t = arena.food().zoneTimer();
        const int   cyc = static_cast<int>(t / sch.cycleDuration());
        const int   act = arena.activeZoneIndex();
        const bool  sp  = sch.spawning(t);

        std::printf(" %3d  %3d   %2d  %-8s", f / 60, cyc, act,
                    sp ? "是" : "否(空窗)");
        for (std::size_t i = 0; i < nz; ++i) {
            const auto& z = arena.food().zones()[i];
            std::printf("  (%6.0f,%6.0f) %6d", z.x, z.y, z.foodCount);
        }
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    //  ★★ 换轮那一刻的【逐帧】数据 —— 只看一个 Arena 实例，不交错
    //
    //  ★ 为什么必须逐帧、而且必须只有一个实例
    //    负责人报："A 亮 → 出现 B（不亮）→ A 不亮，却冒出一个全新的 C 在亮，
    //              而 B 始终不亮"。
    //    逐【秒】输出会把换轮那一秒压成一行，看不到位置是在哪一帧变的。
    //    而且如果同时跑两个 Arena，两边的 [zone] 日志会交错，
    //    读起来像是"同一帧里发生了两件事"—— 那次就把我误导了一轮。
    //
    //  ★ 判据
    //    盯住 cycle 号从 0 变 1 的那一帧：
    //      · 正确：活跃圈坐标【不变】，就是上一段（空窗期）预告的那个位置
    //      · 缺陷：活跃圈坐标在这一帧【变了】⇒ 预告圈被丢掉 = "假圈 + 瞬移"
    // -----------------------------------------------------------------------
    {
        arena::Arena a2(ac);                     // 全新一局，从 t=0 干净地跑
        const float cyc = sch.cycleDuration();

        std::printf("----------------------------------------------------------------------\n");
        std::printf(" ★ 换轮逐帧（第 0 轮 → 第 1 轮，cycleDuration = %.1fs）\n", cyc);
        std::printf("   只看这一个实例；只在换轮前后 0.6 秒打印\n");
        std::printf(" 帧      t    轮 活跃 生成  槽0坐标          槽1坐标        活跃圈坐标\n");

        // 记录上一帧的活跃坐标，用来标出"变了吗"
        float lastActX = -1.0f, lastActY = -1.0f;
        int   lastCycle = -1;
        int   prevActiveIdx = -1;

        const int fEnd = static_cast<int>((cyc + 0.4f) / DT);
        for (int f = 1; f <= fEnd; ++f) {
            a2.step(DT);

            const float t  = a2.food().zoneTimer();
            if (t < cyc - 0.6f) {                       // 还没到观察窗口
                const int ai = a2.activeZoneIndex();
                if (ai >= 0) {
                    lastActX = a2.food().zones()[static_cast<std::size_t>(ai)].x;
                    lastActY = a2.food().zones()[static_cast<std::size_t>(ai)].y;
                }
                lastCycle = static_cast<int>(t / cyc);
                prevActiveIdx = ai;
                continue;
            }

            const int  c2 = static_cast<int>(t / cyc);
            const int  ai = a2.activeZoneIndex();
            const bool sp = sch.spawning(t);
            const auto& zs = a2.food().zones();
            const auto& za = zs[static_cast<std::size_t>(ai < 0 ? 0 : ai)];

            const bool cycleChg  = (c2 != lastCycle);
            const bool idxChg    = (ai != prevActiveIdx);

            // ★ 判据要排除"换轮那一帧"本身：
            //   换轮时活跃圈【本来就该】从旧圈切到预告的那个 —— 那是正常的。
            //   真正的缺陷是【换轮之后还在变】，那才是"瞬移"。
            const bool actMoved  = cycleChg ? false
                                            : ((za.x != lastActX || za.y != lastActY)
                                               && lastActX >= 0.0f);

            std::printf(" %4d %7.3f  %2d  %2d  %-4s (%6.0f,%6.0f)  (%6.0f,%6.0f)  (%6.0f,%6.0f)%s%s%s\n",
                        f, static_cast<double>(t), c2, ai,
                        sp ? "是" : "否",
                        zs[0].x, zs[0].y,
                        zs.size() > 1 ? zs[1].x : 0.0f,
                        zs.size() > 1 ? zs[1].y : 0.0f,
                        za.x, za.y,
                        cycleChg ? "  <=换轮" : "",
                        idxChg   ? "  <=活跃下标变" : "",
                        actMoved ? "  ★活跃圈坐标变了！" : "");

            lastActX = za.x; lastActY = za.y;
            lastCycle = c2;  prevActiveIdx = ai;
        }

        std::printf("----------------------------------------------------------------------\n");
        std::printf(" 读法：换轮那一行【不应该】出现「★活跃圈坐标变了」\n");
        std::printf("----------------------------------------------------------------------\n");
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // ★ 先把控制台输出代码页切成 UTF-8，否则下面的中文全是乱码。
    //   见 src/utils/ConsoleUtf8.h 的说明（默认代码页 936 会花屏）。
    textutil::enableUtf8Console();
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // ★ 内置算法必须注册，否则 BrainRegistry 是空的 →
    //   Arena::reset() 里 create() 返回 nullptr → 解引用崩溃。
    //   症状是 0xC0000005，一句提示都没有。
    //
    //   ⚠️ 这里【只在 main 里注册一次】。以前是每个模式各写一遍，
    //      结果新增 zones 模式时忘了加，就崩了一次（同一个坑踩两次）。
    //   游戏本体是在它自己的 main.cpp 里注册的，任何工具程序都得自己来。
    arena::registerBuiltinBrains();

    // 模式：默认 "manual"（手写循环）；arena = 真正的 arena::Arena；zones = 食物圈
    const std::string mode = (argc > 1) ? std::string(argv[1]) : std::string("manual");
    const bool useArena = (mode == "arena");
    const bool useZones = (mode == "zones");
    const bool useRandZones = (mode == "randomzones" || mode == "rz");

    std::printf("======================================================================\n");
    std::printf(" 世界指纹（replay_check）\n");
    std::printf(" 用途：证明重构【没有改变世界推进的行为】\n");
    std::printf(" 用法：replay_check.exe [manual|arena|zones|randomzones] > out.txt\n");
    std::printf("       manual      = 手写循环（重构前的参考实现）\n");
    std::printf("       arena       = 真正的 arena::Arena   ← 与 manual 必须逐字节相同\n");
    std::printf("       zones       = 食物圈机制（固定位置，逐秒观察食物分布与轮换）\n");
    std::printf("       randomzones = ★ 位置随机模式的圈轮换（诊断“假圈”缺陷）\n");
    std::printf("======================================================================\n");
    std::printf(" 模式: %s\n", mode.c_str());
    std::printf(" 配置: %dx%d, %d 队 x %d 蛇, %d 帧, 种子 %u\n",
                static_cast<int>(MAP_W), static_cast<int>(MAP_H),
                TEAMS, PER_TEAM, FRAMES, SEED);
    std::printf(" 覆盖: 移动 / 碰撞 / 饥饿 / 尸体转食物 / 感知 / 决策 / 食物更新\n");
    std::printf(" 不含: 限时 / 胜负 / 结算（那些还在 GameScene，见 docs/45）\n");
    std::printf("----------------------------------------------------------------------\n");

    if (useRandZones) return runWithRandomZones();
    if (useZones) return runWithZones();
    if (useArena) return runWithArena();
    World w;
    w.build();

    Hash  total;
    std::uint64_t lastHash = 0;

    std::printf(" 初始: food=%zu alive=%d\n", w.food.getFoods().size(),
                static_cast<int>(w.snakes.size()));

    for (int f = 1; f <= FRAMES; ++f) {
        w.step();

        const FramePrint fp = fingerprint(w);
        total.addBits(fp.hash);
        lastHash = fp.hash;

        if ((f % PRINT_EVERY) == 0 || f == FRAMES) {
            std::printf(" %s  h=%016llx\n", fp.line.c_str(),
                        static_cast<unsigned long long>(fp.hash));
        }
    }

    std::printf("----------------------------------------------------------------------\n");
    std::printf(" 最终世界哈希 : %016llx\n", static_cast<unsigned long long>(lastHash));
    std::printf(" 全程累积哈希 : %016llx\n", static_cast<unsigned long long>(total.h));
    std::printf("======================================================================\n");
    return 0;
}
