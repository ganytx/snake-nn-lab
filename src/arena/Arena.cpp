#include "Arena.h"

// HumanBrain 只在 reset() 里 new 一下，用来给玩家那条蛇挂大脑。
// 注意：HumanBrain::think() 会自己读窗口鼠标键盘（见 docs/45 第 4 节），
// Arena 不碰输入。
#include "../brain/BrainRegistry.h"
#include "../brain/HumanBrain.h"
// ★ 2026-09-27：造脑时按队设权重要用 NeuralBrain::setPendingWeights
//   ⚠️ 这个 include 是必需的 —— 漏了会报 "'NeuralBrain' has not been declared"，
//   而错误位置在 Arena.cpp 里，很容易误以为是命名空间写错。
#include "../brain/NeuralBrain.h"
#include "../utils/Constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

// ===========================================================================
//  ★ 圈诊断日志开关（2026-09-26 新增）
//
//  ★ 为什么要给"游戏本体"也加这个
//    负责人报"游戏里有一个从始至终不产食物的圈"，而这个现象在**命令行工具**里
//    复现不出来（同一套 Arena、同样的配置，活跃圈正常交替）。
//    所以必须让**游戏自己**把圈的状态打出来 —— 而不是靠读代码推测。
//
//  ★ 怎么开（`run.bat` 用 `start` 启动、没有控制台，所以要命令行启动）
//        cd build
//        set SNAKE_ZONE_DEBUG=1
//        snake_game.exe
//
//  ★ 开销
//    只在**换轮那一刻**打一行（每 14 秒一次），默认关闭 —— 不影响性能。
// ===========================================================================
bool zoneDebug() {
    static const bool on = (std::getenv("SNAKE_ZONE_DEBUG") != nullptr);
    return on;
}

}  // namespace

namespace arena {

namespace {

// 8 个队伍颜色。放在这里而不是 GameScene，是因为**颜色是 Arena 造出来的**——
// Agent 的构造函数要一个 sf::Color。GameScene 渲染时读的是蛇自己的颜色，
// 所以两边的取色天然一致，不会出现"小地图和实际蛇颜色不一样"这类问题。
//
// ⚠️ 这段数组必须与 GameScene 里原来的定义**逐值相同**，
//    否则重构会改变外观（虽然不影响仿真，但属于"行为变了"）。
const sf::Color TEAM_COLORS[ArenaConfig::MAX_TEAMS] = {
    sf::Color(78, 205, 196),   // 青
    sf::Color(255, 107, 107),  // 红
    sf::Color(255, 200, 60),   // 黄
    sf::Color(140, 110, 255),  // 紫
    sf::Color(90, 220, 120),   // 绿
    sf::Color(255, 140, 60),   // 橙
    sf::Color(255, 130, 200),  // 粉
    sf::Color(120, 200, 255),  // 天蓝
};

}  // namespace

// 队伍颜色查询（越界自动绕回，调用方不用自己取模）
sf::Color Arena::teamColor(int teamIndex) noexcept {
    if (teamIndex < 0) teamIndex = 0;
    return TEAM_COLORS[teamIndex % ArenaConfig::MAX_TEAMS];
}

// ===========================================================================
//  构造
//
//  ★ 构造顺序有讲究：成员按【声明顺序】初始化，而下面这些成员带构造参数，
//    所以必须写在初始化列表里，不能等构造函数体。
// ===========================================================================
Arena::Arena(const ArenaConfig& cfg)
    : m_cfg(cfg)
    , m_food(cfg.mapWidth, cfg.mapHeight)
    , m_collision(cfg.mapWidth, cfg.mapHeight)
{
    applyConfigToSystems();
    reset();
}

// ===========================================================================
//  把配置分发到各个子系统
//
//  ⚠️ 食物上限 / 碰撞参数 / 视距这些是**子系统构造期或 applyConfig 期**生效的，
//     只调 reset() 不会让它们更新。所以换配置必须走这里。
// ===========================================================================
void Arena::applyConfigToSystems() {
    // ---- 食物圈与供食时间表 ----
    m_sched = m_cfg.zoneSchedule;
    const std::vector<FoodZone> zones =
        (m_cfg.zoneLayout.count > 0)
            ? m_cfg.zoneLayout.build(m_cfg.mapWidth, m_cfg.mapHeight)
            : std::vector<FoodZone>{};
    m_food.setZones(zones, m_sched);

    // ⚠️ 这里**不要**调 `beginRoundSetup()`！
    //    "本局种子 + 圈位"是【局级】生命周期，挂在 `Arena::reset()` 上。
    //    挂在这里（配置级）会导致：训练/游戏走无参 `reset()` 时圈位永远不变 ——
    //    那正是负责人报的"每一局开始位置都一样"。
    //    构造顺序是 `applyConfigToSystems()` → `reset()`，所以 reset 里那一次就够了。
}

// ===========================================================================
//  换配置并重建一局
// ===========================================================================
void Arena::reset(const ArenaConfig& cfg) {
    const bool mapChanged = (cfg.mapWidth  != m_cfg.mapWidth ||
                             cfg.mapHeight != m_cfg.mapHeight);

    m_cfg = cfg;

    // ★ 这里【不能】只靠 reset() 里的 m_food.reset() 来清食物 ——
    //   因为下面 applyConfigToSystems() 会重新 applyConfig，
    //   而 reset() 是在它之后才调的。顺序上没冲突，但地图变了要重建
    //   FoodManager（它缓存了地图尺寸），重建本身就等于清空，所以这里
    //   只在"地图没变"时才需要显式清 —— 统一交给 reset() 处理，更不容易漏。
    if (mapChanged) {
        // 地图尺寸变了，FoodManager / CollisionSystem 内部缓存的尺寸也得重建
        m_food      = FoodManager(cfg.mapWidth, cfg.mapHeight);
        m_collision = CollisionSystem(cfg.mapWidth, cfg.mapHeight);
    }
    applyConfigToSystems();
    reset();
}

// ===========================================================================
//  ★★ 每局开始：播世界种子 + 把圈摆到"本轮该在的位置"
//
//  ★★ 为什么必须抽出来、而且必须在【每次 reset()】里调用
//
//    这段逻辑原来只在 `applyConfigToSystems()` 里，而那个函数**只在
//    `Arena` 构造 / `reset(cfg)` 时被调用**。训练和游戏走的却是
//    `Arena::reset()`（无参）—— 于是 **同一个 Arena 的所有局，圈位完全一样**。
//
//    ⚠️ 这个缺陷被"每局换世界种子"这件事暴露出来：种子明明每局都换
//       （日志里 `worldSeed` 一直在涨），可圈位**一帧都不动** ——
//       因为圈根本没被重新挑过。负责人看到的就是"每一局开始位置都一样"。
//
//    ★ 教训：`reset()` 的语义是"重建一局"，**凡是"一局一变"的东西
//      都必须挂在它身上**。只挂在 `applyConfigToSystems()` 上，
//      就等于"只有换配置时才变" —— 那是配置级的生命周期，不是局级的。
// ===========================================================================
void Arena::beginRoundSetup() {
    // ---- ① 播种：本局的世界种子 ----
    //
    //  ★ 负责人要求（2026-09-27）："训练模式，游戏模式请每局种子不同"
    //
    //  ★ 世界里的随机来自【四处】，它们必须来自同一个种子，否则
    //    "同一局可复现"这条性质在调试时就不成立：
    //        · 出生点 / 出生角度            -> 全局 rand()
    //        · 食物位置                     -> 全局 rand()
    //        · 圈的轮换顺序（randomOrder）  -> ZoneSchedule::orderSeed
    //        · 圈的坐标   （randomPosition）-> ZoneLayout::posSeed
    //
    //  ★ 两种模式
    //      advanceSeedOnReset = false（回归工具）：永远用 spawnSeed，
    //          "同一个种子跑出同一局"成立 -> replay_check 的世界指纹才有意义。
    //      advanceSeedOnReset = true （训练 / 游戏）：每 reset 一次换一个种子。
    //
    //  ⚠️ 调用它的位置必须在【撒初始食物】和【建蛇】之前 ——
    //     它内部的 std::srand 重置的是全局 rand() 序列，放在后面就等于没重置。
    if (m_cfg.spawnSeed != 0u) {
        m_worldSeed = m_cfg.spawnSeed;
        if (m_cfg.advanceSeedOnReset) m_worldSeed += m_resetCount;
        std::srand(m_worldSeed);
        // 圈的种子跟着走：顺序（轮换）与位置各派生一支，避免两支撞在一起
        m_sched.orderSeed        = m_worldSeed;
        m_cfg.zoneLayout.posSeed = m_worldSeed + 0x9E3779B9u;
        m_food.setSchedule(m_sched);   // ★ FoodManager 持的是【副本】，要同步
    } else {
        // 固定种子模式（回归工具）：什么都不动，完全保持旧行为
        m_worldSeed = 0u;
    }
    ++m_resetCount;

    // ---- ② 位置随机模式：把两个槽位摆到【各自轮次要用的位置】----
    //
    //  ★ 为什么必须每局都摆
    //    Arena::step 第 0 步的约定是"第 k 轮只挑第 k+1 轮的位置、活跃圈不动"。
    //    所以开局必须满足：
    //        槽位 0 = 第 0 轮的活跃圈
    //        槽位 1 = 第 1 轮的活跃圈 <- 它同时就是第 0 轮的"预告"
    //    否则第 1 轮会用一个和预告不一样的位置。
    //
    //  ★ "槽位 1 在哪"依赖"槽位 0 在哪"（距离约束），所以必须按顺序来。
    //  ★ m_posCycle 一起重置成 0 —— 这样第 0 轮本身不会再触发一次换轮。
    //  ★ 只在位置随机模式做；固定位置模式用的是 build() 算好的四方位。
    if (m_cfg.zoneLayout.randomPosition && m_food.zones().size() >= 2) {
        auto& zs = m_food.mutableZones();
        // ① 第 0 轮的活跃圈：现在挑（参照点用地图中心，此时还没有别的圈）
        m_cfg.zoneLayout.pickRandomCenter(zs[0], m_cfg.mapWidth, m_cfg.mapHeight,
                                          /*cycle=*/0,
                                          m_cfg.mapWidth * 0.5f,
                                          m_cfg.mapHeight * 0.5f);
        // ② 第 1 轮的活跃圈 —— 它同时就是第 0 轮的"预告"。
        //    参照点用①刚定下来的位置 => 两圈距离落在约束区间内。
        m_cfg.zoneLayout.relocateRandom(zs[1], m_cfg.mapWidth, m_cfg.mapHeight,
                                        /*cycle=*/1, zs[0].x, zs[0].y);
        m_posCycle = 0;
    }
}
void Arena::reset() {
    const int teamCount = m_cfg.effectiveTeamCount();
    const int perTeam   = std::max(1, m_cfg.snakesPerTeam);

    const float cx = m_cfg.mapWidth  * 0.5f;
    const float cy = m_cfg.mapHeight * 0.5f;

    // ★ 出生半径必须【留出到墙的安全距离】。
    //
    //   原来写死 mapMin * 0.32：2000 图得 640px（离墙 360px，没问题），
    //   但训练用的 400 图只得 128px —— 蛇头离墙只有 72px，
    //   刚出生几步就撞墙，"一出生就进下一局"。
    //
    //   这里按"至少留 wallClear 的余量"夹一下，并且半径本身给个下限
    //   （太小时所有队挤在一起，开局就互相撞）。
    const float mapMin    = std::min(m_cfg.mapWidth, m_cfg.mapHeight);
    const float wallClear = 100.0f;                       // 出生点到墙的最小距离
    const float maxRadius = std::max(0.0f, mapMin * 0.5f - wallClear);
    const float spawnRadius = std::max(40.0f, std::min(mapMin * 0.32f, maxRadius));

    // ★ 清空上一局留下的【世界状态】——食物、尸体、生成计时器。
    //
    //   这一步是必须的：下面只重建蛇，食物不归蛇管。
    //   少了它，"重开一局"时上一局的尸体和没吃完的食物会全部留在场上
    //   （这是抽 Arena 时引入的回归 —— 以前 GameScene 是整个重建 FoodManager）。
    //
    //   注意顺序：在 srand 之前无所谓，但必须在【撒初始食物】之前。
    m_food.reset();

    // -----------------------------------------------------------------------
    //  ★★ 本局的世界种子 + 圈位 —— 唯一入口是 beginRoundSetup()
    //
    //  ⚠️ 这段逻辑【曾经写在 applyConfigToSystems() 里】，那是个严重的错位：
    //     那个函数只在换配置时被调用，而训练 / 游戏走的是本函数（无参 reset）
    //     ⇒ **同一个 Arena 的所有局，圈位一帧都不换**。
    //     现象就是负责人报的：每一局开始位置都一样。
    //
    //  ★ 位置必须在【撒初始食物】和【建蛇】之前 ——
    //    它内部的 std::srand 重置的是全局 rand() 序列，放在后面就等于没重置。
    // -----------------------------------------------------------------------
    beginRoundSetup();

    // ★ 初始食物：必须在【构造蛇之前】撒。
    //   蛇的构造函数会消耗 rand()（抽转向起伏相位），所以"先建蛇再撒食物"
    //   和"先撒食物再建蛇"会产生完全不同的随机序列。这里固定成前者。
    //
    //   食物圈玩法下 initialFoodCount = 0，这段不执行（圈会自己逐秒生成）；
    //   只有"没有圈的世界"（对照 / 回归测试）才会用到它。
    if (m_cfg.initialFoodCount > 0) m_food.seedUniform(m_cfg.initialFoodCount);

    m_agents.clear();
    m_ptrs.clear();
    m_tick = 0;
    m_humanIndex = -1;
    m_deathCauses.clear();
    // ★ 新的一局要从"第 0 轮还没处理"开始 —— 否则重开一局时
    //   位置随机模式的圈会停在上一局的坐标上
    m_posCycle = -1;

    m_agents.reserve(32);

    for (int t = 0; t < teamCount; ++t) {
        // 各队从【均分圆周的方位】出生，而不是全挤在地图中心 ——
        // 否则开局就互相撞死，学不到"合作"，只学到"乱斗"。
        const float baseAngle = arena::limits::TWO_PI * static_cast<float>(t)
                              / static_cast<float>(teamCount);

        for (int k = 0; k < perTeam; ++k) {
            float x = 0.0f, y = 0.0f, heading = 0.0f;

            if (m_cfg.spawnMode == SpawnMode::Random) {
                // 随机撒点，朝向也随机。
                // 训练用：避免网络过拟合到固定的几个出生方位。
                //
                // ★ spawnMargin 必须【随地图缩放】。
                //   实测踩过的坑：训练用小地图 400x400，而 spawnMargin 写死 200，
                //   于是 lo = 200、hi = 200 —— 出生范围退化成一个点（正中心），
                //   蛇头离墙只有 72px，几步就撞墙死，"一出生就进下一局"。
                //   这里取"地图短边的 15%"，并保证至少留 60px 余量。
                const float minSide = std::min(m_cfg.mapWidth, m_cfg.mapHeight);
                float m = std::min(m_cfg.spawnMargin,
                                   std::max(60.0f, minSide * 0.15f));

                // ★ 定点食物任务有额外的硬约束：出生边距必须 >= 食物距离。
                //
                //   否则食物会被 FoodManager::spawnAt 夹进地图内 —— 实际距离
                //   和方向都变了，而蛇仍然按"朝前冲"去走，于是直接撞墙。
                //   实测现象："会朝食物前进，但基本每次都撞墙"。
                if (m_cfg.spawnFixedFoodAhead) {
                    m = std::max(m, m_cfg.fixedFoodDistance + 20.0f);
                }

                const float loX = m, hiX = m_cfg.mapWidth  - m;
                const float loY = m, hiY = m_cfg.mapHeight - m;
                auto u = [](float lo, float hi) {
                    const float r = static_cast<float>(std::rand())
                                  / static_cast<float>(RAND_MAX);
                    return lo + r * (hi - lo);
                };
                x = u(loX, hiX);
                y = u(loY, hiY);
                heading = u(0.0f, arena::limits::TWO_PI);
            } else {
                // 同队成员在同一方位附近散开，整体朝向地图中心
                const float a = baseAngle +
                                (static_cast<float>(k) - (perTeam - 1) * 0.5f) * 0.22f;
                const float r = spawnRadius * (0.82f + 0.18f * static_cast<float>(k % 2));
                x = cx + std::cos(a) * r;
                y = cy + std::sin(a) * r;
                heading = std::atan2(cy - y, cx - x);
            }

            const bool isHuman = (m_cfg.playerTeam >= 0 && t == m_cfg.playerTeam && k == 0);

            // -------------------------------------------------------------------
            //  ★★ 队 0 的【非学习者】用 "nn-mate" 而不是 "nn"（2026-09-26 深夜）
            //
            //  ★ 为什么需要这段
            //    `teamAlgorithms` / `teamBrains` 是【按队】配的，而队 0 有 2 条蛇。
            //    所以"把队 0 设成 nn"实际上是**两条都挂 nn**：
            //        k == 0 → 学习中的那条（EpisodeRunner 只学 index 0）
            //        k == 1 → 队友
            //    而两者该读的文件完全不同：
            //        "nn"      读 weights/policy.txt —— 用户选的/自动找到的成品权重
            //        "nn-mate" 读 saves/enemy_snapshot.txt —— 每局同步的训练快照
            //    队友挂 "nn" 的话，训练时那个文件不存在 → 它要么是随机蛇、
            //    要么是一份与训练无关的旧权重，而且**整场都不变**。
            //
            //  ★ 为什么判断写在这里而不是 TrainTask
            //    TrainTask 只能表达"第 t 队用什么"，表达不了"同队内第 k 条不同"。
            //    而"哪条是学习者"这件事 Arena 本来就不知道（是 EpisodeRunner 的
            //    约定：index 0 = 队 0 第 0 条）。所以这里用最小、最局部的规则：
            //    **队 0 挂了 nn 时，它的非首条改读快照。**
            //    游戏模式下队 0 通常挂的是 "rule"/"nn"，这个分支不会误伤 ——
            //    它只在"某队确实是神经网络"时才生效，且只改非首条。
            //
            //  ★ 为什么这个约定必须只有一份
            //    "index 0 = 学习者" 是 EpisodeRunner / TrainTask / 这里三处共同
            //    依赖的约定。三处都写了字面量 0，所以这里也注明来源，
            //    避免以后有人改了 learnerIndex 却漏掉这里。
            // -------------------------------------------------------------------
            std::string algo = m_cfg.teamBrains[t];
            if (algo == "nn" && t == 0 && k > 0) {
                algo = "nn-mate";
            }

            std::unique_ptr<Brain> brain;
            if (isHuman) {
                brain = std::unique_ptr<Brain>(new HumanBrain());
            } else {
                // -----------------------------------------------------------
                //  ★★ 每队加载哪份权重（2026-09-27 新增，负责人要求）
                //
                //  ★ 为什么在这里设、而不是在 BrainRegistry 里
                //    因为"加载哪一份"是【这一队】的属性，只有 Arena 知道现在
                //    在给第几队造脑。BrainRegistry 只认 id（那是它的全部意义：
                //    UI 只认 id，新增算法不改 UI）。
                //
                //  ★ 机制：`setPendingWeights` 是全局的"下一份要加载的文件"，
                //    NeuralBrain 构造函数会读它。所以必须【造脑之前设、造完就清】——
                //    忘了清就会让后面所有神经网络都去读这一份
                //    （这正是缺陷 #43 的形态：全局待定值没清）。
                //
                //  ⚠️ 顺带说一个边界：玩家那条蛇挂的是 HumanBrain，
                //     所以第 t 队的权重对它【没有意义】。这是对的 ——
                //     玩家操控的蛇不该有 AI 权重。
                // -----------------------------------------------------------
                const bool setW = !m_cfg.teamWeights[t].empty();
                if (setW) NeuralBrain::setPendingWeights(m_cfg.teamWeights[t]);
                brain = BrainRegistry::instance().create(algo);
                if (setW) NeuralBrain::setPendingWeights(std::string());   // ★ 造完必清

                // 配置里的算法 id 不存在 → 退回规则 AI，别让整局崩掉
                if (!brain) brain = BrainRegistry::instance().create("rule");
            }

            const sf::Color color = TEAM_COLORS[t % ArenaConfig::MAX_TEAMS];

            m_agents.emplace_back(x, y, heading, color, isHuman, std::move(brain), t);
            agent::Agent& agent = m_agents.back();

            // 死因表要和 agents 同步增长，下标才对得上
            m_deathCauses.push_back(DeathCause::None);

            agent.body.applyConfig(m_cfg.snakeSpeed, m_cfg.snakeBoostSpeed,
                                   m_cfg.snakeTurnRate, m_cfg.segmentSpacing,
                                   m_cfg.foodSegments, m_cfg.corpseSegments);
            // ★ 出生时身体堆成一个点还是笔直铺开（见 ArenaConfig::compactSpawn）
            agent.body.setCompactSpawn(m_cfg.compactSpawn);
            // ★ 顺序讲究：setSegmentCount 会按 m_segmentSpacing 重建出生轨迹，
            //   所以必须在 applyConfig 之后调用。
            agent.body.setSegmentCount(m_cfg.snakeInitSegs);
            agent.body.resetStarvation();

            agent.brain->onEpisodeStart(static_cast<std::uint32_t>(1000 + t * 97 + k));

            if (isHuman) m_humanIndex = static_cast<int>(m_agents.size()) - 1;
        }
    }

    // ★ 指针表只在【全部 agent 建完之后】填一次。
    //   之后 m_agents 不再增删 → 指针永不失效，彻底不需要 rebuildSnakeList()。
    for (auto& a : m_agents) m_ptrs.push_back(&a.body);

    // ★ 最小学习验证：在第一条蛇【正前方】放一个食物。
    //
    //   必须在这里做（建蛇之后）—— 因为要读蛇的出生朝向 getAngle()。
    //   撒食物本身不影响 rand() 序列（位置是我们算好的，不抽随机数），
    //   所以不会破坏 replay_check 的确定性。
    if (m_cfg.spawnFixedFoodAhead && !m_agents.empty()) {
        const Snake& s0 = m_agents[0].body;
        const float a  = s0.getAngle();
        m_food.spawnAt(s0.getX() + std::cos(a) * m_cfg.fixedFoodDistance,
                       s0.getY() + std::sin(a) * m_cfg.fixedFoodDistance);
    }

    // 感知层需要知道"世界长什么样"
    m_view.snakes    = &m_ptrs;
    m_view.foods     = &m_food.getFoods();
    m_view.mapWidth  = m_cfg.mapWidth;
    m_view.mapHeight = m_cfg.mapHeight;
    m_view.tick      = 0;
    m_view.dt        = 0.0f;
}

// ===========================================================================
//  把已经死掉的蛇从场上移除
//
//  ★ 三个必须一起修的容器（漏一个就是静默的错位，不报错）
//      ① m_agents       本体
//      ② m_ptrs         &agent.body，给碰撞 / 感知用
//      ③ m_deathCauses  与 agents 下标一一对应（诊断用）
//    外加 m_humanIndex —— 玩家那条的下标。
//
//  ★ 顺序上的一个坑：`m_agents` 和 `m_deathCauses` 必须【同步压缩】。
//    只缩前者会让死因表整体错位 —— 而那个错误不会报错，
//    只会让 train_probe 的死因统计悄悄变成假的。
// ===========================================================================
int Arena::removeDeadAgents() {
    const std::size_t before = m_agents.size();

    // ---- 1) 一次性压缩（erase-remove 惯用法，O(n)）----
    //
    //  ★ 为什么不用循环 erase：那是 O(n²)，而且每次 erase 都会让
    //    后面所有下标平移，写起来极易错。
    //
    //  ★★ 为什么结尾必须用 `erase()` 而**不能**用 `resize(k)`
    //    `agent::Agent` 的默认构造函数是 **= delete** 的
    //    （Snake 没有默认构造，必须给出生参数）。而 `vector::resize(k)`
    //    在 k < size() 时要求元素**可默认构造**（标准如此，即使只是缩容）——
    //    于是编译期直接报：
    //        error: use of deleted function 'agent::Agent::Agent()'
    //    这个错误信息指向 `stl_construct.h`，离"我该用 erase"很远，会白花时间。
    //    `erase()` 只做移动赋值 + 析构，不要求默认构造。
    //
    //  ⚠️ 顺带一条：`m_deathCauses` 的元素是枚举、**可以**默认构造，
    //     所以那边 resize 是合法的 —— 但两个容器必须**一起**缩到 k，
    //     所以这里统一都用 erase，语义上一目了然。
    std::size_t k = 0;
    for (std::size_t i = 0; i < m_agents.size(); ++i) {
        if (!m_agents[i].isAlive()) continue;      // 丢掉这一条
        if (k != i) {
            m_agents[k]      = std::move(m_agents[i]);
            m_deathCauses[k] = m_deathCauses[i];   // ★ 死因表同步搬
        }
        ++k;
    }
    m_agents.erase(m_agents.begin() + static_cast<std::ptrdiff_t>(k),
                   m_agents.end());
    m_deathCauses.erase(m_deathCauses.begin() + static_cast<std::ptrdiff_t>(k),
                        m_deathCauses.end());

    const int removed = static_cast<int>(before - k);
    if (removed == 0) return 0;

    // ---- 2) 修正 humanIndex ----
    //
    //  ★ 玩家那条被移除了 → 置 -1
    //    玩家死了就是死了，没有"新下标"可言。置 -1 之后所有调用方
    //    自动走"观战 / 没有玩家"的分支（它们本来就有 hi < 0 的判断）。
    //    ⚠️ 调用方要在**移除之前**把观战状态切好 —— 见 GameScene。
    //
    //  ★ 玩家还活着 → 重新数一遍，而不是"按下标推算"
    //    推算要维护"它前面死了几条"这种平移关系，容易写错；
    //    而 `Snake::isPlayer()` 是蛇自己的属性，与下标无关 ——
    //    在新表里找一遍就得到正确答案，代价 O(n)。
    //    **用不可能错的做法**（本项目的一贯原则）。
    m_humanIndex = -1;
    for (std::size_t i = 0; i < m_agents.size(); ++i) {
        if (m_agents[i].isHuman()) { m_humanIndex = static_cast<int>(i); break; }
    }

    // ---- 3) 重建指针表（下标全变了，必须重填）----
    //
    //  ★ 为什么每次都重建而不是"删掉对应的那个"
    //    重建是 O(n)、n ≤ 几十，代价可忽略；而"按下标删一个"要自己维护
    //    下标的平移关系，正是最容易错的地方。**用不可能错的做法。**
    m_ptrs.clear();
    m_ptrs.reserve(m_agents.size());
    for (auto& a : m_agents) m_ptrs.push_back(&a.body);

    // ---- 4) 感知层持有的指针表要跟着更新 ----
    //
    //  ⚠️ 这一条最容易漏：`m_view.snakes` 指向的是 `&m_ptrs` 这个 vector
    //     本身（不是它的内容），所以 clear/reserve 不会让它失效；
    //     但仍然显式写一遍，避免以后有人把 m_view 改成存值。
    m_view.snakes = &m_ptrs;

    return removed;
}

// ===========================================================================
//  推进一个固定步长
//
//  逐行对照 GameScene::fixedUpdate() 的 1~6 步，**顺序和数值都不改**。
//
//  第 7~9 步（相机 / 小地图 / 队伍统计与胜负）留在 GameScene —— 见 docs/45。
//  对局结束后冻结世界也留在 GameScene：不调 step() 就是冻结，
//  所以这里不需要 matchOver 状态。
// ===========================================================================
//  ★★ 中途加入一条蛇（无限模式用，2026-09-27）
//
//  ★ 与 reset() 的关系
//    这里**重复**了 reset() 里"建一条蛇"的那几步（建脑 / applyConfig /
//    setSegmentCount / resetStarvation / 死因表 / 指针表）。
//
//    ⚠️ 为什么不抽成一个共用的私有函数
//      因为 reset() 里那几步是夹在"出生几何"和"撒食物"之间的，抽取要动
//      reset 的结构；而 reset 一改就要重跑 replay_check（世界指纹）。
//      权衡之后：把"两条路径必须一致"这件事用注释锁住（下面每一步都标了
//      它对应 reset() 的哪一行），等以后真要做"更多加蛇方式"时再抽。
//      **判据：如果哪天这里和 reset() 的行为不一致了，现象是"开局出生的蛇
//      和后来加入的蛇参数不同"——那会很难查，所以两边字段列表要一起改。**
// ===========================================================================
int Arena::addAgent(const std::string& brainId, int teamId) {
    // ---- 出生位置：随机撒点（和 reset 的 Random 分支同一套约束）----
    const float minSide = std::min(m_cfg.mapWidth, m_cfg.mapHeight);
    float m = std::min(m_cfg.spawnMargin, std::max(60.0f, minSide * 0.15f));
    const float loX = m, hiX = m_cfg.mapWidth  - m;
    const float loY = m, hiY = m_cfg.mapHeight - m;
    if (hiX <= loX || hiY <= loY) return -1;      // 地图太小，加不进去

    // -----------------------------------------------------------------------
    //  ★★ 挑一个【离所有现存蛇都够远】的出生点（2026-09-27 新增）
    //
    //  ★ 为什么必须有这一步（负责人报的"不会生成新的蛇"）
    //     原来就是一行随机撒点。问题是蛇出生时身体会沿朝向的**反方向
    //     铺 12 × 20 × 1.2 = 288px**（见 Snake::prefillTrail）。
    //     如果落点离某条蛇太近，就相当于**把一条 288px 长的身体横穿到
    //     别人身上** —— 要么它自己的头撞上别人，要么别人一头撞在它身上，
    //     两种都是 1 秒内就死。
    //     然后 `removeDeadAgents()` 会把尸体立刻删掉 → **画面上什么都没发生**。
    //     现象正是"倒计时在走，但永远不出现新蛇"。
    //
    //  ★ 判据为什么是"头到对方任意体节的距离"
    //     碰撞检测用的就是这一条（见 CollisionSystem）。出生点只要满足
    //     它、并留出余量，新蛇就不可能一出生就撞死。
    //
    //  ★ 为什么用"退而求其次"的循环而不是"找不到就不生成"
    //     蛇多的时候可能真的挑不出完美位置（20 条蛇 × 288px 的身体
    //     铺在 3000² 上，空间是够的，但随机试探会失败）。
    //     如果找不到就放弃，倒计时会一直空转 —— 那又是"看起来卡住"。
    //     所以保留**最好的那个候选**：宁可挤一点，也不要"永远不生成"。
    // -----------------------------------------------------------------------
    auto u = [](float lo, float hi) {
        const float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        return lo + r * (hi - lo);
    };

    // 需要的出生余量：蛇身全长（≈288px）+ 一点缓冲。
    //  ⚠️ 这两个数必须和 `snakeInitSegs` × `segmentSpacing` 挂钩 ——
    //     写死的话，以后把蛇调长就会重新出现"出生即撞死"。
    const float bodyLen = static_cast<float>(m_cfg.snakeInitSegs)
                        * m_cfg.segmentSpacing * 1.2f;
    const float safeDist = bodyLen + 60.0f;
    // 退让下限：不能低于"两条蛇的半径之和"，否则等于允许重叠出生
    const float minDist  = std::max(80.0f, safeDist * 0.5f);

    float x = 0.0f, y = 0.0f, heading = 0.0f;
    {
        const int attempts = 64;
        float bestClear = -1.0f;
        float bx = 0.0f, by = 0.0f;

        for (int i = 0; i < attempts; ++i) {
            const float cx = u(loX, hiX);
            const float cy = u(loY, hiY);

            // 到最近一条蛇的任意【身体】的距离（逐体节采样，不是只看头）
            float nearest = 1.0e9f;
            for (const auto& other : m_agents) {
                if (!other.isAlive()) continue;
                const auto& segs = other.body.getSegments();
                // 每隔 1 个取一个点就够了：体节间距 20px，采样步长 40px
                for (std::size_t s = 0; s < segs.size(); s += 2) {
                    const float dx = cx - segs[s].x;
                    const float dy = cy - segs[s].y;
                    const float d2 = dx * dx + dy * dy;
                    if (d2 < nearest) nearest = d2;
                }
                // 头单独也算一个点（它是碰撞的主体）
                const sf::Vector2f hp = other.body.getHeadPosition();
                const float hdx = cx - hp.x, hdy = cy - hp.y;
                const float hd2 = hdx * hdx + hdy * hdy;
                if (hd2 < nearest) nearest = hd2;
            }
            const float clear = (nearest >= 1.0e8f) ? 1.0e9f : std::sqrt(nearest);

            // 已经够安全 → 直接用，不再试探（省掉多余的 rand() 消耗）
            if (clear >= safeDist) { bx = cx; by = cy; bestClear = clear; break; }

            if (clear > bestClear) {
                bestClear = clear;
                bx = cx; by = cy;
            }
        }

        x = bx;
        y = by;
        heading = u(0.0f, arena::limits::TWO_PI);

        // ★ 实在挤不下：把朝向也调一下，让身体铺向最空的方向，
        //   而不是随机朝向（随机朝向有一半概率正好铺向最近的那条蛇）。
        if (bestClear < minDist && !m_agents.empty()) {
            float bestAng = heading, bestAngClear = -1.0f;
            for (int k = 0; k < 12; ++k) {
                const float ang = heading + static_cast<float>(k) * 0.5235988f;  // +30°
                // 身体铺在朝向的反方向 —— 采样身体中段和尾端
                const float dirX = -std::cos(ang), dirY = -std::sin(ang);
                float localMin = 1.0e9f;
                for (float d = 0.0f; d <= bodyLen; d += bodyLen * 0.25f) {
                    const float px = x + dirX * d, py = y + dirY * d;
                    for (const auto& other : m_agents) {
                        if (!other.isAlive()) continue;
                        const auto& segs = other.body.getSegments();
                        for (std::size_t s = 0; s < segs.size(); s += 2) {
                            const float dx = px - segs[s].x, dy = py - segs[s].y;
                            localMin = std::min(localMin, dx * dx + dy * dy);
                        }
                    }
                }
                if (localMin > bestAngClear) { bestAngClear = localMin; bestAng = ang; }
            }
            heading = bestAng;
        }
    }

    // ---- 建脑（和 reset 里那段一致：按队设权重、造完必清）----
    std::unique_ptr<Brain> brain;
    if (teamId >= 0 && teamId < ArenaConfig::MAX_TEAMS
        && !m_cfg.teamWeights[teamId].empty()) {
        NeuralBrain::setPendingWeights(m_cfg.teamWeights[teamId]);
        brain = BrainRegistry::instance().create(brainId);
        NeuralBrain::setPendingWeights(std::string());   // ★ 造完必清（缺陷 #43 的形态）
    } else {
        brain = BrainRegistry::instance().create(brainId);
    }
    if (!brain) brain = BrainRegistry::instance().create("rule");

    const int t = (teamId >= 0) ? (teamId % ArenaConfig::MAX_TEAMS) : 0;
    const sf::Color color = TEAM_COLORS[t % ArenaConfig::MAX_TEAMS];

    // ★ 不能是玩家（玩家只有一条，在 reset 时就定好了）
    m_agents.emplace_back(x, y, heading, color, /*isHuman=*/false,
                          std::move(brain), t);
    agent::Agent& agent = m_agents.back();
    const int index = static_cast<int>(m_agents.size()) - 1;

    // 死因表和 agents 同步增长（下标才对得上）
    m_deathCauses.push_back(DeathCause::None);

    agent.body.applyConfig(m_cfg.snakeSpeed, m_cfg.snakeBoostSpeed,
                           m_cfg.snakeTurnRate, m_cfg.segmentSpacing,
                           m_cfg.foodSegments, m_cfg.corpseSegments);
    agent.body.setCompactSpawn(m_cfg.compactSpawn);
    agent.body.setSegmentCount(m_cfg.snakeInitSegs);
    agent.body.resetStarvation();

    // onEpisodeStart 的种子：用下标派生，保证每条蛇的"转向起伏相位"不同
    // （和 reset 里 `1000 + t*97 + k` 同一个意图；这里 k 用蛇序号代替）
    agent.brain->onEpisodeStart(static_cast<std::uint32_t>(1000 + index * 97));

    // ★ 指针表：新蛇要补一条，否则碰撞系统与感知看不到它
    m_ptrs.push_back(&agent.body);

    return index;
}

// ===========================================================================
void Arena::step(float dt) {
    if (m_agents.empty()) return;

    ++m_tick;

    // -----------------------------------------------------------------------
    //  ---- 0. 位置随机模式：每轮产生一个新的圈 ----
    //
    //  ★ 负责人要的规则（原话）
    //    "不存在固定的四个圈位置，而是随机选择一个合法坐标，作为下一个圈的
    //      位置。然后上个食物圈还有一段时间失效，下一个食物位置已经选好并出现，
    //      然后上一个食物圈失效一段时间后这个食物圈激活。"
    //
    //  ★★ 2026-09-27 第二次修"假圈"（负责人第三次报同一个现象）
    //
    //    现象（负责人原话）：
    //      "A 亮 → 出现一个 B（不亮）→ A 不亮，却突然出现一个全新的 C 在亮，
    //        而 B 始终不亮。就好像是这个圈会瞬移。"
    //
    //    ★ 根因：这一段的 `std::swap` 与【活跃下标算法】互相矛盾
    //
    //      活跃下标由 `ZoneSchedule::activeZone(t,n)` 算：`cycle % n`，
    //      也就是【槽位固定轮转】：轮 0 用槽 0、轮 1 用槽 1、轮 2 用槽 0 …
    //      而原来的写法每轮都 `swap` 一次，把"预告的那个"换到槽 0
    //      —— 语义是【活跃圈永远在槽 0】。
    //
    //      这两个约定不可能同时成立：到奇数轮时 `cycle % 2 = 1` 指向槽 1，
    //      而槽 1 正好是刚 relocate 出来的"下一个圈"。于是：
    //        · 槽 0 里那个预告的圈【永远不会被点亮、也永远不产食物】
    //        · 被点亮、并且真的产食物的是槽 1 里那个【全新的圈】
    //      逐帧日志（`SNAKE_ZONE_DEBUG=1 replay_check randomzones`）抓到的正是：
    //          [BEFORE]  cycle=1 槽0(659,757) 槽1(571,1367)
    //          [SWAPPED]        槽0(571,1367) 槽1(659,757)   ← swap 本身没错
    //          [AFTER]          槽0(571,1367) 槽1(530,915)   ← 只改了槽1
    //                   活跃=1  ← 但活跃下标指向【槽1】，即那个全新的圈
    //
    //    ★ 修法（负责人选的"甲"）：**去掉 swap**，改成
    //      "每轮把【下一轮要用的那个槽位】重挑一次坐标"。
    //      于是 `cycle % n` 这个下标算法【一行都不用改】，
    //      而"预告的圈"与"下一个活跃圈"必然是同一个槽位 —— 两者统一了。
    //
    //    ★ 位置必须【提前一轮】选好，这一条靠随机数的 cycle 号保证
    //      第 k 轮的圈位置用 `pickRandomCenter(cycle = k)`。
    //      所以：
    //        · `ZoneLayout::build()` 给槽位 i 传的是 `cycle = i + 1`（见那边的说明）
    //        · 这里在第 k 轮时给"第 k+1 轮的槽位"重挑，传 `cycle = k + 1`
    //      两处用的是【同一个 cycle 号】，所以得到的是同一个坐标 ——
    //      "第 k 轮预告的位置"就是"第 k+1 轮真正用的位置"。
    //
    //  ★ 为什么恒定 2 个槽位而不是保留历史圈
    //    保留 2 个正好表达"当前 + 下一个"，再多就是噪音了。
    //
    //  ★ 为什么放在 step 的【最开头】
    //    必须在感知（第 5 步）和食物生成（第 6 步）之前 ——
    //    这样新位置从这一帧起就同时对"蛇看到的"和"食物长在哪"生效。
    //
    //  ★ 只在随机位置模式生效，顺序模式【一行都不走】——
    //    固定位置玩法的行为完全不变（replay_check 可证）。
    // -----------------------------------------------------------------------
    if (m_cfg.zoneLayout.randomPosition) {
        auto& zones = m_food.mutableZones();
        const int n = static_cast<int>(zones.size());
        if (n >= 2) {
            const float cycleF = m_food.zoneTimer() / m_sched.cycleDuration();
            const int cycle = static_cast<int>(cycleF);
            if (cycle != m_posCycle) {
                const float W = m_cfg.mapWidth, H = m_cfg.mapHeight;

                // 本轮的活跃槽位（= 上一轮预告的那个）
                const int curSlot  = ((cycle % n) + n) % n;
                // 下一轮要用的槽位 —— 本轮把它选好并显示出来，这就是"预告"
                const int nextSlot = ((cycle + 1) % n + n) % n;

                // -------------------------------------------------------------------
                //  ★★ 这一轮【只做一件事】：把【下一轮】的圈选好并显示出来
                //
                //  ★ 为什么【不能】在这里重挑"本轮活跃圈"的位置（踩过的坑）
                //
                //    我一度在这里加了一句"给当前活跃槽位挑新位置"，
                //    想表达"每轮随机换位置"。结果是：
                //        · 它把【上一轮预告的位置】当场覆盖掉了
                //        · 于是预告的那个圈**永远没机会变成活跃圈**
                //        · 逐帧看就是"活跃圈在换轮后一帧又跳走了"
                //    负责人一眼看出来："食物圈只有两个，也不轮换" /
                //    "预告圈和活跃圈必须是相同的圈"。
                //
                //  ★ 正确的结构：活跃圈的位置【提前一轮】就定好了
                //
                //      第 k 轮开始时：
                //        ① 活跃圈【不动】—— 它的位置是第 k−1 轮就预告好的
                //        ② 重挑【下一轮】的槽位 → 新的"预告"，
                //           而它就是第 k+1 轮的活跃圈
                //
                //    于是两条要求同时成立：
                //      · 预告圈 = 下一轮的活跃圈（同一个槽位、同一个坐标，见 ②）
                //      · 位置每轮都在换（新位置在第 k+1 轮生效）
                //    而且**没有瞬移**：活跃圈在整个轮次里一动不动，
                //    换轮只是"原来那个淡色的圈变亮了"。
                //
                //  ★ 随机数用 `cycle + 1`：同一个轮次永远得到同一个坐标。
                //    这个数【就是它成为活跃圈那一轮的号】，所以
                //    "预告的位置"必然等于"它真正被使用的位置"，不会被重挑掉。
                //
                //  ★ 参照点 = 本轮活跃圈的位置 ⇒ 两圈距离落在
                //    minDistRatio ~ maxDistRatio 之间（"跑过去"才有意义）。
                // -------------------------------------------------------------------
                m_cfg.zoneLayout.relocateRandom(
                    zones[static_cast<std::size_t>(nextSlot)], W, H, cycle + 1,
                    zones[static_cast<std::size_t>(curSlot)].x,
                    zones[static_cast<std::size_t>(curSlot)].y);

                m_posCycle = cycle;

                if (zoneDebug()) {
                    std::printf("[zone] cycle=%d t=%.2f  活跃槽=%d(%.0f,%.0f)  "
                                "预告槽=%d(%.0f,%.0f) | 活跃=%d 下一个=%d\n",
                                cycle, static_cast<double>(m_food.zoneTimer()),
                                curSlot,
                                zones[static_cast<std::size_t>(curSlot)].x,
                                zones[static_cast<std::size_t>(curSlot)].y,
                                nextSlot,
                                zones[static_cast<std::size_t>(nextSlot)].x,
                                zones[static_cast<std::size_t>(nextSlot)].y,
                                activeZoneIndex(), nextZoneIndex());
                    std::fflush(stdout);
                }
            }
        }
    }

    // ---- 0. 食物生成速率（随【存活蛇数】缩放）----
    //
    // ★ 为什么按存活数而不是配置数：一局打到后面蛇会死。如果还按初始数量供食，
    //   剩下的蛇就吃得过饱、淘汰压力消失。按存活数缩放能保持恒定的稀缺度。
    //
    // ★ 下限保底至少按 1 条的份供 —— 否则最后剩一条蛇时会供不上、尴尬地饿死。
    //   （这是"雪崩"风险：蛇越少食物越少。保住下限就没有这个问题。）
    {
        int alive = 0;
        for (const auto& a : m_agents) if (a.isAlive()) ++alive;
        const int effective = (alive > 1) ? alive : 1;
        m_food.setFoodPerSecond(static_cast<float>(effective)
                                * m_cfg.foodPerSnakePerSecond);
    }

    // ---- 1. 移动 ----
    for (auto& a : m_agents) a.body.update(dt);

    // ---- 2. 碰撞（撞墙 / 撞蛇 / 吃食物）----
    //
    // ★ 死因诊断：碰撞系统只调 die()，不说是哪条路。
    //   这里在调用【前后对比 isAlive】，变了就是碰撞杀的，
    //   再结合"离墙多远"区分撞墙 / 撞蛇（碰撞前的位置还没动，用它判断）。
    {
        std::vector<bool> aliveBefore;
        aliveBefore.reserve(m_agents.size());
        for (const auto& a : m_agents) aliveBefore.push_back(a.isAlive());

        m_collision.checkAll(m_ptrs, m_food);

        for (std::size_t i = 0; i < m_agents.size(); ++i) {
            if (aliveBefore[i] && !m_agents[i].isAlive()) {
                // 碰撞前蛇头离墙的距离：<= 头半径 + 2px 就认为撞的是墙
                const float x = m_agents[i].body.getX();
                const float y = m_agents[i].body.getY();
                const float dWall = std::min(
                    std::min(x, m_cfg.mapWidth - x),
                    std::min(y, m_cfg.mapHeight - y));
                const float r = m_agents[i].body.getHeadRadius() + 2.0f;
                m_deathCauses[i] = (dWall <= r) ? DeathCause::Wall
                                                : DeathCause::Snake;
            }
        }
    }

    // ---- 3. 饥饿 ----
    //
    // ★ starvationTime <= 0 时【跳过整条规则】。
    //
    //   为什么加这个保护：判定是 `timeSinceEat > starvationTime`，
    //   所以一个负数阈值会让它【恒成立】—— "永不饿死"变成"第一帧就饿死"，
    //   正好是反效果。实测踩过这个坑（现象：蛇一出生就死），
    //   查了好几轮才用死因诊断定位到。
    //
    //   这里让它"小于等于 0 就整条跳过"，语义变成"不启用饥饿" ——
    //   即使有人再写负数，也只会得到"不饿死"，不会得到"立刻死"。
    //   （想真正禁用饥饿，更推荐把单局时长设得比饥饿时间短，见 TrainTask.h）
    if (m_cfg.starvationTime > 0.0f) {
        for (std::size_t i = 0; i < m_agents.size(); ++i) {
            agent::Agent& a = m_agents[i];
            if (a.isAlive() && a.body.getTimeSinceEat() > m_cfg.starvationTime) {
                a.body.die();
                m_deathCauses[i] = DeathCause::Starvation;
            }
        }
    }

    // ---- 4. 死亡 → 尸体转食物（每条蛇只做一次）----
    for (agent::Agent& a : m_agents) {
        if (a.isAlive() || a.deathHandled) continue;
        m_food.spawnCorpseFood(a.body);
        a.deathHandled = true;
        a.brain->onDeath();
    }

    // ---- 5. 感知 + 决策 ----
    m_view.tick = static_cast<int>(m_tick);
    m_view.dt   = dt;

    // ★ 圈信息接进感知通道（全局信息，不走视野网格）。
    //   注意此时食物统计还是【上一帧末】的 —— 因为 m_food.update() 在第 6 步。
    //   差一帧对决策没有影响，但要知道这一点，否则会误以为"统计错了"。
    m_view.zones      = m_food.zones().empty() ? nullptr : &m_food.zones();
    m_view.activeZone = activeZone();
    // ★ 当前 / 下一个圈的【下标】—— 给观测编码用（见 Arena.h 的说明）
    m_view.activeZoneIndex = activeZoneIndex();
    m_view.nextZoneIndex   = nextZoneIndex();
    // ★ 本轮供食已过几秒 + 调度规则（2026-09-26）
    //   —— 供"提前出发"用：算法据此算"当前圈还能供多久"。
    //   ⚠️ 仍然不在感知层重推"下一个是谁"，那个只认 nextZoneIndex。
    m_view.cycleElapsed = m_sched.elapsedInCycle(m_food.zoneTimer());
    m_view.zoneSchedule = &m_sched;
    // ★ 当前是否处于供食阶段（决定"下一个圈"可不可见，见 PerceptionBuilder.h）
    m_view.zoneSpawning = m_sched.spawning(m_food.zoneTimer());

    // ★★ 是否处于【预告期】—— 决定"下一个圈"可不可见（2026-09-27）
    //
    //  ★ 规则唯一实现在 `ZoneSchedule::previewing()`：供食期最后
    //    `previewLeadSeconds` 秒 + 整个空窗期（默认 3 + 5 = 8 秒）。
    //    负责人定义："预告圈是在活跃圈产食物还剩 3s 后出现的。"
    //  ⚠️ 这里【不重推】那个算式 —— 那是本项目栽过两次的"两份实现"。
    m_view.zonePreviewing = m_sched.previewing(m_food.zoneTimer());

    for (std::size_t i = 0; i < m_agents.size(); ++i) {
        agent::Agent& a = m_agents[i];
        if (!a.isAlive()) continue;

        m_builder.build(m_view, i, a.perception);
        // 饥饿剩余与配置有关，感知层拿不到配置，这里补上
        a.perception.self.hungerLeft = m_cfg.starvationTime - a.body.getTimeSinceEat();

        // ★ 决策委托给 controller（训练）或蛇自己的大脑（游戏）。
        //   这是"游戏"与"训练"唯一的分岔点。
        Action action = m_controller ? m_controller(i, a.perception)
                                     : a.brain->think(a.perception);

        // ★ 动作量化 —— 属于"世界规则"，所以在这里做，不在调用方做。
        //
        //   离散动作空间：量到最近的档位。规则 AI 的连续输出也一并量化 ——
        //   否则它对只能出 5 档的神经网络形成了不公平的优势，
        //   "哪个算法强"的比较结论就不可信。
        //
        //   玩家那条【不量化】：它是给人玩的，手感优先，而且不参与算法对比。
        //   （量化公式本身也是双保险 —— Brain 接口约定输出已量化，
        //     但这里再兜一次，防止未来某个算法直接吐越界值。）
        if (a.teamId >= 0 && !a.body.isPlayer()) {
            action.steer = quantizeSteer(action.steer, ActionSpace::Discrete5);
        }

        // ★ 必须用 getSteerLead：蛇的转向是缓动的，目标必须领先足够远，
        //   否则实际角速度只有设定值的 1/(turnEase×dt) —— 玩家会完全控不住。
        a.body.setTargetAngle(a.body.getAngle() + a.body.getSteerLead(action.steer, dt));
        a.body.setBoosting(action.boost);
    }

    // ---- 6. 食物更新 ----
    m_food.update(dt);
}

}  // namespace arena
