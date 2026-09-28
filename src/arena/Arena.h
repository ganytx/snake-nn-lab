#pragma once

// ============================================================================
//  Arena.h — 「一局怎么跑」的唯一定义
//
//  ★ 为什么要有这个类
//
//  在这个类出现之前，"一局怎么跑"有【两份】：
//
//      GameScene::fixedUpdate()      ← 游戏本体
//      train_probe::runEpisode()     ← 训练器（还自带一个 struct World）
//
//  两份必须完全一致，但没有任何机制保证这一点。而它已经真的出过事了：
//  观测编码曾经也有两份，靠"碰巧一样"维持，结果在一个扇区边界判断上分叉，
//  造成 0.96 的观测差异（见 devlogs/2026-09-22.md 第 7 节）。
//  食物圈玩法比那个复杂得多，靠人工同步两份是不可能的。
//
//  所以：**每帧干什么，只在这里写一次。**
//
//  ★ 职责边界（与 docs/45-Arena设计.md 第 2 节一致）
//
//  Arena 负责：
//      agents（蛇 + 大脑 + 感知缓冲）· FoodManager · CollisionSystem
//      PerceptionBuilder · 出生布局 · 每帧的 1~6 步 · 决策调用
//      饥饿判定 · 死亡转尸体 · reset()
//
//  Arena 不负责（留在 GameScene）：
//      渲染 / 纹理 / 字体 / 相机 / 小地图 / 鼠标键盘
//      观战切换 / 队伍统计 / 限时 / 胜负 / 结算
//
//  最后几条"不负责"是有意的边界：将来要把结算也搬进来，是给 Arena
//  【加方法】，而不是重构调用方。
//
//  ★ 关于「无头」的准确含义
//
//  本类会持有 Snake，而 Snake.h include 了 SFML，所以本项目里
//  「无头」= **不创建窗口、不加载资源**，不是「不依赖 SFML」。
//  验收判据：本文件与 Arena.cpp 里不出现
//  sf::RenderWindow / sf::Texture / sf::Font（sf::Color 只是数据，允许）。
// ============================================================================

#include "../agent/Agent.h"
#include "../agent/PerceptionBuilder.h"
#include "../entities/FoodManager.h"
#include "../entities/FoodZone.h"
#include "../systems/CollisionSystem.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace arena {

// ---------------------------------------------------------------------------
//  出生布局
//
//  ★ 为什么需要两种
//
//    Ring  —— 各队从均分圆周的方位出生、朝向地图中心。**游戏用这个**：
//             否则开局全挤在中心互相撞死，学不到"合作"，只学到"乱斗"。
//
//    Random—— 每局在安全边距内随机撒点。**训练用这个**：
//             要是每局都从同一个环上同一个点出发，网络会过拟合到那几个
//             出生方位；随机出生才能逼它学"朝食物走"而不是"背地图"。
//
//    这个差别以前是【两份世界实现各写各的】。现在明确成一个配置项，
//    写在一处，谁用哪种一目了然。
// ---------------------------------------------------------------------------
enum class SpawnMode {
    Ring,     // 均分圆周（游戏默认）
    Random    // 随机撒点（训练默认）
};

// ---------------------------------------------------------------------------
//  死因
//
//  ★ 为什么要它：负责人报"蛇刚出生就死了"，而"这一局结束了"分不清是
//    【撞墙】【撞蛇】【饿死】还是【撞自己】—— 四种的修法完全不同。
//    实测就卡在这里：出生点离墙 94px、死在离墙 95px，明明没靠近墙却死了，
//    只能靠代码自己说是哪条路杀的。
// ---------------------------------------------------------------------------
enum class DeathCause {
    None = 0,
    Wall,       // 撞墙
    Snake,      // 撞到别的蛇（含对头相撞）
    Starvation, // 饿死
    SelfBody,   // 撞到自己身体
};

inline const wchar_t* deathCauseName(DeathCause c) {
    switch (c) {
        case DeathCause::Wall:       return L"撞墙";
        case DeathCause::Snake:      return L"撞蛇";
        case DeathCause::Starvation: return L"饿死";
        case DeathCause::SelfBody:   return L"撞自己";
        default:                     return L"存活";
    }
}


// ---------------------------------------------------------------------------
//  一局所需的全部配置
//
//  刻意【不用 GameConfig】——GameConfig 在 src/core/ 里，而且带着一堆
//  跟"世界"无关的东西（窗口、字体、对局 UI）。Arena 只认得下面这些。
//  由调用方负责把 GameConfig 翻译成 ArenaConfig。
// ---------------------------------------------------------------------------
struct ArenaConfig {
    static constexpr int MAX_TEAMS = 8;

    // ---- 地图 ----
    float mapWidth  = 2000.0f;
    float mapHeight = 2000.0f;

    // ---- 队伍 ----
    int teamCount     = 1;      // 分几队
    int snakesPerTeam = 1;      // 每队几条蛇
    int playerTeam    = -1;     // 玩家在第几队；-1 = 没有玩家（纯 AI / 训练）
    // 每队用什么算法（对应 BrainRegistry 的 id）。长度 MAX_TEAMS，不足的忽略。
    std::string teamBrains[MAX_TEAMS] = {"rule", "rule", "rule", "rule",
                                         "rule", "rule", "rule", "rule"};

    // -----------------------------------------------------------------------
    //  ★★ 每队加载哪份权重文件（2026-09-27 新增，负责人要求）
    //
    //  ★ 为什么必须按队指定
    //    负责人要"不同队伍可以加载存档里的不同权重"。
    //    而 `NeuralBrain::setPendingWeights` 是【全局】的 —— 它只够表达
    //    "这一次 reset 所有网络队都用同一个文件"。所以这里按队存一份，
    //    Arena 造每一队的脑之前把它喂给 `NeuralBrain`（见 Arena.cpp）。
    //
    //  ★ 空字符串 = 走原来的自动查找（weights/policy.txt 等）
    //
    //  ⚠️ 它和 `teamBrains` 一样是【按队】的：同队多条蛇共用同一份权重。
    //     这是刻意的（负责人选了"每队一个权重文件"）。
    // -----------------------------------------------------------------------
    std::string teamWeights[MAX_TEAMS];

    // ---- 出生 ----
    SpawnMode spawnMode = SpawnMode::Ring;
    // Random 模式：距地图边界至少留多少像素（免得一出生就贴墙）
    float spawnMargin = 200.0f;
    // 固定种子。0 = 不重新播种；非 0 = 每次 reset() 用它 srand，
    // 这样"同一个种子跑出同一局"成立（跑分 / 回归要用）。
    unsigned spawnSeed = 0;

    // -----------------------------------------------------------------------
    //  ★★ 每局是否换一个种子（负责人要求，2026-09-27）
    //
    //    "训练模式，游戏模式请每局种子不同"
    //
    //  ★ false（默认）= 旧行为：每次 reset 都用同一个 spawnSeed
    //      replay_check 这类回归工具必须用这个 —— 世界指纹靠它才可复现。
    //  ★ true = 每次 reset 把种子往后推一格
    //      于是每一局的【出生点 / 出生角度 / 食物 / 圈轮换顺序 / 圈坐标】
    //      全都不同。训练和游戏用这个。
    //
    //  ⚠️ 它只在 `spawnSeed != 0` 时生效：种子为 0 是"不动全局序列"的
    //     约定（见上面那行），那种情况下 Arena 不碰任何种子。
    // -----------------------------------------------------------------------
    bool advanceSeedOnReset = false;

    // ---- 蛇 ----
    float snakeSpeed      = 250.0f;
    float snakeBoostSpeed = 450.0f;
    float snakeTurnRate   = 8.0f;
    float segmentSpacing  = 20.0f;
    int   snakeInitSegs   = 12;
    int   foodSegments    = 5;   // 吃几个普通食物 +1 体节
    int   corpseSegments  = 2;   // 吃几个尸体食物 +1 体节

    // ---- 食物 ----
    //
    // ⚠️ `foodMaxCount` 于 2026-09-27 删除：它被拷进来之后
    //    **全项目没有任何读取点**，是个死字段（登记在 docs/30 缺陷清单）。
    //    食物数量的实际上限由"生成速率 × 寿命"决定，不需要单独的阀门。
    int   foodSpawnCount    = 2;
    float foodSpawnInterval = 2.0f;
    int   corpseScore       = 3;

    // -----------------------------------------------------------------------
    //  ★ 食物圈（玩法核心，见 docs/70-食物圈玩法设计.md）
    //
    //  zoneCount = 0  → 关闭圈，退化成"全图随机撒食物"（老行为，对照用）
    // -----------------------------------------------------------------------
    ZoneLayout   zoneLayout;
    ZoneSchedule zoneSchedule;

    // 每【9 条蛇】每秒生成多少食物。生成速率 = 存活蛇数 / 9 * 这个值。
    //
    // ★ 为什么必须随蛇数缩放：「每轮 9 个」这个数字不能写死 ——
    //   蛇少了食物过剩（没人抢，玩法退化），蛇多了必然大面积饿死。
    //   按"每条蛇每秒 1/9 个"恒定配比，2 条蛇和 18 条蛇都自动适配。
    float foodPerSnakePerSecond = 1.0f / 9.0f;

    // 普通食物寿命（秒）。食物圈玩法要求普通食物也会过期，
    // 否则一轮没吃完的食物无限堆积，玩法退化。
    float foodLifetime   = 30.0f;
    float corpseLifetime = 30.0f;

    // -----------------------------------------------------------------------
    //  ★★ 普通食物的数量上限（2026-09-27 恢复）
    //
    //  ⚠️ 这个字段 2026-09-27 白天曾被删掉过一次，理由是"拷进来之后没有任何
    //     读取点，是个死字段"。**那个判断对当时是对的**（当时没人设它），
    //     但负责人随后要求"食物存在上限"，所以现在它有了真正的读取点：
    //     `Arena::applyConfigToSystems()` 把它转交给 `FoodManager`，
    //     由 `FoodManager::update()` 在生成时判断。
    //
    //  ★ 与 `initialFoodCount` 的关系：两者都是"没有圈的世界"才用得到的
    //    参数（模式 2 / 模式 3）。模式 1 有圈，食物由圈逐秒产，上限留 0。
    //
    //  0 = 不限制（训练任务与旧对照都保持这个语义）
    // -----------------------------------------------------------------------
    int foodMaxCount = 0;

    // ★ 初始化时在【全图随机位置】撒多少个普通食物。
    //
    //   食物圈玩法用不到它（圈会自己逐秒生成），默认 0。
    //   它的用途是"没有圈的世界"（对照 / 回归测试）需要一个初始食物量。
    //
    //   ★ 为什么做成配置项而不是让调用方在构造后自己调 FoodManager：
    //     那样调用时机会影响 rand() 序列 —— Arena 构造里会创建 N 个 RuleBrain，
    //     构造后再撒食物就已经晚了一步，和手写世界的随机序列对不上。
    //     做成配置项，撒食物的时机就固定为"食物系统刚配好、还没建蛇"，
    //     两边自然一致。这个坑实际踩过一次（见 devlogs/2026-09-22.md）。
    //
    //   ⚠️ 受 `foodMaxCount` 限制（要到上限就少撒）。
    int initialFoodCount = 0;

    // -----------------------------------------------------------------------
    //  ★ 最小学习验证用：把【一个】食物放在蛇出生点【正前方】
    //
    //  ★ 为什么需要它（解决的是一个方法论问题）
    //
    //    食物圈任务太复杂 —— 地图大、视野小、圈会轮换、还会饿死。
    //    学不会的时候分不清是"奖励设计不对"还是"学习链路根本不通"，
    //    实测调了七八轮奖励都没分清。
    //
    //    这个任务把问题剥到只剩核心：
    //        · 地图很小（400x400），食物就在正前方 300px
    //        · 没有圈、没有障碍、没有尸体
    //        · 单局 10 秒，**短于饥饿时间**，所以局内不会饿死
    //          （用"时长比饥饿短"来保证，不靠任何特殊开关 —— 见 TrainTask.h）
    //    最优策略是【显而易见】的：直行。
    //
    //    于是判据变得干脆：
    //        能学会直行 → 学习链路是通的，问题在奖励设计
    //        学不会     → 问题在链路本身，跟奖励无关
    // -----------------------------------------------------------------------
    bool  spawnFixedFoodAhead = false;
    float fixedFoodDistance   = 300.0f;

    // 出生时身体堆成一个点（负责人提的）
    //
    // ★ 解决三个问题（详见 Snake::setCompactSpawn 的说明）：
    //    1) 出生时不可能撞到自己 → "学觅食"和"学不撞自己"两个变量分开
    //    2) 小地图上不再一出生就把身体伸出地图外
    //    3) 视觉上更符合直觉（刚出生就是一小团）
    //
    // ⚠️ 默认【关】：游戏本体保持原来的"笔直铺开"，不动既有手感。
    //    只有训练任务会打开。
    bool compactSpawn = false;

    // ---- 规则 ----
    //
    // ⚠️ 想让某个任务"局内不会饿死"，请把【单局时长】设得比它短，
    //    不要把 starvationTime 设成负数。
    //
    //    判定是 `if (timeSinceEat > starvationTime) die();` ——
    //    负数会让它【恒成立】，也就是"第一帧就饿死"，正好是反效果。
    //    实测踩过这个坑（现象：蛇一出生就死），详见 TrainTask.h 里 AheadFood 的注释。
    float starvationTime   = 20.0f;
    float collisionForgive = 5.0f;
    bool  friendlyFire     = false;

    // ---- 感知 ----
    float visionCellSize   = 32.0f;   // 32 格 x 32px = 视野 1024px
    int   allyVisibleCount = 2;

    int effectiveTeamCount() const {
        int n = teamCount;
        if (n < 1) n = 1;
        if (n > MAX_TEAMS) n = MAX_TEAMS;
        return n;
    }
};

// ---------------------------------------------------------------------------
//  Arena
// ---------------------------------------------------------------------------
class Arena {
public:
    // 每条蛇该出什么动作。
    //   index : 该蛇在 agents() 里的下标
    //   perc  : 刚为它填好的感知
    // 返回要施加到这条蛇身上的动作。
    using Controller = std::function<Action(std::size_t, const Perception&)>;

    // 默认构造：地图取默认尺寸，**不建任何 agent**（agent 数为 0，step() 直接返回）。
    // 给"构造完再 reset(配置)"这种用法准备的（GameScene 就是这么用的）。
    Arena() : Arena(ArenaConfig{}) {}

    explicit Arena(const ArenaConfig& cfg);

    // 换一套配置并重建全部 agent。
    //
    // ★ 为什么需要它：GameScene「重开一局」和"改了设置再开一局"都要用。
    //   而配置（地图尺寸、食物上限、碰撞参数…）是**构造期**就分发到
    //   FoodManager / CollisionSystem / PerceptionBuilder 里的，
    //   只调 reset() 不会让这些改动生效 —— 那会造成
    //   "设置里改了但没起作用"这种最难查的一类 bug。
    void reset(const ArenaConfig& cfg);

    // 用当前配置重建全部 agent（首次构造之后已经自动调过一次）
    void reset();

    // -----------------------------------------------------------------------
    //  ★★ 中途加入一条蛇（2026-09-27 新增，无限模式用）
    //
    //  ★ 为什么需要它
    //    「无限模式」要求"每隔一段时间有一条新蛇加入"。
    //    而 Arena 原来只能 `reset()` 时一次性建好全部蛇 —— 世界跑起来之后
    //    没有任何办法再加一条。
    //
    //  ★ 它比 reset() 多做/少做了什么
    //    做：建脑（含按队权重）→ 出生位置 → 体节 → 饥饿计时 → 死因表 → 指针表
    //    不做：清空世界。**已有的蛇、食物、圈、计时器全部保留** ——
    //          这正是"加入"与"重开"的区别。
    //
    //  ★ 出生位置
    //    随机撒点（和训练用的随机出生一致：距边界至少 spawnMargin）。
    //    不走 Ring 出生 —— 那是"开局按队排开"用的，半路加入不需要队形。
    //
    //  ★ 返回新蛇在 agents() 里的下标；失败（地图太小等）返回 -1。
    //
    //  ⚠️⚠️ 调用者必须知道：本函数会【重新分配 m_agents】，
    //      于是 `m_ptrs` 里所有 `&a.body` 都会失效 —— 内部的 reserve 就是
    //      为这件事准备的。**所以绝不能让外部长期持有 Agent* 或 Snake*。**
    //      （`GameScene` 里的 `m_humanBrain` 因此也在 spawn 之后可能失效，
    //        见那里的说明。）
    // -----------------------------------------------------------------------
    int addAgent(const std::string& brainId, int teamId);

    // ---- 推进 ----

    // 推进一个固定步长：移动 → 碰撞 → 饥饿 → 尸体 → 感知 → 决策 → 食物
    //
    // ★ 决策发生在【感知刚填好】的那一刻，中间没有任何其它世界状态变化。
    //   把决策留在 step() 内部，调用方就插不进"改世界"的代码，
    //   也就无法让训练环境与游戏环境跑偏 —— 这是刻意的设计。
    void step(float dt);

    // 换掉决策来源。传空（默认）表示走 a.brain->think()。
    // 训练器用它来接管决策（编码观测 → 采样 → 记录轨迹）。
    void setController(Controller c) { m_controller = std::move(c); }

    // 每条蛇的【死因】（下标与 agents() 对应）。
    // 只在诊断时看 —— 世界推进本身不依赖它。
    const std::vector<DeathCause>& deathCauses() const { return m_deathCauses; }

    // -----------------------------------------------------------------------
    //  ★★ 把已经死掉的蛇从场上移除（2026-09-27 新增，负责人要求）
    //
    //  ★ 负责人报的问题
    //      "第二种模式，蛇死了仍然占有 20 条上限。"
    //    根因：无限模式的生蛇上限判的是 `agents().size()`，而**死蛇一直留在
    //    列表里**（Arena 从来不删死的 agent）—— 一旦累计生过 20 条，
    //    上限就永久卡死，世界再也不出新蛇。
    //
    //  ★ 为什么由 Arena 提供这个操作，而不是让 GameScene 自己 erase
    //    `m_agents` 一改，下面三样东西**全部失效**，而且都不会报错：
    //      ① `m_ptrs`  —— 存的是 `&agent.body`，下标一变就指错蛇
    //      ② `m_deathCauses` —— 与 agents 下标一一对应
    //      ③ `m_humanIndex`  —— 玩家那条的下标
    //    这三样都是 Arena 的私有状态，只有 Arena 自己能一致地修好。
    //    **调用方拿到的是一个已经自洽的世界**，不需要知道内部有几个容器。
    //
    //  ★ 调用时机：必须在 `step()` **之后**、下一次 `step()` 之前。
    //    本函数会重建 `m_ptrs`，而 `step()` 里感知/碰撞全程依赖它 ——
    //    中途改它（比如在 step 内部）会让同一步里前后两次读到的表不一致。
    //
    //  ★ 为什么可以安全 erase（与 `reserve(32)` 那个坑的关系）
    //    erase 是**删除**元素、不会扩容，所以 `m_ptrs` 里那些指向
    //    **幸存者**的指针依然有效 —— 但下标会变，所以照样重建一遍最稳。
    //    真正危险的是"无限往里加"，那个由调用方限流（见 GameScene 的生蛇逻辑）。
    //
    //  ★ 返回值：移除掉的条数（给调用方打日志 / 判断"这一帧有没有蛇死"）
    // -----------------------------------------------------------------------
    int removeDeadAgents();

    // ---- 查询 ----

    std::vector<agent::Agent>&       agents()       { return m_agents; }
    const std::vector<agent::Agent>& agents() const { return m_agents; }

    // 供碰撞 / 感知使用的指针表。只在 reset() 末尾填一次，
    // 之后 m_agents 不再增删 → 指针永不失效。
    const std::vector<Snake*>& snakePtrs() const { return m_ptrs; }

    FoodManager&       food()       { return m_food; }
    const FoodManager& food() const { return m_food; }

    const std::vector<Food>& foods() const { return m_food.getFoods(); }

    float mapWidth()  const { return m_cfg.mapWidth; }
    float mapHeight() const { return m_cfg.mapHeight; }

    std::uint64_t tick() const { return m_tick; }

    // 当前正在供食的那个圈；没有圈时返回 nullptr
    const FoodZone* activeZone() const {
        const int i = m_sched.activeZone(m_food.zoneTimer(),
                                         static_cast<int>(m_food.zones().size()));
        if (i < 0) return nullptr;
        return &m_food.zones()[static_cast<std::size_t>(i)];
    }

    // ★ 当前 / 下一个要供食的圈的【下标】（−1 = 没有圈）
    //
    //  ★ 为什么需要它（负责人要求：不要把"4 个圈"写死在观测里）
    //
    //    观测里原来给的是"前 4 个圈的方位"，这是一个**写死的数量假设**：
    //    圈数不是 4 的时候（1 个、6 个、0 个）它就不对了。
    //    换成"当前圈 + 下一个圈"之后，观测与圈数【解耦】——
    //    不管世界里有几个圈，蛇看到的永远是"现在该去哪、接下来去哪"。
    //
    //  ★ 为什么"下一个"有用
    //    它让蛇有机会学【提前出发】：本轮快结束时先往下一个圈挪，
    //    而不是等当前圈关了才开始跑。这正是轮换玩法想要的预判行为。
    int activeZoneIndex() const {
        return m_sched.activeZone(m_food.zoneTimer(),
                                  static_cast<int>(m_food.zones().size()));
    }
    int nextZoneIndex() const {
        // ⚠️ 必须问 ZoneSchedule，不能写 (active+1)%n ——
        //    随机模式下"下一个"是哈希算出来的，写死 +1 就是第二份实现。
        return m_sched.nextZone(m_food.zoneTimer(),
                                static_cast<int>(m_food.zones().size()));
    }
    const ZoneSchedule& zoneSchedule() const { return m_sched; }

    // -----------------------------------------------------------------------
    //  ★★ 只读访问【本帧的世界视图】（2026-09-27 新增）
    //
    //  ★ 为什么需要它
    //    诊断工具要回答"此时此刻这个世界会喂给网络什么" ——
    //    那就必须自己跑一次 `PerceptionBuilder::build(view, ...)` + `encodeObservation`。
    //
    //  ⚠️ 为什么不能拿 `EpisodeRunner::lastObservation()`
    //    那是"学习那条蛇【最后一次决策】的观测"。蛇一死，
    //    `runner.decide()` 就不再被调用，那份观测**冻结在死前那一帧**。
    //    用它去验"某个时间点的可见性"会读到旧值，看起来像"改动没生效" ——
    //    这个坑本轮踩过一次，浪费了一轮排查。
    //
    //  ⚠️ `m_view` 在 `step()` 里每帧重填，但**只在有活蛇时**才填
    //    （见 Arena::step 第 5 步）。所以拿到的是"最近一次感知构建时"的快照，
    //    这一点调用方要知道。
    // -----------------------------------------------------------------------
    const agent::WorldView& view() const { return m_view; }

    // 玩家那条蛇在 agents() 里的下标；没有玩家时返回 -1
    int humanIndex() const { return m_humanIndex; }

    const ArenaConfig& config() const { return m_cfg; }

    // -----------------------------------------------------------------------
    //  队伍颜色
    //
    //  ★ 颜色是 Arena 造蛇时给它的（Agent 构造要一个 sf::Color），
    //    所以定义放在 Arena 这边；HUD / 结算界面想画"队伍色块"就来这里查。
    //
    //    这样就不会出现"渲染代码里另抄一份颜色表、以后改了一边忘了另一边"
    //    —— 那种 bug 的表现是"小地图上的点和蛇的颜色对不上"。
    // -----------------------------------------------------------------------
    static sf::Color teamColor(int teamIndex) noexcept;

private:
    // 把 m_cfg 分发到 FoodManager / CollisionSystem / PerceptionBuilder
    void applyConfigToSystems();

    // ★★ 每局开始：播世界种子 + 把圈摆到本轮该在的位置（见 Arena.cpp 的说明）
    void beginRoundSetup();

    ArenaConfig m_cfg;

    std::vector<agent::Agent> m_agents;
    std::vector<Snake*>       m_ptrs;

    // 每条蛇的死因（诊断用，下标与 m_agents 对应）
    std::vector<DeathCause>   m_deathCauses;

    FoodManager     m_food;
    CollisionSystem m_collision;

    agent::PerceptionBuilder m_builder;
    agent::WorldView         m_view;

    ZoneSchedule m_sched;

    // ★ 位置随机模式：上一次处理过的【轮次号】。
    //   −1 = 还没处理过（第一帧就把第 0 轮的圈挪到它的随机位置）。
    //   只在 zoneLayout.randomPosition 为真时被读写。
    int m_posCycle = -1;

    // ★★ 每局换种子用（见 ArenaConfig::advanceSeedOnReset）
    unsigned m_resetCount = 0;   // reset 了几次（用来把种子往后推）
    unsigned m_worldSeed  = 0;   // 本局实际用的世界种子（诊断可读）

public:
    // 本局的世界种子（0 = 未播种）。给诊断/日志用。
    unsigned worldSeed() const { return m_worldSeed; }
private:

    std::uint64_t m_tick = 0;
    int           m_humanIndex = -1;

    Controller m_controller;   // 空 = 用每条蛇自己的 brain
};

}  // namespace arena
