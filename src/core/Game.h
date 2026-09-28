#pragma once

#include <SFML/Graphics.hpp>
#include <memory>
#include <vector>
#include <string>
#include "Scene.h"
#include "GameConfigIO.h"
#include "GameModes.h"     // ★ GameMode 枚举（每模式一份配置用）

// ==========================================
// 可保存的游戏配置项
//
// ★ 字段本身写在 GameConfigIO.h 的 GAME_CONFIG_FIELDS 表里
//   （用 X-macro 展开成这里的成员声明），
//   目的是让「字段声明 / 读配置 / 写配置」三处永远不可能跑偏。
//   加字段 → 只改那张表。
//
// ★ 三层配置（负责人定的职责边界，见 GameModes.h 的说明）
//     ① 通用世界规则  → 下面这些直接字段（速度/转向/间距/饥饿/碰撞…）
//     ② 每模式默认值  → ModeConfig modes[3]
//     ③ 这一局的配置  → 开局时从 ② 拷一份，小设置页只改那份
// ==========================================
struct GameConfig {
    static constexpr int MAX_TEAMS = 8;   // 队伍数上限

    // -----------------------------------------------------------------------
    //  ★ 一个模式的完整配置（= 大设置里能改的"该模式的默认值"）
    //
    //  ★ 为什么是"完整"而不是"只存差异"
    //    只存差异的话，读的时候要"基线 + 差异"拼起来，而拼接逻辑一旦和
    //    "恢复初始值"不一致就会出现"按了 R 但某项没回去"这类问题。
    //    每模式存一份完整的，最直白、最好验证。
    //
    //  ★ 为什么权重路径是【每队一份】
    //    负责人要求"不同队伍可以加载存档里的不同权重"。
    //    以前的 `NeuralBrain::setPendingWeights` 是【全局】的，一次 reset 只能
    //    让所有网络队加载同一个文件 —— 所以必须变成"按队指定"。
    //      空字符串 = 走原来的自动查找（weights/policy.txt 等）
    // -----------------------------------------------------------------------
    struct ModeConfig {
        // ★ 这三个的初值必须和 `GameModes.cpp::applyModeParams()` 里设的一致
        //   （那边是"恢复默认值"用的，这里是"零配置启动"用的）。
        //   ⚠️ 现状只对【模式 0】成立；模式 1/2 的真正初值由
        //      `Game::loadConfig()` 末尾的"补基线"负责 ——
        //      因为模式定义（圈数/食物速率）不止这三个字段，
        //      在这里复制一份只会多一个不一致的机会。
        int teamCount     = 3;
        int snakesPerTeam = 3;
        float foodPerSnake = 1.0f / 9.0f;   // 每蛇每秒（总速率 = 本值 × 存活数）

        // -------------------------------------------------------------------
        //  ★★ 食物存量（2026-09-27 新增，负责人要求）
        //
        //  ★ 为什么这两个必须【按模式】而不能进通用配置表
        //    "开局撒多少食物""场上最多留多少食物"正是模式之间的分界线：
        //      模式 2 无限模式 —— 开局就撒一大片、上限很高（食物管够）
        //      模式 3 大混战   —— 上限很低（食物稀缺，靠尸体）
        //    放进通用配置表就只有一个值，三个模式必然长得一样 ——
        //    那正是负责人报过的"大混战和无限模式完全混淆"。
        //
        //  ★ 口径
        //    initialFoodCount：开局在全图随机撒多少个【普通食物】
        //    foodMaxCount    ：场上最多同时存在多少个【普通食物】（不含尸体）
        //                      0 = 不限制
        //
        //  ⚠️ 只有"没有食物圈"的模式（2 / 3）用得到它们 ——
        //     模式 1 的食物由圈逐秒产，两个值都留 0。
        // -------------------------------------------------------------------
        int initialFoodCount = 0;
        int foodMaxCount     = 0;

        // 每队的算法 id（对应 BrainRegistry 里注册的 id）
        std::string teamBrains[MAX_TEAMS] = {"nn", "rule", "nn-enemy", "rule",
                                             "rule", "rule", "rule", "rule"};
        // 每队加载哪份权重（空 = 自动查找）
        std::string teamWeights[MAX_TEAMS];

        int effectiveTeamCount() const {
            int n = teamCount;
            if (n < 1) n = 1;
            if (n > MAX_TEAMS) n = MAX_TEAMS;
            return n;
        }
        int effectiveSnakesPerTeam() const {
            return (snakesPerTeam < 1) ? 1 : snakesPerTeam;
        }
        int totalSnakes() const {
            return effectiveTeamCount() * effectiveSnakesPerTeam();
        }
    };

#define X_FLOAT(name, key, def) float name = (def);
#define X_INT(name, key, def)   int   name = (def);
#define X_BOOL(name, key, def)  bool  name = ((def) != 0);
    GAME_CONFIG_FIELDS(X_FLOAT, X_INT, X_BOOL)
#undef X_FLOAT
#undef X_INT
#undef X_BOOL

    // ---- ② 每模式的默认值（下标 = static_cast<int>(GameMode)）----
    ModeConfig modes[game::MODE_COUNT];

    // -----------------------------------------------------------------------
    //  ★ 每模式配置的文本读写（键名约定，Game.cpp 调用）
    //
    //      mode<M>_teams=3              该模式默认几队
    //      mode<M>_per_team=3           该模式默认每队几条
    //      mode<M>_food_per_snake=0.222 该模式默认食物速率（每蛇每秒）
    //      mode<M>_brain_<T>=nn         第 T 队用什么算法
    //      mode<M>_weight_<T>=xxx.txt   第 T 队加载哪份权重（空 = 自动查找）
    //
    //  ★ 为什么键名带模式号而不是分段落
    //    解析器是"逐行 key=value"，没有段落概念。带前缀的平铺键名
    //    让解析保持一行一个判断，不需要引入"当前在哪个段"这种状态。
    //  ⚠️ 前缀常量必须和 Game.cpp 里用的一致 —— 定义在这里，只有一处。
    // -----------------------------------------------------------------------
    static constexpr const char* MODE_KEY_PREFIX = "mode";

    // ⚠️ 原来这里还有一个 `GameConfig::effectiveTeamCount()` ——
    //    2026-09-27 删掉了：队伍规模已经搬进 `ModeConfig`，
    //    留在 GameConfig 上只会让人误以为"这里还能拿到队数"。
    //    现在要队数请用 `cfg.modes[mi].effectiveTeamCount()`。

    // -----------------------------------------------------------------------
    //  ★ 把所有字段夹到"能真的跑起来"的范围里
    //
    //  为什么需要它：配置文件是人可以手改的，设置界面也允许把地图调到 20。
    //  而地图 20x20 会造出一个"看起来在跑、其实什么都不会发生"的世界：
    //  蛇一出生就撞墙，食物根本放不下（生成范围是 [20, mapW-20]，会变成负数）。
    //
    //  以前只靠设置界面的 min/max 挡着，**手改配置文件就绕过去了**。
    //  所以这里做一道【与界面无关】的兜底，loadConfig / 开新对局时都会调用。
    //
    //  ★★ 2026-09-27：队伍规模与每蛇食物速率搬进了 ModeConfig，
    //     所以那几个字段的夹取改在 `sanitizeModes()` 里（下面）。
    // -----------------------------------------------------------------------
    void sanitize() {
        if (mapWidth  < 400.0f) mapWidth  = 400.0f;   // 至少要装得下蛇和一点活动空间
        if (mapHeight < 400.0f) mapHeight = 400.0f;
        if (mapWidth  > 40000.0f) mapWidth  = 40000.0f;
        if (mapHeight > 40000.0f) mapHeight = 40000.0f;

        if (snakeSpeed      < 1.0f)   snakeSpeed      = 1.0f;
        if (snakeBoostSpeed < snakeSpeed) snakeBoostSpeed = snakeSpeed;  // 加速不能比常速慢
        if (snakeTurnRate   < 0.1f)   snakeTurnRate   = 0.1f;
        if (snakeInitSegs   < 1)      snakeInitSegs   = 1;
        if (segmentSpacing  < 2.0f)   segmentSpacing  = 2.0f;

        if (foodSpawnCount   < 1)   foodSpawnCount   = 1;   // 0 = 食物永远不再生成
        if (foodSpawnInterval < 0.1f) foodSpawnInterval = 0.1f;
        if (foodSegments     < 1)   foodSegments     = 1;   // 吃几个长一节，不能是 0
        if (foodLifetime     < 1.0f) foodLifetime    = 1.0f;
        if (corpseLifetime   < 1.0f) corpseLifetime  = 1.0f;
        if (corpseSegments   < 1)   corpseSegments   = 1;
        if (corpseScore      < 1)   corpseScore      = 1;

        // ---- 食物圈 ----
        if (zoneCount < 0) zoneCount = 0;                   // 0 = 关闭圈
        if (zoneCount > 8) zoneCount = 8;                   // 上限：观测里只留了 8 个圈的位置
        if (zoneCenterRatio < 0.05f) zoneCenterRatio = 0.05f;
        if (zoneCenterRatio > 0.48f) zoneCenterRatio = 0.48f;  // 再大就出地图了
        if (zoneRadiusRatio < 0.01f) zoneRadiusRatio = 0.01f;
        if (zoneRadiusRatio > 0.30f) zoneRadiusRatio = 0.30f;
        if (zoneSpawnSeconds < 0.5f) zoneSpawnSeconds = 0.5f;
        if (zoneGapSeconds   < 0.0f) zoneGapSeconds   = 0.0f;

        if (starvationTime < 1.0f)  starvationTime = 1.0f;
        if (collisionForgive < 0.0f) collisionForgive = 0.0f;

        if (playerTeam >= MAX_TEAMS) playerTeam = MAX_TEAMS - 1;   // -1 = 不参战，保留
        if (matchTimeLimit < 0.0f)  matchTimeLimit = 0.0f;
        if (allyVisibleCount < 0)   allyVisibleCount = 0;

        sanitizeModes();
    }

    // 每模式配置的兜底夹取（手改配置文件能绕过设置界面）
    //
    //  ★ 为什么单独一个函数：`modes[]` 是数组，逻辑上和标量字段是两件事；
    //    混在一起会让上面那个函数越来越长、越难核对。
    void sanitizeModes() {
        for (int mi = 0; mi < game::MODE_COUNT; ++mi) {
            ModeConfig& mc = modes[mi];
            if (mc.teamCount     < 1) mc.teamCount     = 1;
            if (mc.teamCount     > MAX_TEAMS) mc.teamCount = MAX_TEAMS;
            if (mc.snakesPerTeam < 1) mc.snakesPerTeam = 1;
            if (mc.snakesPerTeam > 8) mc.snakesPerTeam = 8;
            // 每蛇速率：上限 20 是防手滑写成 111（那会让食物瞬间铺满全图）
            //
            // ⚠️ 2026-09-27 从 5 提到 20：无限模式要"食物生成速度大约 10 倍"，
            //    而 20 条蛇时 8/蛇/秒 = 160 个/秒，已经远超任何合理需求；
            //    真正兜住"铺满全图"的是 `foodMaxCount`（存量上限），
            //    不是这个速率上限。
            if (mc.foodPerSnake  < 0.0f) mc.foodPerSnake = 0.0f;
            if (mc.foodPerSnake  > 20.0f) mc.foodPerSnake = 20.0f;
            // 食物存量：负数 = 没意义，夹到 0（= 不限制）
            if (mc.initialFoodCount < 0) mc.initialFoodCount = 0;
            if (mc.initialFoodCount > 2000) mc.initialFoodCount = 2000;
            if (mc.foodMaxCount < 0) mc.foodMaxCount = 0;
            if (mc.foodMaxCount > 5000) mc.foodMaxCount = 5000;
            // 算法 id 不能是空串（空串会让 BrainRegistry 找不到而退回规则 AI，
            // 那种"设了没用"的失败最难查 —— 所以直接兜成 "rule"）
            for (int t = 0; t < MAX_TEAMS; ++t) {
                if (mc.teamBrains[t].empty()) mc.teamBrains[t] = "rule";
            }
        }
    }
};

// ==========================================
// Game 类 — 掌管主循环、场景栈、窗口、配置
// ==========================================
class Game {
public:
    Game();
    ~Game() = default;

    void run();

    // 场景管理
    void pushScene(std::unique_ptr<Scene> scene);
    void popScene();
    void switchScene(std::unique_ptr<Scene> scene);

    sf::RenderWindow& getWindow() { return m_window; }
    void closeWindow() { m_window.close(); }

    // 配置读写
    GameConfig& getConfig() { return m_config; }
    const GameConfig& getConfig() const { return m_config; }
    void loadConfig();                     // 从文件加载
    void saveConfig() const;               // 保存到文件

    // =======================================================================
    //  ★★★ 第三层配置：这一局的配置（2026-09-27 新增）
    //
    //  ★ 三层的关系（负责人定的职责边界）
    //      ① 通用世界规则  → m_config 的标量字段（m_config 就是"①+②的容器"）
    //      ② 每模式默认值  → m_config.modes[3]
    //      ③ 这一局的配置  → m_matchCfg（下面这两个）
    //
    //  ★ 为什么必须单独一份
    //    「模式选择页 → 设置」（小设置）只允许改【这一局】，不能改默认值。
    //    如果小设置直接改 m_config.modes[mid]，那就变成"改了默认值"，
    //    下次进同一模式会带上上次的临时改动 —— 与"只影响这一局"矛盾。
    //    所以开局时从 ② 拷一份到 ③，小设置只改 ③。
    //
    //  ★ 为什么把 mode 和 config 放在一起
    //    它们必须【同时有效】：读 m_matchCfg 的人一定也需要知道是哪个模式
    //    （例如"这个模式有没有时间限制"）。分开存会出现"模式变了但配置没换"
    //    这种半更新状态 —— 那是最难查的一类 bug。
    // =======================================================================
    game::GameMode matchMode() const { return m_matchMode; }

    GameConfig&       getMatchConfig()       { return m_matchCfg; }
    const GameConfig& getMatchConfig() const { return m_matchCfg; }

    // 开始一局：把 ① + 该模式的 ② 拷进 ③（m_matchCfg）
    //
    //  ★ 这是【唯一】初始化 ③ 的地方。任何"开始游戏"的入口都必须先调它，
    //    否则 ③ 里是上一次的残留（或者根本是默认构造的空配置）。
    void beginMatch(game::GameMode mode);

    // 当前这一局的模式配置（唯一入口）
    //
    //  ★ 为什么要一个访问函数而不是让调用方自己写 `modes[static_cast<int>(...)]`
    //    那个下标表达式一旦写错（忘了 cast、或者读的是全局 m_config 而不是
    //    这一局的 m_matchCfg），现象就是"改了模式没生效"或"读到别的模式的参数"，
    //    很难查。收成一个函数之后，全项目只有一处做这个转换。
    GameConfig::ModeConfig& activeModeConfig();

private:
    void processEvents();
    void update(float dt);
    void render();

    sf::RenderWindow m_window;
    std::vector<std::unique_ptr<Scene>> m_sceneStack;
    sf::Clock m_clock;
    float m_accumulator = 0.0f;
    GameConfig m_config;

    // ---- 第三层：这一局的配置（见上面 beginMatch 的说明）----
    game::GameMode m_matchMode = game::GameMode::ZoneRandom;
    GameConfig     m_matchCfg;   // 开局时从 m_config 拷一份
};
