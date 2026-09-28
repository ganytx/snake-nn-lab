#pragma once

// ============================================================================
//  GameScene.h — 对局场景：**表现层**
//
//  ★ 与上一版的根本区别（2026-09-22 抽 Arena）
//
//  上一版：GameScene 自己持有 agents / FoodManager / CollisionSystem /
//          PerceptionBuilder，并且**自己写了一遍"每帧干什么"**。
//          而 train_probe 里另有一份几乎一样的循环 —— 两份靠人工同步。
//
//  这一版：世界的推进全部交给 arena::Arena（见 src/arena/Arena.h）。
//          GameScene 只剩"看得见、听得见"的那部分：
//
//              渲染 · 输入采集 · 相机 · 小地图 · HUD · 观战 · 限时 · 结算
//
//  ★ 分工的判据：**「如果训练时也应该发生，它就不该在 GameScene 里」。**
//
//      世界推进（移动/碰撞/饥饿/尸体/感知/决策/食物）  → Arena
//      渲染、相机、小地图、HUD                          → GameScene
//      鼠标键盘 → HumanInput → HumanBrain               → GameScene 采集后喂进去
//      观战、限时、胜负、结算                           → GameScene
//
//  最后几条"留在 GameScene"是有意的边界。将来要把结算也搬进 Arena，
//  是给 Arena【加方法】，而不用重构调用方。
//
//  ★ 每帧流程（本文件负责的部分）
//
//      fixedUpdate:  m_arena.step()          ← 世界推进，唯一一份
//                    → 相机 / 小地图
//                    → 队伍统计 / 限时判定
// ============================================================================

#include "../core/Scene.h"
#include "../core/Game.h"
#include "../arena/Arena.h"
#include "../systems/Camera.h"
#include "../systems/Minimap.h"
#include "../brain/HumanBrain.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

class GameScene : public Scene {
public:
    // ★★ 现在必须指定【在玩哪个模式】（2026-09-27）
    //
    //  ★ 为什么模式不能在开局后再改
    //    模式决定了队伍规模、算法、食物规则、有没有时间限制 ——
    //    这些都是"建世界"的输入。开打之后再换模式等于重建整个世界，
    //    那不是"切换模式"而是"重开一局"，语义上应该由调用方重新 new 一个场景。
    //  ⚠️ 而 `rebuildArena()` 每次重开一局都会读它，所以它是【这一局的模式】，
    //     不是"当前全局模式"。
    explicit GameScene(Game* game, game::GameMode mode = game::GameMode::ZoneRandom);

    // 当前在玩哪个模式（HUD / 结算 / 重开一局时要用）
    game::GameMode mode() const { return m_mode; }

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // 把 GameConfig 翻译成 ArenaConfig，然后重建一局
    void rebuildArena();

    // 把鼠标键盘翻译成 arena::HumanInput 交给玩家的 HumanBrain
    void pushHumanInput();

    // -----------------------------------------------------------------------
    //  ★ 取玩家那个脑（HumanBrain*），没有则返回 nullptr
    //
    //  ★★ 为什么是"按索引查"而不是缓存一个裸指针（2026-09-27 改）
    //    以前这里缓存了 `m_humanBrain`。而无限模式会调用 `Arena::addAgent()`
    //    往 `m_agents` 里加蛇 —— **vector 一扩容，缓存的指针就悬垂了**。
    //    那会表现成"玩着玩着鼠标控制突然失灵"或者更糟的越界写。
    //    （Arena 内部已经 `reserve(32)` 让扩容不会发生，但那是 Arena 的实现
    //      细节，GameScene 不该依赖它 —— 依赖了就是把两个类的内部假设绑死。）
    //
    //  ★ 代价：每次调用做一次 dynamic_cast。一帧一次，可忽略。
    // -----------------------------------------------------------------------
    arena::HumanBrain* humanBrain();

    // 玩家那条蛇死了就转观战。
    // ★ 用【查询】而不是"在死亡处理里判断" —— 死亡处理已经搬进 Arena，
    //    GameScene 不该再去插手世界内部的状态机。
    void checkSpectatorSwitch();

    // -----------------------------------------------------------------------
    //  ★★ 无限模式：定时加入新蛇（2026-09-27 新增）
    //
    //  ★ 为什么在 GameScene 而不是 Arena
    //    Arena 只提供"加一条蛇"的机制（`addAgent`）；**什么时候加、加几条、
    //    用什么权重** 是【玩法】，属于上层。放进 Arena 就等于把"无限模式"
    //    这个概念塞进世界层 —— 那正是分层要避免的。
    //
    //  ★ 只在 Infinite 模式生效（见实现里的模式判断）
    // -----------------------------------------------------------------------
    void updateSpawning(float dt);

    // 统计每队存活数 / 总长度 / 击杀数
    void updateTeamStats();
    int  winningTeam() const;

    // -----------------------------------------------------------------------
    //  ★★ 无限模式的排行榜：按【单条蛇】排名，并显示它的神经网络名字
    //     （负责人 2026-09-27 要求）
    //
    //  ★ 负责人原话
    //      "然后旁边的排行榜改一下，不要队伍，要具体的蛇的排名，
    //        然后把蛇的神经网络名字也显示出来。"
    //
    //  ★ 为什么不能复用 m_teamStats
    //    在无限模式里"队"只是"每条蛇的身份槽位"（`snakesPerTeam == 1`），
    //    而且蛇数上限 20 > `MAX_TEAMS` 8 → **队号会重复**。
    //    按队聚合会把两条不同的蛇合成一行，排名直接是错的。
    //    所以必须**逐条蛇**一行，用 agents 的下标当身份（它在本局内稳定且唯一）。
    // -----------------------------------------------------------------------
    struct SnakeEntry {
        int         index  = 0;      // agents 下标（本局内唯一的身份）
        int         teamId = -1;     // 只为取颜色
        bool        alive  = false;
        int         length = 0;      // 体节数
        int         kills  = 0;
        std::wstring brain;          // 神经网络名字 / 规则 AI / 你
    };
    std::vector<SnakeEntry> snakeEntries() const;

    void renderHud(sf::RenderWindow& window);
    void renderTeamPanel(sf::RenderWindow& window);
    void renderSnakePanel(sf::RenderWindow& window);   // ★ 无限模式专用
    void renderDebug(sf::RenderWindow& window);
    void renderMatchOver(sf::RenderWindow& window);

    // ---- 世界（唯一一份，在 Arena 里）----
    arena::Arena m_arena;

    // ★ 这一局在玩哪个模式（构造时定，之后只读）
    game::GameMode m_mode = game::GameMode::ZoneRandom;

    // ---- 表现层 ----
    Camera  m_camera;
    Minimap m_minimap;

    // 玩家那个大脑的裸指针（不拥有）。
    //
    // ⚠️⚠️ 2026-09-27：**不要再缓存它**。无限模式会往 m_agents 里加蛇，
    //     而 vector 扩容会让这个指针悬垂。改用上面的 `humanBrain()`
    //     （每次按索引现查）。这个成员保留只是为了让旧代码能编译，
    //     新代码一律用 humanBrain()。
    arena::HumanBrain* m_humanBrain = nullptr;

    // ---- 无限模式的生蛇状态（只有 Infinite 模式用）----
    //
    // ★★ 规则（负责人 2026-09-27 纠正后定稿）
    //
    //      · 存活数上限 `m_spawnMaxAlive`（20）—— 到了就完全不加
    //      · 存活数 **< `m_spawnFastBelow`（8）** → 每 `m_spawnIntervalFast`（2）秒补一条
    //      · 否则                            → 每 `m_spawnIntervalNormal`（7）秒补一条
    //      · 死掉的蛇**从场上移除**（`Arena::removeDeadAgents`），不占名额
    //
    //  ★★ 我上一版把语义搞反了（负责人原话）
    //      "无限模式你理解错了，存活上限不是 8，是20！只是未满8 速度加快而已！"
    //
    //    我做成了"到 8 就停手"，于是世界上永远只有 8 条蛇，
    //    而"上限 20"被我拿去当成了"累计生成上限"—— 那个概念是我自己
    //    发明出来的，负责人从来没要过。
    //
    //    **两个数各管一件事，别混：**
    //        20 = 上限（到顶就停）
    //         8 = 加速门槛（只决定"用 2 秒还是 7 秒的节奏"）
    //    世界会一路涨到 20 条并稳定在那儿。
    //
    //  ★ 间隔的改动史：20s → 8s → 7s → 2s →（本次）**2s / 7s 双节奏**
    //    单节奏时代的问题是"不管场上还有几条活的，都按同一个节奏加"；
    //    现在按存活数在快慢两档之间切换。
    //
    // ⚠️ 这四个数目前仍写在代码里，设置页还没有对应的行。
    float m_spawnTimer = 0.0f;              // 距下次补蛇还有多久
    int   m_spawnMaxAlive       = 20;       // ★ 存活数上限（到顶停手）
    int   m_spawnFastBelow      = 8;        // ★ 低于它就切快节奏
    int   m_spawnIntervalFast   = 2;        // ★ 快节奏间隔（秒）
    int   m_spawnIntervalNormal = 7;        // ★ 常规节奏间隔（秒）
    // 候选权重文件（开局时读一次 saves/ 列表）—— 新蛇从里面随机挑一份
    std::vector<std::string> m_spawnWeights;
    std::mt19937 m_spawnRng{ 20260927u };   // 专用随机数，不动全局 rand()
    float m_lastSpawnNotice = 0.0f;         // "新蛇加入"提示的剩余显示时间

    float m_mapW = 2000.0f;
    float m_mapH = 2000.0f;

    // ---- 对局状态（留在表现层）----
    float m_matchTimer = 0.0f;
    bool  m_matchOver  = false;
    float m_overTimer  = 0.0f;

    // ---- 观战（玩家那条蛇死了之后）----
    bool         m_spectator = false;
    float        m_spectatorZoom = 1.0f;
    sf::Vector2f m_spectatorCenter;

    // ---- 资源与 UI ----
    sf::Font    m_font;
    sf::Texture m_headTexture;
    bool        m_debugMode = false;
    bool        m_mouseAimReady = false;
    float       m_fps = 0.0f;
    int         m_frameCount = 0;
    sf::Clock   m_fpsClock;

    struct TeamStat { int alive = 0; int members = 0; int kills = 0; int length = 0; };
    std::vector<TeamStat> m_teamStats;
};
