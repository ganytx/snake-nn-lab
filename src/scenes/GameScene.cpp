#include "GameScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "PauseScene.h"
#include "MenuScene.h"
#include "../core/Game.h"
#include "../core/GameModes.h"     // ★ game::GameMode / applyModeParams
#include "../utils/Constants.h"
#include "../utils/Text.h"
#include "../brain/BrainRegistry.h"
#include "../brain/NeuralBrain.h"   // ★ brainDisplayName() 要 dynamic_cast 到它
#include "../rl/Checkpoint.h"       // ★ ckpt::list() —— 无限模式新生蛇随机挑权重

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <memory>
#include <sstream>

namespace {

// ⚠️ 队伍颜色原来定义在这里，已经搬进 src/arena/Arena.cpp ——
//    因为 **颜色是 Arena 造蛇时给的**（Agent 的构造函数要一个 sf::Color）。
//    渲染时读的是每条蛇自己的颜色，所以两处天然一致，不存在"小地图和实际
//    蛇颜色不一样"这类问题。

// ---------------------------------------------------------------------------
//  ★★ "这条蛇的脑子叫什么"（2026-09-27 新增，负责人要求显示神经网络名字）
//
//  ★ 负责人原话
//      "然后把蛇的神经网络名字也显示出来。"
//
//  ★ 为什么需要 dynamic_cast 而不能只读 Agent 的某个字段
//    `Agent` 只持有 `std::unique_ptr<arena::Brain>`（基类指针），
//    而"加载了哪份权重"是 `NeuralBrain` **独有**的信息。
//    本项目里"按具体类型取扩展信息"的既有做法就是 dynamic_cast
//    （见 `GameScene::humanBrain()`，它用同一招取 HumanBrain）。
//
//  ★ 为什么要做【一层显示名转换】而不是直接把 brain->info().displayName 画上去
//    `BrainInfo::displayName` 是"给人挑算法用"的说明文字
//    （"神经网络 (已加载权重)"），挂在排行榜上太长（10~14 个汉字），
//    20 条蛇会把面板撑爆。排行榜要的是**短且能区分**的名字 ——
//    所以这里取权重文件的**文件名**（`saves/喜欢转.txt` → `喜欢转`）。
//
//  ★ 四种情况分开标注，不合并
//    玩家 / 规则 AI / 神经网络 / 加载失败 分不清的话，
//    "排行榜上有个一直不动的蛇"就无从判断是哪种原因。
//    ⚠️ 尤其"未加载"必须显式写出来 —— 它表现为"这条蛇在乱走"，
//       而这正是本项目最该一眼看穿的状态（缺陷 #43/#49 都是这个形态）。
// ---------------------------------------------------------------------------
std::wstring brainDisplayName(const agent::Agent& a) {
    // 玩家优先：他挂的是 HumanBrain，和"用哪份权重"无关
    if (a.isHuman()) return std::wstring(L"你");

    const arena::Brain* b = a.brain.get();
    if (!b) return std::wstring(L"(无脑)");

    if (const auto* nb = dynamic_cast<const arena::NeuralBrain*>(b)) {
        if (!nb->weightsLoaded()) return std::wstring(L"NN·未加载");
        const std::string& p = nb->weightsPath();
        const std::size_t slash = p.find_last_of("\\/");
        std::string name = (slash == std::string::npos) ? p : p.substr(slash + 1);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".txt") == 0) {
            name.erase(name.size() - 4);
        }
        if (name.empty()) return std::wstring(L"NN");
        // 转宽字符：中文文件名在这里会走 UTF-8 → 宽字符（textutil 负责）
        return textutil::utf8ToWide(name);
    }

    // 其余算法（规则 AI 等）用注册表里的显示名
    const std::string& dn = b->info().displayName;
    if (!dn.empty()) return textutil::utf8ToWide(dn);
    return std::wstring(L"?");
}

}  // namespace

// ===========================================================================
//  把 GameConfig 翻译成 ArenaConfig，然后（重）建一局
//
//  ★ 这是 GameScene 与 Arena 之间【唯一】的配置接口。
//    Arena 不认得 GameConfig（那是 core/ 里的东西，还带着一堆跟世界无关的
//    窗口 / 字体 / UI 字段），GameScene 也不该知道 FoodManager 怎么配。
// ===========================================================================
void GameScene::rebuildArena() {
    // ★★ 读【这一局的配置】，不是全局配置（2026-09-27）
    //
    //  ★ 为什么
    //    三层配置里的第三层就是"这一局"。
    //    「模式选择页 → 设置」（小设置）只允许改这一层 ——
    //    如果这里读全局的 `getConfig()`，小设置改的东西就完全不起作用
    //    （而它又不会报错，只会表现为"改了没用"）。
    //
    //  ⚠️ 全局配置（`getConfig()`）只有两个正当用途：
    //      ① 主菜单的设置界面改它  ② 开新局时拷进这一局的配置。
    //    GameScene 里**不应该**再出现 `getConfig()`。
    auto& cfg = m_game->getMatchConfig();

    // ★ 兜一道：手改配置文件可以绕过设置界面的 min/max，
    //   而地图 20x20 这类退化配置会造出"看起来在跑、其实什么都不会发生"的世界。
    cfg.sanitize();

    m_mapW = cfg.mapWidth;
    m_mapH = cfg.mapHeight;

    arena::ArenaConfig ac;
    ac.mapWidth  = cfg.mapWidth;
    ac.mapHeight = cfg.mapHeight;

    // ★★ 队伍配置从【当前模式的那份配置】读
    //
    //  ★ 为什么不再读 cfg.teamCount / cfg.teamBrains
    //    现在有三个模式，每个模式有自己的队伍规模与算法（GameConfig::ModeConfig）。
    //    全局那份 cfg 只保留【通用世界规则】。
    //  ⚠️ `m_mode` 由构造函数传入（开局时就定了），这里不依赖全局状态 ——
    //     否则"改了模式但 GameScene 读到旧的"会是一个很难查的 bug。
    //
    // ⚠️⚠️ 这里**不能加 const**（2026-09-27 编译错误换来的教训）
    //     原先写的是 `const GameConfig::ModeConfig& mc = ...`，因为当时只往下
    //     【读】配置。后来新增的"无限模式开局随机挑权重"要**写**
    //     `mc.teamWeights[t]` —— 对 const 引用赋值会报
    //        no match for 'operator=' (operand types are 'const std::string'
    //        and 'std::string')
    //     这个错误的报错位置离"忘了去掉 const"很远，会白花时间。
    //     **加写操作时，先回头看这个引用是不是 const。**
    GameConfig::ModeConfig& mc = cfg.modes[static_cast<int>(m_mode)];

    // -----------------------------------------------------------------------
    //  ★★ 无限模式：开局那几条蛇也要"随机搭载存档里的神经网络"（负责人 2026-09-27）
    //
    //  ★ 负责人原话
    //      "而且所有蛇随机搭载 save 里的神经网络！"
    //    也就是【不只是】中途加入的蛇随机挑，**开局就在场上的每一条**都随机挑。
    //
    //  ★ 为什么必须在这两件事【之前】
    //    ① 必须在 `Arena::reset()` 之前 —— 造脑时才读 `teamWeights`。
    //    ② 必须在【下面那个 teamWeights 拷贝循环】之前 —— 那个循环把
    //       `mc.teamWeights` 拷进 `ac.teamWeights`，而 Arena 只认 `ac` 里那份。
    //    写在任意一处之后就等于没做，**而且不会报错**，只表现为"权重还是原来那个"。
    //    ⚠️ 所以这段代码的位置是**有讲究的**，不要往下挪。
    //
    //  ★ 为什么"已经是空的才填"
    //    模式设置页允许玩家逐条指定权重。玩家指定过就是有意的选择，
    //    不能被这里的随机覆盖 —— 那会变成"设了没用"（本项目最忌讳的形态）。
    //    留空 = 交给随机，正是无限模式的默认语义。
    //
    //  ★ 玩家那条蛇（队 0 第 0 条）挂的是 HumanBrain，权重对它没有意义。
    //    这里照样给它填一份 —— 因为"队 0 的权重"在无限模式下还可能
    //    被别的用途读到，填一份随机值比留一个空串更不容易出意外。
    // -----------------------------------------------------------------------
    if (m_mode == game::GameMode::Infinite) {
        // ★ 只在开局读一次存档列表：中途新存了档不会立刻生效，但那是小概率事件，
        //   而"每生一条蛇就扫一次目录"是没必要的磁盘 IO。
        //
        // ★★ 过滤掉"自动产物"（2026-09-27 负责人要求）
        //    `saves/` 里混着训练器每局覆盖写的两份文件
        //    （`auto_latest.txt` / `enemy_snapshot.txt`）——
        //    它们**不是"谁挑好的一份网络"**，而是"最新进度"。
        //    让它们参与随机抽，5 条蛇里就有两条抽到同一份，
        //    界面上看起来"随机"跟没做一样。
        //    判据在 `ckpt::isAutoGenerated()`（唯一一份）。
        //
        //  ⚠️ 只影响【随机挑选】，不影响 `nn-enemy` / `nn-mate` 读快照 ——
        //     那两个算法 id 走的是固定路径，和这里无关。
        m_spawnWeights.clear();
        for (const std::string& f : ckpt::list()) {
            if (ckpt::isAutoGenerated(f)) continue;
            m_spawnWeights.push_back(f);
        }
        m_spawnTimer = 0.0f;         // 距第一条新蛇的计时从 0 开始

        int filled = 0;
        if (!m_spawnWeights.empty()) {
            std::uniform_int_distribution<std::size_t> pick(0,
                m_spawnWeights.size() - 1);
            for (int t = 0; t < mc.effectiveTeamCount(); ++t) {
                if (!mc.teamWeights[t].empty()) continue;   // 玩家指定过 → 不动
                mc.teamWeights[t] = m_spawnWeights[pick(m_spawnRng)];
                ++filled;
            }
        }
        std::printf("[无限模式] 开局 %d 条蛇的权重：从 saves/ 里 %d 份存档中随机挑"
                    "（本次指定了 %d 条）\n",
                    mc.totalSnakes(),
                    static_cast<int>(m_spawnWeights.size()),
                    filled);
    }

    ac.teamCount     = mc.effectiveTeamCount();
    ac.snakesPerTeam = mc.effectiveSnakesPerTeam();
    ac.playerTeam    = cfg.playerTeam;
    for (int i = 0; i < arena::ArenaConfig::MAX_TEAMS; ++i) {
        ac.teamBrains[i]  = mc.teamBrains[i];
        ac.teamWeights[i] = mc.teamWeights[i];   // ★ 每队加载哪份权重
    }

    ac.snakeSpeed      = cfg.snakeSpeed;
    ac.snakeBoostSpeed = cfg.snakeBoostSpeed;
    ac.snakeTurnRate   = cfg.snakeTurnRate;
    ac.segmentSpacing  = cfg.segmentSpacing;
    ac.snakeInitSegs   = cfg.snakeInitSegs;
    ac.foodSegments    = cfg.foodSegments;
    ac.corpseSegments  = cfg.corpseSegments;

    ac.foodSpawnCount    = cfg.foodSpawnCount;
    ac.foodSpawnInterval = cfg.foodSpawnInterval;
    ac.corpseScore       = cfg.corpseScore;

    // ---- ★ 食物圈（玩法核心，见 docs/70-食物圈玩法设计.md）----
    //
    //  ★★ 2026-09-27：圈相关参数现在【由模式决定】
    //
    //    · 模式 1（随机食物圈）：2 个圈、位置每轮随机 —— 就是原来那套
    //    · 模式 2/3（无限 / 大混战）：`zoneCount == 0` → 食物全图随机撒
    //      （这是 FoodManager 里已经存在的退化路径，见它的 update()）
    //
    //  ⚠️ 以前这里把 `count = 2` 和 `randomPosition = true` 写死，
    //     所以三个模式没法区分。现在读 `cfg.zoneCount` —— 它由
    //     `game::applyModeParams()` 按模式写进 GameConfig，是唯一来源。
    //
    //  ★ 为什么"位置随机"仍然写死为 true（对模式 1 而言）
    //    固定四方位时蛇和玩家可以靠"背地图"取胜，位置随机之后
    //    "读当前圈/下一个圈在哪"才成为真正的能力。
    //    圈数为 0 时这个字段根本不被读（没有圈可放）。
    ac.zoneLayout.count           = cfg.zoneCount;
    ac.zoneLayout.randomPosition  = true;   // ★ 每轮随机挑一个新的合法圈心
    ac.zoneLayout.minDistRatio    = 0.40f;  // 与上一个圈的距离：短边的 40%~60%
    ac.zoneLayout.maxDistRatio    = 0.60f;  //   → 3000 图上 1200~1800px（跑 4.8~7.2 秒）
                                            //   ⚠️ 预告圈提前 8 秒出现（供食还剩 3s + 空窗 5s），
                                            //      所以最远 7.2 秒仍赶得上 —— 但余量只剩 0.8 秒。
                                            //      再往上调就会变成"永远在赶路、一口吃不上"。
    ac.zoneLayout.radiusRatio     = cfg.zoneRadiusRatio;

    // -----------------------------------------------------------------------
    //  ★★ 游戏：每局换一个【世界种子】（负责人要求，2026-09-27）
    //
    //    "训练模式，游戏模式请每局种子不同"
    //
    //  ★ 以前这里 `spawnSeed` 是 0（"不动全局随机序列"）+ 只给圈位一个
    //    `posSeedOffset`。后果：
    //      · 出生点 / 出生角度 / 食物 走全局 `rand()`，那是**一条跨局连续的流**，
    //        单独重开一局时它们**不重新随机**；
    //      · 圈的轮换顺序 `orderSeed` 没有偏移 ⇒ 顺序序列每局都一样。
    //
    //  ★ 现在整局由【一个种子】决定：出生 / 食物 / 圈轮换顺序 / 圈坐标
    //    全部来自它，而且 `advanceSeedOnReset = true` ⇒ 每 reset 一次换一个。
    //
    //  ★ 起始种子用 `std::clock()` 而不是 `rand()`
    //    `rand()` 是全局序列，在 reset() 之前调它会**改变世界的随机序列**
    //    （出生点、食物位置都抽它）→ replay_check 的世界指纹会对不上。
    //    时钟不碰全局序列，纯表现层的行为差异，不影响任何对拍。
    //
    //  ⚠️ `posSeedOffset` 现在不必再单独设了：`Arena::reset` 会把
    //     `zoneLayout.posSeed` 对齐到本局的世界种子。留着它反而会让
    //     "圈的位置"和"世界的种子"来自两个不同的源。
    // -----------------------------------------------------------------------
    ac.spawnSeed          = static_cast<unsigned>(std::clock());
    ac.advanceSeedOnReset = true;   // ★ 每局换一个

    ac.zoneSchedule.spawnDuration = cfg.zoneSpawnSeconds;
    ac.zoneSchedule.gapDuration   = cfg.zoneGapSeconds;
    ac.foodPerSnakePerSecond      = mc.foodPerSnake;   // ★ 每蛇速率（模式决定）
    ac.foodLifetime               = cfg.foodLifetime;
    ac.corpseLifetime             = cfg.corpseLifetime;
    // ★★ 开局存量与存量上限（2026-09-27 新增，负责人要求）
    //    · 模式 2 无限模式：开局 200 个、上限 400 个
    //    · 模式 3 大混战  ：0 个、上限 60 个（食物稀缺，靠尸体）
    //    · 模式 1 有圈    ：都是 0（食物由圈逐秒产）
    ac.initialFoodCount           = mc.initialFoodCount;
    ac.foodMaxCount               = mc.foodMaxCount;

    ac.starvationTime   = cfg.starvationTime;
    ac.collisionForgive = cfg.collisionForgive;
    ac.friendlyFire     = cfg.friendlyFire;

    ac.visionCellSize   = 32.0f;      // 32 格 x 32px = 视野 1024px
    ac.allyVisibleCount = cfg.allyVisibleCount;

    m_arena.reset(ac);

    // ---- 玩家那条蛇的头部贴图（表现层的东西，Arena 不该知道）----
    //
    // ★ 2026-09-27：这里**不再缓存 HumanBrain 指针**
    //   （无限模式会 addAgent → vector 可能扩容 → 缓存指针悬垂）。
    //   改成每次用 humanBrain() 现查。这里只负责贴图。
    m_humanBrain = nullptr;          // 旧成员，保留仅为兼容；新代码用 humanBrain()
    const int hi = m_arena.humanIndex();
    if (hi >= 0 && m_headTexture.getSize().x > 0) {
        agent::Agent& me = m_arena.agents()[static_cast<std::size_t>(hi)];
        me.body.setHeadTexture(m_headTexture);
    }

    // ---- 无限模式：存档列表已在上面读过（见 rebuildArena 前半段的说明）----
    //
    // ⚠️ 不要在这里再读一次 `ckpt::list()`。那不但多一次磁盘 IO，
    //    更要紧的是会和上面那份"开局随机挑权重"用到的列表**不是同一个快照** ——
    //    中间要是有人存了新档，两边就会不一致，而现象是"某条蛇的权重来源不明"。

    // ---- 队伍统计容器 ----
    m_teamStats.assign(static_cast<std::size_t>(m_arena.config().effectiveTeamCount()),
                       TeamStat{});

    // ---- 观战起点：玩家（或第 0 条蛇）当前的位置 ----
    const auto& agents = m_arena.agents();
    if (!agents.empty()) {
        const std::size_t focus = (hi >= 0) ? static_cast<std::size_t>(hi) : 0u;
        m_spectatorCenter = agents[focus].body.getHeadPosition();
    }
}

// ===========================================================================
GameScene::GameScene(Game* game, game::GameMode mode)
    : Scene(game)
    , m_arena()      // 先建一个空 Arena，下面 rebuildArena() 会按配置真正建局
    , m_mode(mode)
    // ★ 相机/小地图的初始尺寸也读【这一局的配置】
    //
    //  ⚠️ 以前这里读的是 `game->getConfig()`（全局）。值恰好相同（两个配置的
    //     地图都是 3000²），所以看不出问题 —— 但这是"潜在不一致"：
    //     一旦哪天地图尺寸按模式不同，初始相机就会用错尺寸，
    //     而现象是"开局视角不对、按一下 R 就好了"，极难查。
    //     判据：GameScene 里凡是读配置，一律 getMatchConfig()。
    , m_camera(game->getMatchConfig().mapWidth, game->getMatchConfig().mapHeight,
               Constants::WINDOW_WIDTH, Constants::WINDOW_HEIGHT)
    , m_minimap(game->getMatchConfig().mapWidth, game->getMatchConfig().mapHeight,
                Constants::MINIMAP_WIDTH, Constants::MINIMAP_HEIGHT)
{
    // ---- 字体 / 贴图（必须在 rebuildArena 之前，那里要用 m_headTexture）----
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "GameScene");
    }
    m_headTexture.loadFromFile("assets/picture/cathead.png");

    // ★ 世界只在这里建一次，之后"重开一局"也走同一个函数 ——
    //   不会出现"开局那局"和"按 R 重开那局"配置不一样的情况。
    rebuildArena();
}

// ===========================================================================
void GameScene::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::MouseMoved ||
        event.type == sf::Event::MouseButtonPressed) {
        m_mouseAimReady = true;
    }
    if (event.type != sf::Event::KeyPressed) return;

    switch (event.key.code) {
        case sf::Keyboard::F3:
            m_debugMode = !m_debugMode;
            break;

        case sf::Keyboard::R: {
            // 重开一局：按同样的配置重建
            //
            // ★ 以前这里手写了十行"重建食物管理器 / 碰撞系统 / 相机 / 小地图"，
            //   和构造函数里的那段是【两份】，很容易改一边忘一边。
            //   现在统一走 rebuildArena()。
            rebuildArena();

            m_camera  = Camera(m_mapW, m_mapH, Constants::WINDOW_WIDTH, Constants::WINDOW_HEIGHT);
            m_minimap = Minimap(m_mapW, m_mapH, Constants::MINIMAP_WIDTH, Constants::MINIMAP_HEIGHT);

            m_matchTimer = 0.0f;
            m_matchOver  = false;
            m_overTimer  = 0.0f;
            m_spectator  = false;
            break;
        }

        case sf::Keyboard::Escape:
            m_game->pushScene(std::make_unique<PauseScene>(m_game));
            break;

        default: break;
    }
}

// ===========================================================================
//  把鼠标键盘翻译成 HumanInput
//
//  屏幕像素 → 世界坐标的换算必须用【渲染时那一个 View】，
//  否则镜头一移动，蛇头就指偏了。
// ===========================================================================
void GameScene::pushHumanInput() {
    arena::HumanBrain* brain = humanBrain();
    if (!brain || m_arena.humanIndex() < 0) return;

    arena::HumanInput in;
    in.boost     = sf::Keyboard::isKeyPressed(sf::Keyboard::Space) ||
                   sf::Mouse::isButtonPressed(sf::Mouse::Left) ||
                   sf::Keyboard::isKeyPressed(sf::Keyboard::LShift);
    in.turnLeft  = sf::Keyboard::isKeyPressed(sf::Keyboard::Left);
    in.turnRight = sf::Keyboard::isKeyPressed(sf::Keyboard::Right);

    sf::RenderWindow& window = m_game->getWindow();
    const sf::Vector2i pixel = sf::Mouse::getPosition(window);
    const int w = static_cast<int>(window.getSize().x);
    const int h = static_cast<int>(window.getSize().y);

    // 观战时用自由摄像机，瞄准也要用同一个 View
    sf::View view = m_camera.getView();
    if (m_spectator || m_arena.humanIndex() < 0) {
        view.setCenter(m_spectatorCenter);
        view.setSize(Constants::WINDOW_WIDTH * m_spectatorZoom,
                     Constants::WINDOW_HEIGHT * m_spectatorZoom);
    }

    // 玩家还没动过鼠标 → 不启用瞄准，避免开局朝窗口角落冲出去
    if (m_mouseAimReady && pixel.x >= 0 && pixel.y >= 0 && pixel.x < w && pixel.y < h) {
        const sf::Vector2f world = window.mapPixelToCoords(pixel, view);
        in.hasAim = true;
        in.aimX = world.x;
        in.aimY = world.y;
    }

    brain->setInput(in);
}

// ===========================================================================
//  玩家那条蛇死了 → 转观战
//
//  ★ 旧版是在 Arena 的"死亡 → 尸体"循环里顺手判断下标，那是 GameScene
//    伸手进世界内部的状态机。现在死亡处理归 Arena，GameScene 只【查询】：
//    "玩家那条蛇还活着吗？"
//
//  行为与旧版一致 —— 旧版也是在同一步（step 内部）把 m_spectator 置 true，
//  所以相机在那一步就已经按观战分支走了。
// ===========================================================================
void GameScene::checkSpectatorSwitch() {
    if (m_spectator) return;

    const int hi = m_arena.humanIndex();
    if (hi < 0) return;   // 没有玩家（纯 AI 对局）→ 本来就走观战分支

    const agent::Agent& me = m_arena.agents()[static_cast<std::size_t>(hi)];
    if (!me.isAlive()) {
        m_spectator = true;
        m_spectatorCenter = me.body.getHeadPosition();
    }
}

// ===========================================================================
//  固定步长更新
//
//  ★ 世界推进只剩这一行：m_arena.step(dt)
//
//    "每帧干什么"（移动 → 碰撞 → 饥饿 → 尸体 → 感知 → 决策 → 食物）
//    现在只有一份实现，在 src/arena/Arena.cpp 里。
//    游戏和训练器都调它 —— 这样才不可能出现"训练世界和游戏世界不一致"。
//
//    本函数剩下的是纯表现层：相机、小地图、观测玩家死亡、限时判定。
// ===========================================================================
// ===========================================================================
//  ★ 取玩家那个脑（每次现查，不缓存 —— 见头文件里的说明）
// ===========================================================================
arena::HumanBrain* GameScene::humanBrain() {
    const int hi = m_arena.humanIndex();
    if (hi < 0) return nullptr;
    const std::size_t i = static_cast<std::size_t>(hi);
    if (i >= m_arena.agents().size()) return nullptr;
    return dynamic_cast<arena::HumanBrain*>(m_arena.agents()[i].brain.get());
}

// ===========================================================================
//  ★★ 无限模式：定时加入新蛇
//
//  ★ 玩法（负责人定的）
//      · 初始 5 条（1 条玩家 + 4 条 AI）—— 由模式设置里的"初始蛇数"决定
//      · **存活数上限 20**；**未满 8 时用 2 秒的快节奏补**，8 条以上用 7 秒的常规节奏
//      · 新蛇的脑子：从存档里【随机挑一份权重】的神经网络
//        ⚠️ 开局在场的那几条也一样随机 —— 那一步在 rebuildArena() 里，
//           因为必须在 Arena::reset() 造脑【之前】把权重写进这一局的配置。
//      · **没有组队**：每条蛇自己占一队，所有蛇互为敌人
//      · 死掉的蛇会被 `Arena::removeDeadAgents()` **从场上移除**，
//        所以"场上蛇数"就是真实存活数 —— 不会再出现"尸体占着名额"
//
//  ★★ 两个节奏的语义（负责人 2026-09-27 纠正过一次，务必别搞反）
//
//  ★ 负责人原话
//      "无限模式你理解错了，存活上限不是 8，是20！只是未满8 速度加快而已！"
//
//  ★ 我上一版理解错的形态
//    把它做成了"存活数到 8 就**停手**"—— 于是世界上永远只有 8 条蛇，
//    而"上限 20"根本没有意义（那个 20 被我拿去当"累计生成上限"了，
//    那是我自己发明的东西，不是负责人要的）。
//
//  ★ 正确的语义（两个数各管一件事）
//      20 = 存活数上限。到了就完全不加。
//       8 = **加速门槛**。低于它 → 2 秒补一条；否则 → 7 秒补一条。
//    也就是说 8 不是"目标"，它只决定"用哪个节奏"。
//    世界最终会一路涨到 20 条并稳定在那儿。
//
//  ★ 为什么上限判的是 `alive`（存活数）而不是 `agents().size()`
//    因为无限模式每帧都会 `removeDeadAgents()` 把尸体清掉，
//    所以两者此时是相等的；但**判存活数才是语义正确的那个** ——
//    万一以后移除逻辑又被改动，判存活数不会突然变成"尸体占名额"。
//
//  ★ 旧实现的另一个错
//    判的是 `agents().size()`（**含尸体**）→ 累计生过 20 条之后
//    永久卡死，世界再也不出新蛇。负责人报的"蛇死了仍然占有 20 条上限"就是它。
void GameScene::updateSpawning(float dt) {
    // 只有无限模式会"不断加蛇"
    if (m_mode != game::GameMode::Infinite) return;

    // ---- 存活数：既是上限判据，也是节奏判据 ----
    int alive = 0;
    for (const auto& a : m_arena.agents()) if (a.isAlive()) ++alive;

    // 到上限 → 不再加（计时器停在 0，满了就没有"下一条"可言）
    if (alive >= m_spawnMaxAlive) { m_spawnTimer = 0.0f; return; }

    // ★ 选节奏：少于 8 条用快节奏
    const int interval = (alive < m_spawnFastBelow) ? m_spawnIntervalFast
                                                    : m_spawnIntervalNormal;

    m_spawnTimer += dt;
    if (m_spawnTimer < static_cast<float>(interval)) return;
    m_spawnTimer = 0.0f;

    // ---- 挑一份权重（随机）----
    //
    // ★ 用"先进 nn 工厂 + setPendingWeights"这条既有链路，不另开一条
    //   权重加载路径（那会变成第二份实现）。
    std::string weights;
    if (!m_spawnWeights.empty()) {
        std::uniform_int_distribution<std::size_t> pick(0, m_spawnWeights.size() - 1);
        weights = m_spawnWeights[pick(m_spawnRng)];
    }

    // ---- 队号：队 0 留给玩家，其余轮转 ----
    //
    //  ★ 为什么队号要"从 1 开始轮转"
    //    队 0 永远是玩家。如果新蛇也用队 0，它会和玩家【同队】——
    //    而同队之间可以互相穿过、不会撞死，那就不是"互为敌人"了。
    //    所以新蛇只在队 1..MAX-1 之间轮转。
    //
    //  ⚠️ 队数上限是 8，而在场蛇数可以到 20 —— 所以队号一定会重复。
    //     在"没有组队"的语义下这无害：队号只是身份槽位，
    //     真正决定"会不会互相撞死"的是 `friendlyFire` + 队号是否相同。
    //     **重复队号的两条 AI 蛇会被当成友军（可穿过）** —— 这是这个实现的
    //     已知取舍。要彻底消除它就得把 MAX_TEAMS 提到 ≥20，
    //     那会影响 ArenaConfig 的数组大小、观测里的 8 个圈槽位等一串东西。
    //
    //  ★ 队号轮转用【存活数】就够，不需要"累计生成数"
    //    尸体每帧被移除，所以存活数不会因为"死了还在"而虚高；
    //    而队号只要在**当前在场的这几条**之间不撞车即可 ——
    //    死掉的蛇已经不在场上了，它用过哪个队号没有意义。
    const int span = arena::ArenaConfig::MAX_TEAMS - 1;   // 1..MAX-1
    const int teamId = 1 + (alive % (span > 0 ? span : 1));

    // ---- 让 Arena 按这个队号 + 这份权重造一条神经网络蛇 ----
    //
    // ★ 做法：把权重临时写进**这一局的配置**里那一队的 teamWeights，
    //   再调 addAgent —— 这样权重走的还是 Arena 里那唯一一条加载路径
    //   （`Arena::addAgent` 会读 `m_cfg.teamWeights[teamId]`）。
    //
    // ⚠️ 这会**覆盖那一队的默认权重**。在无限模式下可以接受：
    //    每条蛇自己一队、且队号会轮转复用，"某队的权重"本来就只是
    //    "这条蛇用哪份权重"的载体。
    {
        GameConfig::ModeConfig& mc = m_game->getMatchConfig()
                                        .modes[static_cast<int>(m_mode)];
        if (teamId >= 0 && teamId < GameConfig::MAX_TEAMS) {
            mc.teamWeights[teamId] = weights;
            mc.teamBrains[teamId]  = "nn";
        }
    }

    const int idx = m_arena.addAgent("nn", teamId);
    if (idx < 0) return;

    // 新蛇加入的提示（给人看的；控制台在 WIN32 子系统下一般看不到）
    m_lastSpawnNotice = 2.5f;
}

// ===========================================================================
void GameScene::fixedUpdate(float dt) {
    // ★★ 读【这一局的配置】（不是全局配置）—— 见 rebuildArena 里的说明
    auto& cfg = m_game->getMatchConfig();

    // 对局结束后世界冻结（显示结算界面）。
    // ★ 冻结 = 不调 step()，所以 Arena 里不需要 matchOver 状态。
    if (m_matchOver) {
        m_overTimer += dt;
        return;
    }

    m_matchTimer += dt;

    const bool wasOver = m_matchOver;
    (void)wasOver;

    // ---- 1~6. 世界推进（唯一一份）----
    m_arena.step(dt);

    // ---- 观测玩家是否已死 → 转观战 ----
    //
    // ★★ 顺序是硬要求（2026-09-27）：**必须在移除尸体之前**
    //
    //  无限模式现在会调用 `removeDeadAgents()` 把死蛇从场上删掉，
    //  而玩家那条蛇一死就会被删 → `humanIndex()` 变成 -1。
    //  如果把这一步放在移除之后，玩家死了就**永远不会切到观战**，
    //  相机会卡在最后一帧的位置不动 —— 现象是"我死了，画面定住了"。
    //  （`checkSpectatorSwitch()` 里 `hi < 0` 是直接 return 的，
    //    它把"没有玩家"当成"纯 AI 对局"，那个分支在游戏模式下不该发生。）
    //
    // ★ 这里用【查询】而不是"在死亡处理里判断"：
    //   死亡处理已经搬进 Arena，GameScene 不该再去插手世界内部的状态机。
    checkSpectatorSwitch();

    // ---- 移除已经死掉的蛇 ----
    //
    // ★ 为什么放在 step 之后、updateSpawning 之前
    //   · 必须在 step 之后：本操作会重建 Arena 里的指针表（m_ptrs），
    //     而 step 内部全程依赖它 —— 中途改会让同一步前后读到不一致的表。
    //   · 必须在 updateSpawning 之前：生蛇的上限判的是**场上蛇数**，
    //     先清掉尸体才能立刻补上（否则"死了还占名额"那个 bug 就回来了）。
    //
    // ★ 为什么只在无限模式做
    //   模式 1/3 靠"只剩一队"分胜负，而"某队全灭"这件事需要那条蛇
    //   **留在场上被统计到**（`winningTeam()` 读的是 agents 里的 teamId）。
    //   删掉尸体不会改变"哪队活着"，但会让"刚死那一下"的统计数据少一条 ——
    //   没必要为无限模式的需求去动另外两个模式的结算。
    if (m_mode == game::GameMode::Infinite) m_arena.removeDeadAgents();

    // ---- 无限模式：按间隔加入新蛇 ----
    //
    // ★ 放在 step 之后：新蛇会从【下一帧】开始参与世界推进
    //   （addAgent 只是把它加进 agents 列表，不推进世界）。
    updateSpawning(dt);
    if (m_lastSpawnNotice > 0.0f) m_lastSpawnNotice -= dt;

    // ---- 7. 相机 ----
    const int hi = m_arena.humanIndex();
    if (m_spectator || hi < 0) {
        // 观战：自由摄像机（四个方向键漫游、PgUp/PgDn 缩放）
        const float speed = 500.0f * m_spectatorZoom;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  m_spectatorCenter.x -= speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) m_spectatorCenter.x += speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))    m_spectatorCenter.y -= speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))  m_spectatorCenter.y += speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::PageUp))
            m_spectatorZoom = std::max(0.3f, m_spectatorZoom - 0.6f * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::PageDown))
            m_spectatorZoom = std::min(4.0f, m_spectatorZoom + 0.6f * dt);

        // 别飘到地图外面去
        m_spectatorCenter.x = std::max(0.0f, std::min(m_mapW, m_spectatorCenter.x));
        m_spectatorCenter.y = std::max(0.0f, std::min(m_mapH, m_spectatorCenter.y));
    } else {
        agent::Agent& me = m_arena.agents()[static_cast<std::size_t>(hi)];
        m_camera.setTarget(me.body.getHeadPosition());
        m_camera.setFollowSpeed(me.body.isBoosting() ? Constants::CAMERA_BOOST_FOLLOW
                                                     : Constants::CAMERA_NORMAL_FOLLOW);
        m_camera.update(dt);
    }
    // ⚠️ 2026-09-27：这里原来还有【第二份】相机代码（L325~347），
    //    与上面那段逐字重复 —— 是两次编辑叠加留下的。已删除。
    //    **判据**：同一段逻辑出现两次时，改一处忘一处就会表现为
    //    "观战镜头有时能动、有时不能"，很难查。

    // ---- 小地图 ----
    // ★ Minimap 是"update 填数据 / render 画"两段式的，
    //   漏掉 update 的话它会一直画一份从未被填充过的数据
    //   （绿点停在原点、没有红点、视野框为空）——看起来就像"小地图坏了"。
    {
        sf::View miniView = m_camera.getView();
        if (m_spectator || m_arena.humanIndex() < 0) {
            miniView.setCenter(m_spectatorCenter);
            miniView.setSize(Constants::WINDOW_WIDTH * m_spectatorZoom,
                             Constants::WINDOW_HEIGHT * m_spectatorZoom);
        }
        const std::size_t focus =
            (m_arena.humanIndex() >= 0) ? static_cast<std::size_t>(m_arena.humanIndex()) : 0u;
        if (!m_arena.agents().empty()) {
            // ★ 把食物圈信息也传进小地图：符号标位置、颜色标活跃状态。
            //   这是【给人看】的 —— AI 拿的是 Perception::zones，与此无关。
            //
            //  ★ 第 5 个参数从 `spawning` 改成 `previewing`（2026-09-27）：
            //    现在决定"标不标下一个圈"的是**预告期**，不是供食期。
            //    见 ZoneSchedule::previewing 的说明。
            m_minimap.update(m_arena.agents()[focus].body,
                             m_arena.snakePtrs(), miniView,
                             &m_arena.food().zones(),
                             m_arena.activeZone(),
                             m_arena.zoneSchedule().previewing(
                                 m_arena.food().zoneTimer()));
        }
    }

    // ---- 9. 队伍统计 + 胜负判定 ----
    updateTeamStats();

    // ★★ 2026-09-27：**无限模式永不结束**（负责人明确："永不结束，只能主动退出"）
    //
    //  ★ 原来这里无条件判 `timeUp || oneTeamLeft`，于是无限模式会在
    //    "只剩一队活着"时弹出获胜结算 —— 而无限模式【没有队伍】，
    //    "只剩一队"只是"别的蛇都死了"，世界本该继续（新蛇会不断加入）。
    //    负责人报的"无限模式仍存在获胜界面"就是它。
    //
    //  ★ 所以：无限模式在【统计之后、判定之前】直接返回。
    //    统计仍然要做 —— HUD 和队伍面板要读它。
    //    玩家退出方式是 ESC；玩家那条蛇死了会转观战（见 checkSpectatorSwitch）。
    if (m_mode == game::GameMode::Infinite) return;

    // 限时（三个模式都是 0 = 不限时，所以这一条目前恒为假；
    //       保留它是因为以后可能给某个模式加回限时）
    const bool timeUp = (cfg.matchTimeLimit > 0.0f && m_matchTimer >= cfg.matchTimeLimit);

    // 只剩一队有活蛇
    const bool oneTeamLeft = (winningTeam() >= 0);

    // -----------------------------------------------------------------------
    //  ★★ 结束条件【按模式不同】（2026-09-27）
    //
    //  ★ 负责人报："大混战和无限模式完全混淆 … 都无限时间，都仍然存在获胜界面"
    //
    //  ★ 两者的【正确】行为（负责人指定）
    //      模式 2 无限模式：**永不结束**，只能主动退出（上面已 early return）
    //      模式 3 大混战  ：该有胜负 —— 但"只剩一队"这一天在 15 条蛇里
    //                       几乎到不了（大家会先饿死），所以必须再补一个
    //                       更早、更符合这个模式的结束条件：**场上没有蛇了**。
    //
    //  ★ 为什么用"没有活蛇"而不是"只剩一队"
    //    大混战没有自然食物、尸体是唯一的粮 —— 打到后面是**集体饿死**。
    //    最后一条蛇饿死那一刻，场上一条都不剩，"只剩一队"永远不成立
    //    （0 队活着时 winningTeam() 返回 -1），对局就会无限空转。
    //    "没有活蛇"能抓住这个真实结局。
    //
    //  ★ 为什么还要带 `m_spectator` 条件
    //    玩家还活着的时候不该弹结算 —— 玩家正玩着，凭什么结束？
    //    （模式 3 里玩家那条蛇也算一条，所以"没有活蛇"必然晚于玩家死亡。）
    // -----------------------------------------------------------------------
    int aliveNow = 0;
    for (const auto& a : m_arena.agents()) if (a.isAlive()) ++aliveNow;
    const bool wipedOut = (aliveNow == 0);

    if (timeUp || oneTeamLeft || (m_spectator && wipedOut)) {
        m_matchOver = true;
        m_overTimer = 0.0f;
        for (auto& a : m_arena.agents()) a.brain->onMatchEnd();
    }
}

// ===========================================================================
void GameScene::update(float dt) {
    pushHumanInput();

    ++m_frameCount;
    if (m_fpsClock.getElapsedTime().asSeconds() >= 0.5f) {
        m_fps = static_cast<float>(m_frameCount) / m_fpsClock.restart().asSeconds();
        m_frameCount = 0;
    }
}

// ===========================================================================
void GameScene::updateTeamStats() {
    for (auto& s : m_teamStats) { s.alive = 0; s.length = 0; s.kills = 0; }
    for (const auto& a : m_arena.agents()) {
        const int t = a.teamId;
        if (t < 0 || t >= static_cast<int>(m_teamStats.size())) continue;
        TeamStat& s = m_teamStats[static_cast<std::size_t>(t)];
        s.kills += a.body.getKills();
        if (a.isAlive()) {
            ++s.alive;
            s.length += a.body.getSegmentCount();
        }
    }
}

// 只剩一支队伍还有活蛇时返回它的编号；否则返回 -1
int GameScene::winningTeam() const {
    int aliveTeams = 0, last = -1;
    for (std::size_t t = 0; t < m_teamStats.size(); ++t) {
        if (m_teamStats[t].alive > 0) { ++aliveTeams; last = static_cast<int>(t); }
    }
    if (aliveTeams <= 1 && m_teamStats.size() > 1) return last;
    return -1;
}

// ===========================================================================
//  ★★ 无限模式排行榜的数据：逐条蛇一行（负责人要求"不要队伍，要具体的蛇"）
//
//  ★ 排序规则（就是"排名"的定义，必须写清楚，否则没法判断对不对）
//      ① 活着的排前面           —— 死蛇不该压在活蛇上面
//      ② 同状态按【体节数】降序 —— 长度是贪吃蛇的实力体现
//      ③ 再按击杀数降序
//      ④ 最后按 index 升序      —— 保证顺序**稳定**，不会每帧乱跳
//
//  ★ 为什么第 ④ 条是必须的
//    没有它的话，两条长度相同的蛇会因为 `std::sort` 不稳定而
//    **每帧互换位置** —— 面板上看起来像在闪，实际是排序没定序。
//    （这是个"看起来像渲染 bug、其实是排序缺 tie-break"的经典形态。）
// ===========================================================================
std::vector<GameScene::SnakeEntry> GameScene::snakeEntries() const {
    std::vector<SnakeEntry> out;
    const auto& agents = m_arena.agents();
    out.reserve(agents.size());

    for (std::size_t i = 0; i < agents.size(); ++i) {
        const agent::Agent& a = agents[i];
        SnakeEntry e;
        e.index  = static_cast<int>(i);
        e.teamId = a.teamId;
        e.alive  = a.isAlive();
        e.length = e.alive ? a.body.getSegmentCount() : 0;
        e.kills  = a.body.getKills();
        e.brain  = brainDisplayName(a);
        out.push_back(std::move(e));
    }

    std::sort(out.begin(), out.end(),
              [](const SnakeEntry& a, const SnakeEntry& b) {
                  if (a.alive  != b.alive)  return a.alive;      // ① 活的在前
                  if (a.length != b.length) return a.length > b.length;  // ② 长的在前
                  if (a.kills  != b.kills)  return a.kills  > b.kills;   // ③ 杀多的在前
                  return a.index < b.index;                      // ④ 定序，防抖
              });

    return out;
}

// ===========================================================================
//  渲染
// ===========================================================================
void GameScene::render(sf::RenderWindow& window) {
    const sf::View defaultView = window.getView();

    if (m_spectator || m_arena.humanIndex() < 0) {
        sf::View v = m_camera.getView();
        v.setCenter(m_spectatorCenter);
        v.setSize(Constants::WINDOW_WIDTH * m_spectatorZoom,
                  Constants::WINDOW_HEIGHT * m_spectatorZoom);
        window.setView(v);
    } else {
        window.setView(m_camera.getView());
    }

    // ---- 地图边界 ----
    sf::RectangleShape border;
    border.setFillColor(sf::Color::Transparent);
    border.setOutlineColor(sf::Color(233, 69, 96, 120));
    border.setOutlineThickness(4.0f);
    border.setSize(sf::Vector2f(m_mapW, m_mapH));
    window.draw(border);

    // ---- ★ 食物圈：淡色填充圆 ----
    //
    // ★ 画在【食物之下】：圈是地面标记，被食物和蛇盖住才对。
    //
    // ★ 为什么用"淡色填充"而不是"描边"：
    //   填充能让人一眼看出"这块地是圈"，而细描边在大地图、缩放之后几乎看不见。
    //   透明度压得很低（alpha 16~40），不会遮住食物和蛇。
    //   **不要描边** —— 描边会让圈看起来像"围栏"，而它其实只是"这片地会出食物"。
    //
    // ★ 活跃圈亮一点：让玩家能看出"现在轮到哪个圈供食"。
    //   注意这对 AI **没有额外好处** —— AI 从感知里拿的是方位/距离/食物数，
    //   食物数本身就编码了"轮到谁"（见 encoder.h）。这里纯粹是给人看的。
    //
    //  ★★ 2026-09-27 定稿（负责人定义）：**「下一个圈」只在预告期可见**
    //
    //    "不要一开始就有预告圈，预告圈是在活跃圈产食物还剩 3s 后出现的，
    //      然后中间有一定时间 gap，然后再被激活。"
    //
    //    时间线（供食 9s + 空窗 5s + 预告提前 3s）：
    //      0s ────────── 6s ─────── 9s ─────── 14s ─────── 23s
    //      │← 只有活跃圈 →│←── 预告期 ──→│← 空窗 →│← 新活跃圈供食 →│
    //
    //    ⚠️ 与 `PerceptionBuilder` 的 `v.visible = v.active || world.zonePreviewing`
    //       **完全一致** —— 玩家看到的和 AI 看到的必须是同一个世界。
    //    ⚠️ 判据的唯一实现是 `ZoneSchedule::previewing()`，这里不重推算式。
    //
    //  ★ 颜色规则
    //      活跃 + 供食中   → 暖色、最亮（"现在这里有吃的"）
    //      活跃 + 非供食   → 冷色、稍暗（"轮到它，但这轮产完了"）
    //      非活跃（预告期）→ 很淡的灰（"下一轮轮到它"）
    {
        const std::vector<arena::FoodZone>& zones = m_arena.food().zones();
        const arena::FoodZone* active = m_arena.activeZone();
        const bool spawning = m_arena.zoneSchedule().spawning(m_arena.food().zoneTimer());
        const bool previewing = m_arena.zoneSchedule().previewing(m_arena.food().zoneTimer());

        for (const arena::FoodZone& z : zones) {
            const bool isActive = (&z == active);

            // ★ 非活跃圈只在预告期画；活跃圈永远画
            if (!isActive && !previewing) continue;

            sf::CircleShape ring(z.radius);
            ring.setOrigin(z.radius, z.radius);
            ring.setPosition(z.x, z.y);
            ring.setOutlineThickness(0.0f);      // ★ 不描边，纯填充

            if (isActive && spawning) {
                // 正在供食：暖色、最亮
                ring.setFillColor(sf::Color(255, 220, 140, 40));
            } else if (isActive) {
                // 轮到它但还在空窗期：冷色，稍暗
                ring.setFillColor(sf::Color(140, 200, 255, 26));
            } else {
                // 预告期的"下一个圈"：比当前圈更淡
                ring.setFillColor(sf::Color(200, 200, 200, 20));
            }
            window.draw(ring);
        }
    }

    m_arena.food().render(window);

    // ---- 所有蛇：同队同色；玩家那条最后画，保证在最上层 ----
    for (std::size_t i = 0; i < m_arena.agents().size(); ++i) {
        if (static_cast<int>(i) == m_arena.humanIndex()) continue;
        m_arena.agents()[i].body.render(window);
    }
    if (m_arena.humanIndex() >= 0) {
        m_arena.agents()[static_cast<std::size_t>(m_arena.humanIndex())].body.render(window);
    }

    window.setView(defaultView);

    renderHud(window);

    // ★★ 排行榜按模式分流（2026-09-27）
    //
    //  ★ 无限模式：逐条蛇排名 + 显示它加载的神经网络名字（负责人要求）
    //  ★ 模式 1/3：仍然是队伍面板 —— 那两个模式**真的有队**
    //     （同队可以互相穿过），按队聚合才是对的。
    //  ⚠️ 不要"统一成一个面板"：两种模式的"排名主体"根本不同，
    //     硬合并的结果必然是某一方显示错（这正是缺陷 #55 的形态：
    //     拿无限模式的实现去套界面）。
    if (m_mode == game::GameMode::Infinite) renderSnakePanel(window);
    else                                    renderTeamPanel(window);

    m_minimap.render(window);
    if (m_debugMode)   renderDebug(window);
    if (m_matchOver)   renderMatchOver(window);
}

// ---------------------------------------------------------------------------
void GameScene::renderHud(sf::RenderWindow& window) {
    // ★ 读【这一局的配置】（时间限制等都属于"这一局"）
    auto& cfg = m_game->getMatchConfig();

    // 剩余时间
    std::ostringstream t;
    if (cfg.matchTimeLimit > 0.0f) {
        const int left = static_cast<int>(std::max(0.0f, cfg.matchTimeLimit - m_matchTimer));
        t << "剩余 " << (left / 60) << ":"
          << std::setw(2) << std::setfill('0') << (left % 60);
    } else {
        t << "无限时";
    }
    sf::Text timeText(textutil::utf8(t.str()), m_font, 22);
    timeText.setFillColor(sf::Color(255, 255, 255, 220));
    timeText.setPosition(10.0f, 10.0f);
    window.draw(timeText);

    // ---- ★ 无限模式：存活数 ----
    //
    //  ★★ 这一行的修正史（两次都值得记）
    //    ① 原来显示 `agents().size()`（**含尸体**）→ 尸体从不移除，
    //       于是读数卡在 "20 / 20" 而场上其实只有几条活的。
    //       负责人："蛇死了仍然占有 20 条上限。"
    //    ② 后来我把它当成"到 8 就停手"，分母写成 8 —— 语义整个反了。
    //       负责人："存活上限不是 8，是20！只是未满8 速度加快而已！"
    //       现在分母是真正的上限 20。
    //
    //  ★ 2026-09-24 精简（负责人要求）：删掉了两样东西
    //    · 「下一条 X.Xs（快/常规 Ns）」补蛇倒计时
    //    · 「食物 N / M」地图食物存量
    //    两者都是**世界内部状态**，不是玩家需要看的东西；挤在 HUD 上
    //    反而把真正要看的（长度/击杀/饥饿）淹掉了。
    //    注意：这【只是不显示】，updateSpawning 的补蛇逻辑一行没动。
    if (m_mode == game::GameMode::Infinite) {
        int aliveCnt = 0;
        for (const auto& a : m_arena.agents()) if (a.isAlive()) ++aliveCnt;

        std::ostringstream o;
        o << "存活 " << aliveCnt << " / " << m_spawnMaxAlive;
        if (aliveCnt >= m_spawnMaxAlive) o << "   （已满员）";

        sf::Text st(textutil::utf8(o.str()), m_font, 18);
        st.setFillColor(sf::Color(150, 220, 255, 210));
        st.setPosition(10.0f, 92.0f);
        window.draw(st);
    }

    // 玩家那条蛇的状态
    if (m_arena.humanIndex() >= 0) {
        const agent::Agent& me = m_arena.agents()[static_cast<std::size_t>(m_arena.humanIndex())];
        std::ostringstream s;
        s << "长度 " << me.body.getSegmentCount()
          << "   击杀 " << me.body.getKills();
        // ⚠️ "队伍 N" 只在【真的有队伍】的模式里显示。
        //    无限模式没有队伍（队号只是"每条蛇的身份槽位"），
        //    把它印在 HUD 上等于又犯一次缺陷 #55 的错 ——
        //    界面上出现"队"字就会让人以为这个模式有队伍。
        if (m_mode != game::GameMode::Infinite) s << "   队伍 " << (me.teamId + 1);
        sf::Text st(textutil::utf8(s.str()), m_font, 18);
        st.setFillColor(sf::Color(255, 255, 255, 200));
        st.setPosition(10.0f, 40.0f);
        window.draw(st);

        if (me.isAlive()) {
            const float left = std::max(0.0f, cfg.starvationTime - me.body.getTimeSinceEat());
            std::ostringstream h;
            h << std::fixed << std::setprecision(1) << left << "s";
            const bool urgent = (left < 5.0f);
            sf::Text ht(textutil::utf8("饥饿 " + h.str()), m_font, urgent ? 26 : 16);
            ht.setFillColor(urgent ? sf::Color(255, 60, 60)
                                   : sf::Color(255, 255, 255, 150));
            ht.setPosition(10.0f, 66.0f);
            window.draw(ht);
        }
    }

    if (m_spectator) {
        sf::Text sp(L"观战中  方向键移动  PgUp/PgDn 缩放  R 重开", m_font, 16);
        sp.setFillColor(sf::Color(255, 255, 100, 200));
        sp.setPosition(10.0f, Constants::WINDOW_HEIGHT - 30.0f);
        window.draw(sp);
    }
}

// ---------------------------------------------------------------------------
void GameScene::renderTeamPanel(sf::RenderWindow& window) {
    const float panelX = Constants::WINDOW_WIDTH - 205.0f;
    const float panelY = Constants::MINIMAP_HEIGHT + 50.0f;

    // ⚠️ 这里原来写的是 sf::Text title("-- 队伍 --", ...) ——
    //    窄字符串里含中文，被 sf::String 按本地编码解释，画出来是方块。
    //    凡是【含中文的字符串字面量】要么加 L 前缀，要么过 textutil::utf8。
    sf::Text title(L"-- 队伍 --", m_font, 15);
    title.setFillColor(sf::Color(255, 200, 100));
    title.setStyle(sf::Text::Bold);
    title.setPosition(panelX, panelY);
    window.draw(title);

    float y = panelY + 20.0f;
    for (std::size_t t = 0; t < m_teamStats.size(); ++t) {
        const TeamStat& s = m_teamStats[t];
        std::ostringstream oss;
        oss << (t + 1) << "队 活" << s.alive << "/" << s.members
            << " 长" << s.length << " 杀" << s.kills;

        sf::Text line(textutil::utf8(oss.str()), m_font, 12);
        const bool isHumanTeam =
            (m_arena.humanIndex() >= 0 &&
             m_arena.agents()[static_cast<std::size_t>(m_arena.humanIndex())].teamId == static_cast<int>(t));
        line.setFillColor(isHumanTeam ? sf::Color(120, 255, 120)
                                      : arena::Arena::teamColor(t));
        line.setPosition(panelX, y);
        window.draw(line);

        sf::RectangleShape chip(sf::Vector2f(8.0f, 8.0f));
        chip.setFillColor(arena::Arena::teamColor(t));
        chip.setPosition(panelX - 12.0f, y + 4.0f);
        window.draw(chip);

        y += 16.0f;
    }
}

// ---------------------------------------------------------------------------
//  ★★ 无限模式排行榜（负责人 2026-09-27 要求）
//
//      "旁边的排行榜改一下，不要队伍，要具体的蛇的排名，
//        然后把蛇的神经网络名字也显示出来。"
//
//  ★ 为什么必须另写一个面板，而不是把 renderTeamPanel 改一改
//    在无限模式里"队"只是身份槽位，而且蛇数上限 20 > MAX_TEAMS 8，
//    **队号会重复** —— 按队聚合会把两条不同的蛇并成一行，排名是错的。
//    队面板对模式 1/3 仍然正确，所以保留它、按模式分流。
//
//  ★ 三行的内容分配（宽度是硬约束，面板只有 205px）
//      第一行  #排名  长NN  杀N           ← 数字，窄
//      第二行  缩进的【神经网络文件名】    ← 中文名字，可能较宽
//      第三行  死蛇显示"已阵亡"
//    名字单独占一行是因为中文文件名（"略显智能"）比数字宽得多，
//    挤在同一行会被后面的排名字段截断。
// ---------------------------------------------------------------------------
void GameScene::renderSnakePanel(sf::RenderWindow& window) {
    const float panelX = Constants::WINDOW_WIDTH - 205.0f;
    const float panelY = Constants::MINIMAP_HEIGHT + 50.0f;

    // 数据每帧现算：排行榜是"表现"，不该维护第二份世界状态
    //（维护了就会有"面板和实际不一致"的机会，那是本项目的经典坑）
    const std::vector<SnakeEntry> list = snakeEntries();

    {
        std::wostringstream title;
        title << L"-- 蛇排名 (" << list.size() << ") --";
        sf::Text t(title.str(), m_font, 15);
        t.setFillColor(sf::Color(255, 200, 100));
        t.setStyle(sf::Text::Bold);
        t.setPosition(panelX, panelY);
        window.draw(t);
    }

    const int humanIdx = m_arena.humanIndex();

    // -----------------------------------------------------------------------
    //  ★ 能放几条：先算清楚，别把字画出屏幕外
    //
    //  ★ 为什么必须算（而不是"反正最多 20 条，挤一挤就行"）
    //    每条占 25px、面板从 y=200 起、窗口 720 —— 顶多 20 条多一点，
    //    看似刚好够。但 HUD 的行数、字体、窗口尺寸任何一个变了，
    //    就会变成"最后几条的神经网络名字沉到窗口下面"，
    //    而那种现象看着像"数据没生成"，不像"画不下"，很难往这里想。
    //
    //  ★ 为什么是"少画几条 + 写一句还有几条"而不是"压扁行距"
    //    压扁会在条数多的时候把字挤成一团，等于信息全丢；
    //    明确写"+还有 N 条"至少告诉你**排行榜是完整的**，只是没全画出来。
    //    而排行榜的排序规则保证：少画的一定是**又短又死的**那些。
    // -----------------------------------------------------------------------
    const float listBottom = static_cast<float>(Constants::WINDOW_HEIGHT) - 34.0f;
    const float rowStep    = 25.0f;                       // 与下面 y += 的值必须一致
    // ★ 再留一行给"+ 还有 N 条"那句（它自己 18px）—— 否则那句话本身会出屏
    int maxRows = static_cast<int>((listBottom - (panelY + 20.0f) - 18.0f) / rowStep);
    if (maxRows < 1) maxRows = 1;

    const int shown = std::min(static_cast<int>(list.size()), maxRows);

    float y = panelY + 20.0f;
    for (int r = 0; r < shown; ++r) {
        const SnakeEntry& e = list[static_cast<std::size_t>(r)];
        const sf::Color col = arena::Arena::teamColor(e.teamId);

        // ---- 第一行：排名 + 长度 + 击杀 ----
        {
            std::wostringstream o;
            o << L"#" << (r + 1) << L"  ";
            if (e.alive) o << L"长" << e.length;
            else         o << L"阵亡";
            o << L"  杀" << e.kills;

            sf::Text t(o.str(), m_font, 13);
            // 玩家那条用亮绿色盖过队色，一眼能找到自己
            t.setFillColor(e.index == humanIdx ? sf::Color(120, 255, 120) : col);
            t.setPosition(panelX, y);
            window.draw(t);
        }

        // ---- 色块（和场上的蛇同色，用来对应"哪条是它"）----
        {
            sf::RectangleShape chip(sf::Vector2f(8.0f, 8.0f));
            chip.setFillColor(col);
            chip.setPosition(panelX - 12.0f, y + 3.0f);
            window.draw(chip);
        }

        // ---- 第二行：神经网络名字（缩进，暗一点，不和数字抢注意力）----
        {
            sf::Text t(e.brain, m_font, 12);
            t.setFillColor(e.alive ? sf::Color(170, 205, 235)
                                   : sf::Color(115, 115, 125));
            t.setPosition(panelX + 16.0f, y + 14.0f);
            window.draw(t);
        }

        // ★ 每条 25px = 第一行 13px + 第二行 12px（两行刚好首尾相接）
        //   循环结束后 y 正好是"列表画完的位置"，
        //   下面的"还有 N 条 / 你已阵亡"接在它后面即可。
        y += rowStep;
    }

    // ---- 没画出来的部分，如实说一声 ----
    if (shown < static_cast<int>(list.size())) {
        sf::Text t(L"+ 还有 " + std::to_wstring(list.size() - shown) + L" 条（更短的）",
                   m_font, 12);
        t.setFillColor(sf::Color(140, 140, 150));
        t.setPosition(panelX, y);
        window.draw(t);
        y += 18.0f;
    }

    // ---- 人死了、蛇还在的时候给一句提示 ----
    //
    // ⚠️ 判据是 `m_spectator` 而**不是** "humanIndex 那条还活着"：
    //    无限模式现在会把死蛇从场上移除，玩家那条一死就没了 ——
    //    如果按"它还活着吗"来判断，这句话**永远显示不出来**
    //    （而它恰恰是玩家最需要看到的那一句）。
    //    `m_spectator` 是表现层的状态，不受世界清场影响。
    if (m_spectator) {
        sf::Text t(L"你已阵亡，观战中", m_font, 12);
        t.setFillColor(sf::Color(255, 140, 140));
        t.setPosition(panelX, y + 4.0f);
        window.draw(t);
    }
}

// ---------------------------------------------------------------------------
void GameScene::renderDebug(sf::RenderWindow& window) {
    int alive = 0;
    for (const auto& a : m_arena.agents()) if (a.isAlive()) ++alive;

    std::ostringstream oss;
    oss << "=== DEBUG (F3) ===\n";
    oss << "FPS " << static_cast<int>(m_fps) << "\n";
    oss << "蛇 " << alive << "/" << m_arena.agents().size() << "\n";
    oss << "食物 " << m_arena.food().getFoods().size() << "\n";
    oss << "tick " << m_arena.tick() << "\n";
    oss << "对局 " << std::fixed << std::setprecision(1) << m_matchTimer << "s\n";
    oss << "视野 " << arena::VISION_ROWS << "x" << arena::VISION_COLS << " x32px\n";

    // ---- ★★ 无限模式的生蛇诊断（2026-09-27 加）----
    //
    //  ★ 为什么必须放在调试面板里，而不是"再看一眼 HUD 就行"
    //    负责人报"卡在下一条蛇：3s，不会生成新的蛇"时，我**无法复现**
    //    （编译不了），只能靠读代码猜。而这条链上有五个变量
    //    （计时器 / 存活数 / 目标 / 累计 / 上限），HUD 只显示其中三个的
    //    组合结果 —— 组合值卡住时，看不出是哪个环节停了。
    //    把**原始值**摊在 F3 面板上，下次一眼就能定位。
    if (m_mode == game::GameMode::Infinite) {
        int aliveDbg = 0;
        for (const auto& a : m_arena.agents()) if (a.isAlive()) ++aliveDbg;

        oss << "--- spawn ---\n";
        oss << "alive " << aliveDbg << " / cap " << m_spawnMaxAlive << "\n";
        oss << "timer " << std::fixed << std::setprecision(2) << m_spawnTimer
            << " / " << ((aliveDbg < m_spawnFastBelow) ? m_spawnIntervalFast
                                                      : m_spawnIntervalNormal)
            << ((aliveDbg < m_spawnFastBelow) ? "  (fast)" : "  (normal)") << "\n";
        oss << "fastBelow " << m_spawnFastBelow << "\n";
        oss << "weightFiles " << m_spawnWeights.size() << "\n";
    }

    sf::Text txt(textutil::utf8(oss.str()), m_font, 14);
    txt.setFillColor(sf::Color(220, 220, 220));

    // ★ 背景高度按【实际行数】算，不写死
    //   写死 120px 时，无限模式多出来的 5 行会画在背景外面
    //   （白字压在游戏画面上，很难看而且读不清）。
    //   行数 = 换行符个数 + 1（最后一行没有结尾换行）。
    const std::string s = oss.str();
    const int lines = 1 + static_cast<int>(std::count(s.begin(), s.end(), '\n'));
    const float bgH = 14.0f * static_cast<float>(lines) + 12.0f;

    sf::RectangleShape bg(sf::Vector2f(190.0f, bgH));
    bg.setFillColor(sf::Color(0, 0, 0, 160));
    bg.setPosition(Constants::WINDOW_WIDTH - 200.0f, 300.0f);
    window.draw(bg);
    txt.setPosition(Constants::WINDOW_WIDTH - 195.0f, 305.0f);
    window.draw(txt);
}

// ---------------------------------------------------------------------------
void GameScene::renderMatchOver(sf::RenderWindow& window) {
    sf::RectangleShape overlay(sf::Vector2f(Constants::WINDOW_WIDTH,
                                            Constants::WINDOW_HEIGHT));
    overlay.setFillColor(sf::Color(0, 0, 0, 170));
    window.draw(overlay);

    const int winner = winningTeam();

    std::ostringstream title;
    if (winner >= 0) title << (winner + 1) << " 队获胜";
    else             title << "时间到";

    sf::Text t(textutil::utf8(title.str()), m_font, 54);
    t.setFillColor(sf::Color(255, 210, 90));
    t.setStyle(sf::Text::Bold);
    t.setPosition(Constants::WINDOW_WIDTH / 2.0f - 130.0f,
                  Constants::WINDOW_HEIGHT / 2.0f - 120.0f);
    window.draw(t);

    float y = Constants::WINDOW_HEIGHT / 2.0f - 40.0f;
    for (std::size_t i = 0; i < m_teamStats.size(); ++i) {
        const TeamStat& s = m_teamStats[i];
        std::ostringstream oss;
        oss << (i + 1) << " 队    存活 " << s.alive << "/" << s.members
            << "    总长度 " << s.length << "    击杀 " << s.kills;
        sf::Text line(textutil::utf8(oss.str()), m_font, 20);
        line.setFillColor(arena::Arena::teamColor(i));
        line.setPosition(Constants::WINDOW_WIDTH / 2.0f - 215.0f, y);
        window.draw(line);
        y += 28.0f;
    }

    sf::Text hint(L"R 再来一局    ESC 返回菜单", m_font, 18);
    hint.setFillColor(sf::Color(200, 200, 200));
    hint.setPosition(Constants::WINDOW_WIDTH / 2.0f - 130.0f, y + 20.0f);
    window.draw(hint);
}
