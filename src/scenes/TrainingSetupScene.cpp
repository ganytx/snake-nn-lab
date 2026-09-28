#include "TrainingSetupScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "TrainingTeamScene.h"      // ★「队伍设置」（训练模式的第二个设置界面）
#include "LiveTrainingScene.h"
#include "MenuScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include "../rl/Checkpoint.h"     // ★ 扫 saves/ 列出可选模型

#include <algorithm>
#include <sstream>

namespace {

const sf::Color kSelect = sf::Color(78, 205, 196);
const sf::Color kNormal = sf::Color(185, 185, 185);
const sf::Color kTitle  = sf::Color(255, 210, 90);
const sf::Color kDim    = sf::Color(130, 130, 130);

// ---------------------------------------------------------------------------
//  可选档位
//
//  ★ 用"档位表 + 循环切换"而不是连续加减：
//    这些参数都是少数几个有意义的档位，循环切换更快到位，
//    也不会调到奇怪的值（比如单局 7 秒、学习率 0.037 这种）。
// ---------------------------------------------------------------------------
// 可选档位
//
// ★ 2026-09-24 单局时长下限从 20s 抬到 60s（负责人要求"至少 40s"）
//
//   原因：饥饿时间是 20s，而原来最小档 20s、ZoneMini 预设 15s ——
//   **局比饥饿还短，蛇永远不会饿**，于是：
//     · 饥饿惩罚（③）完全不触发
//     · 吃饱余韵（⑤）变成"吃一次躺一局"的被动收入
//   单局 ≥ 饥饿时间，才跑得完"饿 → 吃 → 余韵 → 又饿"这个循环，
//   两条奖励项也才有意义。这里给 60s = 3 个饥饿周期。
const int   SNAKES[]  = {1, 3, 9};
const float FOODPS[]  = {0.111111f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
// ★ 2026-09-24 短档加回来（负责人要求"更少时间的选项也要保留"）
//   最短 20s 仍然等于饥饿时间 —— 那样局内不会饿死，适合"只练找食物"。
const int   SECONDS[] = {20, 30, 60, 90, 120, 180};
const float LRS[]     = {0.005f, 0.01f, 0.02f, 0.05f};
const float GAMMAS[]  = {0.95f, 0.98f, 0.99f, 0.995f};
const float EPSS[]    = {0.0f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f};

template <typename T, int N>
int cycleIndex(const T (&opts)[N], T cur, int delta) {
    int idx = 0;
    for (int i = 0; i < N; ++i) if (opts[i] == cur) { idx = i; break; }
    idx = ((idx + delta) % N + N) % N;
    return idx;
}

// ---- 各行的显示文本 ----

std::wstring showKind(const rl::TaskConfig& t) {
    return rl::taskKindName(t.kind);
}

std::wstring showSnakes(const rl::TaskConfig& t) {
    std::wostringstream o;
    const int n = t.snakeCount();
    o << n << L" 条";
    if (n == 1) o << L"   （只有学习中的那一条）";
    else        o << L"   （1 条学习 + " << (n - 1) << L" 条规则AI 当对手）";
    return o.str();
}

std::wstring showFood(const rl::TaskConfig& t) {
    std::wostringstream o;
    const double v = t.foodPerSecondOverride;
    o.precision(4);
    o << v << L" 个/秒";
    if (v > 0.0) o << L"   ≈ 每 " << static_cast<int>(1.0 / v + 0.5) << L" 秒一个";
    return o.str();
}

std::wstring showSeconds(const rl::TaskConfig& t) {
    std::wostringstream o;
    const double secs = t.maxTicks / 60.0;
    const double cyc  = t.zoneSpawnSeconds + t.zoneGapSeconds;
    o << static_cast<int>(secs) << L" 秒"
      << L"   （食物圈轮换 " << static_cast<int>(secs / cyc) << L" 轮）";
    return o.str();
}

std::wstring showSpawn(const rl::TaskConfig& t) {
    return (t.spawnMode == arena::SpawnMode::Random)
         ? L"随机撒点   （训练推荐：避免过拟合到固定出生点）"
         : L"均分圆周   （和游戏里一样）";
}

std::wstring showLR(const rl::TaskConfig& t) {
    std::wostringstream o; o.precision(4); o << t.lr; return o.str();
}

std::wstring showGamma(const rl::TaskConfig& t) {
    std::wostringstream o; o.precision(4); o << t.gamma; return o.str();
}

std::wstring showEps(const rl::TaskConfig& t) {
    std::wostringstream o;
    o.precision(3);
    o << t.explorationEps;
    if (t.explorationEps <= 0.0f) {
        o << L"   （关闭 —— 策略可能坍缩成永远同一个动作）";
    } else {
        const int every = static_cast<int>(1.0 / t.explorationEps + 0.5);
        o << L"   （大约每 " << every << L" 步随机试一次别的动作）";
    }
    return o.str();
}

// ---------------------------------------------------------------------------
//  ★「训练蛇数」改成只读显示（2026-09-24）
//
//  它现在是"组队"的结果，不再是一个能单独调的参数 ——
//  要改请进「队伍设置」。所以这里明确写出"去哪改"，
//  而不是显示一个数字却不响应左右键（那会让人以为界面坏了）。
// ---------------------------------------------------------------------------
std::wstring showSnakesLocked(const rl::TaskConfig& t) {
    std::wostringstream o;
    o << t.snakeCount() << L" 条   （在「队伍设置」里改）";
    return o.str();
}

std::wstring showCamera(const rl::TaskConfig& t) {
    // 多蛇也能跟随 —— 默认跟学习中的那条（#0），进去后按 Tab 换目标
    if (!t.followCamera) return L"自由镜头   （默认：能看见整张图）";
    if (t.snakeCount() == 1) return L"跟随学习中的那条";
    return L"跟随蛇 #0   （进去后按 Tab 切换跟哪条）";
}

// ---------------------------------------------------------------------------
//  ★「队伍设置」那一行的摘要（回车进去改）
//
//  显示"几队 × 每条"和"有几种算法"—— 让人一眼看出进去会改什么。
// ---------------------------------------------------------------------------
std::wstring showTeamSummary(const rl::TaskConfig& t) {
    std::wostringstream o;
    o << t.teamCount << L" 队 x " << t.snakesPerTeam << L" 条"
      << L"   （共 " << t.snakeCount() << L" 条）";
    o << L"   按回车进入设置 ->";
    return o.str();
}

// ---------------------------------------------------------------------------
//  ★ 显示"要加载哪个模型"
//
//  ★ 为什么每次调用都重新扫目录
//    从训练模式退回来时，saves/ 里可能刚多了 auto_latest.txt
//    （退出自动存档）。如果这里缓存列表，用户就看不到它。
//    扫一个几十个文件的目录是微秒级的，不值得缓存。
// ---------------------------------------------------------------------------
std::wstring showPreload(const rl::TaskConfig&) {
    const std::string cur = TrainingSetupScene::pendingPreloadRead();
    if (cur.empty()) {
        const int total = static_cast<int>(ckpt::list().size());
        std::wostringstream o;
        o << L"不加载（从头练）";
        o << L"   （saves/ 里有 " << total << L" 份存档可选）";
        return o.str();
    }
    // 文件名是 UTF-8，sf::Text 需要宽字符 —— 用 sf::String 转换
    const sf::String ws = sf::String::fromUtf8(cur.begin(), cur.end());
    std::wstring out = ws.toWideString();
    if (cur == ckpt::kAutoSaveName) {
        out += L"   （退出时自动存的）";
    }
    return out;
}

}  // namespace

// ===========================================================================
TrainingSetupScene::TrainingSetupScene(Game* game)
    : Scene(game)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "TrainingSetupScene");
    }

    buildRows();
}

// ===========================================================================
//  ★ 行表构建（2026-09-24 抽成独立函数）
//
//  ★ 为什么要抽出来
//    加了「队伍设置」界面之后，蛇数是【在那边】改的 ——
//    改完回到本界面，"训练蛇数"那一行的显示必须跟着变。
//    所以需要能从外部触发重建（见头文件里的 refreshRows()）。
//
//  ★ 训练模式现在有【两个】设置界面（负责人要求）
//      ① 训练设置（本界面）—— 加载哪个模型 / 任务 / 食物 / 学习超参
//      ② 队伍设置（TrainingTeamScene）—— 几队 / 每队几条 /
//         以及每队是不是神经网络（列在这里是为了透明，训练时只有 0 号学习）
//    两者共享同一份 pendingConfig()，所以改完天然同步。
// ===========================================================================
void TrainingSetupScene::buildRows() {
    m_rows = {
        // -------------------------------------------------------------------
        //  ★ 加载模型（负责人要求：进训练模式前自己选加载什么模型）
        //
        //  ★ 为什么放在【第一行】
        //    它是"这次训练从哪儿开始"的决定 —— 比任务参数更靠前。
        //    任务参数决定"练什么"，这一行决定"从什么基础练"。
        //
        //  ★ 选项来自 saves/ 的实时扫描 + 一个"不加载"
        //    列表每次调整时重新扫（训练完回来列表会自动更新）。
        //
        //  ★ 不在这里加载，只【记下名字】
        //    真正的加载发生在 LiveTrainingScene 构造时 ——
        //    因为网络那时才建出来。这里只是选择。
        // -------------------------------------------------------------------
        { L"加载模型", &showPreload,
          [](rl::TaskConfig&, int d) {
              // 把"不加载"当成下标 0，存档从 1 开始
              const std::vector<std::string> files = ckpt::list();
              const std::string cur = pendingPreload();
              int idx = 0;                       // 0 = 不加载
              for (std::size_t i = 0; i < files.size(); ++i) {
                  if (files[i] == cur) { idx = static_cast<int>(i) + 1; break; }
              }
              const int total = static_cast<int>(files.size()) + 1;   // +1 是"不加载"
              idx = ((idx + d) % total + total) % total;
              pendingPreload() = (idx == 0) ? std::string() : files[static_cast<std::size_t>(idx - 1)];
          } },

        // ★ 任务种类放在前面 —— 它决定下面哪些参数有意义。
        //   切换它会【换一整套预设】（地图大小、食物、时长都会变），
        //   所以切换时要重新读预设。
        { L"任务", &showKind,
          [](rl::TaskConfig& t, int d) {
              // ★ 用 TASK_KIND_COUNT 而不是字面量 3/4 ——
              //   加任务时忘了改这里，就会有一个任务永远选不到。
              const int cur  = static_cast<int>(t.kind);
              const int n    = rl::TASK_KIND_COUNT;
              const int next = ((cur + d) % n + n) % n;
              // 换任务 = 换一整套预设（保留镜头这个纯表现层设置）
              const bool cam = t.followCamera;
              const unsigned sd = t.seed;
              t = rl::makeTaskConfig(static_cast<rl::TaskKind>(next), sd);
              t.followCamera = cam;
          } },

        // -------------------------------------------------------------------
        //  ★ 训练蛇数：2026-09-24 改成【只读显示】
        //
        //  ★ 为什么不再在这里改
        //    蛇数是【组队】的一部分（几队 x 每队几条），现在已经搬进
        //    「队伍设置」界面。两处都能改会打架 ——
        //    而且这里原来是直接 assign roles，**完全不动 teams 表**，
        //    于是 teams 会停留在上一次的规模、和 roles 对不上
        //    （实测就是这个 bug：改蛇数后 teams 长度不变，角色表和队表错位）。
        //
        //    现在改队数/人数统一走 `TaskConfig::rebuildTeamTables()`，
        //    roles 和 teams 永远一起重算，"两张表对不上"这个类错误从结构上消失。
        // -------------------------------------------------------------------
        { L"训练蛇数", &showSnakesLocked,
          [](rl::TaskConfig&, int) {} },   // 左右键不做事 —— 去队伍设置里改

        { L"食物速率", &showFood,
          [](rl::TaskConfig& t, int d) {
              const float cur = static_cast<float>(t.foodPerSecondOverride);
              const int i = cycleIndex(FOODPS, cur, d);
              t.foodPerSecondOverride = FOODPS[i];
          } },

        { L"单局时长", &showSeconds,
          [](rl::TaskConfig& t, int d) {
              const int secs = t.maxTicks / 60;
              const int i = cycleIndex(SECONDS, secs, d);
              t.maxTicks = SECONDS[i] * 60;
          } },

        { L"出生方式", &showSpawn,
          [](rl::TaskConfig& t, int) {
              t.spawnMode = (t.spawnMode == arena::SpawnMode::Random)
                          ? arena::SpawnMode::Ring
                          : arena::SpawnMode::Random;
          } },

        { L"学习率", &showLR,
          [](rl::TaskConfig& t, int d) {
              const float cur = static_cast<float>(t.lr);
              t.lr = LRS[cycleIndex(LRS, cur, d)];
          } },

        { L"折扣因子", &showGamma,
          [](rl::TaskConfig& t, int d) {
              const float cur = static_cast<float>(t.gamma);
              t.gamma = GAMMAS[cycleIndex(GAMMAS, cur, d)];
          } },

        // ★ 探索率：防"策略坍缩成永远同一个动作"的真正保险。
        //   放在界面里是因为这个数很可能还要再调（0.05 是第一次的估值）。
        { L"探索率", &showEps,
          [](rl::TaskConfig& t, int d) {
              const float cur = static_cast<float>(t.explorationEps);
              t.explorationEps = EPSS[cycleIndex(EPSS, cur, d)];
          } },

        // ★ 镜头跟随也放到这里，而不是只在训练里按 F
        //
        //   原来它只藏在一个快捷键后面，结果负责人以为"跟随功能没了"。
        //   凡是"开局前能定"的东西，都该在这个界面上看得见。
        { L"镜头", &showCamera,
          [](rl::TaskConfig& t, int) {
              t.followCamera = !t.followCamera;
          } },

        // -------------------------------------------------------------------
        //  ★「队伍设置」入口（负责人要求）
        //
        //  ★ 为什么单独一个界面而不是在本页加几行
        //    "几队 / 每队几条 / 每队什么算法"是【组队】这件事，
        //    和"任务 / 食物 / 学习率"是两类不同的东西。
        //    混在一页会越堆越长，而这一页已经有 10 行了。
        //    这也和游戏模式一致（对局设置专门管组队）。
        //
        //  ★ 这一行按回车【跳转】而不是开始训练 —— 判据见 handleEvent 里的
        //    `isTeamRow()`（行号是动态的，所以用"最后一行"判断）。
        // -------------------------------------------------------------------
        { L"队伍设置", &showTeamSummary,
          [](rl::TaskConfig&, int) {} },   // 左右键不做事，只能按回车进去
    };
}

// ===========================================================================
//  待定配置：**跨场景保留**
//
//  放在函数内的 static —— 第一次调用时才构造，避免"静态初始化顺序"问题。
//
//  ★ 2026-09-26 负责人定的训练模式默认任务：**食物圈·位置随机**
//
//    ⚠️ 这是刻意写死的，不是"沿用 makeTaskConfig() 的默认值"（那个是 FoodZone）。
//       理由（负责人原话）："多蛇 + 位置随机才是真正有用的，其它模式不具参考价值"。
//       —— 固定位置时可以靠"背地图"取胜，训练时分不出它是读观测还是背路线。
//
//    任务本身的其他参数（3 队 × 2 条 / nn + rule + nn-enemy / 90 秒）
//    在 rl::makeTaskConfig(TaskKind::ZoneRandom) 里定义，这里不重复写 ——
//    否则就是"同一件事有两个来源"，本项目为此栽过两次。
// ===========================================================================
rl::TaskConfig& TrainingSetupScene::pendingConfig() {
    static rl::TaskConfig s_cfg = rl::makeTaskConfig(rl::TaskKind::ZoneRandom);
    return s_cfg;
}

const rl::TaskConfig& TrainingSetupScene::pendingConfigRead() {
    return pendingConfig();
}

// ===========================================================================
//  待加载的存档文件名（跨场景保留）
//
//  ★ 为什么不放进 TaskConfig
//    TaskConfig 描述"训练任务"（地图/奖励/蛇数），
//    而"加载哪份权重"是【会话选择】。把它塞进 TaskConfig 会让
//    rl/ 那一层知道"磁盘上有什么"，破坏"rl 只管算法、不管磁盘"的分层。
//
//  ★ 空字符串 = 不加载，从头练（默认）
// ===========================================================================
std::string& TrainingSetupScene::pendingPreload() {
    static std::string s_name;      // 默认空 = 不加载
    return s_name;
}

const std::string& TrainingSetupScene::pendingPreloadRead() {
    return pendingPreload();
}

// ===========================================================================
void TrainingSetupScene::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::KeyPressed) return;

    const int n = static_cast<int>(m_rows.size());
    if (n == 0) return;

    rl::TaskConfig& cfg = pendingConfig();

    switch (event.key.code) {
        case sf::Keyboard::Up:
            m_row = (m_row - 1 + n) % n;
            break;
        case sf::Keyboard::Down:
            m_row = (m_row + 1) % n;
            break;

        case sf::Keyboard::Left:
        case sf::Keyboard::Right: {
            const int d = (event.key.code == sf::Keyboard::Right) ? 1 : -1;
            m_rows[static_cast<std::size_t>(m_row)].adjust(cfg, d);
            break;
        }

        case sf::Keyboard::Enter:
        case sf::Keyboard::Space:
            // ★「队伍设置」那一行 → 跳到组队界面；其它行 → 开始训练
            if (m_row == teamRowIndex()) {
                m_game->switchScene(std::make_unique<TrainingTeamScene>(m_game));
            } else {
                m_game->switchScene(std::make_unique<LiveTrainingScene>(m_game));
            }
            break;

        case sf::Keyboard::Escape:
            m_game->switchScene(std::make_unique<MenuScene>(m_game));
            break;

        default: break;
    }
}

void TrainingSetupScene::fixedUpdate(float) {}
void TrainingSetupScene::update(float) {}

// ===========================================================================
void TrainingSetupScene::render(sf::RenderWindow& window) {
    const rl::TaskConfig& cfg = pendingConfigRead();

    sf::Text title(L"训练设置", m_font, 40);
    title.setFillColor(kTitle);
    title.setStyle(sf::Text::Bold);
    title.setPosition(60.0f, 30.0f);
    window.draw(title);

    sf::Text hint(L"上下键 选择    左右键 调整    Enter 开始训练    ESC 返回",
                  m_font, 16);
    hint.setFillColor(kDim);
    hint.setPosition(60.0f, 86.0f);
    window.draw(hint);

    sf::Text hint2(L"[这些参数只影响训练模式，不改游戏本体的平衡]",
                   m_font, 16);
    hint2.setFillColor(sf::Color(110, 150, 130));
    hint2.setPosition(60.0f, 110.0f);
    window.draw(hint2);

    // -----------------------------------------------------------------------
    //  ★ 行区：动态行距，保证【永远不撞到底部说明】
    //
    //  ⚠️ 负责人报过"下面字体叠一起了"。原因是：
    //      行从 y=178 开始、每行 46px，加到 11 行时已经到 y=638；
    //      而底部说明从 `720 − 130 = 590` 起画 —— **重叠了 48px**。
    //
    //      "行数变了但底部起点写死"这类问题会反复出现，所以这里改成【算出来】：
    //      行区可用高度 = 说明区顶部 − 行起点 − 留白，再按行数分行距。
    //      以后再加行，它会自动压紧，而不是悄悄叠上去。
    // -----------------------------------------------------------------------
    const float kFooterTop = static_cast<float>(Constants::WINDOW_HEIGHT) - 96.0f;
    const float kRowsTop   = 172.0f;
    const int   nRows      = static_cast<int>(m_rows.size());
    const float rowGap = (nRows > 0)
        ? std::min(46.0f, (kFooterTop - kRowsTop - 8.0f) / static_cast<float>(nRows))
        : 46.0f;

    float y = kRowsTop;
    for (std::size_t i = 0; i < m_rows.size(); ++i) {
        const bool sel = (static_cast<int>(i) == m_row);
        const Row& row = m_rows[i];

        // 选中行背景条 —— 让"当前在改哪一项"一眼可见
        if (sel) {
            sf::RectangleShape bar(sf::Vector2f(1010.0f, 36.0f));
            bar.setPosition(50.0f, y - 5.0f);
            bar.setFillColor(sf::Color(78, 205, 196, 45));
            bar.setOutlineColor(sf::Color(78, 205, 196, 120));
            bar.setOutlineThickness(1.0f);
            window.draw(bar);

            // 左右箭头，提示"这一项可以调"
            sf::Text arrowL(L"<", m_font, 22);
            arrowL.setFillColor(kSelect);
            arrowL.setPosition(160.0f, y);
            window.draw(arrowL);
            sf::Text arrowR(L">", m_font, 22);
            arrowR.setFillColor(kSelect);
            arrowR.setPosition(1040.0f, y);
            window.draw(arrowR);
        }

        sf::Text label(row.label, m_font, 22);          // 宽字符串直接构造，中文没问题
        label.setFillColor(sel ? sf::Color::White : kNormal);
        label.setPosition(60.0f, y);
        window.draw(label);

        sf::Text value(row.get(cfg), m_font, 22);
        value.setFillColor(sel ? kSelect : kNormal);
        value.setPosition(190.0f, y);
        window.draw(value);

        y += rowGap;
    }

    // ---- 底部说明（压成 2 行；起点由上面的 kFooterTop 统一决定）----
    {
        float fy = kFooterTop + 8.0f;
        sf::Text f1(L"· 食物速率 / 单局时长 —— 决定一局里能不能见到【食物圈轮换】；"
                    L"少或短就学不到「跟着圈跑」",
                    m_font, 15);
        f1.setFillColor(kDim);
        f1.setPosition(60.0f, fy);
        window.draw(f1);
        fy += 21.0f;

        sf::Text f2(L"· 队伍与蛇数在「队伍设置」里改；"
                    L"训练中按 S 命名保存、O 覆盖当前、L 加载存档（saves/）",
                    m_font, 15);
        f2.setFillColor(sf::Color(120, 200, 160));
        f2.setPosition(60.0f, fy);
        window.draw(f2);
    }
}
