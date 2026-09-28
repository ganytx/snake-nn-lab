#include "LiveTrainingScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "MenuScene.h"
#include "TrainingSetupScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include "../utils/Text.h"
#include "../brain/BrainRegistry.h"
#include "../rl/Checkpoint.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>

namespace {

constexpr float DT = 1.0f / 60.0f;   // 固定步长，和游戏一致

// ---------------------------------------------------------------------------
//  速度档位：**2 的幂**（负责人要求"一定倍数以下固定为 2 的幂，更整洁"）
//
//  index 0..6   → 每帧正好 2^k 步（精确、可验证）
//  index 7..11  → 按时间预算跑（取实测能达到的最大倍速）
//  index -1..-2 → 慢动作（每 n 帧跑 1 步）
//
//  ⚠️ 显示用的倍速【不是】从 k 硬算的，而是实测
//     （仿真时间 / 墙钟时间）。因为高倍速段受渲染与机器负载影响，
//     硬算的值和实际情况对不上 —— 那样就又变成"档位≠速度"了。
// ---------------------------------------------------------------------------
constexpr int   SPEED_FIXED_MAX = 6;                       // 0..6 → 1x..64x
constexpr float SPEED_BUDGET_MS[5] = {8.0f, 12.0f, 16.0f, 24.0f, 32.0f};
constexpr int   SPEED_BUDGET_FIRST = SPEED_FIXED_MAX + 1;  // 7
constexpr int   SPEED_MAX_INDEX    = SPEED_BUDGET_FIRST + 4;  // 11
constexpr int   SPEED_SLOW_STEPS[2] = {2, 4};              // 每 n 帧 1 步

}  // namespace

// ===========================================================================
LiveTrainingScene::LiveTrainingScene(Game* game)
    : Scene(game)
    , m_rng(20260921u)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "LiveTrainingScene");
    }

    // ★ 任务参数来自【训练设置界面】。直接进本场景（比如以后加了快捷键）
    //   也能工作 —— 那会用到 makeTaskConfig() 的默认值。
    m_task = TrainingSetupScene::pendingConfigRead();

    // ★ 内置算法必须先注册 —— Arena::reset() 里 create() 要靠它。
    //   游戏本体在 main.cpp 里注册过一次，这里再调一次是幂等的，
    //   但显式写出来是为了让"从主菜单直接进训练模式"这条路径也自洽。
    arena::registerBuiltinBrains();

    refreshCheckpoints();

    restartTraining();
}

// ===========================================================================
//  权重存档
// ===========================================================================
void LiveTrainingScene::refreshCheckpoints() {
    ckpt::ensureDir();
    m_checkpoints = ckpt::list();
    if (m_loadCursor >= static_cast<int>(m_checkpoints.size())) {
        m_loadCursor = std::max(0, static_cast<int>(m_checkpoints.size()) - 1);
    }
}

void LiveTrainingScene::saveCheckpoint(const std::string& name) {
    if (!m_trainer) return;

    // ★ ckpt::path 现在返回 filesystem::path（中文名必须靠它才能正确落盘）。
    //   MLP::save 收 path，所以直接传；显示时才转成字符串。
    const std::filesystem::path full = ckpt::path(name);

    if (m_trainer->net().save(full)) {
        m_statusText = std::string("已保存  ") + full.string()
                     + "   (episode " + std::to_string(m_episodes) + ")";
        m_currentModelName = name;      // ★ 存过一次之后，它就成了"当前模型"
        refreshCheckpoints();
    } else {
        m_statusText = std::string("保存失败：") + full.string();
    }
    m_statusTimer = 4.0f;
}

// ---------------------------------------------------------------------------
//  ★ 直接覆盖"当前模型"（负责人要求）
//
//  ★ 和 saveCheckpoint() 的区别
//      saveCheckpoint()          新起一个带时间戳的名字 → 每次都是【新版本】
//      overwriteCurrentModel()   写回同一个文件       → 【原地更新】当前进度
//
//  ★ 两种都需要，用途不同
//      · 想留档对比（"调奖励前那版"）→ 用 S 命名
//      · 训练中途存一下、接着练      → 用覆盖，不产生新文件
//
//  ★ 没有"当前模型"时不猜一个名字
//    提示用户先用 S 命名一次。猜名字会造成"以为覆盖了 X，其实多了个 Y"。
// ---------------------------------------------------------------------------
void LiveTrainingScene::overwriteCurrentModel() {
    if (!m_trainer) return;

    if (m_currentModelName.empty()) {
        m_statusText  = "还没有「当前模型」——先按 S 命名保存一次，之后就能直接覆盖";
        m_statusTimer = 6.0f;
        return;
    }
    saveCheckpoint(m_currentModelName);
    m_statusText  = std::string("已覆盖当前模型  ") + ckpt::path(m_currentModelName).string();
    m_statusTimer = 4.0f;
}

void LiveTrainingScene::saveCheckpoint() {
    // ★ 按 S 的默认行为：名字自动生成（局数 + 时间戳），不覆盖旧存档。
    //   想自己命名请用 S 的输入模式（见 keyPressed 里的 m_nameInputMode）。
    saveCheckpoint(ckpt::makeName(m_episodes));
}

// ---------------------------------------------------------------------------
//  ★ 自动存档：退出训练模式时存一份，固定名、每次都覆盖
//
//  ★ 为什么必须有它（负责人要求）
//
//    以前按 ESC 退出**什么都不存**，而进训练模式时网络是用固定种子
//    重新初始化的 —— 所以"练了半小时、按 ESC 全没了"。
//
//  ★ 为什么固定名 + 覆盖，而不是带时间戳
//
//    它表达的是"**这次训练的最新进度**"，语义上就该只有一份。
//    要留多个版本请用 `S` 自己命名存档。
//    两者分工明确：auto_latest 用来"接着练"，命名存档用来"留档对比"。
//
//  ★ 为什么只在【确实练过】的时候才存
//
//    一进来就退出（episode 还是 0）时存一份随机初始化的权重，
//    会把上一次真正练过的 auto_latest 覆盖掉 —— 那是**净损失**。
// ---------------------------------------------------------------------------
void LiveTrainingScene::autoSave() {
    if (!m_trainer) return;
    if (m_episodes <= 0) return;      // 没练过任何一局 → 不动上次的存档
    saveCheckpoint(ckpt::kAutoSaveName);
}

void LiveTrainingScene::openLoadList() {
    refreshCheckpoints();
    m_loadListOpen = true;
    m_paused = true;              // 选存档时先停住，别让局数继续跑
    m_loadCursor = std::max(0, static_cast<int>(m_checkpoints.size()) - 1);
}

void LiveTrainingScene::loadSelected() {
    if (!m_trainer || m_checkpoints.empty()) {
        m_statusText = "没有存档可加载";
        m_statusTimer = 4.0f;
        m_loadListOpen = false;
        return;
    }
    const std::string name = m_checkpoints[static_cast<std::size_t>(m_loadCursor)];
    const std::filesystem::path full = ckpt::path(name);

    if (m_trainer->net().load(full)) {
        // ★ 加载权重后要【重置基线】。
        //
        //   基线是"最近回报的滑动平均"，属于【这次训练的状态】，不属于权重。
        //   不重置的话，新加载的策略会拿旧策略的基线去算 advantage，
        //   前期梯度方向会偏，看起来就是"加载完反而变笨了一阵"。
        m_trainer->resetBaseline();

        // 局数从 0 起算，但文件名里保留了它原来是第几局练出来的
        m_episodes = 0;
        m_recentReward.clear();
        m_recentEaten.clear();
        m_runner->begin(rl::RunMode::Train);

        m_currentModelName = name;    // ★ 它成了"当前模型"，可以直接覆盖

        m_statusText = std::string("已加载  ") + full.string()
                     + "   (baseline 已重置)";
    } else {
        m_statusText = std::string("加载失败：") + full.string()
                     + "   (维度不匹配？改了 OBS_SIZE 的旧存档会读不了)";
    }
    m_statusTimer = 4.0f;
    m_loadListOpen = false;
}

LiveTrainingScene::~LiveTrainingScene() {
    // ★ 兜底自动存档（负责人要求"退出时保存"）
    //
    //   ESC 那条路径已经显式调过 autoSave()，这里是**第二条出口**：
    //   直接关窗口 / 切到别的场景 / 进程被杀之前，析构仍会跑到。
    //
    //   ★ 为什么两道都要有
    //     ESC 那条是"用户主动退出"的正常路径，写完还能在状态栏提示"已保存"；
    //     析构这条是"没有提示机会"的路径，只能静默写。
    //     只有一条的话，另一种退出方式就会丢进度 —— 而"丢进度"正是要修的问题。
    //
    //   autoSave() 内部判了"没练过就不存"，所以重复调用是安全的。
    autoSave();
}

// ===========================================================================
//  重新开始训练
// ===========================================================================
void LiveTrainingScene::restartTraining() {
    m_episodes   = 0;
    m_lastReward = 0.0;
    m_lastTicks  = 0;
    m_lastEaten  = 0;
    m_recentReward.clear();
    m_recentEaten.clear();

    // ★ 重建顺序要和依赖方向一致：
    //   先 Trainer（网络），再 Arena（世界），最后 EpisodeRunner（把两者串起来）。
    //   析构时成员按声明顺序的【反序】销毁，所以 declaration 那边
    //   m_runner 声明在最前面 —— 它会最先被销毁，不会悬空引用另外两个。
    m_trainer = std::make_unique<rl::Trainer>(m_task.seed);
    m_trainer->gamma            = m_task.gamma;
    m_trainer->lr               = m_task.lr;
    m_trainer->baselineMomentum = m_task.baselineMomentum;
    m_trainer->entropyCoef      = m_task.entropyCoef;

    // ★★★ 2026-09-26：接上 actor-critic（价值网络 baseline）
    //
    //  ★ 为什么必须在这里也接（原来只有命令行 `--learner critic` 能开）
    //    负责人看的是**游戏内训练模式**，而它以前**没有这个开关** ——
    //    于是"换成 actor-critic 试试"这件事在训练模式里根本做不到。
    //    ⚠️ 这和"画圈有三份实现只改一份"是同一类问题：
    //      **一个能力只在一条路径上接了，另一条路径就没有。**
    //
    //  ★ 为什么现在切成 Critic
    //    实测单局回报**标准差 44.5**，而均值只有 28.9 ——
    //    噪声是"吃一个食物(+3)"信号量级的 **15 倍**。
    //    纯 REINFORCE 的标量基线减不掉"这个状态本来就该得多少分"，
    //    所以 advantage 被局间波动淹没。价值网络给的是**按状态**的基线。
    m_trainer->mode = rl::LearningMode::Critic;

    // 价值网络的超参（默认值与命令行一致，见 Trainer.h 的说明）
    m_trainer->valueLr      = 0.02;    // 与策略同值：两个网络输入同维、隐藏层同宽
    m_trainer->gaeLambda    = 0.95;    // GAE 折中：λ=1 高方差、λ=0 高偏差
    m_trainer->normalizeAdv = true;    // 按批 z-score，让步长与回报量级脱钩

    // -----------------------------------------------------------------------
    //  ★★★ 快照改成【每局同步】（2026-09-26 深夜，负责人报行为差异）
    //
    //  ★ 负责人报的现象
    //      "我现在选择 3 队，3 条蛇，学习那一队的其余两条蛇
    //       仍然与第一条蛇存在明显行为差异"
    //
    //  ★ 根因：`mirrorInterval` 默认是 **100**，即快照每 100 局才写一次
    //      而学习者（index 0）**每局**都在更新自己那份 `m_net`。
    //      队友（队 0 的第 2、3 条）读的是快照 → 最多落后 **100 局**。
    //      100 局在 90 秒/局的设置下是 2.5 小时的训练量，行为当然完全不同。
    //
    //  ★ 改成 1：每局结束都写一次快照
    //      于是队友与敌方每局都能拿到"上一局结束时"的最新权重。
    //
    //  ★ 为什么是"上一局结束时"而不是"本局开始时的同一个对象"
    //      这是分层铁律的结果，不是选择：
    //        · `Arena`（arena/ 层）在 `reset()` 里创建大脑，
    //        · 而"当前网络"在 `Trainer`（rl/ 层）里，
    //        · `arena/` 不能 include `rl/`。
    //      所以只能通过【文件】同步，而文件只能在局末写。
    //      **代价：队友恒定落后学习者 1 局。** 这是当前架构下能做到的最小差距。
    //      （要让 lag = 0 就必须让 Arena 直接持有 Trainer 的网络指针，
    //        那会破坏分层 —— 属于另一个量级的改动，需要负责人点头。）
    //
    //  ★ 代价评估
    //      每局多一次文件写：网络约 2917 个参数 × 8 字节 ≈ 23 KB，
    //      相对每局 5400 帧的感知开销（占 51~69%）可以忽略。
    //  ⚠️ 它同时影响敌方（"nn-enemy"）—— 对手也会每局变强，这是好事
    //      （原来每 100 局跳一次，对 learner 是非平稳环境的突跳）。
    // -----------------------------------------------------------------------
    m_trainer->mirrorInterval = 1;

    // -----------------------------------------------------------------------
    //  ★ 加载【训练设置界面】里选好的模型（负责人要求）
    //
    //  ★ 为什么加载完必须 resetBaseline()
    //    基线是"最近回报的滑动平均"，属于【这一次训练的过程状态】，
    //    不在权重文件里。拿旧策略的基线给新策略算 advantage 会让前期梯度方向偏，
    //    现象是"加载完反而变笨了一阵"。见 Trainer::resetBaseline 的说明。
    //
    //  ★ 加载失败不静默：明确告诉用户是"文件不存在"还是"维度不匹配"，
    //    否则他只会看到"加载了但好像没生效"。
    // -----------------------------------------------------------------------
    const std::string& preload = TrainingSetupScene::pendingPreloadRead();
    if (!preload.empty()) {
        const std::filesystem::path full = ckpt::path(preload);
        if (m_trainer->net().load(full)) {
            m_trainer->resetBaseline();
            m_currentModelName = preload;      // ★ 它成了"当前模型"，可以直接覆盖
            m_statusText  = std::string("已加载  ") + full.string();
            m_statusTimer = 4.0f;
        } else {
            m_statusText  = std::string("加载失败（文件不存在或维度不匹配）：") + full.string();
            m_statusTimer = 6.0f;
        }
    }

    // -----------------------------------------------------------------------
    //  ★★ 敌方神经网络：开训前先写一次【初始快照】
    //
    //  ★ 为什么这里是必须的
    //    敌方脑（"nn-enemy"）在 Arena::reset() 里创建，那时会去读快照文件。
    //    如果这个文件还不存在（第一次训练），敌方就会"加载失败"→ 变成随机蛇，
    //    而且它会一直随机到第 100 局第一次同步为止 ——
    //    前 100 局的对手是无意义的，那段数据也白跑了。
    //
    //    所以在这里先写一次：敌方一开始就是"当前网络的拷贝"。
    //    因为 m_trainer 是刚构造的（可能刚加载过模型），这一步的语义是
    //    "敌方从和我一样的水平开始" —— 这正是自对弈该有的起点。
    // -----------------------------------------------------------------------
    ckpt::ensureDir();                       // 快照写在 saves/ 下，目录得先存在
    if (!m_trainer->writeMirrorSnapshot()) {
        m_statusText  = "警告：敌方快照写入失败，敌方本局会是随机网络";
        m_statusTimer = 8.0f;
    }

    m_arena = std::make_unique<arena::Arena>(rl::makeArenaConfig(m_task));

    m_runner = std::make_unique<rl::EpisodeRunner>(*m_arena, m_trainer.get(), m_task);
    // ★ 学习对象固定是第 0 条（Arena 里第一条出生的蛇）
    m_runner->setLearnerIndex(0);

    // 把 Runner 接到 Arena 的 controller 上
    m_arena->setController([this](std::size_t i, const arena::Perception& p) {
        return m_runner->decide(i, p, m_rng);
    });

    m_runner->begin(rl::RunMode::Train);

    // 镜头回到地图中心，缩放设成"整张图刚好放进视口"
    m_camCenter = sf::Vector2f(m_arena->mapWidth() * 0.5f,
                               m_arena->mapHeight() * 0.5f);
    m_camZoom = fitZoom();

    // 镜头跟随由【训练设置】界面决定。多蛇也能跟随 —— 默认跟学习中的那条（#0），
    // 想看别的按 Tab 切换。
    m_followCamera = m_task.followCamera;
    m_followIndex  = 0;
}

// ===========================================================================
//  让整张地图刚好放进视口的缩放系数
// ===========================================================================
float LiveTrainingScene::fitZoom() const {
    if (!m_arena) return 1.0f;

    const float W = static_cast<float>(Constants::WINDOW_WIDTH);
    const float H = static_cast<float>(Constants::WINDOW_HEIGHT);
    const float zx = m_arena->mapWidth()  / W;
    const float zy = m_arena->mapHeight() / H;
    // 取【大的】那个才能保证两个方向都装得下
    const float z = std::max(zx, zy);
    return std::max(0.2f, std::min(8.0f, z));
}

// ---------------------------------------------------------------------------
//  把命名缓冲（码点序列）转成 UTF-8 字符串
//
//  ★ 为什么要过 sf::String
//    项目铁律：sf::String 从 std::string 构造时按【本地编码】解释，不是 UTF-8。
//    反过来，要把 Unicode 码点变成【UTF-8 字节】给文件系统用，
//    最可靠的就是 sf::String::toUtf8() —— 它做的是真正的 Unicode → UTF-8 编码，
//    而不是按本地代码页猜。
//
//  ★ 为什么文件名用 UTF-8
//    本项目的源码、权重文件、日志全是 UTF-8，文件名保持同一套编码，
//    才不会出现"游戏里存的名字，命令行 ls 出来是乱码"。
// ---------------------------------------------------------------------------
std::string LiveTrainingScene::nameBufferUtf8() const {
    if (m_nameBuffer.empty()) return std::string();
    sf::String s;
    for (sf::Uint32 cp : m_nameBuffer) s += cp;
    // ⚠️ toUtf8() 返回的是 basic_string<sf::Uint8>（unsigned char），
    //    不是 std::string —— SFML 2.6 就是这样。所以要显式转一次，
    //    不能直接 return（编译器报 "could not convert basic_string<unsigned char>"）。
    const std::basic_string<sf::Uint8> u8 = s.toUtf8();
    return std::string(u8.begin(), u8.end());
}

// ===========================================================================
void LiveTrainingScene::handleEvent(const sf::Event& event) {
    // -----------------------------------------------------------------------
    //  ★ 命名输入模式：优先吃掉所有按键（和 SettingsScene 的做法一致）
    //
    //  ★ 必须先处理 TextEntered 再处理 KeyPressed
    //    因为打字依赖 TextEntered 事件，而 Enter/ESC/退格是 KeyPressed。
    //    两个都在这一个函数里分流，所以顺序不能反。
    // -----------------------------------------------------------------------
    if (m_nameInputMode) {
        if (event.type == sf::Event::TextEntered) {
            const sf::Uint32 ch = event.text.unicode;

            // ★ 只允许"当文件名安全"的字符
            //
            //   Windows 文件名里不能出现 \ / : * ? " < > |，
            //   而中文/空格/连字符/下划线/点/数字/字母都可以。
            //   所以这里用【白名单】而不是黑名单 —— 黑名单永远会漏。
            const bool ok = (ch >= L'0' && ch <= L'9')
                         || (ch >= L'a' && ch <= L'z')
                         || (ch >= L'A' && ch <= L'Z')
                         || (ch >= 0x4E00 && ch <= 0x9FFF)   // CJK 汉字
                         || ch == L'_' || ch == L'-' || ch == L'.' || ch == L' ';
            if (ok && m_nameBuffer.size() < 40) {
                m_nameBuffer.push_back(ch);
            }
            return;
        }
        if (event.type != sf::Event::KeyPressed) return;

        switch (event.key.code) {
            case sf::Keyboard::Enter:
                if (!m_nameBuffer.empty()) {
                    // ★ 重名直接覆盖（负责人定的）—— 不做二次确认，
                    //   因为多一层确认就多一个按键状态，而这里要的是"存得快"。
                    saveCheckpoint(nameBufferUtf8() + ".txt");
                }
                m_nameInputMode = false;
                m_nameBuffer.clear();
                m_paused = false;          // 命名结束，继续训练
                break;
            case sf::Keyboard::Escape:
                m_nameInputMode = false;
                m_nameBuffer.clear();
                m_paused = false;
                m_statusText  = "已取消命名";
                m_statusTimer = 2.0f;
                break;
            case sf::Keyboard::Backspace:
                // ★ 存的是码点，所以退格就是弹一个元素 —— 不会切坏汉字
                if (!m_nameBuffer.empty()) m_nameBuffer.pop_back();
                break;
            default: break;
        }
        return;   // ★ 不能穿透到下面的普通按键
    }

    if (event.type != sf::Event::KeyPressed) return;

    // ---- 正在选存档：按键先给这个面板 ----
    if (m_loadListOpen) {
        const int n = static_cast<int>(m_checkpoints.size());
        switch (event.key.code) {
            case sf::Keyboard::Up:
                if (n > 0) m_loadCursor = (m_loadCursor - 1 + n) % n;
                break;
            case sf::Keyboard::Down:
                if (n > 0) m_loadCursor = (m_loadCursor + 1) % n;
                break;
            case sf::Keyboard::Enter:
            case sf::Keyboard::Space:
                loadSelected();
                break;
            case sf::Keyboard::Escape:
                m_loadListOpen = false;
                break;
            default: break;
        }
        return;   // ★ 不能穿透到下面的普通按键 —— 否则 ESC 会连带退回主菜单
    }

    switch (event.key.code) {
        case sf::Keyboard::Enter:
        case sf::Keyboard::Space:
            m_paused = !m_paused;
            break;

        case sf::Keyboard::N:
            // 暂停时单步 —— 一帧一帧看它怎么决策
            if (m_paused) m_runner->tick(DT);
            break;

        case sf::Keyboard::R:
            restartTraining();
            m_paused = false;
            break;

        case sf::Keyboard::V:
            m_renderWorld = !m_renderWorld;
            break;

        case sf::Keyboard::U:
            // 不限速：每帧跑满 4ms。想尽快看到"学会了没有"时用
            m_unlimited = !m_unlimited;
            break;

        case sf::Keyboard::F:
            // 跟随镜头。多蛇时可以跟随（用 Tab 切换跟哪条）
            m_followCamera = !m_followCamera;
            break;

        case sf::Keyboard::S:
            // -------------------------------------------------------------------
            //  ★ 保存当前权重 —— 2026-09-24 改成"先命名再存"
            //  ★ 为什么不再直接存
            //    原来按 S 存出来的名字是 `ep000123_20260923_091530.txt`：
            //    它能分辨版本，但**没法按用途命名**（"学会觅食那版"/"调奖励前对照版"）。
            //    所以改成弹出输入行，输完 Enter 才真的写盘。
            //
            //  ★ 保留"不命名也能存"的退路吗？—— 不留。
            //    两条路径会让"按 S 到底会发生什么"变得不确定，
            //    而按 ESC 退出本来就会自动存一份 auto_latest（见 autoSave）。
            //    所以：想留档就命名，想接着练就什么都不用做。
            // -------------------------------------------------------------------
            m_nameInputMode = true;
            m_nameBuffer.clear();
            {
                // 预填自动名（去掉 .txt），用户可以直接回车，也可以改掉
                std::string autoName = ckpt::makeName(m_episodes);
                if (autoName.size() > 4 &&
                    autoName.compare(autoName.size() - 4, 4, ".txt") == 0) {
                    autoName.erase(autoName.size() - 4);
                }
                const sf::String ws = sf::String::fromUtf8(autoName.begin(), autoName.end());
                for (std::size_t i = 0; i < ws.getSize(); ++i) {
                    m_nameBuffer.push_back(ws[i]);
                }
            }
            m_paused = true;    // 命名时先停住，别让局数继续跑
            break;

        case sf::Keyboard::O:
            // ★ 直接覆盖"当前模型"（负责人要求）—— 不产生新文件，原地更新
            overwriteCurrentModel();
            break;

        case sf::Keyboard::L:
            // 打开存档列表
            openLoadList();
            break;

        case sf::Keyboard::Tab:
            // 切换跟随的目标（多蛇时用）
            if (m_followCamera && m_task.snakeCount() > 1) {
                m_followIndex = (m_followIndex + 1)
                              % static_cast<int>(m_arena->agents().size());
            }
            break;

        // ---- 蛇数等环境参数已经移到【训练设置】界面 ----
        //
        // 原来把 1/2/3 绑在这里改蛇数，有两个毛病：
        //   1. 改蛇数会重建世界、清空网络 —— 训练到一半按到，进度全没了
        //   2. 按键改参数不好发现（按了没反应，分不清是没编译还是没生效）
        // 现在改成"开始训练前先在设置界面定好"，和「开始游戏 → 对局设置」一致。
        // 本场景只保留"运行中随时会用"的操作（暂停/单步/速度/镜头/画面）。

        // ---- 速度：用 [ 和 ]，不再占方向键 ----
        //
        // ★ 原来绑在 ↑↓ 上，和"视角漫游"打架 —— 负责人要求把方向键让出来。
        //   现在 ↑↓ 是纯镜头上下，[ ] 是速度。
        case sf::Keyboard::LBracket:
            setSpeedIndex(std::max(-2, m_speedIndex - 1));
            break;
        case sf::Keyboard::RBracket:
            setSpeedIndex(std::min(SPEED_MAX_INDEX, m_speedIndex + 1));
            break;

        case sf::Keyboard::Escape:
            // -------------------------------------------------------------------
            //  ★ 退出前【自动存档】（负责人要求 2026-09-24）
            //
            //  ★ 修的是什么
            //    以前这里直接就 switchScene 回主菜单，**什么都不存**。
            //    而进训练模式时网络是用固定种子重新初始化的 ——
            //    于是"练了半小时、按 ESC 全没了"。
            //
            //  ★ 存档名固定 auto_latest.txt、每次覆盖
            //    它表达"这次训练的最新进度"，语义上就该只有一份。
            //    要留多个版本用 S 自己命名（上面那一套）。
            //
            //  ★ 没练过就不存（autoSave 里判的）
            //    免得"进来就退出"用随机初始权重覆盖掉上次真正练过的进度。
            // -------------------------------------------------------------------
            autoSave();
            m_game->switchScene(std::make_unique<MenuScene>(m_game));
            break;

        default: break;
    }
}

void LiveTrainingScene::fixedUpdate(float) {}

// ===========================================================================
//  速度档位
// ===========================================================================
void LiveTrainingScene::setSpeedIndex(int idx) {
    m_speedIndex = std::max(-2, std::min(SPEED_MAX_INDEX, idx));
    m_unlimited  = false;

    // ★ 把实测倍速的两个累计量同时清零。
    //   倍速 = 仿真时间 / 墙钟时间，两个都从 0 起算，得到的才是【新档位】的真实倍速；
    //   不清零就会把旧档位跑过的时间混进来。
    m_simSeconds  = 0.0;
    m_frameCounter = 0;
    m_speedClock.restart();
    m_speedX = 0.0;
}

// ---------------------------------------------------------------------------
//  这一帧跑几步
//
//    > 0 ：就跑这么多步（慢动作段的"不跑"由 <0 表示）
//    < 0 ：这一帧不跑
//     0 ：走时间预算（吞吐段 / 不限速）
// ---------------------------------------------------------------------------
long LiveTrainingScene::stepsThisFrame() {
    if (m_speedIndex < 0) {
        // ---- 慢动作：每 n 帧跑 1 步 ----
        const int n = SPEED_SLOW_STEPS[(-m_speedIndex) - 1];
        ++m_frameCounter;
        if (m_frameCounter < n) return -1;
        m_frameCounter = 0;
        return 1;
    }
    if (m_speedIndex <= SPEED_FIXED_MAX) {
        // ---- 固定步数：每帧正好 2^k 步 ----
        return 1L << m_speedIndex;
    }
    // ---- 吞吐段：走时间预算 ----
    return 0;
}

// ===========================================================================
//  自由镜头：方向键漫游、PgUp/PgDn 缩放
//
//  ★ 和观战模式（GameScene 里玩家死后的分支）同一套手感与键位。
//    **不跟踪 AI** —— 训练时蛇会到处乱跑，跟镜头既看不清全局
//    （食物圈分布），镜头也一直在晃。
// ===========================================================================
void LiveTrainingScene::updateCamera(float dt) {
    if (!m_arena || m_arena->agents().empty()) return;

    // ---- 跟随（单蛇、多蛇都行；多蛇用 Tab 切换跟哪条）----
    //
    // ★ 默认是关的：开局给"整张地图刚好放进视口"的自由镜头，
    //   这样才看得见食物圈的分布（那是全局信息）。
    if (m_followCamera) {
        const int n = static_cast<int>(m_arena->agents().size());
        if (m_followIndex < 0 || m_followIndex >= n) m_followIndex = 0;
        m_camCenter = m_arena->agents()[static_cast<std::size_t>(m_followIndex)]
                          .body.getHeadPosition();
        return;
    }

    const float speed = 500.0f * m_camZoom;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))   m_camCenter.x -= speed * dt;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))  m_camCenter.x += speed * dt;
    // ★ ↑↓ 现在【纯粹给镜头上下】—— 速度改到 [ ] 了，不再抢这两个键
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))     m_camCenter.y -= speed * dt;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))   m_camCenter.y += speed * dt;

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::PageUp))
        m_camZoom = std::max(0.2f, m_camZoom - 0.6f * dt);
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::PageDown))
        m_camZoom = std::min(8.0f, m_camZoom + 0.6f * dt);

    // 别飘到地图外面去
    m_camCenter.x = std::max(0.0f, std::min(m_arena->mapWidth(),  m_camCenter.x));
    m_camCenter.y = std::max(0.0f, std::min(m_arena->mapHeight(), m_camCenter.y));
}

// ===========================================================================
//  推进
//
//  ★ 速度分三段（见 stepsThisFrame() 与头文件里的 SPEED_* 常量）
//
//      慢动作段        每 n 帧跑 1 步
//      固定步数段      每帧正好 2^k 步    ← 倍速精确、可验证
//      吞吐段          按时间预算跑，取实测能达到的最大倍速
//
//  ★ 为什么低倍速必须按"每帧几步"精确控制
//
//  第一版全程用"每帧的时间预算 ÷2 / ×2"，结果**降不下去**：
//  预算分到 0.5ms 本来就该很慢，但自适应批量有 `batch = max(batch, 1)`
//  的下限（每帧至少跑 1 步），于是实际速度被"每帧步数 × 60 帧"兜底住了。
//
//  ★ 为什么高倍速必须用时间预算
//
//  "每帧 N 步"有个天花板：主循环被 vsync 锁在 60 帧/秒，
//  N=64 时就是 3840 tick/秒 —— **关掉渲染也快不了**
//  （省的是绘制，loop 仍然只有 60 帧）。
//
//  ★ 每一次推进仍然是【固定步长 DT】
//    所以无论跑多快，物理和训练时看到的世界完全一致。
//    这和"把 dt 乘 N"有本质区别（后者会让转向、碰撞、缓动全部变形）。
// ===========================================================================
void LiveTrainingScene::update(float /*dt*/) {
    updateCamera(1.0f / 60.0f);      // 镜头用固定 dt，与帧率无关

    // 状态提示倒计时（暂停时也要走，否则提示永远不消失）
    if (m_statusTimer > 0.0f) m_statusTimer -= 1.0f / 60.0f;

    if (m_paused || !m_runner) return;

    // 这一帧该跑几步：见 stepsThisFrame() 的说明
    const long fixed = stepsThisFrame();
    if (fixed < 0) return;           // 慢动作段的"这一帧不跑"

    auto doTick = [this]() {
        m_runner->tick(DT);

        // ★ 每 tick 抓一次【实时】饥饿值（局末那一段只是再同步一次）
        //   这样面板上"还剩几秒会饿死"才是活的，而不是一局只变一次。
        {
            const rl::EpisodeStats& cur = m_runner->stats();
            m_curHungerLeft   = cur.curHungerLeft;
            m_curTimeSinceEat = cur.curTimeSinceEat;
        }

        if (m_runner->done()) {
            const rl::EpisodeStats& s = m_runner->stats();

            // ★ 局末才更新网络 —— 这一步就是"学习"发生的地方
            m_trainer->finishEpisode();
            m_trainer->countEpisode();

            ++m_episodes;
            m_lastReward = s.totalReward;
            m_lastTicks  = s.ticks;
            m_lastEaten  = s.eaten;

            // 出生/死亡诊断
            m_lastSpawnX        = s.spawnX;
            m_lastSpawnY        = s.spawnY;
            m_lastSpawnWallDist = s.spawnDistanceToWall;
            m_lastDeathX        = s.deathX;
            m_lastDeathY        = s.deathY;
            m_lastDeathWallDist = s.deathDistanceToWall;
            m_lastMaxSegs       = s.maxSegments;
            m_lastCause         = s.cause;
            m_lastStarveTime    = s.starvationTime;
            m_lastTimeSinceEat  = s.timeSinceEat;
            m_lastHeadR         = s.headRadius;
            // ★★ ⑪ 的本局累计（给"这一项到底扣了多少"一个不含平滑的读数）
            m_lastWallD         = s.wallDangerGain;
            m_lastEnemyD        = s.enemyDangerGain;
            // ★ 实时饥饿值也要跟着走 —— 否则一局刚结束的那一瞬间，
            //   面板上显示的会是上一帧的旧值（看着像"卡住了"）
            m_curHungerLeft     = s.curHungerLeft;
            m_curTimeSinceEat   = s.curTimeSinceEat;

            m_recentReward.push_back(s.totalReward);
            m_recentEaten.push_back(s.eaten);
            if (static_cast<int>(m_recentReward.size()) > HISTORY) {
                m_recentReward.erase(m_recentReward.begin());
                m_recentEaten.erase(m_recentEaten.begin());
            }

            // 奖励构成的滑动平均（α = 0.02 ≈ 最近 50 局的均值）
            auto ema = [](double& acc, double v) { acc = acc * 0.98 + v * 0.02; };
            ema(m_avgDie,      s.diePenalty);
            ema(m_avgEat,      s.eatGain);
            ema(m_avgHunger,   s.hungerGain);
            // ④ approach 已整项删除（2026-09-26）
            ema(m_avgFed,      s.fedGain);
            ema(m_avgEscape,   s.escapeGain);  // ⑥′ 逃生奖励
            ema(m_avgHunt,     s.huntGain);    // ⑦ 猎杀优势
            ema(m_avgKill,     s.killGain);    // ⑧ 击杀奖励
            ema(m_avgDisp,     s.dispGain);    // ⑩′ 饥饿时位移不足扣分（负数）
            ema(m_avgWallD,    s.wallDangerGain);   // ⑪ 墙危险距离扣分（负数）
            ema(m_avgEnemyD,   s.enemyDangerGain);  // ⑪ 敌身危险距离扣分（负数）

            m_runner->begin(rl::RunMode::Train);
        }
    };

    long ran = 0;

    if (fixed > 0) {
        // ---- 慢动作段 / 固定步数段：跑【正好这么多步】----
        //
        // 这是唯一能让倍速"说到做到"的做法：
        // 1x = 每帧 1 步、2x = 每帧 2 步 …… 与渲染快慢无关。
        for (long i = 0; i < fixed; ++i) doTick();
        ran = fixed;
    } else {
        // ---- 吞吐段 / 不限速：按时间预算跑，跑多少算多少 ----
        //
        // ★ 自适应批量：用【实测单步成本】反推"这点预算还能跑几步"，
        //   跑一小批就重新查表，并随实际耗时动态收缩。
        //
        //   第一版是"每 256 步查一次时钟"。单步约 6 微秒时没问题，
        //   但渲染开着时单步成本会上升，一次 overshoot 就能吃掉几百毫秒，
        //   于是界面卡成"未响应"。
        const double frameBudgetMs = m_unlimited
            ? 4.0
            : static_cast<double>(SPEED_BUDGET_MS[m_speedIndex - SPEED_BUDGET_FIRST]);

        const auto t0 = std::chrono::steady_clock::now();
        long batch = 64;

        while (true) {
            const long n = (batch < 1) ? 1 : batch;
            for (long i = 0; i < n; ++i) doTick();
            ran += n;

            const double used = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (used >= frameBudgetMs) break;

            const double measuredUs = used * 1000.0 / static_cast<double>(n);
            m_costPerTickUs = m_costPerTickUs * 0.7 + measuredUs * 0.3;

            const double canDo = ((frameBudgetMs - used) * 1000.0 / m_costPerTickUs) * 0.75;
            batch = static_cast<long>(canDo);
            if (batch < 1) batch = 1;
            if (batch > 50000) batch = 50000;   // 单批别太久，保证能及时回来刷界面
        }
    }

    // ---- 实测倍速 ----
    //
    // ★ 倍速 = 仿真时间 / 墙钟时间。两个累计量在改档位时同时清零，
    //   所以这个数就是【当前档位】的真实倍速。
    //
    //   为什么不用"每帧步数 × 60"硬算：高倍速段实际帧率会被渲染拖下来，
    //   硬算的值和真相差很远 —— 那就又变成"档位 ≠ 速度"了。
    m_simSeconds += static_cast<double>(ran) * static_cast<double>(DT);
    const double wall = m_speedClock.getElapsedTime().asSeconds();
    if (wall >= 0.4) {
        m_speedX = m_simSeconds / wall;
        m_simSeconds = 0.0;
        m_speedClock.restart();
    }
}

// ===========================================================================
//  画世界
// ===========================================================================
void LiveTrainingScene::renderWorld(sf::RenderWindow& window, const sf::View& view) const {
    if (!m_arena) return;

    const float mapW = m_arena->mapWidth();
    const float mapH = m_arena->mapHeight();

    // 地图底色
    sf::RectangleShape bg(sf::Vector2f(mapW, mapH));
    bg.setFillColor(sf::Color(22, 22, 38));
    window.draw(bg);

    // 边界
    sf::RectangleShape border(sf::Vector2f(mapW, mapH));
    border.setFillColor(sf::Color::Transparent);
    border.setOutlineColor(sf::Color(233, 69, 96, 120));
    border.setOutlineThickness(4.0f);
    window.draw(border);

    // ---- 食物圈（淡色填充，和 GameScene 一致；活跃圈更亮）----
    //
    // ★★ 2026-09-26 修"假圈"（负责人反复报的问题）
    //  ★★ 2026-09-27 定稿（负责人定义）：**「下一个圈」只在预告期可见**
    //
    //    "不要一开始就有预告圈，预告圈是在活跃圈产食物还剩 3s 后出现的，
    //      然后中间有一定时间 gap，然后再被激活。"
    //    判据的唯一实现在 `ZoneSchedule::previewing()`，这里不重推算式。
    //
    //    ⚠️ 与 `PerceptionBuilder` / `GameScene` / `Minimap` 必须一致 ——
    //       这是本项目惯犯级的坑（同一渲染规则多份实现）。
    //
    //  ★ 教训（仍然有效）：画圈这件事**有两处实现**（GameScene / LiveTrainingScene），
    //     改这里必须**两处一起改**。
    {
        const std::vector<arena::FoodZone>& zones = m_arena->food().zones();
        const arena::FoodZone* active = m_arena->activeZone();
        const bool spawning = m_arena->zoneSchedule().spawning(
            m_arena->food().zoneTimer());
        const bool previewing = m_arena->zoneSchedule().previewing(
            m_arena->food().zoneTimer());

        for (const arena::FoodZone& z : zones) {
            const bool isActive = (&z == active);
            if (!isActive && !previewing) continue;   // ★ 非活跃圈只在预告期画

            sf::CircleShape ring(z.radius);
            ring.setOrigin(z.radius, z.radius);
            ring.setPosition(z.x, z.y);
            ring.setOutlineThickness(0.0f);
            if (isActive && spawning)      ring.setFillColor(sf::Color(255, 220, 140, 40));
            else if (isActive)             ring.setFillColor(sf::Color(140, 200, 255, 26));
            else                           ring.setFillColor(sf::Color(200, 200, 200, 20));
            window.draw(ring);
        }
    }

    // 食物
    m_arena->food().render(window);

    // ---- 蛇 ----
    //
    // ★ 学习中的那条【最后画】并加一圈白边 —— 多蛇时必须一眼看出
    //   "哪条是我在训的"。否则 9 条蛇混在一起完全分不清。
    const std::size_t learner = m_runner ? m_runner->learnerIndex() : 0u;
    const auto& agents = m_arena->agents();

    for (std::size_t i = 0; i < agents.size(); ++i) {
        if (i == learner) continue;
        agents[i].body.render(window);
    }
    if (learner < agents.size()) {
        // 先用一个略大的白圈垫底，形成描边效果
        const Snake& s = agents[learner].body;
        if (s.isAlive()) {
            sf::CircleShape halo(s.getHeadRadius() + 5.0f);
            halo.setOrigin(s.getHeadRadius() + 5.0f, s.getHeadRadius() + 5.0f);
            halo.setPosition(s.getX(), s.getY());
            halo.setFillColor(sf::Color::Transparent);
            halo.setOutlineColor(sf::Color(255, 255, 255, 200));
            halo.setOutlineThickness(2.0f);
            window.draw(halo);
        }
        s.render(window);
    }

    (void)view;
}

// ===========================================================================
//  左上角数据面板
// ===========================================================================
// ===========================================================================
//  左上角数据面板
//
//  ★ 版式原则：**每个分节用固定坐标，不累加**
//
//  第一版是让一个 line() 助手自己往下推 y。加了几个诊断之后，
//  累计高度算不准，而后面"状态提示"又把 y 硬设回 320 ——
//  结果提示叠在曲线和诊断上，字全挤在一起（负责人报的问题）。
//
//  所以现在改成分节常量：每节从自己的 Y 开始，节内再逐行累加。
//  这样加/删任何一行，最多影响它所在的那一节，不会串到别处。
// ===========================================================================
namespace {
constexpr float PANEL_X   = 22.0f;    // 正文左边距
constexpr float PANEL_W   = 620.0f;   // 面板宽度

// ---------------------------------------------------------------------------
//  ★★★ 版面：从"手写分节坐标"改成"游标自动下推"（2026-09-26 深夜 第二次修叠字）
//
//  ★ 历史（两次同一个坑，必须记住）
//    1) 第一版：一个 `line()` 助手自己往下推 y，但后面的状态提示又把 y
//       **硬设回**一个常量 → 两段文字重叠。
//    2) 第二版：改成"固定分节常量"（每个 SECTION_* 一个写死的 y）。
//       这解决了"硬设回去"的问题，但引入了新问题：
//       **每加一行就要人手改后面所有常量，漏改一次就叠字。**
//       ⑪ 危险距离扣分让"奖励构成"从 1 行变 2 行，我只往下挪了后面的节，
//       却漏算了"核心数字"那一节本身已经变高 → 负责人第二次看到叠字。
//
//  ★ 根本原因不是某个数写错，而是**这个版面允许"节与节互相盖住"**。
//    只要位置靠人手维护，就一定会再错。
//
//  ★ 现在的做法：**游标**
//      · `panelCursor` = "下一节可以从这里开始"
//      · 每节开始时取 `nextSectionY()`，画完把 y 推回去
//      · 节与节之间由 `SECTION_GAP` 统一给间距
//    于是"某节变高"会自动把后面全部推下去 —— **叠字在结构上不可能发生**，
//    面板高度也由内容量出来（见 renderPanel 里的 contentBottom）。
//
//  ⚠️ 保留一条人工判断：面板不能高过窗口（720）。
//    内容太多时宁可删信息，也不要让它顶到屏幕外。
// ---------------------------------------------------------------------------
constexpr float PANEL_TOP   = 56.0f;   // 第一节的起点（标题在 18~46）
constexpr float SECTION_GAP = 10.0f;   // 节与节之间的空白

// 回报曲线高度。
// ★ 2026-09-26 深夜从 84 压到 64：面板内容变多，曲线是"看趋势"用的，
//   压缩它代价最小。**现在压缩它不再需要同步改别的常量**（游标会自动推）。
constexpr float PLOT_H      = 64.0f;
}  // namespace

void LiveTrainingScene::renderPanel(sf::RenderWindow& window) {
    // =======================================================================
    //  ★★★ 面板背景改成【先画内容、量出实际高度、最后补背景】
    //
    //  ★ 为什么必须这样（2026-09-26 深夜，负责人第二次报"字又叠一起"）
    //
    //    以前是**先按一个写死的 PANEL_H 画黑底**，再往里填字。
    //    于是只要内容比 PANEL_H 高，多出来的字就**画出面板之外**、
    //    落在游戏画面上 —— 看起来就是"字叠在一起、看不清"。
    //
    //    根因不是某一个 y 常量写错，而是**这个设计允许"内容超过背景"发生**：
    //    每加一行就要人手去改所有 SECTION_* 常量，而漏改一次就是这个 bug。
    //    这个坑已经踩过两次（第一次是 SECTION_HINT 被两段共用，第二次是现在）。
    //
    //  ★ 现在改成"内容驱动"：
    //      · `put()` 记下每一行的底部 → `contentBottom`（所有 text 的最大值）
    //      · 面板高度 = max(PANEL_H, contentBottom + 余量)
    //      · 背景在【最后】画，所以一定盖得住全部文字
    //    ★ 于是"加一行忘了改常量"不再可能造成叠字 —— 面板会自己变高。
    //
    //  ⚠️ 唯一剩下的约束是**窗口高度**：面板不能高过屏幕。
    //    如果 contentBottom 逼近窗口高度，说明面板该精简了（或该分页），
    //    那时下面的 clamp 会让它贴底，而不是溢出。
    // =======================================================================
    float contentBottom = 0.0f;

    // -----------------------------------------------------------------------
    //  ★ 版面游标：下一节从哪开始（见文件头 PANEL_* 常量的说明）
    //
    //    `nextSectionY()`  取下一节的起点（并留出节间距）
    //    `commitSection(y)` 把这一节画完的底部交回去 —— 下一节据此下推
    //
    //  ★ 为什么必须是"画完再交回底部"而不是"进入时就算好"
    //    因为一节里有几行、每行多高，只有画的时候才知道。
    //    先算后画就又回到了"人手维护坐标"的老路。
    // -----------------------------------------------------------------------
    float panelCursor = PANEL_TOP;
    auto nextSectionY = [&]() -> float {
        return panelCursor + (panelCursor > PANEL_TOP ? SECTION_GAP : 0.0f);
    };
    auto commitSection = [&](float bottom) {
        if (bottom > panelCursor) panelCursor = bottom;
    };

    // 往 (x, y) 画一行；返回下一行的 y
    //
    // ★ 每一行都会更新 contentBottom —— 这样"面板多高"是【量出来的】，
    //   不是猜出来的。注意 putS 走 put，所以两条路径都覆盖到。
    auto put = [&](float x, float y, const std::wstring& w,
                   const sf::Color& c, int size) -> float {
        sf::Text t(w, m_font, size);
        t.setFillColor(c);
        t.setPosition(x, y);
        window.draw(t);
        const float next = y + static_cast<float>(size) + 5.0f;
        if (next > contentBottom) contentBottom = next;
        return next;
    };
    auto putS = [&](float x, float y, const std::string& s,
                    const sf::Color& c, int size) -> float {
        return put(x, y, textutil::utf8(s), c, size);
    };

    // ---- 标题 + 任务名 ----
    put(PANEL_X, 18.0f, L"训练模式", sf::Color(78, 205, 196), 23);
    put(PANEL_X + 118.0f, 24.0f, rl::taskKindName(m_task.kind),
        sf::Color(255, 210, 90), 18);

    // ---- 任务参数（两行）----
    {
        float y = nextSectionY();
        std::ostringstream a;
        a << "map " << static_cast<int>(m_task.mapWidth)
          << "x" << static_cast<int>(m_task.mapHeight)
          << "   snakes " << m_task.snakeCount()
          << "   episode " << (m_task.maxTicks / 60) << "s";
        y = putS(PANEL_X, y, a.str(), sf::Color(170, 170, 195), 15);

        const double foodPerSec = (m_task.foodPerSecondOverride > 0.0)
                                ? m_task.foodPerSecondOverride
                                : m_task.foodPerSnakePerSecond * m_task.snakeCount();
        std::ostringstream b;
        b.precision(3);
        b << "food " << foodPerSec << "/s";
        switch (m_task.kind) {
            case rl::TaskKind::AheadFood:
                b << "   fixed " << static_cast<int>(m_task.fixedFoodDistance) << "px ahead x" << m_task.fixedFoodCount;
                break;
            case rl::TaskKind::ZoneMini:
                b << "   zone r" << static_cast<int>(m_task.mapWidth * m_task.zoneRadiusRatio) << " centered";
                break;
            default:
                b << "   zones " << m_task.zoneCount << "   cycle " << static_cast<int>(m_task.zoneSpawnSeconds + m_task.zoneGapSeconds) << "s";
                break;
        }
        y = putS(PANEL_X, y, b.str(), sf::Color(170, 170, 195), 15);
        commitSection(y);
    }

    // ---- 核心数字 ----
    {
        float y = nextSectionY();
        std::ostringstream a;
        a << "episode " << m_episodes;
        y = putS(PANEL_X, y, a.str(), sf::Color(235, 235, 235), 17);

        std::ostringstream b;
        b.precision(3);
        b << "reward " << m_lastReward
          << "   food " << m_lastEaten
          << "   survival " << m_lastTicks << "/" << m_task.maxTicks;
        y = putS(PANEL_X, y, b.str(), sf::Color(235, 235, 235), 17);

        // 平滑值（单局噪声太大，看不出趋势）
        double avgR = 0.0, avgE = 0.0;
        for (double v : m_recentReward) avgR += v;
        for (int v : m_recentEaten)     avgE += v;
        if (!m_recentReward.empty()) {
            avgR /= static_cast<double>(m_recentReward.size());
            avgE /= static_cast<double>(m_recentEaten.size());
        }
        std::ostringstream c;
        c.precision(3);
        c << "avg(200) reward " << avgR << "   food " << avgE;
        y = putS(PANEL_X, y, c.str(), sf::Color(255, 200, 60), 17);

        std::ostringstream d;
        d.precision(3);
        d << "baseline " << m_trainer->baseline()
          << "   params " << m_trainer->paramCount();
        y = putS(PANEL_X, y, d.str(), sf::Color(150, 200, 255), 17);

        // ★★ 2026-09-26 深夜：熵项占了多少梯度
        //
        //  ★ 为什么要看这一行
        //    负责人报的"越训 reward 越小、最后全是原地旋转扣分"属于
        //    **策略被训坏**，而不是"奖励设计不对"。而策略梯度 = 任务梯度 + 熵项，
        //    两者混在一个总范数里分不出来。
        //
        //  ★ 判据：熵项应当【远小于】总梯度。
        //    占比一旦不小，说明策略很大程度上是被熵项推着走的 ——
        //    而熵项只喊"别太确定"，不含任何任务信息。
        //
        //  ⚠️ 为了不把面板各行往下挤（曾经因为 y 坐标打架叠过一次字），
        //     这一条【并进 baseline 那一行】而不是另起一行。
        {
            const double gTot = m_trainer->lastGradNorm;
            const double gEnt = m_trainer->lastEntropyGradNorm;
            const double pct  = (gTot > 1.0e-12) ? 100.0 * gEnt / gTot : 0.0;
            std::ostringstream f;
            f.precision(2);
            f << "grad " << gTot << "   entropy " << gEnt
              << " (" << static_cast<int>(pct + 0.5) << "%)";
            y = putS(PANEL_X, y, f.str(),
                     (pct > 30.0) ? sf::Color(255, 120, 120) : sf::Color(170, 170, 170), 15);
        }

        // ★★ 2026-09-26：critic 诊断（actor-critic 模式下才显示）
        //
        //  ★ 为什么必须看这一行
        //    **如果价值网络学不起来，GAE 就是空的** —— 那时"换了算法"
        //    只是换了个说法，没有任何实质改善。
        //    判据：`V↔目标` 应该明显为正（V 与实际回报正相关）；
        //         持续接近 0 就说明 critic 没学到，该停下来查，而不是继续堆机制。
        //
        //  ⚠️ 但**它只衡量 critic，不衡量 actor**：
        //     corr = 0.5 只说明 V 能预测 25% 的回报方差，很容易靠"预测饥饿程度"拿到。
        //     判断"有没有在学"仍然要看吃食数（见下面的 avg food）。
        if (m_trainer->mode == rl::LearningMode::Critic) {
            std::ostringstream e;
            e.precision(3);
            e << "critic: V-target corr " << m_trainer->lastValueCorr
              << "   MSE " << m_trainer->lastValueLoss
              << "   advSD " << m_trainer->lastAdvStd;
            y = putS(PANEL_X, y, e.str(), sf::Color(180, 255, 180), 17);
        }
        commitSection(y);
    }

    // ---- ★ 看门狗：轨迹有没有被缓冲上限截断（缺陷 #46，2026-09-26 新增）----
    //
    //  ★ 为什么要在面板上【恒显示】而不是只在出问题时提示
    //    截断是**静默**的：不崩、不报错，现象只是"学不会"——
    //    正因如此它一直没被发现（90 秒的局只学了前 4096 帧，
    //    尾部 24% 的奖励连同死亡罚全丢）。
    //    恒显示一行绿色的"正常"，才能让人确信"这个前提检查过了"，
    //    而不是"没看到警告大概就是没问题"。
    {
        const float y = nextSectionY();
        std::wostringstream w;
        w << L"trace " << m_trainer->steps() << L"/" << m_trainer->maxStepsPerEpisode;
        float bottom;
        if (m_trainer->lastEpisodeTruncated) {
            // ★ 中文走宽字符串（SimHei 有这些字形；避开 ↑↓←→ 和全角括号）
            w << L"   警告: 轨迹被截断, 尾部奖励未学习";
            bottom = put(PANEL_X, y, w.str(), sf::Color(255, 90, 90), 15);
        } else {
            w << L"   ok";
            bottom = put(PANEL_X, y, w.str(), sf::Color(120, 190, 120), 15);
        }
        commitSection(bottom);
    }

    // ---- 奖励构成（⑪ 加入后是 2 行）----
    //
    // ★ 调奖励时最有用的信息：只看总回报分不清是哪个信号在推动学习。
    //   某一项恒为 0.00 就说明它从不触发。
    {
        float y = nextSectionY();
        std::ostringstream a;
        a.precision(2);
        a << std::fixed
          << "eat " << m_avgEat
          << "  die " << m_avgDie
          << "  fed " << m_avgFed
          << "  hunger " << m_avgHunger;
        y = putS(PANEL_X, y, a.str(), sf::Color(255, 180, 120), 15);

        // ⑥′⑦⑧⑩′⑪ 单独一行 —— 它们都是"事件式/对抗类"的项，放一起才好对照着看
        //   escape 应该 ≥ 0（每次从敌人身边成功拉开 +0.45）
        //   disp   应该 ≤ 0（饿着却 3 秒内几乎没动才扣）
        //   wall(11)/enemy(11) 应该 ≤ 0（贴墙/贴敌身，越近扣越多）
        //   恒为 0 说明"从不触发" —— 那就要去查触发条件，而不是调系数
        //
        //  ⚠️ 2026-09-26 深夜：这两行**分开写**（不再并成一行挤着）。
        //     以前怕"往下挤"所以硬塞进一行，结果是自己看不清；
        //     现在版面是游标驱动、面板高度是量出来的，**不再需要挤**。
        std::ostringstream b;
        b.precision(3);
        b << std::fixed
          << "escape(6) " << m_avgEscape
          << "   disp(10) " << m_avgDisp;
        y = putS(PANEL_X, y + 22.0f, b.str(), sf::Color(200, 170, 255), 15);

        std::ostringstream c;
        c.precision(3);
        c << std::fixed
          << "wall(11) " << m_avgWallD
          << "   enemy(11) " << m_avgEnemyD;
        y = putS(PANEL_X, y + 22.0f, c.str(), sf::Color(255, 150, 150), 15);

        // ★★ ⑪ 的【本局累计】—— 不带任何平滑
        //
        //  ★ 为什么必须单独一行显示它
        //    上面那行的 `wall(11)` 是 **EMA 平均（α=0.02 ≈ 最近 50 局）**，
        //    它会把单局的极端值抹平。负责人报过"撞死后才两三分"，
        //    而按常数算撞一次的累计应该是三位数 —— 这个矛盾
        //    只有把"本局累计"直接打出来才能一眼判断：
        //        · 这里显示 −120 量级 → 项在正常工作，之前看的是平均值
        //        · 这里也只有 −2~−3 → 判据确实没生效，得去查实现
        //
        //  ⚠️ 它和一局里"撞了几次墙"也有关：撞 1 次 ≈ 一次完整穿越的积分。
        std::ostringstream e2;
        e2.precision(2);
        e2 << std::fixed
           << "  last ep: wall " << m_lastWallD
           << "   enemy " << m_lastEnemyD
           << "   <-- 本局累计（无平滑）";
        y = putS(PANEL_X, y + 22.0f, e2.str(), sf::Color(255, 210, 120), 15);

        // ⚠️ 2026-09-27：这里曾经有一行 "mate agree xx%"（队友一致性探针）。
        //    已随探针一起删除 —— 它在 rl/ 层里用了 brain/ 的类型，违反分层铁律。
        //    负责人也已确认不再追这个问题。

        std::ostringstream d;
        d.precision(3);
        d << std::fixed
          << "hunt(7) " << m_avgHunt
          << "   kill(8) " << m_avgKill;
        y = putS(PANEL_X, y + 22.0f, d.str(), sf::Color(255, 190, 140), 15);
        commitSection(y);
    }

    // ---- ★ 实时饥饿（负责人要求：看得到还剩几秒会饿死）----
    //
    //  ★ 为什么单独占一节、而且醒目
    //    饥饿是主要死因之一，但以前面板上只有两个【局末】值
    //    （starvationTime / timeSinceEat），一局进行中它们不变 ——
    //    等于训练时根本看不到"它现在有多饿"。
    //
    //  ★ 颜色编码
    //      剩余 > 5s   → 绿（安全）
    //      剩余 2~5s   → 黄（该找吃的了）
    //      剩余 < 2s   → 红（马上饿死）
    {
        float y = nextSectionY();
        const float hl = m_curHungerLeft;
        const sf::Color c = (hl > 5.0f) ? sf::Color(150, 220, 150)
                          : (hl > 2.0f) ? sf::Color(255, 210, 90)
                                        : sf::Color(255, 110, 110);
        std::ostringstream a;
        a.precision(1);
        a << std::fixed
          << "hunger left " << hl << "s"
          << "    since eat " << m_curTimeSinceEat << "s";
        if (hl > 5.0f)       a << "    [ok]";
        else if (hl > 2.0f)  a << "    [go find food]";
        else                 a << "    [STARVING]";
        y = putS(PANEL_X, y, a.str(), c, 17);
        commitSection(y);
    }

    // ---- 出生 / 死亡诊断（压成 1 行）----
    //
    // ★ 为什么要有：负责人报"蛇刚出生就死了"，而"这一局结束了"分不清是
    //   【撞墙】【撞蛇】【饿死】【撞自己】—— 四种的修法完全不同。
    //
    // ★ 2026-09-24 精简：去掉了 headR（蛇头半径，只在"贴墙死"时要查）
    //   和 starve/tEat（它们已经被上面的【实时饥饿】一节取代）。
    {
        float y = nextSectionY();
        std::ostringstream a;
        a.precision(0);
        a << std::fixed
          << "spawn w" << m_lastSpawnWallDist
          << " -> death w" << m_lastDeathWallDist
          << "   segs " << m_lastMaxSegs;
        // cause 是中文，必须走宽字符串（sf::Text 不把窄字符串当 UTF-8）
        const std::wstring line = std::wstring(L"last: ") + arena::deathCauseName(m_lastCause)
                                + L"   " + textutil::utf8(a.str());
        y = put(PANEL_X, y, line,
                m_lastCause == arena::DeathCause::None
                    ? sf::Color(150, 220, 150) : sf::Color(230, 170, 90), 15);
        commitSection(y);
    }

    // ---- 回报曲线 ----
    if (m_recentReward.size() >= 2) {
        const float px = PANEL_X;
        const float py = nextSectionY();
        const float pw = PANEL_W - 44.0f;
        const float ph = PLOT_H;
        commitSection(py + ph + 6.0f);   // ★ 曲线也要"占位"，否则下一节会压在它上面

        sf::RectangleShape frame(sf::Vector2f(pw, ph));
        frame.setPosition(px, py);
        frame.setFillColor(sf::Color(18, 18, 32, 210));
        frame.setOutlineColor(sf::Color(70, 70, 90));
        frame.setOutlineThickness(1.0f);
        window.draw(frame);

        float lo = 1e30f, hi = -1e30f;
        for (double v : m_recentReward) {
            lo = std::min(lo, static_cast<float>(v));
            hi = std::max(hi, static_cast<float>(v));
        }
        if (hi - lo < 1e-6f) { lo -= 1.0f; hi += 1.0f; }

        sf::VertexArray poly(sf::LineStrip, m_recentReward.size());
        const std::size_t n = m_recentReward.size();
        for (std::size_t i = 0; i < n; ++i) {
            const float fx = px + pw * static_cast<float>(i)
                                    / static_cast<float>(n > 1 ? n - 1 : 1);
            const float fy = py + ph * (1.0f - (static_cast<float>(m_recentReward[i]) - lo)
                                              / (hi - lo));
            poly[i].position = sf::Vector2f(fx, fy);
            poly[i].color    = sf::Color(78, 205, 196);
        }
        window.draw(poly);
    }

    // ---- 运行状态：速度 + 策略概率 + 镜头 ----
    {
        float y = nextSectionY();

        // ★ 同时显示【档位】和【实测倍速】—— 差距本身是有用信息
        std::ostringstream a;
        if (m_paused) {
            a << "PAUSED";
        } else if (m_unlimited) {
            a << "UNLIMITED";
        } else if (m_speedIndex < 0) {
            a << "1/" << SPEED_SLOW_STEPS[(-m_speedIndex) - 1] << "x (slow)";
        } else if (m_speedIndex <= SPEED_FIXED_MAX) {
            a << (1L << m_speedIndex) << "x";
        } else {
            a << SPEED_BUDGET_MS[m_speedIndex - SPEED_BUDGET_FIRST] << " ms/frame (max)";
        }
        a.precision(1);
        a << std::fixed << "    measured x" << m_speedX;
        y = putS(PANEL_X, y, a.str(),
                 m_paused ? sf::Color(255, 140, 90) : sf::Color(150, 200, 150), 16);

        // ★ 策略概率：判断"有没有学到东西"最直接的窗口
        if (m_runner && m_runner->hasPolicyProbs()) {
            std::ostringstream b;
            b.precision(2);
            b << std::fixed
              << "policy  L " << m_runner->probLeft()
              << "  F "        << m_runner->probFwd()
              << "  R "        << m_runner->probRight()
              << "    camera " << (m_followCamera ? "follow" : "free");
            // 越接近均匀越暗，有偏好时变亮
            const double peak = std::max(m_runner->probLeft(),
                                std::max(m_runner->probFwd(), m_runner->probRight()));
            const sf::Uint8 c = static_cast<sf::Uint8>(
                std::min(255.0, 90.0 + (peak - 0.333) * 500.0));
            y = putS(PANEL_X, y, b.str(), sf::Color(c, 220, 200), 16);
        }
        commitSection(y);
    }

    // ---- 当前模型 + 快捷键（压缩到 2 行）----
    //
    // ★ 为什么把"当前模型名"放在这里
    //    按 O 是"覆盖当前模型"，用户必须看得见"当前是哪一个"，
    //    否则覆盖键是个盲操作。
    {
        float y = nextSectionY();

        if (!m_currentModelName.empty()) {
            // 模型名是 UTF-8 字符串 → 走 putS（它内部过 textutil::utf8 转宽字符）
            y = putS(PANEL_X, y, std::string("model: ") + m_currentModelName
                                    + "   (O = overwrite)",
                     sf::Color(120, 200, 160), 15);
        } else {
            y = put(PANEL_X, y, L"model: none   (S to name & save)",
                    sf::Color(140, 140, 140), 15);
        }
        y += 22.0f;

        // ★ 快捷键压成 2 行：把"常用"和"偶尔用"分开，而不是平铺 5 行
        y = put(PANEL_X, y, L"Enter 暂停   N 单步   [ ] 速度   V 画面   R 重训   F 镜头",
                sf::Color(115, 115, 115), 14);
        y += 18.0f;
        y = put(PANEL_X, y, L"S 命名保存   O 覆盖当前   L 加载存档   ESC 退出(自动存)",
                sf::Color(120, 200, 160), 14);
        commitSection(y);
    }

    // =======================================================================
    //  ★★★ 面板背景【最后】画 —— 高度由内容量出来
    //
    //  见函数开头的说明：先画背景、再填字，只要内容超高就会"字跑到面板外"，
    //  看起来就是叠字。这里在全部内容画完之后才补背景，所以：
    //      · 面板一定盖得住全部文字（高度 = contentBottom + 余量）
    //      · 加一行不需要改任何常量
    //
    //  ⚠️ 唯一的人工约束：不能高过窗口底部。超了就 clamp 到贴底 ——
    //     那时该做的是精简面板内容，而不是继续加高。
    // =======================================================================
    {
        const float kPadBottom = 8.0f;
        float h = contentBottom - 10.0f + kPadBottom;      // 背景从 y=10 开始
        const float maxH = static_cast<float>(Constants::WINDOW_HEIGHT) - 20.0f;
        if (h > maxH) h = maxH;
        if (h < 60.0f) h = 60.0f;                          // 兜底：至少能看
        sf::RectangleShape bg(sf::Vector2f(PANEL_W, h));
        bg.setPosition(10.0f, 10.0f);
        bg.setFillColor(sf::Color(0, 0, 0, 165));
        window.draw(bg);
    }

    // ---- ★ 命名输入行（正在给存档起名字时显示）----
    //
    //  ★ 为什么单独画在最上层、而且是醒目的黄色
    //    输入模式下所有普通按键都被吃掉了（连暂停都不响应），
    //    如果屏幕上没有明显提示，用户会以为"按 S 之后游戏卡住了"。
    //    所以这里必须显眼地告诉它："现在在等一个文件名"。
    if (m_nameInputMode) {
        const float W = static_cast<float>(Constants::WINDOW_WIDTH);
        const float H = static_cast<float>(Constants::WINDOW_HEIGHT);

        sf::RectangleShape box(sf::Vector2f(W - 120.0f, 96.0f));
        box.setPosition(60.0f, H * 0.5f - 48.0f);
        box.setFillColor(sf::Color(20, 20, 40, 240));
        box.setOutlineColor(sf::Color(255, 210, 90));
        box.setOutlineThickness(2.0f);
        window.draw(box);

        put(80.0f, H * 0.5f - 36.0f, L"给这份权重起个名字（重名会覆盖）",
            sf::Color(255, 210, 90), 18);

        // 当前输入内容 + 一个"光标"（用下划线表示）
        sf::String shown;
        for (sf::Uint32 cp : m_nameBuffer) shown += cp;
        shown += sf::String(L"_");
        put(80.0f, H * 0.5f - 4.0f, shown.toWideString(), sf::Color(255, 255, 255), 20);

        put(80.0f, H * 0.5f + 30.0f, L"Enter 保存    ESC 取消    退格 删除",
            sf::Color(160, 160, 160), 14);
    }

    // ---- 底部状态提示（存/读的结果，几秒后消失）----
    if (m_statusTimer > 0.0f && !m_statusText.empty()) {
        put(PANEL_X, static_cast<float>(Constants::WINDOW_HEIGHT) - 30.0f,
            textutil::utf8(m_statusText), sf::Color(255, 230, 140), 16);
    }
}

// ===========================================================================
//  "选存档"面板
// ===========================================================================
void LiveTrainingScene::renderLoadList(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);
    const float H = static_cast<float>(Constants::WINDOW_HEIGHT);

    // 遮罩 + 面板
    sf::RectangleShape mask(sf::Vector2f(W, H));
    mask.setFillColor(sf::Color(0, 0, 0, 170));
    window.draw(mask);

    const float pw = 720.0f, ph = 460.0f;
    sf::RectangleShape panel(sf::Vector2f(pw, ph));
    panel.setPosition((W - pw) * 0.5f, (H - ph) * 0.5f);
    panel.setFillColor(sf::Color(24, 24, 40, 245));
    panel.setOutlineColor(sf::Color(78, 205, 196));
    panel.setOutlineThickness(2.0f);
    window.draw(panel);

    const float px = panel.getPosition().x + 26.0f;
    float py = panel.getPosition().y + 22.0f;

    {
        sf::Text t(L"加载存档", m_font, 26);
        t.setFillColor(sf::Color(78, 205, 196));
        t.setPosition(px, py);
        window.draw(t);
        py += 38.0f;
    }
    {
        sf::Text t(L"上下键 选择    Enter 加载    ESC 取消", m_font, 15);
        t.setFillColor(sf::Color(150, 150, 150));
        t.setPosition(px, py);
        window.draw(t);
        py += 30.0f;
    }

    if (m_checkpoints.empty()) {
        sf::Text t(L"（还没有存档 —— 训练中按 S 就能存一份）", m_font, 18);
        t.setFillColor(sf::Color(255, 170, 120));
        t.setPosition(px, py);
        window.draw(t);
        return;
    }

    // 列表（最多显示 12 条，跟随光标滚动）
    const int total = static_cast<int>(m_checkpoints.size());
    const int visible = 12;
    int first = 0;
    if (total > visible) {
        first = m_loadCursor - visible / 2;
        if (first < 0) first = 0;
        if (first > total - visible) first = total - visible;
    }

    for (int i = first; i < std::min(total, first + visible); ++i) {
        const bool sel = (i == m_loadCursor);
        if (sel) {
            sf::RectangleShape bar(sf::Vector2f(pw - 60.0f, 26.0f));
            bar.setPosition(px - 8.0f, py - 2.0f);
            bar.setFillColor(sf::Color(78, 205, 196, 50));
            window.draw(bar);
        }
        sf::Text t(textutil::utf8(m_checkpoints[static_cast<std::size_t>(i)]),
                   m_font, 17);
        t.setFillColor(sel ? sf::Color::White : sf::Color(185, 185, 185));
        t.setPosition(px, py);
        window.draw(t);
        py += 28.0f;
    }

    {
        std::ostringstream oss;
        oss << "共 " << total << " 份    目录：saves/";
        sf::Text t(textutil::utf8(oss.str()), m_font, 14);
        t.setFillColor(sf::Color(130, 130, 130));
        t.setPosition(px, panel.getPosition().y + ph - 30.0f);
        window.draw(t);
    }
}

// ===========================================================================
void LiveTrainingScene::render(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);
    const float H = static_cast<float>(Constants::WINDOW_HEIGHT);

    // ---- 世界：自由镜头（方向键漫游、PgUp/PgDn 缩放）----
    //
    // ★ 不跟踪 AI —— 和观战模式一致。
    //   第一版写成"跟着蛇头"，那是错的：训练时蛇到处乱跑，
    //   跟镜头既看不清食物圈分布（全局信息），镜头也一直在晃。
    if (m_renderWorld && m_arena && !m_arena->agents().empty()) {
        sf::View view;
        view.setSize(W * m_camZoom, H * m_camZoom);
        view.setCenter(m_camCenter);

        window.setView(view);
        renderWorld(window, view);

        // 恢复默认视图再画 UI（否则面板会跟着镜头乱跑、缩放时字会变形）
        sf::View def(sf::FloatRect(0.0f, 0.0f, W, H));
        window.setView(def);
    } else {
        window.clear(sf::Color(26, 26, 46));
    }

    renderPanel(window);

    // 选存档面板画在最上面（它会盖住数据面板）
    if (m_loadListOpen) renderLoadList(window);
}
