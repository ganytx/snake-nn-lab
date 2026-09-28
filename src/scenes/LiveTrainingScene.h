#pragma once

// ============================================================================
//  LiveTrainingScene.h — 游戏内「训练模式」：亲眼看模型一步步学
//
//  ★ 和「训练数据」界面（TrainingScene）的区别
//
//      TrainingScene    读 train_metrics.csv 画曲线 —— 看的是【跑完的结果】
//      本场景           在游戏进程里【真的训练】 —— 看的是【正在发生的过程】
//
//  负责人要的是后者：亲眼看见模型怎么从乱撞变成会找吃的。
//
//  ★ 为什么游戏里能直接训练
//
//  因为"一局怎么跑"已经收敛到 arena::Arena 这一份，
//  所以把驱动的节奏从"每渲染帧推进一次"换成"每渲染帧推进 N 次"，
//  就是原地训练 —— 不需要另写一套世界。
//
//  分工：
//      arena::Arena       世界
//      rl::Trainer        网络怎么更新
//      rl::EpisodeRunner  两者怎么串 + 奖励
//      本场景             跑多快、画什么、怎么操作
//
//  ★ 操作
//
//      Enter   暂停 / 继续
//      N       只推进一步（暂停时用，一帧一帧看）
//      R       重新训练（网络清零）
//      V       开关画面渲染
//      U       不限速（每帧跑满 4ms）
//      [ ]     速度（2 的幂）
//      S       保存当前权重（带局数+时间戳，不会覆盖旧的）
//      L       打开存档列表，↑↓ 选、Enter 加载、ESC 取消
//      F       镜头跟随开关
//      Tab     切换跟随哪条蛇（多蛇时）
//      方向键  漫游镜头（和观战模式一致）
//      PgUp/PgDn  缩放镜头
//      ESC     返回主菜单
//
//  ★ 权重存档
//
//  以前进训练模式【不会保存任何东西】—— 每次都用固定种子重新初始化网络，
//  练到一半退出就全没了（负责人问过这个）。
//  现在按 S 存一份到 `saves/`，按 L 从列表里选一份加载，可以存很多版本。
//
//  ★ 镜头：默认**不跟踪 AI**，和观战模式一样是自由镜头
//
//  第一版写成"跟着蛇头"，那是错的 —— 训练时蛇会到处乱跑，
//  跟镜头既看不清全局（食物圈分布），也会因为镜头一直移动而晃。
//  现在默认开局就是"整张地图刚好放进视口"的自由镜头。
//
//  （例外：只有 1 条蛇、且用 F 打开跟随时才跟。多蛇时跟随没有意义。）
// ============================================================================

#include "../core/Scene.h"
#include "../arena/Arena.h"
#include "../rl/Trainer.h"
#include "../rl/EpisodeRunner.h"

#include <memory>
#include <random>
#include <string>
#include <vector>
class LiveTrainingScene : public Scene {
public:
    explicit LiveTrainingScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

    ~LiveTrainingScene() override;

private:
    void restartTraining();

    // ---- 权重存档 ----
    //
    //  ★ 2026-09-24 改成"一个底层实现 + 三个调用场景"：
    //      saveCheckpoint(name)  ← 底层：存到指定文件名（自动/手动共用）
    //      saveCheckpoint()      ← 按 S 的默认行为：自动生成"局数_时间戳"名字
    //      autoSave()            ← 按 ESC 退出时存 auto_latest.txt
    //    以前只有中间那个，于是"退出就丢"和"不能命名"两件事都没法表达。
    void saveCheckpoint(const std::string& name);   // 底层：存成这个名字
    void saveCheckpoint();          // 存一份（文件名带局数+时间戳，不覆盖旧的）
    void overwriteCurrentModel();   // ★ 直接覆盖"当前模型"（Ctrl+S / O）
    void autoSave();                // 退出时自动存一份（auto_latest.txt，覆盖）
    void openLoadList();            // 打开"选存档"面板
    void loadSelected();            // 加载列表里选中的那份
    void refreshCheckpoints();      // 重新扫描 saves/ 目录

    // 画世界（蛇 / 食物 / 圈 / 边界）
    void renderWorld(sf::RenderWindow& window, const sf::View& view) const;

    // 画左上角的数据面板
    void renderPanel(sf::RenderWindow& window);

    // 画"选存档"面板
    void renderLoadList(sf::RenderWindow& window);

    // 自由镜头的键盘漫游（每帧调用，和观战模式同一套手感）
    void updateCamera(float dt);

    // 初始缩放：让【整张地图】刚好放进视口
    //
    // ★ 为什么要这样：地图 2000x2000、视口只有 1280x720，
    //   1:1 只能看到地图的 41% x 36% —— 边界都看不见，
    //   训练时也看不到全局（食物圈分布）。
    //   默认给"看全图"，想看细节用 PgUp 放大。
    float fitZoom() const;

    // 当前档位对应的"每帧跑几步"。
    //   返回值 > 0 ：这一帧就跑这么多步
    //   返回值 < 0 ：这一帧不跑（慢动作段用来做分频）
    //   返回值 = 0 ：走时间预算（吞吐段 / 不限速）
    long stepsThisFrame();

    // 改档位（同时把实测倍速的累计量清零 —— 这样测出来的就是新档位的倍速）
    void setSpeedIndex(int idx);

    // ---- 训练三件套 ----
    //
    // ★ 全用 unique_ptr 持有所有权。
    //   注意【声明顺序 = 析构顺序的反序】：
    //   m_runner 依赖 m_trainer 和 m_arena，所以它必须【最后被析构】——
    //   也就是要【最先声明】。
    rl::TaskConfig     m_task;
    std::unique_ptr<rl::EpisodeRunner> m_runner;
    std::unique_ptr<rl::Trainer>       m_trainer;
    std::unique_ptr<arena::Arena>      m_arena;

    std::mt19937 m_rng;

    // ---- 运行状态 ----
    bool  m_paused      = false;
    bool  m_renderWorld = true;

    // ★ 速度档位：**2 的幂**，而且分三段用不同的机制
    //
    //  第一版用"每帧的时间预算 ÷2 / ×2"，结果**降不下去**：
    //  预算分到 0.5ms 本来就该很慢，但自适应批量有 `batch = max(batch, 1)`
    //  的下限（每帧至少跑 1 步），于是实际速度被"每帧步数 × 60 帧"兜底住了。
    //
    //  而且低倍速本来就不该用时间算 —— **低倍速应该按"每帧几步"精确控制**：
    //      1x   → 每帧 1 步      （真实速度）
    //      0.5x → 每两帧 1 步
    //      0.25x→ 每四帧 1 步
    //
    //  所以分三段：
    //      慢动作段（k < 0）   ：按"每帧几步"的分数控制
    //      固定步数段（0..6）  ：每帧正好 2^k 步   ← 倍速精确、可验证
    //      吞吐段（k > 6）     ：按时间预算跑，取实测能达到的最大倍速
    //
    //  ⚠️ 每帧要跑多少步必须【数出来】，不能靠"用完了多少时间"估 ——
    //     倍速 = 仿真时间 / 墙钟时间，按下键时把两个累计量同时清零就能量准。
    //  ★ 2026-09-24 默认档位从 6（64x）改成 0（**1x**）：
    //    负责人要"进入训练时速度为默认 1x"。
    //    64x 起步的问题是——**看不清它在干什么**，而训练模式的全部意义就是看。
    //    想加速按 `]` 即可（档位是 2 的幂，按几下就到 64x / 吞吐段）。
    int  m_speedIndex  = 0;       // 默认 1x（真实速度）
    bool m_unlimited   = false;   // 不限速：每帧跑满 4ms

    // 实测倍速 = 累计仿真时间 / 累计墙钟时间
    // 累计量在下键改变速度时清零，所以这个数就是"当前档位的真实倍速"
    double    m_simSeconds = 0.0;
    sf::Clock m_speedClock;
    double    m_speedX     = 0.0;

    // 自适应批量在吞吐段用的单步成本估计（微秒）
    double m_costPerTickUs = 6.0;

    // 慢动作段用：每 n 帧跑 1 步（配一个帧计数器）
    int m_frameCounter = 0;

    // ---- 自由镜头（默认**不跟踪 AI**）----
    sf::Vector2f m_camCenter;
    float        m_camZoom = 1.0f;

    // 跟随哪一条（单蛇、多蛇都可以；多蛇时用 Tab 切换）
    bool m_followCamera = false;
    int  m_followIndex  = 0;

    // ---- 统计（画在面板上）----
    long   m_episodes   = 0;
    double m_lastReward = 0.0;
    // ★★ ⑪ 危险距离扣分的【本局累计】（2026-09-26 深夜新增）
    //
    //  ★ 为什么必须单独存"本局值"而不是只看 EMA 平均
    //    负责人报："撞死后的危险地带惩罚也才两三分" —— 而按常数算撞一次
    //    应该是 −120 量级。**这个矛盾必须能一眼定位**：
    //        · 本局累计 ≈ −120  → 项在正常工作，之前看到的是 EMA 平均（看不清全貌）
    //        · 本局累计 ≈ −2    → 确实有问题，得去查判据为什么没生效
    //    EMA 平均是"最近 50 局左右的平均"，它会把单局的极端值抹平，
    //    **不适合用来判断"这一项到底扣了多少"**。所以两个都显示。
    double m_lastWallD  = 0.0;
    double m_lastEnemyD = 0.0;
    // ⚠️ 2026-09-27：m_lastMateAgree / m_lastMateN（队友一致性探针的读数）
    //    已随探针一起删除 —— 那段代码在 rl/ 层里用了 brain/ 的类型，违反分层铁律。
    int    m_lastTicks  = 0;
    int    m_lastEaten  = 0;

    // 上一局的【奖励构成】滑动平均 —— 用来回答"到底是哪个信号在起作用"
    //
    // ★ 为什么要看这个：只看总回报根本分不清是哪个分量在推动学习。
    //   正是靠它才发现过"总回报 99.66% 来自一个哨兵值 bug"，
    //   以及"某个塑形项从不触发"。
    //
    // ★ 2026-09-26：⑥ 从"每帧贴敌身惩罚"改成"事件式逃生奖励"，
    //   并新增 ⑨ 无食物区久留扣分，所以这里对应变成 8 项。
    double m_avgDie      = 0.0;   // ① 死亡扣分
    double m_avgEat      = 0.0;   // ② 吃到食物
    double m_avgHunger   = 0.0;   // ③ 饥饿扣分
    // ④「靠近最近食物」于 2026-09-26 整项删除（它会惩罚"吃掉食物"）
    double m_avgFed      = 0.0;   // ⑤ 吃饱后的余韵
    double m_avgEscape   = 0.0;   // ⑥′ 逃生奖励（正数）
    double m_avgHunt     = 0.0;   // ⑦ 敌方头贴到我身体的奖励
    double m_avgKill     = 0.0;   // ⑧ 击杀奖励
    double m_avgDisp     = 0.0;   // ⑩′ 饥饿时位移不足扣分（负数）
    double m_avgWallD    = 0.0;   // ⑪ 墙危险距离扣分（负数）★ 2026-09-26 深夜
    double m_avgEnemyD   = 0.0;   // ⑪ 敌身危险距离扣分（负数）★ 2026-09-26 深夜

    // 出生/死亡诊断（用来定位"一出生就死"这类问题）
    float  m_lastSpawnX        = 0.0f;
    float  m_lastSpawnY        = 0.0f;
    float  m_lastSpawnWallDist = 0.0f;
    float  m_lastDeathX        = 0.0f;
    float  m_lastDeathY        = 0.0f;
    float  m_lastDeathWallDist = 0.0f;
    int    m_lastMaxSegs       = 0;
    arena::DeathCause m_lastCause = arena::DeathCause::None;

    // ★ 实时饥饿（每 tick 更新，见 EpisodeRunner::EpisodeStats::curHungerLeft）
    //
    //   ⚠️ 和下面的 m_lastStarveTime / m_lastTimeSinceEat 不是一回事：
    //      那两个是【局末快照】（一局进行中不变），这两个是【当前值】。
    //      面板要显示"现在还剩几秒会饿死"，必须用这两个。
    float  m_curHungerLeft    = 0.0f;
    float  m_curTimeSinceEat  = 0.0f;
    float  m_lastStarveTime   = 0.0f;
    float  m_lastTimeSinceEat = 0.0f;
    float  m_lastHeadR        = 0.0f;

    // ---- 权重存档 ----
    std::vector<std::string> m_checkpoints;   // saves/ 下的文件名
    int    m_loadCursor = 0;                  // 列表里选中的下标
    bool   m_loadListOpen = false;            // 是否正在选存档
    std::string m_statusText;                 // 底部状态提示（存/读的结果）
    float  m_statusTimer = 0.0f;              // 提示显示多久后消失

    // -----------------------------------------------------------------------
    //  ★ 存档【自己命名】的输入模式（2026-09-24 负责人要求）
    //
    //    按 `S` 进入输入模式 → 打字 → Enter 确认保存 / ESC 取消。
    //
    //  ★ 为什么要有它
    //    原来 `S` 存出来的名字是自动生成的 `ep000123_20260923_091530.txt`：
    //    它解决了"多版本能分辨"的问题（见 Checkpoint.h），
    //    但**没法按用途命名** —— 比如"这是学会觅食那版""这是调奖励前的对照版"。
    //
    //  ★ 命名冲突：**直接覆盖**（负责人定的）。不做确认交互，
    //    因为多一层确认就要再多一个按键状态，而这里的目标是"存得快"。
    //    万一覆盖错了，退出去看文件列表就知道，不存在"静默丢失"。
    // -----------------------------------------------------------------------
    bool        m_nameInputMode = false;   // 是否正在输入文件名

    // ★ 用【码点序列】存输入的字符，而不是 std::string
    //
    //  ★ 为什么不用 std::string
    //    1) sf::Text 要宽字符/UTF-32 才能正确画中文
    //       （项目铁律：sf::String 从 std::string 构造时按本地编码解释，不是 UTF-8）
    //    2) 退格必须退【一个字符】。UTF-8 里一个汉字占 3 字节，
    //       pop_back() 一次只会切掉一个字节 → 名字变成半个汉字、显示成乱码方块。
    //       存码点的话退格就是 pop_back() 一个元素，天然正确。
    //
    //  ★ 保存时再统一转成 UTF-8（sf::String::toUtf8）
    std::vector<sf::Uint32> m_nameBuffer;

    // 把当前缓冲转成 UTF-8 字符串（保存时用）
    std::string nameBufferUtf8() const;

    // -----------------------------------------------------------------------
    //  ★ 自动存档（2026-09-24 负责人要求）
    //
    //  ★ 解决什么问题
    //    以前**按 ESC 退出不保存任何东西** —— 而进训练模式每次都用固定种子
    //    重新初始化网络，所以"练了半小时、按 ESC 全没了"。
    //
    //  ★ 为什么固定名 + 每次覆盖
    //    它要表达的是"**这次训练的最新进度**"，语义上就该只有一份。
    //    想留多个版本用 `S` 自己命名存档（上面那一套）。
    //    两者分工：`auto_latest.txt` 是"接着练"，命名存档是"留档对比"。
    //
    //  ⚠️ 名字定义在 ckpt::kAutoSaveName（Checkpoint.h）——
    //     因为"训练设置"界面也要显示它，放这儿会变成私有、那边读不到。
    // -----------------------------------------------------------------------
    bool m_autoSaved = false;              // 本次会话是否已自动存过（避免重复刷提示）

    // -----------------------------------------------------------------------
    //  ★ "当前模型"的文件名 —— 用来支持【直接覆盖】（负责人要求）
    //
    //  ★ 为什么需要记它
    //    "覆盖当前模型"必须知道"当前"是哪一个。这个信息只有两处来源：
    //      · 进来时预加载的那份（TrainingSetupScene 选的）
    //      · 用 L 加载的那份
    //    两处都往这里写，覆盖键才有目标。
    //
    //  ★ 为空时（从头练、没加载过任何存档）
    //    覆盖键没有目标，此时应该**提示用户先命名一次**，
    //    而不是自作主张写一个名字 —— 否则用户以为覆盖了某个文件，其实多出来一个。
    // -----------------------------------------------------------------------
    std::string m_currentModelName;

    // 最近若干局的平滑值 —— 单局噪声太大，看不出趋势
    std::vector<double> m_recentReward;
    std::vector<int>    m_recentEaten;

    // 面板上的小曲线用（最近 N 局的回报）
    static constexpr int HISTORY = 200;

    sf::Font m_font;
};
