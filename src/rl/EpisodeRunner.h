#pragma once

// ============================================================================
//  EpisodeRunner.h — 「一局怎么跑 + 奖励怎么算」的共用外壳
//
//  ★ 为什么要有这个文件
//
//  训练这件事有两处会用到：
//      ① tools/train_probe.exe        —— 命令行跑 2000 局、写 CSV
//      ② 游戏内的「训练模式」          —— 在游戏里亲眼看它一步步学
//
//  如果两处各写一遍"每 tick 干什么、奖励怎么结算"，就一定会跑偏 ——
//  这个项目已经因为"两份实现"栽过两次（观测编码分叉 0.96、
//  世界推进两份导致训练与游戏不一致）。所以这一层也抽出来共用。
//
//  ★ 分工（每一层都只有一份实现）
//
//      arena::Arena      世界怎么推进          —— src/arena/Arena.h
//      rl::Trainer       网络怎么更新          —— src/rl/Trainer.h
//      rl::EpisodeRunner 两者怎么串 + 奖励     —— 本文件
//      调用方            什么时候跑、跑多快、怎么显示
//
//  ★ 用法（train_probe 和游戏内训练完全一样）
//
//      EpisodeRunner runner(arena, &trainer, task);
//      arena.setController([&](std::size_t i, const arena::Perception& p) {
//          return runner.decide(i, p, rng);       // 内部按 mode 选动作
//      });
//
//      runner.begin(RunMode::Train);
//      while (!runner.done()) runner.tick(1.0f / 60.0f);
//      if (mode == RunMode::Train) trainer.finishEpisode();   // ← 学一次
// ============================================================================

#include "TrainTask.h"
#include "Trainer.h"
#include "../agent/Agent.h"
#include "../arena/Arena.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <random>

namespace rl {

// ---------------------------------------------------------------------------
//  一局的统计
// ---------------------------------------------------------------------------
struct EpisodeStats {
    double totalReward = 0.0;
    int    ticks = 0;
    int    eaten = 0;
    bool   died  = false;

    // ---- 诊断：这一局是怎么结束的 ----
    //
    // ★ 为什么要记：负责人报"蛇刚出生就死了"，但只看"这一局结束了"
    //   分不清是【撞墙】【撞自己】还是【饿死】—— 三种的修法完全不同。
    //   把'出生位置'和'死亡位置'都记下来，一对比就知道问题在哪一段。
    float spawnX = 0.0f;      // 出生时蛇头位置
    float spawnY = 0.0f;
    float spawnDistanceToWall = 0.0f;
    float deathX = 0.0f;      // 死时蛇头位置
    float deathY = 0.0f;
    float deathDistanceToWall = 0.0f;
    float mapW   = 0.0f;      // 地图尺寸（用来判断是不是贴墙死的）
    float mapH   = 0.0f;
    int   maxSegments = 0;    // 这一局身体最长到过几节

    // -----------------------------------------------------------------------
    //  ★ 最近的【实时】饥饿状态（每 tick 更新，不是局末快照）
    //
    //  ★ 为什么要单独记这个
    //    原来面板上只有 `starvationTime` 和 `timeSinceEat` 两个【局末】值，
    //    它们在一局进行中是不变的 —— 于是"现在还剩几秒会饿死"根本看不到。
    //    而饥饿是主要死因之一，训练时看不到它等于瞎调。
    // -----------------------------------------------------------------------
    float curHungerLeft   = 0.0f;   // 距饿死还剩几秒（Arena 每帧算好的）
    float curTimeSinceEat = 0.0f;   // 距上次进食过了几秒

    // ★ 死因 —— 由代码判定，不靠人看画面猜
    arena::DeathCause cause = arena::DeathCause::None;

    // ---- 原始数值诊断（用来钉死"为什么第 1 帧就死"）----
    //
    // ★ 为什么要它：上面那个"死因"是按规则反推的，也可能推错。
    //   这几个数都是【规则真正比较的原始值】，一看就知道是哪条条件成立：
    //       starvationTime < 0        → 饥饿这条路根本不该触发
    //       timeSinceEat > starveTime → 那就是它
    //       wallDist <= headR         → 那就是撞墙
    float starvationTime = 0.0f;
    float timeSinceEat   = 0.0f;
    float headRadius     = 0.0f;
    float deathWallDistRaw = 0.0f;

    // -----------------------------------------------------------------------
    //  奖励构成（用来回答"到底哪个信号在起作用"）
    //
    //  ★ 为什么要记这个：调奖励的时候，只看"总回报"根本分不清是
    //    哪一个分量在推动学习。实测正是靠它才发现"总回报 99.66% 来自
    //    一个哨兵值 bug"，以及"某个塑形项从不触发"。
    //
    //  ★ 2026-09-24 奖励重写后【只剩 5 项】，所以这里也只有 5 个字段。
    //    旧字段全部删除 —— 不留"恒为 0 的僵尸字段"，那会让人以为它还在起作用。
    // -----------------------------------------------------------------------
    double diePenalty   = 0.0;   // ① 死亡扣分累计（记成【正数】= 被扣了多少）
    double eatGain      = 0.0;   // ② 吃到食物累计
    double hungerGain   = 0.0;   // ③ 饥饿扣分累计（负数）
    // ④「靠近最近食物」于 2026-09-26 整项删除（详见 tick() 里的说明）
    double fedGain      = 0.0;   // ⑤ 吃饱后的余韵累计（**只算吃过之后的**）
    // ★ ⑤ 诊断：**被跳过的"开局自带余韵"**（2026-09-26 深夜新增）
    //
    //  ★ 它是什么
    //    出生时 `timeSinceEat == 0`，按旧公式会被当成"刚吃过"，
    //    于是前 10 秒白送 `rFedMax` 满额（约 +25/局）。
    //    现在这部分**不加分**，但**照旧记账**放在这里。
    //
    //  ★ 为什么留着而不是删掉
    //    它是"修复是否生效"的唯一证据：**这个数应当恒为 0**（因为不加分了）。
    //    如果哪天它又变成正数，说明门控被改坏了 —— 这比读代码可靠。
    double fedSkippedBirth = 0.0;
    double escapeGain   = 0.0;   // ⑥′ 逃生奖励累计（正数）★ 2026-09-26 取代 snakeGain
    double huntGain     = 0.0;   // ⑦ 敌方头贴到我身体的奖励累计
    double killGain     = 0.0;   // ⑧ 击杀奖励累计
    double dispGain     = 0.0;   // ⑩′ 饥饿时位移不足扣分累计（负数）★ 2026-09-26
    double wallDangerGain  = 0.0;   // ⑪ 墙危险距离扣分累计（负数）★ 2026-09-26 深夜
    double enemyDangerGain = 0.0;   // ⑪ 敌身危险距离扣分累计（负数）★ 2026-09-26 深夜

    // ★ ⑥′ 诊断：本局成功脱离危险区几次（每次 +rEscape）
    int escapeCount = 0;
    // ★ ⑩′ 诊断：本局真的被"位移不足"扣分了多少帧
    int dispPenaltyFrames = 0;
    // ★ ⑪ 诊断：本局各自有多少帧处于危险距离内
    int wallDangerFrames  = 0;
    int enemyDangerFrames = 0;

    // -----------------------------------------------------------------------
    // ⚠️ 2026-09-27：这里曾经有"队友一致性探针"的一组统计字段
    //    （mateSamples / mateAgree / learnerActHist / mateActHist …）。
    //    已随探针一起删除 —— 它在 rl/ 层里用了 brain/ 的类型，违反分层铁律。
    //    详见 decide() 里那段说明。

    // -----------------------------------------------------------------------
    //  ★ 塑形项的探针（2026-09-26：approach 那一组已随该项删除）
    //
    //  保留下来的历史教训（值得记住，所以留在这里而不是删掉）：
    //  正是靠 `approachFirstDist` 打印出 31622.8（= sqrt(1e9)，一个哨兵值）
    //  才定位到"有食物分支的门限写错"那个把 99.66% 的回报变成 bug 常数的
    //  严重问题。**教训：看到一个"物理上不可能"的数就立刻停下来查。**
    // -----------------------------------------------------------------------
    // =======================================================================
    //  ★★ 诊断：策略锁死 / 奖励速率 / 饥饿状态（2026-09-25 新增）
    //
    //  ★ 为什么必须加这几个
    //
    //  上面那些「奖励构成」给的是【整局总和】，而总和 = 速率 × 持续帧数。
    //  两者混在一起时，**分不清"某一项很大"是因为它速率高，
    //  还是因为它持续得久**。而"策略锁死"这个现象的机制正好落在速率上：
    //
    //      绕圈时 approach 的差分恒为 0（闭合回路），
    //      于是唯一还在持续给分的正项是 fed —— 它成了"每帧的存活性带宽"。
    //
    //  所以这里把【每帧速率】和【状态帧占比】分开记。
    //  只看总和永远查不出这件事，这正是本轮要建立的判据。
    // =======================================================================

    // ---- 实际执行的动作次数（左 / 直 / 右）----
    //
    //  ★ 这是"策略锁死"最直接的客观判据：三项里某一项占了 >90%，
    //    就是刻板行为。用【实际执行】而不是网络输出的概率，
    //    因为它还包含了 ε-greedy 的随机步 —— 那才是真正作用在世界上的动作。
    //   ★ 用 {} 值初始化而不是写死个数 —— ACT_SIZE 从 6 回滚到 3 之后，
    //     "写死 6 个 0" 会直接编译失败（too many initializers）。
    //     `{}` 永远按 ACT_SIZE 归零，以后改动作空间不用再改这里。
    int actCount[ACT_SIZE] = {};

    // ---- 本局处于三种饥饿状态的帧数 ----
    //
    //  ★ 三个桶的判据（阈值都来自 TaskConfig，不在这里写死）
    //      fed        : hungerLeft > starvationTime − fedDuration  → 在余韵窗口内
    //      starving   : hungerLeft < hungerWarnSec                 → 正在被扣分
    //      normal     : 其余
    //
    //  ⚠️ 注意 fedDuration 与 hungerWarnSec 的【相对大小】会改变分桶形状：
    //      两者相等时（默认都是 10s）中间那一桶恒为 0，这是正常的，不是 bug。
    //      它们是两个独立的设计参数，不该为了"桶好看"去强行凑关系。
    //
    //  ★ 判据：如果 fed 帧占比接近 1，说明"吃饱"这件事已经饱和 ——
    //    蛇大部分时间泡在余韵里，余韵就不再是"奖励吃东西"，
    //    而变成了"奖励活着"。
    int framesFed      = 0;
    int framesStarving = 0;
    int framesNormal   = 0;

    // =======================================================================
    //  ★★★ 2026-09-27 新增：网络的【响应幅度】与【贪心动作是否恒定】
    //
    //  ★ 为什么要单独加这一组 —— 这是"AI 完全随机 / 失去一切行为"的直接判据
    //
    //    现有的判据全都是**关于输出的**（动作概率、动作熵、回报）。
    //    但它们都答不了一个更基础的问题：
    //
    //        **网络的输出，到底有没有随输入变化？**
    //
    //    一个"永远输出同一个动作"的网络，和一个"健康但偏好某个动作"的网络，
    //    在【动作概率面板】上可以长得一模一样 —— 都是 `0.07 0.07 0.60`。
    //    区别在于前者对任何观测都给这一组概率，后者随局面变化。
    //
    //  ★ 实测背景（这就是加它的原因）
    //    实测一条 800 局的训练：`policy` 输出的三项之和只有 0.54，
    //    也就是 46% 的概率压在【加速档】上，其中"直行加速"独占 ~0.60。
    //    反推 logits：ln(0.60/0.07) ≈ 2.15 —— **整局所有帧的 logits 幅度
    //    只有 2 量级**，而帧与帧之间 logits 的变化只有 0.1 量级。
    //    结果：argmax 几乎永远落在同一个动作上。
    //
    //  ★ 这同时解释了"为什么只有学习中的蛇看起来还有行为"
    //    学习中的蛇走 `RunMode::Train` → **按概率采样**，
    //    所以那 40% 的非最大概率动作仍会被抽到，看起来会拐弯。
    //    而敌方/队友走 `NeuralBrain::think` → **argmax 确定性**，
    //    概率最大的那个永远赢 —— 于是它们**整齐划一地笔直前进**。
    //    两者用的是同一份退化网络，差别只在"采样 vs 取最大"。
    //
    //  ★ 怎么读这三个数
    //      greedySwitchRate : 贪心动作在相邻帧之间改过多少次 / 总帧数
    //         健康 → 明显大于 0（局面一变就换动作，几帧一换很正常）
    //         退化 → **≈ 0**（整局死咬同一个动作）
    //      topActionFrac    : 出现最多的那个贪心动作占了多少帧
    //         退化 → 接近 1.0
    //      logitSpread*     : max(logits) − min(logits)，即"网络对自己多有主见"
    //         它大不代表策略对（可能自信地做错事），但它**小到 2 以下**
    //         就说明网络对输入的响应极弱 —— 那是最根本的问题。
    //
    //  ⚠️ 这三个数【不参与任何奖励、不影响任何行为】——
    //     纯只读诊断。加它们的目的是让"网络有没有在看输入"这件事
    //     从"靠画面猜"变成"读一个数"。
    // =======================================================================
    int    greedySwitchCount = 0;    // 贪心动作变化的次数
    int    greedyFrames      = 0;    // 参与统计的帧数（第一帧不算：没有前一帧可比）
    int    greedyActionCount[ACT_SIZE] = {};   // 贪心动作直方图（{} = 按 ACT_SIZE 归零）
    double logitSpreadSum    = 0.0;  // Σ (max − min)
    double logitSpreadMin    = 1.0e30;
    double logitSpreadMax    = -1.0e30;

    // 出现最多的贪心动作占多少帧（0..1）
    double topGreedyFraction() const {
        int tot = 0, best = 0;
        for (int i = 0; i < ACT_SIZE; ++i) {
            tot += greedyActionCount[i];
            if (greedyActionCount[i] > best) best = greedyActionCount[i];
        }
        return (tot > 0) ? (static_cast<double>(best) / static_cast<double>(tot)) : 0.0;
    }

    // 贪心动作的切换率（每帧换几次）。≈0 = 网络无视输入
    double greedySwitchRate() const {
        return (greedyFrames > 0)
             ? (static_cast<double>(greedySwitchCount) / static_cast<double>(greedyFrames))
             : 0.0;
    }

    // logit 幅度均值
    double logitSpreadAvg() const {
        return (greedyFrames > 0) ? (logitSpreadSum / static_cast<double>(greedyFrames)) : 0.0;
    }

    // ---- 关键项的【每帧速率】累加（已乘 dt，未乘任何整局系数）----
    //
    //  ★ 于是 `fedRateSum / ticks` 就是"平均每帧从这一项拿到多少分"，
    //    可以直接和别的项比量级 —— 这是判断"谁在主导每帧信号"的唯一办法。
    double fedRateSum      = 0.0;   // ⑤ 吃饱余韵：每帧实际加的分
    double hungerRateSum   = 0.0;   // ③ 饥饿扣分：每帧实际扣的分（负数）

    // ---- 每一项【实际生效】了多少帧（2026-09-25 新增）----
    //
    //  ★ 为什么光有"总和"不够 —— 本轮真实踩到的推理错误
    //
    //  实测 `snake` 项在关掉 fed 的组里是 −10.66/局，于是我想反推
    //  "它有多少帧处在危险区内"，算出来是 71 帧（满额）。
    //  但这个反推【依赖"每帧都是满额"这个假设】，而梯度是线性的，
    //  贴着边界待 1000 帧也能凑出同样的总和 —— 两种情况的成因和修法完全不同。
    //
    //  所以必须分开记【帧数】和【总和】：
    //      帧数占比  → 这个危险区到底常不常触发（决定它是不是主要压力）
    //      总和/帧数 → 触发时平均有多深（决定梯度信号强不强）
    //  只看总和永远分不清这两件事。
    int dangerFrames = 0;   // ⑥′ 我的头处于敌方危险区内的帧数（诊断：常不常进危险区）
    int hungerFrames = 0;   // ③ 处于饥饿扣分窗口内的帧数
    // ★ 本局是否真的死过（用于区分"饿死"和"饿到局末"）
    //   已有的 died 字段在 tick() 里置位，clear() 会把它清掉 —— 行为不变。
    void clear() { *this = EpisodeStats{}; }
};

// ---------------------------------------------------------------------------
//  这一局用哪种方式选动作
//
//  ★ 用枚举而不是两个 bool：`learning=true, greedy=true` 这种自相矛盾的组合
//    在 bool 下是合法代码、能编译、行为还很怪；枚举下根本写不出来。
//
//  ★ 注意：这些模式只作用于【学习中的那条蛇】（learnerIndex）。
//    场上其它的蛇一律走它们自己的大脑（默认是规则 AI）——
//    它们是"环境的一部分"，不是学习对象。
// ---------------------------------------------------------------------------
enum class RunMode {
    Train,    // 采样动作 + 记录轨迹（调用方在局末调 finishEpisode 才会更新网络）
    Eval,     // argmax 动作，不记录、不更新（给人看它现在的水平）
    Random    // 均匀随机动作（基线，用来做"零点"对比）
};

// ---------------------------------------------------------------------------
class EpisodeRunner {
public:
    // trainer 可以为 nullptr（只跑 Random / Eval，不学习）
    EpisodeRunner(arena::Arena& arena, Trainer* trainer, const TaskConfig& cfg)
        : m_arena(arena)
        , m_trainer(trainer)
        , m_cfg(cfg)
    {}

    // 开一局（会 reset 世界）
    void begin(RunMode mode) {
        m_mode  = mode;
        m_stats.clear();
        m_stepHungerLeft = 1.0e9;
        m_done  = false;

        // ★ 事件式奖励/计时的状态必须【每局清零】
        //   否则新一局会带着上一局的状态：
        //     · m_inDanger 残留   → 开局就白送一次逃生奖励
        //     · 位移窗口 残留     → 开局就拿着上一局的位移去判定"没动"
        //   （和帧堆叠必须每局 reset 是同一类错误）
        m_inDanger   = false;
        m_dispHist.clear();
        m_stepHeadX = 0.0;
        m_stepHeadY = 0.0;
        // ★★ 本局"吃过没有"必须清零 —— 它就是 ⑤ 余韵的门控
        //    不清的话，新一局开局就会带着上一局的"吃过了"标记，
        //    于是**开局白送 fed 的 bug 会以另一种形式复活**。
        //    （和 m_inDanger / 位移窗口是同一类错误，所以放在一起。）
        m_hasEatenThisEpisode = false;

        // ★ 帧堆叠的历史必须【每局清零】
        //   否则新一局的头几帧会带着【上一局】的画面当"记忆" ——
        //   现象是每局开头几十帧行为诡异，而且越往后越难看懂。
        m_frameStack.reset();

        // ★ 响应幅度探针的"上一帧贪心动作"也必须每局清零 ——
        //   否则两局之间的接缝会被记成一次假的"动作切换"。
        m_prevGreedyAction = 0;
        m_hasPrevGreedy    = false;

        m_arena.reset();

        // 诊断：记录【出生】时的位置与离墙距离。
        // 和"死亡位置"一对比就能分清是"出生几何就有问题"还是"移动后才出问题"。
        if (m_learnerIndex < m_arena.agents().size()) {
            const Snake& s = m_arena.agents()[m_learnerIndex].body;
            m_stats.spawnX = s.getX();
            m_stats.spawnY = s.getY();
            m_stats.mapW   = m_arena.mapWidth();
            m_stats.mapH   = m_arena.mapHeight();
            m_stats.spawnDistanceToWall = std::min(
                std::min(m_stats.spawnX, m_stats.mapW - m_stats.spawnX),
                std::min(m_stats.spawnY, m_stats.mapH - m_stats.spawnY));
        }

        // ★ 只有 Train 模式才开一条新轨迹 —— Eval / Random 不该碰训练缓冲
        if (m_trainer && m_mode == RunMode::Train) m_trainer->beginEpisode();
    }

    // -----------------------------------------------------------------------
    //  选动作。调用方把它接到 Arena 的 controller 上。
    //
    //  ★ index 不是学习中的那条 → 交给它自己的大脑（规则 AI），
    //    这样它们就是"环境的一部分"：行为固定，学习信号才可归因。
    //
    //  ★ 学习中的那条：编码观测 → 采样/取最大 → 记进训练缓冲。
    //    "编码"和"记录"必须发生在同一时刻（感知刚填好的那一瞬间），
    //    所以都放在这里，不给调用方漏记的机会。
    // -----------------------------------------------------------------------
    arena::Action decide(std::size_t index, const arena::Perception& p,
                         std::mt19937& rng)
    {
        // ---- 不是学习中的那条：用它自己的大脑 ----
        //
        //  ⚠️ 2026-09-27：这里曾经有一段"队友一致性探针"（用学习者的网络对同一个
        //     观测算贪心动作，和队友实际动作比对，用来判断"队友行为不同"是权重
        //     不同还是采样发散）。**已整段删除**，原因有两条：
        //
        //     ① 负责人已经明确"不再考虑这个问题"，它对当前任务没有任何用处；
        //     ② 它里面用了 `dynamic_cast<arena::NeuralBrain*>` ——
        //        而 `rl/` 层**不允许依赖 `brain/`**（分层铁律：把 brain/ 和 rl/
        //        删掉，仿真层必须照样能编译）。所以那段代码从写下的那一刻起
        //        就是违反架构的，只是拖到编译时才暴露（'NeuralBrain' does not
        //        name a type）。
        //
        //  ★ 教训：加"临时诊断"之前先想清楚它属于哪一层。
        //    诊断代码同样受分层约束 —— 因为它在同一个 .cpp 里被编译。
        if (index != m_learnerIndex) {
            agent::Agent& a = m_arena.agents()[index];
            return a.brain ? a.brain->think(p) : arena::Action{};
        }

        // ---- 编码观测（唯一的编码函数，与游戏共用）----
        //
        // ★ 2026-09-24 清理：原来这里接住了 encodeObservation 的返回值
        //   （"视野内最近食物的距离"），存进 m_stepFoodDist 供"视野内食物塑形"用。
        //   那次奖励重写把那一项删掉之后，这个成员和我这行赋值都成了残留 ——
        //   而成员声明早就删了，只剩赋值 → 编译报"未声明"。
        //   现在返回值显式丢弃：它唯一的使用者已经不存在了。
        (void)encodeObservation(p, m_stepObs.data());

        // ★★ 帧堆叠：把这一帧压进历史，拿到网络真正吃的输入
        //
        //   ⚠️ 必须在 encodeObservation 之后、forward 之前 ——
        //      顺序反了就会拿【上一帧】的堆叠向量去决策，
        //      现象是"决策慢一帧"，很难查。
        m_frameStack.push(m_stepObs.data(), m_stackedObs.data());

        // ★ 记下最近的 logits，供界面显示"策略到底学没学到东西"
        //
        //   为什么要看这个：蛇"原地转圈"这个现象有两种完全不同的原因，
        //   只看画面分不清 ——
        //     · 动作概率接近 33/33/33 → 策略【什么都没学到】（梯度没生效）
        //     · 动作概率明显偏向某一类 → 策略学到了，但学的是错的行为
        //   这两种的处理方式完全不同（一个是调奖励/学习率，一个是改任务）。
        m_haveLogits = false;

        // ===================================================================
        //  ★★ 2026-09-26：④「靠近最近食物」塑形【整项已删除】
        //
        //  ★ 负责人原话："为什么会有这种东西？改掉"
        //    （指的是"刚吃到一个、目标换成更远的"时候 approach 是负数）
        //
        //  ★ 为什么必须删，而不是再加补丁
        //
        //    ④ 是【差分】项：`+0.01 × (上一步距离 − 这一步距离)`。
        //    于是"远离食物"必然得负分 —— 而这带来一个**结构性的荒谬**：
        //
        //        吃掉眼前那个食物 → "最近食物"换成更远的一个 → 距离跳变
        //        → 这一步拿到一个大负数 → **吃食物本身被惩罚**
        //
        //    旧代码为此打过一个补丁（"目标换了就跳过这一步的差分"，
        //    用 `m_prevApproachTargetX/Y` 判定）。但补丁只是**掩盖**了症状：
        //      · 它依赖于"位置浮点相等"来判定"是不是同一个目标"，本来就脆
        //      · 它并没有消除"远离食物就扣分"这条规则本身
        //      · 实测它每局要拦下 5.8~24 步 —— 说明这个荒谬**频繁发生**
        //
        //    ★ 更根本的问题：这条塑形**和 `feed`/`eat` 的目标是矛盾的**
        //      "吃到食物"必然伴随"下一个目标更远"，所以奖励在
        //      **同时奖励和惩罚同一个动作**。这不是调系数能解决的。
        //
        //  ★ 删掉之后的损失 —— 诚实评估
        //    实测 ④ 只占总回报的 **2.4%~5%**，是次要项。
        //    所以删掉它的代价很小，而消除"吃被罚"的价值很大。
        //    （历史上它占比高过，但那是因为当时有哨兵值 bug 撑起了虚假的量级。）
        //
        //  ★ 相关的诊断探针（首帧/末帧距离、跳变次数、换目标步数）也一并删除 ——
        //    留着"恒为 0 的僵尸字段"会让后来人以为这一项还在起作用，
        //    那正是本项目反复踩过的坑（见 docs/30 缺陷 #30 的三个死字段）。
        // ===================================================================

        // ---- ⑥⑦ 用的两个【方向相反】的量（负责人要求，2026-09-24 恢复）----
        //
        //  ★ 直接读感知里算好的字段，不在这里重推
        //    它们由 `PerceptionBuilder` 算好（而且是【只在视野内】算，
        //    队友不算敌方）—— 感知层重推一遍就是"两份实现"。
        //
        //  ★ 为什么必须是两个量而不是一个
        //    规则是「我的头碰对方身体才死」，所以：
        //        我的头 → 对方身体 ：危险，要躲        （headToEnemyBodyDist）
        //        对方头 → 我的身体 ：优势，要利用      （enemyHeadToMyBodyDist）
        //    用"头到头"或"中心距"当代理会把这两件事混成同一个信号，
        //    结果既惩罚危险、也惩罚包围。
        m_stepSnakeDist = static_cast<double>(p.headToEnemyBodyDist);
        m_stepHuntDist  = static_cast<double>(p.enemyHeadToMyBodyDist);

        // ---- ⑪ 危险距离扣分用的"头到最近墙的距离" ----
        //
        //  ★ 为什么要在这里【抄一份】而不是等 tick() 里再去读感知
        //    因为 `tick()` 拿不到 `p` —— 感知对象只在这一次 decide() 调用
        //    里存在，tick() 是在 `arena.step()` 之后跑的。
        //    这和 `m_stepSnakeDist` / `m_stepHungerLeft` 是同一个模式：
        //    **decide() 负责抄下本帧要用的感知字段，tick() 只结算奖励。**
        //
        //  ⚠️ 抄错了会很隐蔽：它不会被重置成 0，而是"上一帧的值"，
        //    现象是扣分整体慢一帧（几乎看不出来）。所以和上面两个字段
        //    放在一起、用同样的命名（m_stepXxx），一眼能看出是同一批。
        //  ⚠️ 访问路径是 `p.headToWallDist`（Perception 自己的字段），
        //    **不是** `p.self.headToWallDist` —— 它不在 SelfState 里。
        //    2026-09-26 就因为这个写错编译失败过一次：原因是
        //    `resetFrameState()` 里写的是裸名字 `headToWallDist = ...`，
        //    看不出它属于哪一层。**加字段时先确认它挂在哪个结构体下。**
        m_stepWallDist = static_cast<double>(p.headToWallDist);

        // ---- 剩余饥饿时间（③ 饥饿扣分 与 ⑤ 吃饱余韵 共用）----
        //
        // ★ 直接读感知里的字段 —— 它是 Arena 在推进时算好的
        //   （starvationTime − 距上次进食的时间），所以不用在这里重算，
        //   也就不会出现"两处算法不一致"的问题。
        m_stepHungerLeft = static_cast<double>(p.self.hungerLeft);

        int action = 0;

        if (m_mode == RunMode::Random || m_trainer == nullptr) {
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            action = static_cast<int>(uni(rng) * ACT_SIZE) % ACT_SIZE;
            for (int i = 0; i < ACT_SIZE; ++i) m_lastProbs[i] = 1.0 / ACT_SIZE;
            m_haveLogits = true;
        } else {
            // ★ 必须 forward —— 网络的缓存只保存最后一次前向
            const auto& logits = m_trainer->net().forward(toInput(m_stackedObs.data()));

            // 记下概率分布（不管哪种模式都记 —— 评估时更需要看）
            const auto probs = nn::softmax(logits);
            double sum = 0.0;
            for (int i = 0; i < ACT_SIZE; ++i) {
                m_lastProbs[i] = probs.a[i];
                sum += probs.a[i];
            }
            if (sum > 1.0e-9) {
                for (int i = 0; i < ACT_SIZE; ++i) m_lastProbs[i] /= sum;
            }
            m_haveLogits = true;

            // ★ 把两个速度档合并成"转向偏好"（给界面/日志用）
            //   probLeft = P(左转普通) + P(左转加速)
            //
            //  ⚠️⚠️ 2026-09-28 修越界：这里是"转向档 × 速度档"的二维展开，
            //     档数必须由 ACT_SIZE 决定，**不能写死 2 个速度档**。
            //     回滚后 ACT_SIZE = 3 = ACT_STEER_COUNT，若仍按 `b < 2` 展开，
            //     `m_lastProbs[s + 1*3]` 就会读到下标 3..5 ——
            //     而 m_lastProbs 只有 3 个元素，**越界读**。
            //     （本项目因为"用一个常量当两个概念"越界写过 96 字节，
            //       所以这类展开一律用算式推，不写死数字。）
            const int boostSlots = (ACT_SIZE + ACT_STEER_COUNT - 1) / ACT_STEER_COUNT;
            for (int s = 0; s < ACT_STEER_COUNT; ++s) {
                m_steerProbs[s] = 0.0;
                for (int b = 0; b < boostSlots; ++b) {
                    const int idx2 = s + b * ACT_STEER_COUNT;
                    if (idx2 >= 0 && idx2 < ACT_SIZE) m_steerProbs[s] += m_lastProbs[idx2];
                }
            }

            // ★★ 这一步是不是 ε-greedy 抽出来的随机步（2026-09-27）
            //
            //  ★ 为什么要单独记一个局部变量、而不是在 pushStep 那里重新判一次
            //    因为"是不是随机步"这个信息**只在这里存在** ——
            //    `pushStep` 只拿到 (obs, action)，它无法反推出这个动作
            //    是采样来的还是掷硬币来的。重判一次就会引入第二份实现
            //    （本项目在"观测编码"和"世界推进"上已经各栽过一次）。
            //
            //  ★ 它拿去干什么：Trainer 会用这个标志把那一步**排除在梯度之外**。
            //    原因见 `Step::fromExploration`（用策略自己的概率给均匀抽样出来的
            //    动作加权，等于假设重要性比为 1，方向是错的）。
            bool fromExploration = false;

            if (m_mode == RunMode::Eval) {
                action = nn::argmax(logits);
            } else {
                // ---- ★ ε-greedy：以固定概率随机选动作 ----
                //
                //  ★ 为什么不能只靠熵正则（实测踩过的坑）
                //    策略概率曾经锁死成 (0,0,1)：蛇永远右转、原地绕圈。
                //    熵正则拦不住，因为它的梯度 ∝ p —— p 越小越拉不动，
                //    而策略梯度还在把 p 推得更小，形成【自我强化的坍缩】。
                //
                //    ε-greedy 的探索强度【不随 p 衰减】，所以能打破那个循环：
                //    无论策略多极端，每 N 步总会试一次别的动作。
                //
                //  ★ 随机选的那一步**照常记进轨迹**（保留诊断价值），
                //    但会打上 fromExploration 标记，由 Trainer 在局末
                //    【不把它累加进梯度】—— 见上面那段说明。
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                const double r = uni(rng);

                if (r < m_cfg.explorationEps) {
                    // 随机探索：另取一个随机数（r 已经用来判定了）
                    action = static_cast<int>(uni(rng) * ACT_SIZE) % ACT_SIZE;
                    fromExploration = true;
                } else {
                    // 按策略采样：★ 用同一个 r，所以要先把它映射回 [0,1)
                    // （因为已经扣掉了 [0, eps) 那一段，否则会偏向"直行"）
                    const double q = (r - m_cfg.explorationEps)
                                   / (1.0 - m_cfg.explorationEps);
                    double acc = 0.0;
                    action = ACT_SIZE - 1;
                    for (int i = 0; i < ACT_SIZE; ++i) {
                        acc += probs.a[i];
                        if (q < acc) { action = i; break; }
                    }
                }
            }

            // ===============================================================
            //  ★★★ 响应幅度探针（2026-09-27）—— 纯只读，不影响任何决策
            //
            //  ★ 为什么量的是【贪心动作】而不是【采样出来的动作】
            //    训练模式下 `action` 是采样结果，它每一步都在随机变，
            //    拿它算"切换率"会恒等于 1，什么也说明不了。
            //    而敌方/队友在游戏里用的是 `argmax` —— 要复现它们的行为，
            //    就必须也看 argmax。**两者必须是同一个量才可比。**
            //
            //  ★ 这三个数合起来回答一个二元问题
            //      "网络到底有没有在看输入？"
            //        · 看：局面一变 argmax 就换 → switchRate 明显 > 0
            //        · 不看：整局死咬一个动作 → switchRate ≈ 0，topFrac ≈ 1
            //
            //  ⚠️ 必须在 forward 之后、且用同一组 logits ——
            //     重跑一次 forward 会白白多花一次前向（本项目感知已占 51~69%）。
            // ===============================================================
            {
                const int gAct = nn::argmax(logits);

                double zmax = logits.a[0], zmin = logits.a[0];
                for (int i = 1; i < ACT_SIZE; ++i) {
                    if (logits.a[i] > zmax) zmax = logits.a[i];
                    if (logits.a[i] < zmin) zmin = logits.a[i];
                }
                const double spread = zmax - zmin;
                m_stats.logitSpreadSum += spread;
                if (spread < m_stats.logitSpreadMin) m_stats.logitSpreadMin = spread;
                if (spread > m_stats.logitSpreadMax) m_stats.logitSpreadMax = spread;

                // 第一帧没有"前一帧"，所以第一帧只进直方图、不进切换率
                if (m_hasPrevGreedy && gAct != m_prevGreedyAction) {
                    ++m_stats.greedySwitchCount;
                }
                if (m_hasPrevGreedy) ++m_stats.greedyFrames;
                m_prevGreedyAction = gAct;
                m_hasPrevGreedy    = true;

                if (gAct >= 0 && gAct < ACT_SIZE) ++m_stats.greedyActionCount[gAct];
            }

            if (m_mode == RunMode::Train) {
                m_trainer->pushStep(m_stackedObs.data(), action, fromExploration);
            }
        }

        // ★ 诊断：记下【实际执行】的动作（含 ε-greedy 的随机步）。
        //   它才是真正作用在世界上的动作，所以是"策略是否锁死"最可靠的依据 ——
        //   网络输出的概率可能看起来还在变，但执行的若是同一个动作，
        //   行为上就是刻板的。
        //
        //  ⚠️ 动作空间现在是 3 个（左 / 直 / 右）—— 29 维回滚后没有加速档，
        //     所以"转向档位"就是动作本身，`actCount` 直接就是转向分布。
        //     （历史：曾是 6 个 = 3 转向 × 2 速度，那时要看合并后的档位。）
        if (action >= 0 && action < ACT_SIZE) ++m_stats.actCount[action];

        return arena::Action{ steerOfAction(action), boostOfAction(action) };
    }

    // 最近一次决策的动作概率（左转 / 直行 / 右转）——【把两个速度档合并】
    //
    //  ★ 这是判断"策略有没有在学"最直接的窗口：
    //      三项都接近 0.333 → 策略是均匀的，等于没学到东西
    //      某一项占优        → 策略有偏好；再看这个偏好对不对
    //
    //  ★ 为什么合并两个速度档
    //    动作空间**曾经**是 6 个（3 转向 × 2 速度），"锁死"这个现象是转向层面的 ——
    //    "永远左转"无论加速与否都是刻板行为，所以要合并。
    //    合并之后这三项是"策略在转向上的偏好"，语义与历史口径一致，
    //    游戏内的面板也因此一行都不用改。
    //
    //  ★ 2026-09-28 回滚后：ACT_SIZE = 3，**没有加速档**，
    //    所以"合并"是恒等操作，probLeft/Fwd/Right 直接就是三个动作的概率。
    //    保留这层结构是为了以后想再试加速时不用改调用方。
    //    ⚠️ 现在不存在 actCount[3..5] 了（数组只有 3 个元素）。
    // -----------------------------------------------------------------------
    bool hasPolicyProbs() const { return m_haveLogits; }
    double probLeft()  const { return m_steerProbs[0]; }
    double probFwd()   const { return m_steerProbs[1]; }
    double probRight() const { return m_steerProbs[2]; }

    // -----------------------------------------------------------------------
    //  推进一个 tick：世界走一步 → 结算奖励 → 交给训练器 → 判断是否结束
    //
    //  ★ 奖励只算【学习中的那条】。
    //    其它蛇的生死只是世界状态的一部分（会影响食物、会影响碰撞），
    //    但它们不是学习对象，给它们发奖励没有意义。
    // -----------------------------------------------------------------------
    void tick(float dt) {
        if (m_done) return;
        if (m_learnerIndex >= m_arena.agents().size()) { m_done = true; return; }

        const agent::Agent& learner = m_arena.agents()[m_learnerIndex];
        const int scoreBefore = learner.body.getScore();
        const int killsBefore = learner.body.getKills();   // ★ ⑧ 击杀奖励用

        m_arena.step(dt);
        ++m_stats.ticks;

        const agent::Agent& after = m_arena.agents()[m_learnerIndex];

        // ★★ ⑩′ 用得到的：本帧蛇头【坐标】
        //
        //  ★ 为什么存坐标、而不是"这一帧走了多远"
        //    因为判据是**净位移**（有没有换地方），不是**路径长度**（动了多少）。
        //    绕圈的路径长度很大但净位移≈0 —— 用路径长度会漏判（实测 `disp` 恒为 0）。
        //    详见 tick() 里 ⑩′ 那一段与成员声明处的说明。
        //
        //  ⚠️ 必须在 arena.step() 【之后】取坐标，否则拿到的是上一步的位置。
        m_stepHeadX = static_cast<double>(after.body.getX());
        m_stepHeadY = static_cast<double>(after.body.getY());

        // 诊断：身体最长到过几节
        if (after.body.getSegmentCount() > m_stats.maxSegments) {
            m_stats.maxSegments = after.body.getSegmentCount();
        }

        // ===================================================================
        //  ★★★ 奖励结算（2026-09-24 全量重写；2026-09-26 修正食物分值标定）
        //
        //  完整设计与常数依据见 docs/90-奖励机制.md。这里只列实现要点：
        //
        //    ① 死亡       −10.0   死了就基本白干，严格劣于任何其它选择
        //    ② 吃到食物   +3.0    唯一的、压倒性的正收益来源
        //                          （按分值折算：普通食物 = FOOD_SCORE = 1 → +3.0）
        //    ③ 饥饿       0 → −3  剩余 10s 内线性加深，封顶 −3
        //    ④ 靠近食物   【已整项删除，见 TrainTask.h 的说明】
        //    ⑤ 吃饱余韵   +2.5 → 0 吃到后 10s 内每秒递减（原 5.0，负责人要求减半）
        //    ⑩′ 饿着原地转 −3.0   最近 3s 净位移 < 120px 时每帧扣
        //    ⑪ 危险距离   −0.5 → 0 ★ 2026-09-26 深夜新增
        //                          墙：头到边界 < 150px 时线性扣
        //                          敌身：头到敌身 < 150px 时线性扣
        //                          两项独立，可叠加
        //
        //  ★ 与旧机制最大的区别：**没有存活奖励**。
        //    旧机制每帧白送 +0.002，900 帧就是 1.8 分 ——
        //    它奖励的是"耗时间"而不是"做事"，是把策略钉死在苟活上的主因。
        // ===================================================================
        double r = 0.0;

        // ---- ② 吃到食物 ----
        //
        //  ★ 2026-09-24 改成【按实际得分缩放】
        //
        //    原来是无条件 `+rEat`（固定 3.0）。但食物的 `scoreValue` 是不同的：
        //        普通食物 = FOOD_SCORE = 1
        //        尸体食物 = corpseScore（训练任务里设成 5，负责人要"吃尸体更值"）
        //    固定给分的话，吃尸体和吃普通食物在【训练奖励】里完全一样，
        //    "尸体更值"就只体现在游戏分数上、训练里根本学不到。
        //
        //    改成 `rEat × (分值 / 普通食物分值)` 之后：
        //        吃普通食物 → rEat × 1/1 = 3.0
        //        吃尸体     → rEat × 5/1 = 15.0   （更值）
        //    而且**归一化到普通食物分值**，所以以后调 `corpseScore`
        //    不需要同步改 `rEat` —— 只有一个旋钮。
        //
        //    ⚠️ 2026-09-26：上面这两个数是**修正后**的。在此之前
        //      `foodScoreUnit` 被错写成 3.0（见 TrainTask.h 那一项），
        //      于是实际只给到 1.0 / 5.0 —— 吃一个食物只有设计意图的 1/3。
        bool ateThisTick = false;
        const int scoreDelta = after.body.getScore() - scoreBefore;
        if (scoreDelta > 0) {
            // 单位 = 普通食物的分值（归一化基准，改 corpseScore 时不用动 rEat）
            const double unit = (m_cfg.foodScoreUnit > 0.0) ? m_cfg.foodScoreUnit : 1.0;
            const double gain = m_cfg.rEat * (static_cast<double>(scoreDelta) / unit);
            r += gain;
            m_stats.eatGain += gain;
            ++m_stats.eaten;
            ateThisTick = true;
            // ★★★ 本局"吃过东西"标记 —— ⑤ 余韵的门控（2026-09-26 深夜）
            //
            //  ★ 为什么放在这里，而不是用 `timeSinceEat`
            //    `Snake::resetStarvation()` 在【出生】和【吃到】时都把
            //    `timeSinceEat` 归零，所以两者无法区分；而出生后第一帧
            //    它就变成 > 0 —— 用 `> 0` 当门控等于没做。
            //
            //  ★ 为什么用"分数变大"是**唯一正确**的判据
            //    `scoreDelta > 0` 是这一帧真的吃到东西的**定义本身**
            //    （② 的结算就用它）。所以它不可能和"吃"这件事不一致 ——
            //    这比任何间接推断（计时器回退、hungerLeft 跳变）都可靠。
            //
            //  ★ 一旦置位本局不再清除：余韵的语义是"吃完之后的满足感"，
            //    而不是"刚吃的那一帧"。所以第二次吃之后当然还要照常给。
            m_hasEatenThisEpisode = true;
        }

        // ★ 最小学习验证：吃掉之后，在前方再放【一串】—— 让"走向食物"
        //   【在一局里反复发生】，而且一次能连吃好几个。
        //
        //   不这么做的话，一个食物吃完这一局就没事可做了，
        //   一局只能产生 1 个正样本，学得极慢。
        //
        //   ★ 为什么放【一串 3 个】而不是 1 个（负责人要"增加食物的量"）
        //     · 食物变多 → 吃到不再是稀缺事件，它不必为了一个食物冒险
        //     · 排成一串 → 最优策略是"一直往前走"，而不是"小心翼翼地蹭"
        //
        //   重放在【蛇当前朝向的正前方】，所以它必须持续朝前走才吃得到。
        //
        //   ⚠️ 放在这里（局内）而不是 Arena::reset()，是因为只有训练才知道
        //      "什么是可以重复的课程"。Arena 只管世界规则，不管教学法。
        if (ateThisTick && m_cfg.fixedFoodAhead && !m_arena.foods().empty()) {
            const Snake& s = after.body;
            const float a = s.getAngle();
            const float ux = std::cos(a), uy = std::sin(a);

            // 先把场上剩的都清掉 —— 否则会越积越多，任务越来越简单
            m_arena.food().reset();

            const int cnt = (m_cfg.fixedFoodCount > 0) ? m_cfg.fixedFoodCount : 1;
            for (int k = 0; k < cnt; ++k) {
                const float d = m_cfg.fixedFoodDistance
                              + static_cast<float>(k) * m_cfg.fixedFoodSpacing;
                m_arena.food().spawnAt(s.getX() + ux * d, s.getY() + uy * d);
            }
        }

        // ---- ④ 靠近最近食物的塑形 —— 【2026-09-26 整项已删除】----
        //
        //  ★ 负责人原话："为什么会有这种东西？改掉"
        //    （指的是"刚吃到一个、目标换成更远的"时候 approach 是负数）
        //
        //  ★ 删除的理由（不是"效果不好"，是"逻辑上就是错的"）
        //
        //    这是【差分】项：`+0.01 × (上一步距离 − 这一步距离)`。
        //    所以"远离食物"必然得负分。而"吃到食物"这个最该被鼓励的动作，
        //    恰恰**必然导致距离变大**（最近的那个没了，目标换成更远的）——
        //    于是奖励在**同时奖励和惩罚同一个动作**。
        //
        //    旧代码为此打过一个补丁：用"目标位置是否相同"来判定"换没换目标"，
        //    换了就跳过这一步的差分。
        //    **但那只是掩盖症状**：
        //      · 它依赖浮点位置相等来判定"同一个目标"，本来就脆
        //      · 它没有消除"远离食物就扣分"这条规则本身
        //      · 实测每局要拦下 5.8~24 步 —— 说明这个荒谬**频繁发生**
        //
        //  ★ 删掉之后的损失 —— 诚实评估
        //    实测 ④ 只占总回报 **2.4%~5%**，是次要项；而 `fed`(57~72%)
        //    与 `eat` 才是主导。所以删它的代价小，消除"吃被罚"的价值大。
        //
        //  ★ 一并删除的诊断探针
        //    approach 的首帧/末帧距离、单帧最大变化、跳变次数、换目标步数 ——
        //    它们只为这一项存在。留着"恒为 0 的僵尸字段"会让后来人以为
        //    这一项还在起作用（本项目踩过这个坑，见 docs/30 缺陷 #30）。
        //
        //  ⚠️ 代价必须承认：删掉一个**密集的**方向信号之后，
        //     策略在"离食物还远"的时候**完全没有方向指引**了，
        //     只能靠走进视野（512px）后从食物扇区读到方向。
        //     这可能让"远距离导航"更难 —— 但那是**真实的信息缺口**，
        //     应该用观测去补（已有的圈方位/距离就是干这个的），
        //     而不是用一个会惩罚"吃"的假信号去填。
        // ===================================================================

        // ---- ③ 快饿死：越饿扣越多，封顶 ----
        //
        //   公式：每秒扣分 = rHungerMax × (1 − 剩余秒数 / hungerWarnSec)
        //   逐帧的话再乘 dt。
        //
        //   例（rHungerMax=−3、hungerWarnSec=10）：
        //       剩余 10s → 每秒   0  → 每帧  0
        //       剩余  5s → 每秒 −1.5 → 每帧 −0.025
        //       剩余  0s → 每秒 −3.0 → 每帧 −0.05   ← 封顶
        //
        //   ★ 为什么封顶取 −3 而不是更多：饿死那一刻本来就有【死亡 −10】，
        //     如果饥饿也爬到 −10，那一刻总扣 −20，会把死亡罚的相对分量稀释掉。
        //     −3 让"饿着"始终是次要压力，主要压力来自"会死"。
        if (m_cfg.hungerWarnSec > 0.0 && m_stepHungerLeft < 1.0e8) {
            const double left = std::max(0.0, m_stepHungerLeft);       // 饿死后是负数，夹掉
            if (left < m_cfg.hungerWarnSec) {
                const double frac = 1.0 - left / m_cfg.hungerWarnSec;  // 0..1，越饿越大
                const double g = m_cfg.rHungerMax * frac * dt;
                r += g;
                m_stats.hungerGain += g;
                m_stats.hungerRateSum += g;      // ★ 诊断：每帧速率
                ++m_stats.hungerFrames;          // ★ 诊断：这一帧真的在扣
            }
        }

        // ---- ⑤ 吃饱后的余韵：吃到后 10s 内每秒加分递减到 0 ----
        //
        //   公式：每秒加分 = rFedMax × (1 − elapsed / fedDuration)
        //         elapsed = 距上次进食过了多久
        //
        //   ★ 为什么能用 hungerLeft 反推 elapsed
        //     Arena 每帧算好 `hungerLeft = starvationTime − timeSinceEat`，
        //     而"刚吃到"时 hungerLeft 恰好等于 starvationTime。
        //     所以   elapsed = starvationTime − hungerLeft
        //     —— 不需要额外维护一个计时器，也就不会和 Arena 的饥饿逻辑不一致。
        //
        //   ★ 单个食物的满额总收益 = 吃(+3.0) + 余韵(最多 +2.5) = +5.5
        //     它【压过】饥饿上限（−3）—— 所以"去吃"永远优于"饿着"。
        //
        //  ★★★ 2026-09-26 深夜：必须区分「吃出来的余韵」和「出生自带的余韵」
        //
        //  ★ 负责人提出的问题
        //      "我需要确认，开局的 10s 是否加 fed？我希望不加"
        //
        //  ★ 确认结果：**原实现在开局白送满额**
        //      `Snake::resetStarvation()` 的注释就是"吃东西【或刚出生】时调用"，
        //      而它把 `m_timeSinceEat` 归零。
        //      于是开局 `hungerLeft = starvationTime` → `fedLeft = 10s`
        //      → `ratio = 1.0` → **前 10 秒白送 2.5/秒 = +25 分**，
        //      而这条蛇**一口都没吃过**。
        //
        //  ★ 为什么这是一个真问题（不只是"送了点分"）
        //    1. 它是**每局恒定的 +25**，会被 baseline 吸收 → 对梯度几乎没用
        //    2. 它污染了「奖励构成」：`fed` 的读数里混进了与"吃"无关的部分，
        //       而 `fed` 正是历史上占总回报 57~72% 的主导项 ——
        //       读数虚高会让人误判"吃饱了"这件事的收益有多大
        //    3. 语义上错的：余韵奖励的是"刚吃到的满足感"，
        //       出生时并没有吃过东西
        //
        //  ★ 修法：用【本局分数是否变过】作为判据（见下面 `m_hasEatenThisEpisode`）
        //    ⚠️ 我第一版用的是 `timeSinceEat > 0`，**那是错的**：
        //       `resetStarvation()` 在【出生】和【吃到】时都把计时归零，
        //       所以出生时是 0、刚吃到也是 0 —— 无法区分；
        //       而出生后第一帧它就会变成 > 0，**门控等于没做**。
        //       正确的判据是"分数变大过"，它在 ② 那一处已经算出来了，
        //       直接置一个本局标记即可（见 tick() 里 ② 的 `ateThisTick`）。
        //
        //  ⚠️ 代价：开局那 10 秒从 +25 变成 0，
        //     所以**所有历史回报数字都会整体下降约 25**（若活满 10 秒）。
        //     比较改动前后的回报必须减掉这个偏移，不能直接比。
        const bool hasEatenThisEpisode = m_hasEatenThisEpisode;
        if (m_cfg.fedDuration > 0.0 && m_stepHungerLeft < 1.0e8) {
            const double st = static_cast<double>(m_arena.config().starvationTime);
            const double fedLeft = st - std::max(0.0, m_stepHungerLeft);   // 剩余余韵秒数
            if (fedLeft > 0.0) {
                const double ratio = std::min(1.0, fedLeft / m_cfg.fedDuration);
                const double g = m_cfg.rFedMax * ratio * dt;
                if (hasEatenThisEpisode) {
                    r += g;
                    m_stats.fedGain += g;
                    m_stats.fedRateSum += g;         // ★ 诊断：每帧速率
                } else {
                    // ★ 单独记账：这是"开局白送"的部分，改动后应恒为 0。
                    //   留着它是为了**验证修复生效**（也是防复活）。
                    m_stats.fedSkippedBirth += g;
                }
            }
        }

        // ===================================================================
        //  ★★ 诊断：三种饥饿状态的帧数分桶（2026-09-25 新增，不影响奖励）
        //
        //  ★ 为什么要单独记这个
        //
        //  ⑤ 余韵的公式是 `fedLeft = starvationTime − timeSinceEat`，
        //  只要 `timeSinceEat < fedDuration` 就一直在加分。
        //  于是"吃一口"买到的不是一次性的分，而是【一段时间的每帧带宽】。
        //
        //  判据：如果 fed 帧占比接近 1，说明蛇大部分时间都泡在余韵里 ——
        //  那时余韵奖励的不再是"吃东西"，而是"活着"，
        //  而"活着"这件事和"往哪走"无关 → 它对策略的区分度是零。
        //  这正是"绕圈不掉分、于是绕圈成为局部最优"的机制假设，
        //  本轮就是要用这个占比去证实或证伪它。
        //
        //  ★ 阈值全部来自 TaskConfig，不在这里写死 —— 换了任务参数也自动跟着走。
        // ===================================================================
        {
            // ★ 用 starvationTime 反推"距上次进食多久"，
            //   和 Arena 的 `hungerLeft = starvationTime − timeSinceEat` 是同一个式子。
            const double st       = static_cast<double>(m_arena.config().starvationTime);
            const double sinceEat = st - m_stepHungerLeft;

            // ⚠️ 分桶的判据必须和【第 ③ 项扣分的条件】逐字一致（`left < hungerWarnSec`），
            //    否则会出现"明明在扣分，却算进 mid 桶"这种自相矛盾的统计。
            //
            //  ★★ 2026-09-26 深夜：`isFed` 必须和 ⑤ 的门控同口径
            //
            //  ★ 为什么（这是一个"统计和奖励不一致"的坑）
            //    ⑤ 现在规定"本局没吃过东西就不给余韵"，
            //    而出生时 `sinceEat == 0`，于是 `isFed` 会算成 true ——
            //    统计说"在余韵里"，实际一分没加。
            //    **统计与现实不符会让「吃饱帧占比」这个判据失去意义。**
            //    （这个项目栽过好几次"两处判据不一致"，所以宁可重复写一遍注释。）
            //
            //  ⚠️ 门控就是 ⑤ 用的 `m_hasEatenThisEpisode`，**不是**某个计时器：
            //     出生与"刚吃到"在计时器上无法区分（见 tick() 里的说明）。
            const bool fedActive  = (m_cfg.fedDuration > 0.0)
                                 && m_hasEatenThisEpisode               // ← 与 ⑤ 同口径
                                 && (sinceEat < m_cfg.fedDuration);
            const bool isFed      = fedActive;
            const bool isStarving = (m_cfg.hungerWarnSec > 0.0)
                                 && (std::max(0.0, st - sinceEat) < m_cfg.hungerWarnSec);

            if (isFed)           ++m_stats.framesFed;
            else if (isStarving) ++m_stats.framesStarving;
            else                 ++m_stats.framesNormal;
        }

        // ---- ⑩′ 饥饿时【净位移不足】惩罚（2026-09-26 负责人要求）----
        //
        //  ★ 负责人原话："把持续转圈的惩罚删掉，改成在饥饿时间（也就是 <10s）时，
        //                3s 内总位移小于一定长度持续扣分（和饥饿扣分相同）。"
        //
        //  ★★ 2026-09-26 修正：判据从【总路径长度】改成【净位移（平均坐标偏移）】
        //
        //    ⚠️ 我第一版用的是"每帧位移的累加"（总路径长度），这是**错的**：
        //      **蛇只要在动，转一圈的路径长度就有 750px** ——
        //      也就是说"一直绕圈"必然满足"路径 ≥ 阈值"，惩罚**永远不会触发**。
        //      实测结果就是 `disp` 恒为 0。
        //
        //    ★ 负责人指出："不应该检测移动总距离（一直绕圈也能满足），
        //      而应该检查总位移，或一段时间平均坐标的偏移。"
        //
        //    ★ 现在用【平均坐标偏移】（比单纯的"首尾坐标差"更稳）：
        //        把 3 秒窗口的坐标样本分成前后两半，各求平均坐标，
        //        净位移 = 两个平均坐标之间的距离。
        //          · 绕圈    → 平均值几乎不动 → 净位移 ≈ 0  → **正确触发**
        //          · 直线赶路 → 平均值明显平移 → 净位移大    → 不触发
        //          · 原地抖  → 平均值不动                  → 正确触发
        //      用"平均"而不是"首尾两点"，是为了不被单帧的位置噪声骗到。
        //
        //  ★ 判据（两个条件【同时】成立）
        //      ① 剩余饥饿 < hungerWarnSec（10s）—— 已经进入饥饿窗口
        //      ② 最近 dispWindowSec（3s）的【净位移】 < dispMinPx
        //
        //  ★ ① 为什么必须有：没有"饥饿"这个前提，
        //    慢慢巡逻找食物也会被罚 —— 那是正常且必要的探索。
        //
        //  ★ 为什么这条能同时抓到"绕圈"和"直行卡死"两种锁死形态
        //    因为它只看结果：**你这 3 秒有没有真的换个地方**。
        if (m_cfg.dispWindowSec > 0.0 && m_cfg.dispPenalty < 0.0) {
            // ① 饿了才算
            const bool hungry = (m_cfg.hungerWarnSec > 0.0)
                             && (m_stepHungerLeft < 1.0e8)
                             && (std::max(0.0, m_stepHungerLeft) < m_cfg.hungerWarnSec);

            if (hungry) {
                m_dispHist.push_back(HeadSample{ m_stepHeadX, m_stepHeadY });

                // 只保留最近 dispWindowSec 秒的样本
                const int want = static_cast<int>(m_cfg.dispWindowSec * 60.0 + 0.5);
                while (static_cast<int>(m_dispHist.size()) > want) {
                    m_dispHist.pop_front();
                }

                // ② 窗口填满之后才判定（否则开局样本不足会误触发）
                if (static_cast<int>(m_dispHist.size()) >= want && want >= 4) {
                    // 前后两半各求平均坐标（用定点累加，窗口只有 180 个样本，开销可忽略）
                    const std::size_t half = m_dispHist.size() / 2;
                    double ax1 = 0.0, ay1 = 0.0, ax2 = 0.0, ay2 = 0.0;
                    for (std::size_t i = 0; i < m_dispHist.size(); ++i) {
                        const HeadSample& s = m_dispHist[i];
                        if (i < half) { ax1 += s.x; ay1 += s.y; }
                        else          { ax2 += s.x; ay2 += s.y; }
                    }
                    const double n1 = static_cast<double>(half);
                    const double n2 = static_cast<double>(m_dispHist.size() - half);
                    if (n1 > 0.0 && n2 > 0.0) {
                        ax1 /= n1; ay1 /= n1;
                        ax2 /= n2; ay2 /= n2;
                        const double ddx = ax2 - ax1;
                        const double ddy = ay2 - ay1;
                        const double netDisp = std::sqrt(ddx * ddx + ddy * ddy);

                        if (netDisp < static_cast<double>(m_cfg.dispMinPx)) {
                            const double g = m_cfg.dispPenalty * static_cast<double>(dt);
                            r += g;
                            m_stats.dispGain += g;
                            ++m_stats.dispPenaltyFrames;
                        }
                    }
                }
            } else {
                // 不饿 → 窗口清零。语义：只在"饿着"的连续时间段里看位移，
                //        吃到东西就重新开始（否则会拿"上一轮饿时"的旧坐标误判）。
                m_dispHist.clear();
            }
        }

        // ---- ⑪ 危险距离扣分：墙 / 敌身（负责人要求，2026-09-26 深夜）----
        //
        //  ★ 负责人原话
        //    "现在我们加上危险距离扣分试试（墙，头距离敌人身体），
        //      扣分大小比逃逸多一些" / "越近扣分越多，线性增加"
        //    "墙和敌人身体都是 150px 危险距离。-0.5 每帧先试试"
        //
        //  ★ 公式：每帧 r × (1 − d/dangerDist)，两项独立可叠加
        //
        //      d >= dangerDist → 不扣（比例 <= 0，下面的 if 直接跳过）
        //      d == 0          → 扣满额 r
        //
        //  ★ 为什么是"每帧"而不是"撞了才罚"
        //    `rDie = −10` 只在撞上那一刻给，而且局就结束了 ——
        //    它只教"别撞"，教不了"该提前多远开始躲"。
        //    每帧线性扣分把"危险程度"变成一个**在撞上之前就生效**的
        //    连续信号：离得越近越亏，于是"提前转向"本身就有收益。
        //
        //  ★ 墙用的是 `Perception::headToWallDist`（垂直距离）
        //    不是从视野射线推 —— 射线只有 4 个正交方向，斜冲墙角量不准。
        //    见 Perception.h 里那个字段的说明。
        //
        //  ★ 敌身用的是 `headToEnemyBodyDist`（已算好，只在视野内）
        //    ⚠️ 它 **只算了敌方，不算队友** —— 队友之间能互相穿过，不构成危险。
        //    ⚠️ 视野外是哨兵 1e9，而 150px << 视野 512px，所以不会漏判。
        {
            // ---- 曲线：peak × (1 − d/lim)^power ----
            //
            //  ★ 为什么必须有 power（而不是纯线性）—— 2026-09-26 深夜实测驱动
            //
            //    负责人第一次看到面板上是 `wall(11) 0.05`、`enemy(11) 0.05`
            //    —— **最小的两项**，和"最强惩罚"完全相反。
            //
            //    原因算一下就清楚：范围锁在 150px 时，蛇穿过整个危险区只要
            //        150px ÷ 250px/s = 0.6 秒 = 36 帧
            //    线性曲线下每局总额 = ½ × peak × 36/60：
            //        peak −0.25 → 总额 −0.075   ← 实测 0.05 量级，吻合
            //        peak −3.0  → 总额 −0.90
            //    而"提前绕开墙"要多走 2 秒 = 饥饿 −1.2。
            //    **所以线性下"撞上去"比"绕开"便宜得多，给不出躲避的梯度。**
            //
            //  ★ 负责人的要求（2026-09-26 深夜）
            //      "不要改范围（允许存在贴边的高难动作），但是太近了要痛
            //       （就像走钢丝一样紧张），我要求如果是撞死，那么这个扣分
            //       能达到 -5 左右"
            //
            //  ★ 这两个要求合起来 = 曲线必须【非线性】
            //    线性做不到：要撞一次 −5 得 peak ≈ −16.7/帧（−1000/秒），
            //    那在离墙 150px 处就已经痛得离谱，和"允许贴边"直接冲突。
            //    改成三次方后：外面几乎无感、贴脸剧痛。
            //
            //        d=150px（刚进区） → 0
            //        d= 75px          → −0.053/帧（ −3.2/秒）
            //        d= 50px          → −0.124/帧（ −7.5/秒）
            //        d= 25px          → −0.243/帧（−14.6/秒）
            //        d≈  0（贴脸）    → −0.42 /帧（−25.0/秒）
            //
            //    标定依据：撞死前走完的比例约 1/3，于是
            //        总额 ≈ peak × (1/3) × 0.6s × 60 = peak × 12
            //        peak = −0.42 → **撞死一次 ≈ −5.0**（负责人指定的目标）
            //
            //  ⚠️ 加速的影响（几何决定，不是额外规则）
            //    450px/s 时穿越只要 0.33 秒 → 撞死总额约 −2.5。
            //    顺带表达了"加速冲墙更亏"，与"加速会加速饥饿"是同一个方向。
            const double power = static_cast<double>(m_cfg.dangerCurvePower);

            // ---- 墙 ----
            if (m_cfg.wallDangerDist > 0.0f && m_cfg.rWallDanger < 0.0) {
                const double dw = m_stepWallDist;
                const double lim = static_cast<double>(m_cfg.wallDangerDist);
                if (dw < lim) {
                    // clamp：距离无效（负数）时比例别超过 1
                    const double u = std::max(0.0, 1.0 - dw / lim);
                    const double ratio = (power == 1.0) ? u : std::pow(u, power);
                    const double g = m_cfg.rWallDanger * ratio * static_cast<double>(dt);
                    r += g;
                    m_stats.wallDangerGain += g;
                    ++m_stats.wallDangerFrames;
                }
            }

            // ---- 敌身 ----
            if (m_cfg.enemyDangerDist > 0.0f && m_cfg.rEnemyDanger < 0.0) {
                const double de = m_stepSnakeDist;
                const double lim = static_cast<double>(m_cfg.enemyDangerDist);
                if (de < lim) {   // 1e9（视野外）必然 >= lim，天然跳过
                    const double u = std::max(0.0, 1.0 - de / lim);
                    const double ratio = (power == 1.0) ? u : std::pow(u, power);
                    const double g = m_cfg.rEnemyDanger * ratio * static_cast<double>(dt);
                    r += g;
                    m_stats.enemyDangerGain += g;
                    ++m_stats.enemyDangerFrames;
                }
            }
        }

        // ---- ⑧ 击杀奖励（负责人要求：击杀其它蛇奖励 5）----
        //
        //  ★ 在此之前"击杀"在训练奖励里【一分不给】—— 只有游戏分数和 ⑥⑦
        //    那对间接信号。于是蛇没有动机去主动制造击杀。
        //
        //  ★ 怎么知道发生了击杀
        //    用 `Snake::getKills()` 的变化量。现有的碰撞系统已经在加这个计数
        //    （`other->addKill()`），所以**碰撞系统一行都不用改**。
        //
        //  ★ 和"吃尸体"的关系（两笔不同的账，都是刻意分开的）
        //    击杀那一刻          +rKill (5)
        //    之后吃掉它的尸体    corpseScore = 5 按分值折算 → 训练里 +3.0 × 5/1 = +15.0
        //    所以"杀了并且回头吃掉"远高于"只杀不吃"—— 鼓励它捡尸体。
        if (m_cfg.rKill > 0.0 && after.body.getKills() > killsBefore) {
            const double g = m_cfg.rKill * static_cast<double>(after.body.getKills() - killsBefore);
            r += g;
            m_stats.killGain += g;
        }

        // ---- ⑥′ 逃生奖励（2026-09-26 取代原来的每帧贴敌身惩罚）----
        //
        //  ★ 负责人原话："贴近敌方身体的惩罚删掉吧。但是离敌方太近，成功离开给奖励。"
        //
        //  ★ 为什么"位置惩罚"换"事件奖励"是对症的
        //
        //    旧版是**每帧**扣分：蛇只要**待**在危险区里就一直扣，
        //    而"待着"**无法归因到具体某一步** —— 每一步都在扣，梯度被摊平。
        //    改成"离开才结算"之后，信号只出现在「危险→安全」这个**转变点**，
        //    而转变点是**可以归因到具体动作的**（就是那一步转向）。
        //
        //  ★ 状态机：硬阈值 + 迟滞（负责人选的方案乙）
        //
        //        进入 100px  → m_inDanger = true
        //        退回 150px  → 结算 +rEscape，m_inDanger = false
        //        每次进入只结算一次 → 蹭边界无法刷分
        //
        //  ★ 为什么必须迟滞（进出两个不同的阈值）
        //    单一阈值下，蛇在边界上蹭来蹭去会**反复触发**"离开" → 刷分。
        //    迟滞让进出是两道不同的门，来回蹭不再产生任何奖励。
        //
        //  ★ 边界情况（两个都已处理）
        //    ① 敌人离开视野：视野 512px ≫ 危险区 100px，所以敌人在危险区内时
        //       **必然在视野内** —— "看不见了"必然是距离变远，算离开是安全的。
        //    ② 敌人死亡/消失：下一帧距离读数是哨兵 1e9 → 同样结算一次离开。
        //       这是**合理的**（危险确实解除了）。
        if (m_cfg.escapeEnterDist > 0.0f) {
            const double enterD = static_cast<double>(m_cfg.escapeEnterDist);
            const double exitD  = enterD * static_cast<double>(m_cfg.escapeExitRatio);

            if (!m_inDanger) {
                // 还没进危险区：只看进入门
                if (m_stepSnakeDist < enterD) m_inDanger = true;
            } else {
                // 已经在危险区：只看离出门
                if (m_stepSnakeDist > exitD) {
                    r += m_cfg.rEscape;
                    m_stats.escapeGain += m_cfg.rEscape;
                    ++m_stats.escapeCount;
                    m_inDanger = false;
                }
            }
        }

        // ---- ⑦ 对方头贴近我的身体：优势，给奖励（与 ⑥ 数值相等）----
        //
        //   距离来源：`p.enemyHeadToMyBodyDist` —— 最近的敌方头 → 我的身体。
        //   它和 ⑥ 是【反方向】的一对量：
        //       我的头 → 对方身体 ：危险（⑥ 扣分）
        //       对方头 → 我的身体 ：优势（⑦ 加分）
        //   两条同时存在时，"用身体别住对手、同时把头让开"才既拿分又不扣分
        //   —— 那就是包围。
        if (m_stepHuntDist < 1.0e8 &&
            m_stepHuntDist < static_cast<double>(m_cfg.huntAdvantageDist)) {
            const double t = 1.0 - m_stepHuntDist
                                 / static_cast<double>(m_cfg.huntAdvantageDist);
            const double g = m_cfg.rHuntAdvantage * t;
            r += g;
            m_stats.huntGain += g;
        }

        if (!after.isAlive()) {
            r += m_cfg.rDie;
            m_stats.diePenalty += -m_cfg.rDie;   // 记成正数
            m_stats.died = true;

            // 诊断：死在哪、离墙多近、以及轮到【代码自己说】的死因
            m_stats.deathX = after.body.getX();
            m_stats.deathY = after.body.getY();
            m_stats.mapW   = m_arena.mapWidth();
            m_stats.mapH   = m_arena.mapHeight();
            m_stats.deathDistanceToWall = std::min(
                std::min(m_stats.deathX, m_stats.mapW - m_stats.deathX),
                std::min(m_stats.deathY, m_stats.mapH - m_stats.deathY));

            const auto& causes = m_arena.deathCauses();
            if (m_learnerIndex < causes.size()) {
                m_stats.cause = causes[m_learnerIndex];
            }

            // 原始数值：规则真正比较的那几个
            m_stats.starvationTime  = m_arena.config().starvationTime;
            m_stats.timeSinceEat    = after.body.getTimeSinceEat();
            m_stats.headRadius      = after.body.getHeadRadius();
            m_stats.deathWallDistRaw = m_stats.deathDistanceToWall;
        }

        m_stats.totalReward += r;

        // ★ 只有 Train 模式才把奖励交回训练器 —— 评估时动网络是 bug
        if (m_trainer && m_mode == RunMode::Train) m_trainer->pushReward(r);

        // ★ 实时饥饿状态（给训练面板显示"还剩几秒会饿死"）
        //   每 tick 都更新，所以面板上看到的是【当前】值而不是局末快照。
        //
        //   ⚠️ 用和 Arena 完全相同的算式（starvationTime − timeSinceEat），
        //      而不是另立一个 —— 这个项目已经因为"同一个东西算两遍"栽过两次。
        m_stats.curTimeSinceEat = after.body.getTimeSinceEat();
        m_stats.curHungerLeft   = m_arena.config().starvationTime
                                - m_stats.curTimeSinceEat;

        // ---- 这一局结束了吗 ----
        // 学习中的那条死了、或到帧数上限，这一局就结束。
        // 其它蛇死了不影响 —— 它们只是环境。
        if (!after.isAlive() || m_stats.ticks >= m_cfg.maxTicks) m_done = true;
    }

    // ---- 查询 ----
    bool                done()  const { return m_done; }
    const EpisodeStats& stats() const { return m_stats; }
    RunMode             mode()  const { return m_mode; }

    // 学习中的那条蛇在 agents() 里的下标
    void setLearnerIndex(std::size_t i) { m_learnerIndex = i; }
    std::size_t learnerIndex() const { return m_learnerIndex; }

    // ★ 最近一次决策时喂给网络的观测（排查"编码给了什么"用）
    //
    //   为什么要暴露它：观测编码是这个项目反复出问题的地方
    //   （两份实现分叉 0.96、越界写 96 字节、哨兵值混进距离）。
    //   能在出问题时【直接把向量打出来】，比读代码猜快得多。
    const std::array<double, OBS_SIZE>& lastObservation() const { return m_stepObs; }

    // ★ 最近一次决策时喂给网络的【堆叠后】输入（NN_INPUT_SIZE 维）
    //
    //  ★ 为什么还要单独暴露这个（而不只用 lastObservation）
    //    `lastObservation()` 是【单帧】观测，而网络真正吃的是 FrameStack
    //    拼出来的 NN_INPUT_SIZE 维向量。要复现"网络当时看到了什么、
    //    于是输出了什么"，必须用后者 —— 用单帧重跑 forward 会得到
    //    和当时【不同】的结果（STACK_FRAMES > 1 时尤其明显）。
    const std::array<double, NN_INPUT_SIZE>& lastStackedInput() const { return m_stackedObs; }

    // ★ 最近一次决策时网络输出的动作概率（左/直/右）
    //
    //   为什么必须暴露它：这是判断"策略到底有没有在学"最直接的窗口
    //   （项目铁律里写着的判据）：
    //      三项都 ≈ 0.333 → 什么都没学到（梯度没生效）
    //      某项明显占优   → 学到了，再看这个偏好对不对
    //      锁死 0/0/1     → 坍缩成永远同一个动作
    //   ⚠️ 命令行工具以前没有这个输出，只能靠游戏界面看 —— 本轮补上。
    double probL() const { return m_lastProbs[0]; }
    double probF() const { return m_lastProbs[1]; }
    double probR() const { return m_lastProbs[2]; }

private:
    arena::Arena& m_arena;
    Trainer*      m_trainer = nullptr;
    TaskConfig    m_cfg;

    EpisodeStats m_stats;
    RunMode      m_mode = RunMode::Train;
    bool         m_done = false;

    // 哪一条是学习对象。默认第 0 条（Arena 里第一条出生的蛇）。
    std::size_t m_learnerIndex = 0;

    // controller 与奖励结算之间的桥：观测/动作在 step 之前定，
    // 奖励要在 step 之后才能算（要看分数有没有涨、蛇有没有死）。
    std::array<double, OBS_SIZE> m_stepObs{};

    // ★ 帧堆叠：最近 K 帧的历史 + 拼好的网络输入
    //
    //   ★ 为什么不把"上一帧观测"简单存一份就够
    //     要 K 帧就得存 K 帧。FrameStack 负责那份历史，
    //     每局开始时要 reset()（否则新一局会带着上一局的尾巴，
    //     "记忆"里混进别的局的画面）。
    FrameStack                        m_frameStack;
    std::array<double, NN_INPUT_SIZE> m_stackedObs{};

    // ---- ④ 靠近最近食物的相关成员 —— 【2026-09-26 整项删除】----
    //
    //  原来这里有 m_stepApproachDist / m_prevApproachDist /
    //  m_stepApproachTargetX/Y / m_prevApproachTargetX/Y / m_stepIsCorpse。
    //  它们只为"差分塑形"服务，而那一项会惩罚"吃掉食物"，已整项删除。
    //  见 tick() 里 ④ 那一段的完整理由。

    // ---- ③⑤ 饥饿 / 吃饱余韵共用的量 ----
    //
    //  ★ 剩余饥饿时间。`Arena::step()` 里算好写进感知
    //    （`hungerLeft = starvationTime − timeSinceEat`），这里只是取用，
    //    **不重算** —— 重算就会出现"两处算法不一致"。
    //    ③ 用它算"有多饿"，⑤ 用它反推"距上次进食过了多久"。
    double m_stepHungerLeft = 1.0e9;

    // ---- ⑥⑦ 贴近敌方身体 / 猎杀优势（负责人要求）----
    //
    //  ★ 两个【方向相反】的量，都直接取自感知（不重算）：
    //      m_stepSnakeDist = 我的头 → 最近的敌方身体/头   （危险，⑥ 扣分）
    //      m_stepHuntDist  = 最近的敌方头 → 我的身体       （优势，⑦ 加分）
    //
    //  ⚠️ 它们在 2026-09-24 的奖励重写里被删过一次（当时那两个奖励项没了），
    //     现在因为负责人要重新加机制而恢复。**删掉成员却忘了删赋值**会导致
    //     编译报"未声明"——这个坑本轮踩过两次，注意别再犯。
    double m_stepSnakeDist  = 1.0e9;
    double m_stepHuntDist   = 1.0e9;

    // ---- ⑪ 危险距离扣分（墙 / 敌身，2026-09-26 深夜）----
    //
    //  ★ 头到最近地图边界的距离，在 decide() 里从感知抄下来。
    //    语义和上面两个一样：**本帧要用的感知字段的快照**。
    //  ★ 哨兵 1.0e9 = "本帧还不知道" —— 取这个值时代码里 `d < limit`
    //    必然为假，也就是【不扣分】，所以即使某帧忘了抄也不会误扣。
    double m_stepWallDist   = 1.0e9;

    // ---- ⑤ 余韵的门控状态（2026-09-26 深夜）----
    //
    //  ★ 为什么需要它
    //    负责人要求"开局那 10 秒不加 fed"。而出生时
    //    `resetStarvation()` 把 `timeSinceEat` 归零，与"刚吃到"完全无法区分，
    //    所以只能靠一个【本局是否真的吃到过】的标记。
    //
    //  ★ 置位点：tick() 的 ② 结算里 `scoreDelta > 0` 那一处
    //    （那是"吃到东西"的定义本身，不可能不一致）
    //  ★ 清零点：begin()（每局开头）—— 忘了清会让 bug 以另一种形式复活
    //
    //  ⚠️ 这里**故意没有** `m_stepTimeSinceEat` 这类中间量：
    //     "距上次进食多久"在本文件里已经有唯一来源 ——
    //     `m_stats.curTimeSinceEat`（在 tick() 里从 `after.body` 取），
    //     而"本局吃过没有"只能靠这个 bool。
    //     2026-09-26 我曾加过一个 `m_stepTimeSinceEat` 想走 `p.self.timeSinceEat`，
    //     结果**编译失败**（SelfState 里根本没有这个字段）——
    //     那次失败正好暴露了"这个量不需要、也不该有第二条来源"。
    bool   m_hasEatenThisEpisode = false;

    // ---- ⑥′ 逃生奖励的状态机（2026-09-26）----
    //
    //  ★ 为什么必须有这个状态：奖励是【事件式】的（离开才给），
    //    所以必须记住"我现在是不是在危险区里"。
    //    没有它就无法区分"离开危险区"和"一直在外面待着"。
    bool   m_inDanger = false;

    // ---- ⑩′ 净位移不足惩罚的状态（2026-09-26）----
    //
    //  ★ 存【坐标样本】而不是"每帧走了多远"
    //
    //    ⚠️ 第一版存的是"每帧位移"，然后用它们的**和**去判定 —— 那是错的：
    //      蛇只要在动，绕一圈的**路径长度**就有 750px，于是"一直绕圈"
    //      必然通过检查，惩罚永远不触发（实测 `disp` 恒为 0）。
    //      **要判的是"有没有换地方"，那就必须存坐标、算净位移。**
    //
    //  ★ 窗口大小 = dispWindowSec × 60 = 180 个样本（3 秒 @60fps）
    struct HeadSample { double x; double y; };
    std::deque<HeadSample> m_dispHist;

    // 本帧蛇头位置（每帧从 Arena 取，供上面那个窗口用）
    double m_stepHeadX = 0.0;
    double m_stepHeadY = 0.0;

    // ---- ★ 响应幅度探针的状态（2026-09-27）----
    //
    //  ★ 为什么必须【每局清零】
    //    "上一帧的贪心动作"如果跨局残留，新一局第一帧会被算成一次"切换"，
    //    而那只是两局之间的接缝，不是策略行为。
    //    这是本项目反复踩过的一类错（m_inDanger / 帧堆叠 / 位移窗口
    //    都是"事件式状态没每局清"），所以放在最显眼处并注明。
    int  m_prevGreedyAction = 0;
    bool m_hasPrevGreedy    = false;


    // 最近一次决策的动作概率 —— 给界面判断"策略有没有在学"
    //
    //  ⚠️ 长度是 ACT_SIZE。**回滚后 ACT_SIZE = 3**（左 / 直 / 右，没有加速档），
    //     所以这一组数现在【就是】转向偏好本身，和 m_steerProbs 数值相同。
    //     两者都保留是为了少改调用方（界面读 m_steerProbs，诊断读 m_lastProbs）。
    double m_lastProbs[ACT_SIZE] = { 1.0 / ACT_SIZE, 1.0 / ACT_SIZE, 1.0 / ACT_SIZE };
    // 把两个速度档合并后的"转向偏好"（左 / 直 / 右），三项和为 1
    //   ★ 3 动作空间下"合并"是恒等操作，但保留它让界面代码一行都不用改；
    //     以后若把 ACT_SIZE 改回 6，这里会自动重新变成真正的合并。
    double m_steerProbs[ACT_STEER_COUNT] = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    bool   m_haveLogits = false;
};

}  // namespace rl
