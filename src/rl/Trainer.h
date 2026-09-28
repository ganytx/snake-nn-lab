#pragma once

// ============================================================================
//  Trainer.h — REINFORCE 策略梯度：**唯一的实现**
//
//  ★ 为什么要有这个文件
//
//  2026-09-22 之前，"跑一局 + 学一次"这套逻辑只有 tools/train_probe.cpp 里有。
//  后来负责人要在游戏里做「训练模式」（亲眼看模型学习），
//  就面临一个选择：再写一份，还是抽出来共用。
//
//  **反向传播的梯度累加顺序是这个项目里最易错的一段代码**：
//
//      net.zeroGrad()          ← 清空累加器
//      从后往前逐时刻 backward ← ★ 必须对每个时刻【重跑 forward】，
//                                网络的缓存只保存最后一次前向
//      net.scaleGrad(1/T)      ← ★ 平均梯度，忘了就把学习率放大 T 倍
//      net.step(lr, momentum)
//
//  四步顺序不能乱、第三步不能忘。写第二份几乎必然会跑偏，
//  而且跑偏的现象很迷惑（能跑、不报错、只是学不会/学歪）。
//
//  所以抽到这里，**游戏和 train_probe 都调它**。
//
//  ★ 这一层不管"世界怎么跑"
//
//  世界推进在 arena::Arena 里（那份也只有一个）。本文件只负责
//  "拿到一串 (观测, 动作, 奖励) 之后怎么更新网络"。
//
//  典型用法（和 train_probe 的 runEpisode 一致）：
//
//      每个 tick：
//          vec x = rl::toInput(obs);
//          logits = net.forward(x);
//          probs  = nn::softmax(logits);           ← :内容 >
//          action = 按 probs 采样（训练）/ argmax（评估）
//          存下 (obs, action)
//          arena.step(dt) 之后拿到 reward
//          trainer.pushReward(reward)
//      一局结束：
//          trainer.finishEpisode()                 ← 这里才真正更新网络
// ============================================================================

#include "encoder.h"
#include "nn/loss.h"
#include "nn/network.h"

#include <algorithm>   // std::min（correlation 里用）
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>      // std::fprintf（轨迹被截断时的告警，见 pushStep）
#include <vector>

namespace rl {

// ---------------------------------------------------------------------------
//  ★★ 用哪种方式算 advantage（2026-09-26 新增）
//
//  ★ 为什么要有这个开关
//
//  负责人怀疑「越训越差」的根因在优化过程而不是奖励设计。实测支持了这个判断：
//  奖励一行没改，只把 lr 从 0.02 降到 0.005，一个原本会永久锁死的种子
//  （动作熵 0.171、每局吃 0.22）就变成不锁死且持续变好（熵 1.076、吃 2.19）。
//
//  所以要把学习侧单独换掉、而奖励一分不动 —— 否则改完分不清是哪一边的作用。
//
//  ★ Baseline（默认）逐位等价于改动前 —— 这是刻意的回滚保证。
// ---------------------------------------------------------------------------
enum class LearningMode {
    Baseline,   // 纯 REINFORCE：adv = G − 跨局标量基线（原行为）
    Critic,     // Actor-Critic：adv = GAE，由价值网络 V(s) 算出来
};

// ---------------------------------------------------------------------------
//  一个 (观测, 动作) 对。
//
//  ★ 只存观测和动作，**奖励单独存**：
//    因为奖励是在 arena.step() 【之后】才能算出来的（要看分数有没有涨、
//    蛇有没有死），而观测和动作是在 step 之前定下的。分成两个容器更直白。
// ---------------------------------------------------------------------------
struct Step {
    // ★ 存的是【堆叠后】的输入（NN_INPUT_SIZE 维），不是单帧观测。
    //   为什么：REINFORCE 在局末要对每个时刻重跑 forward，
    //   而网络吃的是堆叠向量 —— 存单帧的话局末就还原不出当时的输入了
    //   （那需要把整段历史再重建一遍，既麻烦又容易错）。
    //   代价是内存 STACK_FRAMES 倍，对这个规模完全无所谓。
    std::array<double, NN_INPUT_SIZE> obs{};
    int action = 0;

    // -----------------------------------------------------------------------
    //  ★★★ 这一步的动作是不是 ε-greedy 的【随机探索步】（2026-09-27 新增）
    //
    //  ★ 为什么必须记下来 —— 这是一个真实的 bug，不是"精度优化"
    //
    //    `EpisodeRunner` 在 ε 概率下用**均匀分布**抽一个动作，然后把它
    //    和其他步一样记进轨迹。而局末的更新是：
    //
    //        g = softmaxCrossEntropyGrad(probs, action) × advantage
    //
    //    这里的 `probs` 是**策略自己**的分布。也就是说：一个由【均匀分布】
    //    抽出来的动作，被当成【策略自己选的动作】去加权 ——
    //    数学上等价于假设"行为策略 = 目标策略"，重要性采样比恒为 1.0。
    //
    //  ★ 后果不是"多一点噪声"，而是【方向性错误】
    //    举一个具体的例子：
    //        策略此刻输出 (左 0.05, 直 0.90, 右 0.05)
    //        ε-greedy 掷到了"左转" → 执行左转 → 恰好吃到食物 → advantage 为正
    //        → 梯度去【强化左转】，而策略当时认为左转是最差的动作
    //    反复发生之后，策略被推向"当时最不看好的那个动作"。
    //    **它会把随机乱撞出来的成功当成自己的策略来学。**
    //
    //  ★ 为什么表现恰好是"好像完全没在学"
    //    ε=0.05 → 每 20 步有 1 步是这种错误样本。它们的方向是随机的，
    //    但它们**和真正该学的信号混在同一个累加器里**，而且
    //    "乱撞撞对了"这种事件天然带有较大的正 advantage（撞对了才有奖励），
    //    于是错误样本的权重反而偏高。净效果是梯度方向被稀释成噪声。
    //
    //  ★ 处理方式：这类步**照常记进轨迹**（保留诊断和回放价值），
    //    但在 `finishEpisode()` 里【不参与梯度累加】。
    //    —— 这也正是"off-policy 样本要丢掉"最朴素的正确做法。
    bool fromExploration = false;
};

// ---------------------------------------------------------------------------
//  REINFORCE 训练器
//
//  持有网络 + 一局的轨迹缓冲。生命周期：
//      一个 Trainer 管【一个网络】，跨很多局。
// ---------------------------------------------------------------------------
class Trainer {
public:
    // ---- 超参数（默认值和 train_probe 原本用的一致）----
    double gamma     = 0.99;    // 折扣因子
    double lr        = 0.02;    // 学习率
    double baselineMomentum = 0.95;  // 基线的滑动平均系数

    // -----------------------------------------------------------------------
    //  ★ 熵正则系数（**这是防"策略坍缩"的关键**）
    //
    //  ★ 为什么必须有它 —— 实测踩过的坑
    //
    //    训练出来的策略概率曾经变成 `0.00 0.00 1.00`，
    //    也就是【永远右转】→ 蛇绕一个固定半径的圈，原地打转。
    //
    //    一旦走到这一步就【出不来】了，因为纯 REINFORCE 有个结构缺陷：
    //        熵 = 0 → 采样永远出同一个动作 → 永远试不到别的动作
    //        梯度里的 (onehot − p) ≈ 0    → 就算试了也几乎没有梯度
    //    策略被"冻"住，坍缩不可逆。
    //
    //  ★ 加熵奖励之后：策略有动力【保持一点随机性】，
    //    于是它会持续探索"左转/直行/右转"，不会过早锁死在一个动作上。
    //
    //  ★ 梯度为什么是 −entCoef·(log p + 1)：
    //      熵 H = −Σ pᵢ log pᵢ，最大化 H 就是最小化 −H
    //      dH/dz = p · ( −log p − (1/p)·p ) 的逐元素形式化简后
    //            = −p · (log p + 1)
    //      我们要"最大化熵"，等价于对 −H 做梯度下降，
    //      所以送进 backward 的是 +(−(log p + 1)) 乘系数。
    //
    //  ★ 取 0.1：熵最大是 log3 ≈ 1.10，所以这一项最多贡献约 0.11/步。
    //
    //  ⚠️ 但**别指望它单独阻止坍缩**：p=1e-4 时它的梯度只有 1e-5，
    //     而策略梯度约 1e-2 —— 小三个数量级，而且 p 越小它越小，
    //     所以坍缩是自我强化的。真正的保险是 ε-greedy
    //     （见 EpisodeRunner 里的 explorationEps）。
    //     这里留 0.1 是为了给策略一个"别太早变确定"的温和偏好。
    double entropyCoef = 0.1;

    // 一局最多攒多少步（防止死循环时缓冲无限涨）
    //
    // ★★★ 2026-09-26 修正：4096 → 8192
    //
    //  ★ 为什么必须改（这是一个真实的 bug，不是调参）
    //    随机圈任务 `maxTicks = 90 * 60 = 5400`，而旧上限是 4096
    //    → **一局只要活过 4096 帧，后面 1304 帧就全被静默丢掉**。
    //
    //    `pushStep()` 是"满了就 return"（静默丢观测+动作），
    //    `pushReward()` 当时没有上限（奖励照涨到 5400）。
    //    于是 `finishEpisode()` 里 T = 4096、R = 5400，
    //    按 T 对齐 → **后 1304 帧的奖励连同死亡那一笔 −10 全部丢失**。
    //
    //  ★ 后果为什么严重
    //    丢的是"存活久的局"的尾巴 —— 也就是 `hunger` / `disp`
    //    累计最多的那一段。**越是活得久、越需要被纠正的行为，
    //    越是被丢干净**。死亡罚 −10 也只在"恰好死在 4096 帧之前"才学得到。
    //
    //  ★ 现在取 8192：≥ 5400 且留足余量。
    //    内存代价（Step 约 280 字节）：8192 × 280 ≈ 2.2 MB，可忽略。
    //    ⚠️ 以后任何任务的 `maxTicks` 都必须 ≤ 本值，否则会再次截断 ——
    //       `pushStep()` 里现在有告警，出现告警就说明这个前提被违反了。
    int maxStepsPerEpisode = 8192;

    // ★ 本局有没有因为缓冲满而丢掉观测/动作（给面板和 probe 报警用）
    //
    //  ★ 为什么要有这个标志：截断是**静默**的 —— 现象是"学不会"，
    //    而不是"崩了"，所以历史上一直没被发现。有它就能一眼看见。
    bool lastEpisodeTruncated = false;

    // -----------------------------------------------------------------------
    //  ★★ 梯度裁剪上限（2026-09-25 新增）
    //
    //  ★ 为什么必须有它 —— 「训练好好的突然就废了」的标准成因
    //
    //  策略梯度的更新量 = lr × 平均梯度，而平均梯度正比于【整局回报】。
    //  本项目实测单局回报从 −167 到 +499（差 30 倍），于是**同一套超参下
    //  不同局的实际步长能差几十倍**。那种异常大的局会一次性把参数踹飞，
    //  而它造成的破坏是不可逆的 —— 策略被推进一个坏区域后，
    //  后续采样全来自坏策略，于是越训越差。
    //
    //  裁剪把"一步最多走多远"钉死：异常大的局只影响【方向】，不影响【幅度】。
    //
    //  ★ 取 1.0：本网络参数范数量级约 30（实测），梯度范数裁剪到 1.0
    //    意味着单步最多把参数移动 lr × 1.0 = 0.02 ——
    //    相对参数范数只有 0.07%，不可能一脚踹飞。
    //
    //  ★ <= 0 表示关闭，默认【关闭】= 与改动前逐位一致（可回滚保证）。
    // -----------------------------------------------------------------------
    double gradClipNorm = 0.0;

    // -----------------------------------------------------------------------
    //  ★★ Actor-Critic 的超参（2026-09-26 新增）
    //
    //  ⚠️ 这些【只在 mode == Critic 时生效】。Baseline 模式下它们一行都不参与，
    //     所以默认行为与改动前逐位一致。
    // -----------------------------------------------------------------------

    // 用哪种 advantage
    LearningMode mode = LearningMode::Baseline;

    // 价值网络的学习率。
    //
    // ★ 为什么先取和策略一样的 0.02：两个网络吃同一个观测、隐藏层同宽，
    //   梯度量级本来就在一个数量级上。**先用同值，有判据再分开调** ——
    //   一上来就分别取两个互相"配平"的数，会把两个变量同时引入，
    //   出了问题分不清是哪一个。
    double valueLr = 0.02;

    // GAE 的 λ。
    //
    //   λ = 1 → 退化成整局折扣回报 G_t（观测全用上，但方差最大）
    //   λ = 0 → 退化成单步 TD 误差 r + γV(s') − V(s)（方差最小，但偏差最大：
    //           完全依赖 V 的准确性，V 没学好就学不动）
    //   0.95 是标准折中。
    double gaeLambda = 0.95;

    // -----------------------------------------------------------------------
    //  ★★ critic 每个 episode 训练几个 epoch（2026-09-26 新增）
    //
    //  ★ 为什么需要它 —— 实测判据
    //    负责人报 `V-target corr ≈ 0.3`，即 V 只解释回报方差的 **9%**，
    //    **critic 基本没学会**，而 GAE 建立在它上面 →
    //    advantage 里 91% 是噪声 → actor 拿噪声更新。
    //
    //  ★ 为什么"每局 1 次更新"学不动
    //    回报标准差 44.5 → MSE 量级几百；参数需要移动相当大的距离，
    //    而每局只有一次 `step()`，且步长被 ÷T 压掉。
    //    **3000 个样本喂一次更新，critic 收敛不了。**
    //
    //  ★ 取 5：数据已在内存里，多跑 epoch 不增加采样成本，只多几次前向+反向。
    //    ⚠️ 想更充分就调大（10~20 也常见）；太小则 critic 依旧学不动。
    //    ⚠️ 只有 critic 多跑，**actor 仍每局一次** —— 让 actor 多跑需要
    //      重要性采样，是另一个量级的改动。
    int valueEpochs = 5;

    // advantage 是否按整批做 z-score 标准化。
    //
    // ★ 为什么默认开：让 advantage 的【量级】与"整局回报"脱钩。
    //   实测单局回报从 −167 到 +499，于是同一套 lr 在不同局之间的实际步长
    //   能差 30 倍 —— 这正是"lr 敏感、换个种子就锁死"的直接来源。
    //   标准化之后每批的 advantage 都是零均值单位方差，步长只由 lr 决定。
    double normalizeAdv = true;

    // ---- 上一步更新的诊断（给"越训越差"定位用）----
    double lastGradNorm   = 0.0;   // 裁剪【前】的梯度范数
    double lastUpdateNorm = 0.0;   // 参数实际被移动的距离 = lr × 裁剪后梯度范数
    double lastParamNorm  = 0.0;   // 参数自身的范数（判断步长是否过大）
    bool   lastClipped    = false; // 这一步有没有撞到裁剪上限

    // ---- ★ 本局有多少步真的参与了学习（2026-09-27 新增）----
    //
    // ★ 为什么必须能看见这两个数
    //   ε-greedy 的随机步现在被排除在梯度之外（见 Step::fromExploration）。
    //   但如果排除逻辑写错（比如标志位没传过来、或者条件反了），
    //   现象仍然是"学不会"—— 和它要修的那个 bug 一模一样，无法区分。
    //   所以必须能把"排除了多少"直接读出来：
    //       lastPolicySteps + lastExplorationSteps == 本局步数 (T)
    //   成立才算排除逻辑是对的。
    //   判据：lastExplorationSteps / (两者之和) 应当 ≈ explorationEps（5%）。
    //         如果是 0% → 标志没传进来（bug 没修上）；
    //         如果是 100% → 条件写反了（把策略步全排除了，学不到任何东西）。
    int lastPolicySteps      = 0;
    int lastExplorationSteps = 0;

    // ---- 价值网络的诊断（判断 critic 到底有没有学起来）----
    //
    // ★ 这几个数必须看：**如果 V 学不起来，GAE 就是空的**，
    //   那时"换了算法"只是换了个说法，没有任何实质改善。
    //   判据：`valueCorr` 应该明显为正（V 与实际回报正相关）；
    //        持续接近 0 就说明 critic 没学到，要停下来查而不是继续堆。
    double lastValueLoss = 0.0;    // 本局价值网络的 MSE
    double lastValueCorr = 0.0;    // V 预测 与 实际回报 的相关系数
    double lastAdvMean   = 0.0;    // 本局 advantage 的均值
    double lastAdvStd    = 0.0;    // 本局 advantage 的标准差

    // -----------------------------------------------------------------------
    //  ★★ 敌方神经网络：冻结快照（自对弈）
    //
    //  ★ 负责人要求："我想把敌人也改成神经网络"
    //
    //  ★ 为什么用【冻结快照】而不是让敌人也学
    //    当前训练器的能力边界是【一条轨迹 + 一次更新】（REINFORCE）。
    //    让敌人也学需要两套网络、两条轨迹、两套奖励 —— 那是重做训练器，
    //    而且会把已经很大的梯度方差再放大一倍。
    //
    //    冻结快照是自对弈的标准做法（AlphaGo 那套的简化版）：
    //        敌人 = 学习网络的一份【冻结拷贝】，只决策、不学习
    //        每隔 mirrorInterval 局，把当前权重复制给快照
    //    于是对手会随着训练一起变强，但**不引入第二条轨迹** ——
    //    训练器的全部假设（单轨迹、只算一条蛇的奖励）一行都不用改。
    //
    //  ★ 快照怎么传给敌方脑
    //    Trainer 在 rl/ 层，不该 include brain/（分层铁律）。
    //    所以这里只把快照【写成一个文件】，由调用方把路径告诉 NeuralBrain。
    //    多一次磁盘 IO 换掉一条跨层依赖，划算。
    // -----------------------------------------------------------------------
    int  mirrorInterval  = 100;                            // 每多少局同步一次
    // ★ 快照文件路径。必须和 BrainRegistry 里注册 "nn-enemy" 时写的那个名字一致
    //   （两边不能互相 include，所以各自写一遍字面量，并都在这里注明）。
    const char* mirrorPath = "saves/enemy_snapshot.txt";

    // 把当前权重写进快照文件；成功返回 true
    bool writeMirrorSnapshot() {
        return m_net.save(std::filesystem::path(mirrorPath));
    }

    explicit Trainer(unsigned seed = 42u) {
        m_net.init(nn::Init::Xavier, seed);
        // ★ 价值网络用**派生**的种子，而不是同一个 seed：
        //   同一个 seed 会让两个网络拿到完全相同的初始随机数流，
        //   虽然网络形状不同（输出 3 维 vs 1 维）不至于完全相同，
        //   但没必要让它们共享流 —— 派生一个明确的偏移更好排查。
        m_vnet.init(nn::Init::Xavier, seed + 0x9E3779B9u);
    }

    nn::MLP<double, NN_INPUT_SIZE, ACT_SIZE, 1, HIDDEN_WIDTH, nn::tanhAct, nn::dtanhAct>& net() {
        return m_net;
    }
    const nn::MLP<double, NN_INPUT_SIZE, ACT_SIZE, 1, HIDDEN_WIDTH, nn::tanhAct, nn::dtanhAct>& net() const {
        return m_net;
    }

    int paramCount() const { return m_net.paramCount(); }
    double baseline() const { return m_baseline; }

    // =======================================================================
    //  ★★ 诊断：熵项到底贡献了多少梯度（2026-09-26 深夜新增）
    //
    //  ★ 为什么要量这个
    //    负责人报："越训 reward 越小，最后全是原地旋转扣的分"。
    //    这类"越训越差"必须回答一个问题：**是哪一项梯度在推动策略？**
    //    光看总回报分不出来 —— 总梯度 = 任务梯度 + 熵梯度，
    //    两者混在一个 `lastGradNorm` 里。这里把熵项单独量出来做对比。
    //
    //  判据：熵项应当 **远小于** 总梯度范数。
    //        占比一旦不小，说明策略很大程度上是被熵项推着走的 ——
    //        而熵项只喊"别太确定"，不含任何任务信息。
    //
    //  ⚠️ 这两个成员必须是 **public** 的（和 lastGradNorm / lastValueCorr 一样）：
    //     训练面板（scenes/）和 train_probe 都要读它们。
    //     2026-09-26 就是误放进 private 导致编译不过一次，别再挪回去。
    // =======================================================================
    double lastEntropyGradNorm = 0.0;

    // 把一份"只有熵项"的梯度走完 backward，量出它的 L2 范数（**不改主网络**）
    //
    //  ★ 为什么敢用"复制一份网络"这种笨办法
    //    神经网络没有"只对某个参数子集求梯度"的接口，而我又绝不能污染
    //    主网络的 dw/db（那会改掉真正的更新）。复制一份 scratch 走完整
    //    backward 是最直白、最不可能出错的做法，代价是每局多一次
    //    前向+反向（相对感知占的 51~69%，可忽略）。
    //
    //  ⚠️ scratch 必须是【策略网络】的类型（输出 ACT_SIZE 维）：
    //     熵项作用在策略 logits 上。写成 ValueNet（输出 1 维）
    //     会在 backward 那里维度不匹配 —— 两者同宽同激活，很容易看错。
    void measureEntropyGradNorm(const nn::vec<double, ACT_SIZE>& probs, double scale) {
        Policy scratch = m_net;
        scratch.zeroGrad();
        nn::vec<double, ACT_SIZE> g{};
        addEntropyGrad(g, probs);
        scratch.backward(g);
        lastEntropyGradNorm = static_cast<double>(scratch.gradNorm()) * scale;
    }

    // ---- 一局的轨迹 ----

    void beginEpisode() {
        m_steps.clear();
        m_rewards.clear();
        lastEpisodeTruncated = false;   // ★ 每局重置截断标志（见 pushStep）
        // ★ 每局重置"多少步参与了学习"（与 lastEpisodeTruncated 同一类：
        //   不重置的话上一局的读数会残留，面板上看起来"这一局没排除任何随机步"）
        lastPolicySteps      = 0;
        lastExplorationSteps = 0;
    }

    // 记下这一步的观测与动作（在 arena 决策时调用）
    //   obs 必须是【堆叠后】的 NN_INPUT_SIZE 维向量
    //
    // ★ 满了不再【静默】丢弃 —— 见 maxStepsPerEpisode 的说明。
    //   只告警一次（每局），避免刷屏；同时置 lastEpisodeTruncated 给上层看。
    //
    // ★ fromExploration：这一步是不是 ε-greedy 的随机探索步。
    //   默认 false = 与改动前逐位一致（老调用方不用改）。
    //   它的用途见 `Step::fromExploration` 的说明 —— 这类步不进梯度。
    void pushStep(const double* obs, int action, bool fromExploration = false) {
        if (static_cast<int>(m_steps.size()) >= maxStepsPerEpisode) {
            if (!lastEpisodeTruncated) {
                lastEpisodeTruncated = true;
                std::fprintf(stderr,
                    "[Trainer] 警告：本局轨迹已达上限 %d 步，后续观测/动作被丢弃。\n"
                    "          这说明任务的 maxTicks > maxStepsPerEpisode —— "
                    "请把 maxStepsPerEpisode 调大，否则这局的尾部奖励学不到。\n",
                    maxStepsPerEpisode);
            }
            return;
        }
        Step s;
        for (int i = 0; i < NN_INPUT_SIZE; ++i) s.obs[static_cast<std::size_t>(i)] = obs[i];
        s.action = action;
        s.fromExploration = fromExploration;
        m_steps.push_back(s);
    }

    // 记下这一步的奖励（在 arena.step() 返回之后调用）
    //
    // ★ 同样加上限：奖励和观测必须【同长】，否则 T 与 R 不一致，
    //   对齐时会拿错位的奖励去更新（比丢掉更糟）。
    void pushReward(double r) {
        if (static_cast<int>(m_rewards.size()) >= maxStepsPerEpisode) return;
        m_rewards.push_back(r);
    }

    int  steps() const { return static_cast<int>(m_steps.size()); }
    bool bufferFull() const { return static_cast<int>(m_steps.size()) >= maxStepsPerEpisode; }

    // ---- 学习 ----

    // 用当前轨迹更新一次网络。返回本局的折扣回报之和（G，未减基线）。
    //
    // ★ 只有这一处实现 —— 见文件头的说明。
    double finishEpisode() {
        const int T = static_cast<int>(m_steps.size());
        // 奖励比步数少一个是正常的（最后一步之后世界又推进了一次才判定死亡）；
        // 以【步数】为准，缺的补 0。
        const int R = static_cast<int>(m_rewards.size());

        if (T > 1) {
            // 先把这一局的奖励对齐成 T 个（缺的补 0）——
            // 两条学习路径都要用，所以只做一次。
            m_alignedR.assign(static_cast<std::size_t>(T), 0.0);
            for (int t = 0; t < T && t < R; ++t) {
                m_alignedR[static_cast<std::size_t>(t)] = m_rewards[static_cast<std::size_t>(t)];
            }

            // 本局【未折扣】总奖励 —— 这是人看的"这一局得了多少分"
            m_lastEpisodeReward = 0.0;
            for (int t = 0; t < T; ++t) {
                m_lastEpisodeReward += m_alignedR[static_cast<std::size_t>(t)];
            }

            if (mode == LearningMode::Critic) {
                updateCritic(T);
            } else {
                updateBaseline(T);
            }
        } else {
            m_lastReturn = 0.0;
            m_lastEpisodeReward = 0.0;
            // ★ T <= 1 时两条学习路径都不会被调用，所以这两个诊断计数器
            //   必须在这里显式清零 —— 否则面板上读到的是【上一局】的数字，
            //   而"这一局根本没更新"这件事就被掩盖了。
            //   （和 beginEpisode 里重置它们同源：不能留上一局的读数。）
            lastPolicySteps      = 0;
            lastExplorationSteps = T;
        }

        // 基线跟着最新回报走（滑动平均）
        m_baseline = baselineMomentum * m_baseline
                   + (1.0 - baselineMomentum) * m_lastReturn;

        // ★ 到期就把当前权重同步给"敌方快照"（自对弈）
        //
        //   放在 finishEpisode 里是因为它是【两条训练路径的公共点】：
        //   train_probe 和游戏内训练模式都调它，所以同步逻辑只写一处
        //   （这正是 rl::Trainer 存在的理由）。
        const char* syncNotice = nullptr;
        if (mirrorInterval > 0 && ++m_episodesSinceMirror >= mirrorInterval) {
            m_episodesSinceMirror = 0;
            mirrorWritten = writeMirrorSnapshot();
            if (mirrorWritten) syncNotice = "已同步敌方快照";
        }
        lastMirrorNotice = syncNotice;

        return m_lastReturn;
    }

    // 上一次 finishEpisode 是否写了快照（给界面提示用；nullptr = 没写）
    const char* lastMirrorNotice = nullptr;
    bool        mirrorWritten = false;

    // 上一局 finishEpisode() 返回的折扣回报
    double lastReturn() const { return m_lastReturn; }

    // 上一局【未折扣】的总奖励（人看的"这局得了多少分"）
    //
    // ★ 和 lastReturn 的区别：lastReturn 是 Σ γ^t·r_t，γ=0.99 时
    //   60 秒的局会把它压得很小，不适合直接当"成绩"读。
    double lastEpisodeReward() const { return m_lastEpisodeReward; }

    // 价值网络（给诊断和存档用）
    ValueNet&       valueNet()       { return m_vnet; }
    const ValueNet& valueNet() const { return m_vnet; }

    // 清成"全新训练"（重开训练模式时用）
    void reset(unsigned seed = 42u) {
        m_net.init(nn::Init::Xavier, seed);
        m_vnet.init(nn::Init::Xavier, seed + 0x9E3779B9u);   // ★ 和构造函数保持同一个派生规则
        m_baseline   = 0.0;
        m_lastReturn = 0.0;
        m_lastEpisodeReward = 0.0;
        m_episodes   = 0;
        m_steps.clear();
        m_rewards.clear();
        m_alignedR.clear();
        m_vHat.clear();
        m_adv.clear();
        m_vTarget.clear();
    }

    // -----------------------------------------------------------------------
    //  只重置基线（加载了别人的权重时用）
    //
    //  ★ 为什么加载权重后必须重置基线
    //
    //    基线是"最近回报的滑动平均"，它属于【这一次训练的过程状态】，
    //    而不是【网络的参数】—— 所以它没有被 save()/load() 存下来。
    //
    //    不重置的话，新加载的策略会拿【旧策略】的基线去算 advantage
    //        adv = G − baseline
    //    两个不同水平的策略混在一起比，前期梯度方向会偏，
    //    现象就是"加载完反而变笨了一阵，过一会儿才恢复"。
    // -----------------------------------------------------------------------
    void resetBaseline() { m_baseline = 0.0; m_lastReturn = 0.0; }

    // ---- 统计（给训练界面显示）----
    long episodes() const { return m_episodes; }
    void countEpisode() { ++m_episodes; }

private:
    // =======================================================================
    //  学习路径 A：纯 REINFORCE（原行为，逐位不变）
    //
    //      adv = G_t − baseline（跨局标量）
    //
    //  ★ 它的结构性缺陷：baseline 是标量，所以减不掉
    //    "同一个状态在不同局里回报不同"这件事 —— 相同状态在好局里被强化、
    //    在坏局里被削弱，梯度互相抵消。见 LearningMode 的说明。
    // =======================================================================
    void updateBaseline(int T) {
        m_net.zeroGrad();                      // ← 第 1 步

        // ★★★ 2026-09-27：ε-greedy 的随机步【不进梯度】—— 修 off-policy 偏差
        //
        //  ★ 为什么必须排除（不是优化，是纠错）
        //    随机步的动作来自【均匀分布】，下面却用【策略自己的 probs】去算梯度，
        //    等价于假设重要性比 = 1.0。后果见 `Step::fromExploration` 的完整说明：
        //    它会把"随机乱撞撞对了"当成策略自己的成功来强化，方向是错的。
        //
        //  ★ 为什么 normalize 用 m_policySteps 而不是 T
        //    `backward()` 是【累加】语义，最后要除以"累加了几项"才是平均梯度。
        //    既然只累加了 m_policySteps 项，除数就必须是它 ——
        //    用 T 会把梯度整体缩小 1/(1−ε)（ε=5% 时偏小 5%）。
        //    这不是精确的 off-policy 修正（那需要重要性采样），
        //    但它是"把样本数对齐"的最低要求，否则会混进一个系统性的缩放误差。
        int m_policySteps = 0;

        double G = 0.0;
        double gSum = 0.0;                     // 只用于统计（本局 G 的起点）
        for (int t = T - 1; t >= 0; --t) {
            const double r = m_alignedR[static_cast<std::size_t>(t)];
            // ★ 回报的递推【必须对每一步都做】：
            //   随机步的奖励和 γ 折扣同样是"世界里真实发生的事"，
            //   它必须进入 G 才能让后面（更早）的策略步拿到正确的回报。
            //   **跳过的是"这一步的梯度"，不是"这一步的奖励"。**
            G = r + gamma * G;
            if (t == 0) gSum = G;

            if (m_steps[static_cast<std::size_t>(t)].fromExploration) continue;
            ++m_policySteps;

            // ★ 必须对每个时刻重跑 forward —— 网络的缓存只保存最后一次前向
            const auto& logits = m_net.forward(toInput(m_steps[static_cast<std::size_t>(t)].obs.data()));
            const auto probs = nn::softmax(logits);

            auto g = nn::softmaxCrossEntropyGrad(
                probs, m_steps[static_cast<std::size_t>(t)].action);

            const double adv = G - m_baseline;  // ★ 减基线降方差
            g = g * adv;

            addEntropyGrad(g, probs);           // ★ 熵正则（两条路径共用）

            m_net.backward(g);                  // ← 第 2 步：累加
        }

        // ★ 一局里一步策略步都没有（ε 极高 + 局极短）→ 这局没有可学的梯度。
        //   直接返回，**不能往下走** —— `scaleGrad(1/0)` 会写出 inf，
        //   再 `step()` 一次整个网络就变成 nan，而现象只是"突然学废了"。
        if (m_policySteps <= 0) {
            m_lastReturn = gSum;
            lastPolicySteps = 0;
            lastExplorationSteps = T;
            return;
        }

        // ★ 诊断：量出熵项单独的梯度范数（必须在 scaleGrad 之前抓一份 probs，
        //   而且要在主网络 backward 之后、不改主网络 —— 见函数说明）
        measureEntropyGradNorm(nn::softmax(
            m_net.forward(toInput(m_steps[static_cast<std::size_t>(T - 1)].obs.data()))),
            1.0 / static_cast<double>(m_policySteps));

        m_net.scaleGrad(1.0 / static_cast<double>(m_policySteps));   // ← 第 3 步：平均
        stepWithDiagnostics();                           // ← 第 4 步（含裁剪与诊断）

        // ★ 给面板/probe 看的：这一局有多少步真的参与了学习
        lastPolicySteps      = m_policySteps;
        lastExplorationSteps = T - m_policySteps;
        m_lastReturn = gSum;
    }

    // =======================================================================
    //  ★★ 学习路径 B：Actor-Critic + GAE（2026-09-26 新增）
    //
    //  两阶段，**顺序不能反**：
    //      ① 先算 GAE（需要 V 对【每个时刻】的预测）
    //      ② 用 GAE 更新策略网络；同时用 MSE 更新价值网络
    //
    //  ★ 为什么 advantage 必须来自 V 而不是 G − 标量
    //
    //      adv_t = A_t = "这一步比【这个状态下】的平均水平好多少"
    //  相同状态不会再互相抵消 —— 这是本改动唯一的目的。
    //
    //  ★ γ 的有效视野（必须知道）
    //      γ = 0.99、60Hz → γ^100 ≈ 0.37，即**有效视野约 100 帧 ≈ 1.7 秒**。
    //      所以"为了 20 秒后吃到食物而放弃眼前的东西"在数学上本来就传不回去。
    //      这是常数 γ 的性质，不是本函数的 bug；负责人已决定暂不动 γ。
    // =======================================================================
    void updateCritic(int T) {
        const std::size_t n = static_cast<std::size_t>(T);

        // ---- ① 用【当前】价值网络算出每个时刻的 v̂ ----
        //
        //  ⚠️ 必须在这里一次性算完并存下来：
        //     网络的缓存只保存最后一次前向，而 GAE 是倒序递推，
        //     中途还要用 v̂_{t+1} —— 不存下来就取不到。
        m_vHat.assign(n, 0.0);
        for (int t = 0; t < T; ++t) {
            const auto v = m_vnet.forward(toInput(m_steps[static_cast<std::size_t>(t)].obs.data()));
            m_vHat[static_cast<std::size_t>(t)] = v.a[0];
        }

        // ---- ② GAE 倒序递推 ----
        //
        //      δ_t = r_t + γ·v̂_{t+1} − v̂_t        （v̂_T = 0：局末没有后续价值）
        //      A_t = δ_t + γλ·A_{t+1}              （A_T = 0）
        //
        //  ★ v̂_T = 0 的含义（这一条很重要）
        //      本项目一局【死亡即终止】，所以终止态的价值确实是 0。
        //      这让"撞死"被正确地当成**终止**（后面什么都没有），
        //      而不是"扣分事件" —— 这正是原来的标量基线表达不出来的东西。
        //
        //  ★ 这里【不做**自举**】：maxTicks 到点结束也算终止（不是截断）。
        //      严格说应该 bootstrap V(s_T)（因为世界还在，只是这局结束了），
        //      但那需要 EpisodeRunner 把末态观测传进来。当前先按终止处理，
        //      与本项目"回合制"的语义一致；登记为已知简化。
        m_adv.assign(n, 0.0);
        m_vTarget.assign(n, 0.0);
        double nextAdv = 0.0;
        for (int t = T - 1; t >= 0; --t) {
            const std::size_t k = static_cast<std::size_t>(t);
            const double vNext = (t + 1 < T) ? m_vHat[k + 1] : 0.0;
            const double delta = m_alignedR[k] + gamma * vNext - m_vHat[k];
            nextAdv = delta + gamma * gaeLambda * nextAdv;
            m_adv[k]     = nextAdv;
            m_vTarget[k] = nextAdv + m_vHat[k];    // target = A_t + v̂_t
        }

        // ---- ③ advantage 标准化（可选，默认开）----
        //
        //  ★ 为什么要做：让 advantage 的**量级**与"整局回报"脱钩。
        //    实测单局回报 −167 ~ +499，同一套 lr 在不同局里的实际步长能差 30 倍 ——
        //    这正是"换个种子就锁死"的直接来源。
        //    标准化后每批都是零均值单位方差，步长只由 lr 决定。
        //
        //  ⚠️ 标准化**不改变梯度的方向**，只改尺度 —— 所以它不影响"学到什么"，
        //     只影响"走多大步"。这也是它安全的原因。
        {
            double mean = 0.0;
            for (int t = 0; t < T; ++t) mean += m_adv[static_cast<std::size_t>(t)];
            mean /= static_cast<double>(T);
            double var = 0.0;
            for (int t = 0; t < T; ++t) {
                const double d = m_adv[static_cast<std::size_t>(t)] - mean;
                var += d * d;
            }
            var /= static_cast<double>(T);
            const double sd = std::sqrt(var);
            lastAdvMean = mean;
            lastAdvStd  = sd;
            if (normalizeAdv && sd > 1.0e-8) {
                for (int t = 0; t < T; ++t) {
                    m_adv[static_cast<std::size_t>(t)] =
                        (m_adv[static_cast<std::size_t>(t)] - mean) / sd;
                }
            }
        }

        // ---- ④ 更新价值网络（MSE 回归到 vTarget）----
        //
        //  ★ 梯度推导：L = (1/T)·Σ (v̂_t − target_t)²
        //      ∂L/∂v̂_t = 2(v̂_t − target_t)/T
        //    MLP 的输出层是线性的（a_out == z_out），所以这个梯度
        //    直接就是 backward() 要的 dLogits。
        //
        //  ★ 为什么要 /T：backward() 是【累加】语义，累加 T 个之后
        //    不除 T 就等于把学习率偷偷放大 T 倍（3600 倍会很壮观）。
        //    这和策略网络那边 scaleGrad(1/T) 是同一件事，只是这里
        //    直接把系数写进每个时刻的梯度里 —— 因为输出只有 1 维，
        //    放在循环里更省一次遍历。
        //
        //  ★★★ 2026-09-26：改成【每局多个 epoch】
        //
        //  ★ 为什么要改 —— 实测判据
        //    负责人报 `V-target corr ≈ 0.3`，也就是 V 只解释了回报方差的
        //    **R² = 9%** —— **critic 基本没学会**，而 GAE 建立在它上面，
        //    所以 advantage 里 91% 仍是噪声。
        //
        //  ★ 为什么会这样（算一下就知道）
        //    回报的标准差是 44.5 → MSE 量级在几百。
        //    要让 V 学会预测它，参数得移动相当大的距离；
        //    而旧代码**每局只走一步** `step()`，且步长被 ÷T 压掉。
        //    **每局 1 次梯度更新，喂 3000 个样本 —— 学不动是必然的。**
        //
        //  ★ 修法：同一个 episode 的数据上【多跑几个 epoch】
        //    这是 PPO / A2C 的标准做法（"critic 多训几轮"）。
        //    数据已经存在 m_steps / m_vTarget 里，多跑 epoch **不增加采样成本**，
        //    只是多几次前向+反向 —— 相对感知（占每帧 51~69%）可以忽略。
        //
        //  ⚠️ 注意：这里**只有 critic 多跑**，actor 仍然每局只更新一次。
        //    让 actor 多跑需要重要性采样（行为策略与目标策略不同了），
        //    那是另一个量级的改动，不在本次范围。
        const int vEpochs = (valueEpochs > 0) ? valueEpochs : 1;
        double vLoss = 0.0;
        for (int ep = 0; ep < vEpochs; ++ep) {
            m_vnet.zeroGrad();
            double epLoss = 0.0;
            for (int t = T - 1; t >= 0; --t) {
                const std::size_t k = static_cast<std::size_t>(t);
                const auto v = m_vnet.forward(toInput(m_steps[k].obs.data()));   // 重跑前向
                const double err = v.a[0] - m_vTarget[k];
                epLoss += err * err;

                nn::vec<double, 1> g;
                g.a[0] = 2.0 * err / static_cast<double>(T);
                m_vnet.backward(g);
            }
            if (ep == vEpochs - 1) vLoss = epLoss;   // 报最后一个 epoch 的 loss
            m_vnet.step(valueLr, 0.0);
        }
        lastValueLoss = vLoss / static_cast<double>(T);

        // ---- ⑤ 用 GAE 更新策略网络（结构同 Baseline，只有 adv 来源不同）----
        //
        // ★★★ 2026-09-27：同样排除 ε-greedy 的随机步 —— 理由与 `updateBaseline`
        //     里那一段逐字相同（用策略自己的 probs 去给均匀抽样出来的动作加权，
        //     等于假设重要性比为 1.0）。完整说明见 `Step::fromExploration`。
        //     两条学习路径必须【同时】修，否则换一下 --learner 就复现了。
        m_net.zeroGrad();
        nn::vec<double, ACT_SIZE> lastProbs{};
        int policySteps = 0;
        for (int t = T - 1; t >= 0; --t) {
            const std::size_t k = static_cast<std::size_t>(t);
            const auto& logits = m_net.forward(toInput(m_steps[k].obs.data()));
            const auto probs = nn::softmax(logits);
            if (t == T - 1) lastProbs = probs;   // ★ 诊断：留最后一个时刻的分布

            if (m_steps[k].fromExploration) continue;   // ★ 随机步不进梯度
            ++policySteps;

            auto g = nn::softmaxCrossEntropyGrad(probs, m_steps[k].action);
            g = g * m_adv[k];
            addEntropyGrad(g, probs);
            m_net.backward(g);
        }

        // ★ 一步策略步都没有 → 梯度为空。绝不能走到 scaleGrad(1/0)（会写出 inf，
        //   下一次 step() 就把网络变成 nan，而现象只是"突然学废了"，极难查）。
        if (policySteps <= 0) {
            lastPolicySteps      = 0;
            lastExplorationSteps = T;
            double Gr = 0.0;                       // 统计口径仍与 Baseline 一致
            for (int t = T - 1; t >= 0; --t) {
                Gr = m_alignedR[static_cast<std::size_t>(t)] + gamma * Gr;
            }
            m_lastReturn = Gr;
            return;
        }

        measureEntropyGradNorm(lastProbs, 1.0 / static_cast<double>(policySteps));
        m_net.scaleGrad(1.0 / static_cast<double>(policySteps));
        stepWithDiagnostics();

        lastPolicySteps      = policySteps;
        lastExplorationSteps = T - policySteps;

        // ---- ⑥ 诊断：V 到底有没有学起来 ----
        //
        //  ★ 这是本改动**唯一**的"要不要继续"判据：
        //    valueCorr 持续接近 0 = critic 没学到 → GAE 是空的 →
        //    应该停下来查原因，而不是继续往上堆别的机制。
        lastValueCorr = correlation(m_vHat, m_vTarget);

        // 统计用：本局折扣回报（与 Baseline 路径口径一致，便于对比曲线）
        double G = 0.0;
        for (int t = T - 1; t >= 0; --t) {
            G = m_alignedR[static_cast<std::size_t>(t)] + gamma * G;
        }
        m_lastReturn = G;
    }

    // 熵正则项（两条学习路径共用 —— 只有一处实现）
    //
    //  熵 H = −Σ pᵢ·log pᵢ。我们要【最大化】H，等价于对 −H 做梯度下降。
    //
    //  ★★ 2026-09-26 深夜：符号修正（这是一个真 bug，见下面的推导）
    //
    //  设 L = −H（我们要最小化 L，也就是最大化 H）。逐元素求导：
    //
    //      H = −Σⱼ pⱼ log pⱼ
    //      ∂H/∂zᵢ = −pᵢ·(log pᵢ + 1)          ← 已知结果
    //      ∂L/∂zᵢ = ∂(−H)/∂zᵢ = **+pᵢ·(log pᵢ + 1)**
    //
    //  而我们送进 backward() 的向量会被当作 ∂L/∂z 做【梯度下降】
    //  （step 里是 `w -= lr·g`）。所以这里应该加的是 **+pᵢ(log pᵢ + 1)**。
    //
    //  ⚠️ 修正前的代码加的是 `entropyCoef * (−p) * (log p + 1)` —— **符号反了**，
    //     于是 `entropyCoef = 0.1` 这一项实际在【最小化】熵 =
    //     **主动把策略推向确定化**，是一个正反馈的自毁项。
    //
    //  ★ 数值自检（改完可以拿它验证）
    //     6 个动作均匀时 p = 1/6，log p + 1 = 1 − log 6 = −0.7918，
    //     正确项 = +pᵢ(log pᵢ+1) = −0.1320（负），
    //     修正前   = −pᵢ(log pᵢ+1) = +0.1320（正）→ 方向完全相反。
    //
    //  ⚠️ 别指望它单独阻止坍缩：策略接近确定时 p→0，这一项也→0。
    //     真正的探索保险是 ε-greedy（见 EpisodeRunner）。
    void addEntropyGrad(nn::vec<double, ACT_SIZE>& g, const nn::vec<double, ACT_SIZE>& probs) const {
        for (int i = 0; i < ACT_SIZE; ++i) {
            const double p = probs.a[i];
            if (p > 1.0e-12) {
                g.a[i] += entropyCoef * p * (std::log(p) + 1.0);
            }
        }
    }

    // 策略网络第 4 步 + 步长诊断（两条路径共用）
    void stepWithDiagnostics() {
        // ★ 先记下【裁剪前】的梯度范数，再决定裁不裁。
        //   顺序很重要：先量后裁，量到的才是"这一步本来想走多远"；
        //   裁完再量就永远等于上限，看不出异常局的存在。
        lastGradNorm  = m_net.gradNorm();
        lastParamNorm = m_net.paramNorm();
        lastClipped   = m_net.clipGradNorm(gradClipNorm);
        lastUpdateNorm = lr * (lastClipped ? gradClipNorm : lastGradNorm);
        m_net.step(lr, 0.0);
    }

    // 两个序列的皮尔逊相关系数（诊断用；长度不等或方差为 0 时返回 0）
    static double correlation(const std::vector<double>& a, const std::vector<double>& b) {
        const std::size_t n = std::min(a.size(), b.size());
        if (n < 2) return 0.0;
        double ma = 0.0, mb = 0.0;
        for (std::size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
        ma /= static_cast<double>(n);
        mb /= static_cast<double>(n);
        double sab = 0.0, saa = 0.0, sbb = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double da = a[i] - ma, db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        if (saa <= 0.0 || sbb <= 0.0) return 0.0;
        return sab / std::sqrt(saa * sbb);
    }

    nn::MLP<double, NN_INPUT_SIZE, ACT_SIZE, 1, HIDDEN_WIDTH, nn::tanhAct, nn::dtanhAct> m_net;

    // ★ 价值网络 V(s) —— 只有 Critic 模式会用到它（Baseline 模式下它一直是初始值）
    ValueNet m_vnet;

    std::vector<Step>   m_steps;
    std::vector<double> m_rewards;

    // ---- 每局的临时缓冲（避免每局重新分配；都是成员而不是局部变量）----
    std::vector<double> m_alignedR;    // 对齐到 T 个的奖励
    std::vector<double> m_vHat;        // 每个时刻的 V 预测
    std::vector<double> m_adv;         // GAE advantage
    std::vector<double> m_vTarget;     // 价值网络的目标 = A + v̂

    double m_baseline    = 0.0;
    double m_lastReturn  = 0.0;
    double m_lastEpisodeReward = 0.0;
    long   m_episodes    = 0;

    // ★ 距上次写快照过了几局（自对弈用）
    int    m_episodesSinceMirror = 0;
};

}  // namespace rl
