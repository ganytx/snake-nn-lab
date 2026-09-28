#pragma once

// ============================================================================
//  encoder.h — 观测编码：把 Perception 压成神经网络的输入向量
//
//  ★★ 这个文件唯一的使命：让【训练】和【游戏】用同一份编码 ★★
//
//  如果训练时用一套编码、游戏里用另一套（哪怕只差一个归一化系数），
//  训练出来的权重接到游戏里就是废的 —— 而且现象很迷惑：
//  网络能跑、不报错，只是行为完全不对。
//
//  ⚠️ 这个警告不是假想的：2026-09-22 发现 train_probe 里有一份手写的
//     拷贝，和本文件"看起来一样"，实测 240 万次比对里最大差异 0.96 ——
//     原因是扇区边界格的浮点末位不同。见 devlogs/2026-09-22.md。
//     现在全项目只有这一份实现。
//
//  ★★★ 29 维的构成（2026-09-28 回滚定稿；33 维那次是净负结果，见下）：
//      [ 0.. 7]  8 个方向的障碍射线距离（墙|敌方身体|敌方头|自己；**不含队友**）
//      [ 8..15]  8 个扇区里最近食物的距离（/800，无食物 = 1）
//      [16]      速度 / 450
//      [17]      长度 / 60（截断到 1）
//      [18..19]  朝向的 sin / cos
//      [20..22]  当前活跃圈：sin(方位) / cos(方位) / 是否轮到它供食（恒 1）
//      [23..25]  下一个活跃圈：sin(方位) / cos(方位) / 是否轮到它供食（恒 0）
//      [26..28]  队友：最近队友的前向偏移 / 右向偏移 / 队友数量
//
//  ★ 历史（为什么来回改过一次）
//      2026-09-26：29 → 33 维，改了两件事 ——
//        ① 圈的第 3 维从常量换成真正的**距离**
//        ② 障碍拆成"墙 4 方向 + 敌身 8 方向"两个独立通道
//      两件事本身都成立，但 5 种子对照是**净负**（每局吃 3.68 → 2.33、
//      锁死 1/5 → 2/5），所以 2026-09-28 回滚。
//      ★ 回滚时【特意保留】了两个 33 维时代修的真 bug（见下文注释）：
//        · 队友不算障碍（掩码用 BLOCK_MASK_NO_ALLY）
//        · 不可见的圈按"没有圈"编码（修掉"假圈"给错误信号）
//        它们不是"33 维的取舍"，而是与真实规则一致性的修正。
//
//  详细取舍与实测数据见 docs/90 第 4.11 / 8 节。
//
//  ★ 圈【不按数量写死】—— 只给"当前"和"下一个"，与世界里到底有几个圈无关。
//  ★ 队友**不算**墙（队友之间可以穿过），所以它们不在障碍射线里。
// ============================================================================

#include "../brain/Perception.h"
#include "nn/network.h"

#include <algorithm>
#include <array>     // ★ FrameStack 的历史缓冲
#include <cmath>
#include <cstdlib>

namespace rl {

// ---------------------------------------------------------------------------
//  ★ 观测布局：维度算式集中在这一处
//
//  ⚠️ 这里曾经因为"用一个常量当两个概念"而写出**越界 96 字节**的事故：
//     观测尾部只有 12 个槽（放得下 4 个圈），循环上界却用了 MAX_ZONES(=8)，
//     于是每次感知都往调用方栈上踩 96 字节。
//     详见 devlogs/2026-09-23.md 第 2 节。
//
//  所以现在**把每一段的名字和算式都写出来**，并用 static_assert 钉死，
//  新增字段时不可能悄悄写越界。
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  ★★★ 观测空间：33 维 →【29 维】回滚（2026-09-28，负责人决定）
//
//  ★ 为什么回滚
//
//    devlogs/2026-09-26 第 5 节记录了一次 5 种子对照，**29 维平均更好**：
//
//        种子       29 维 末500局吃    33 维
//        11111      0.23（锁死）       3.57
//        20260921   **7.13**          1.07
//        22222      2.13             **6.56**
//        33333      **5.23**          0.21（锁死）
//        44444      3.70              0.21（锁死）
//        ────────────────────────────────────────
//        平均       **3.68 / 锁死1/5**  2.33 / 锁死2/5
//
//    即 29 维每局多吃 58%，锁死种子更少。33 维是**净负结果**。
//
//  ⚠️ 必须如实说明的两点：
//    ① 那次对照是在**旧代码状态**下做的 —— 当时 `foodScoreUnit` 还错着
//       （吃一个只给 +1.0 而非 +3.0）、熵项符号还是反的、ε-greedy 偏差还在。
//       那些 bug 现在都已修，所以那份数字**不能直接外推**到当前代码。
//    ② 回滚后**所有 33 维存档作废**（包括 `略显智能.txt` 等 8 份）——
//       这是改 encoder 的必然代价，项目铁律。
//
//  ★ 29 维布局（沿用旧版，便于与历史数据对照）
//
//      [ 0.. 7]  8 个方向的障碍距离（掩码 = 墙|敌身|敌头|自己，不含队友）
//      [ 8..15]  8 个扇区最近食物距离（/800，无食物 = 1）
//      [16]      速度 / 450
//      [17]      长度 / 60
//      [18..19]  朝向 sin / cos
//      [20..22]  当前圈：sin(方位) / cos(方位) / 是否轮到它供食（恒 1）
//      [23..25]  下一个圈：sin(方位) / cos(方位) / 是否轮到它供食（恒 0）
//      [26..28]  队友：最近队友的前向偏移 / 右向偏移 / 队友数量
//
//  ★ 相对 33 维，放弃了什么（诚实登记）
//    · 敌身不再有独立通道 —— 又和墙混在同一个掩码里
//    · 圈的"距离"退回常量（当前圈恒 1、下一个圈恒 0）
//    · 墙射线从 4 方向回到 8 方向（信息更细，但和敌身混在一起）
//    · **动作空间 6 → 3**：去掉"加速"（见下方 ACT_SIZE）
//
//  ★ 为什么下面的常量名字保持不变（重要）
//    为了把改动面压到最小：`SELF_OBS_END` / `ZONE_VIEW_SLOTS` / `ZONE_OBS_DIM` /
//    `ALLY_OBS_DIM` / `ZONE_BEGIN` / `ALLY_OBS_BEGIN` 这些名字**全部沿用**，
//    只是数值变了。这样 `EpisodeRunner` / `train_probe` / 面板代码里的
//    `rl::ZONE_BEGIN` 之类的引用**一行都不用改**。
//    新增了两个旧版没有的概念名（`OBSTACLE_DIM` / `FOOD_BEGIN`），
//    它们只在本文件内部使用。
// ---------------------------------------------------------------------------
constexpr int OBSTACLE_DIM    = 8;    // 障碍射线方向数（墙|敌身|敌头|自己）
constexpr int FOOD_SECTOR_DIM = 8;    // 食物扇区
constexpr int SELF_SCALAR_DIM = 4;    // 速度 / 长度 / 朝向 sin / 朝向 cos
constexpr int ZONE_OBS_DIM    = 3;    // 每个圈：sin / cos / 是否轮到它供食
constexpr int ALLY_OBS_DIM    = 3;    // 队友：前向偏移 / 右向偏移 / 数量

// 各块的起始下标（集中在这里算，函数里只用常量，避免手写下标算错）
constexpr int FOOD_BEGIN       = OBSTACLE_DIM;                          // = 8
constexpr int SELF_BEGIN       = FOOD_BEGIN + FOOD_SECTOR_DIM;          // = 16
constexpr int SELF_OBS_END     = SELF_BEGIN + SELF_SCALAR_DIM;          // = 20
constexpr int ZONE_BEGIN       = SELF_OBS_END;                          // = 20

constexpr int ZONE_VIEW_SLOTS = 2;   // 观测里放几个圈：当前 + 下一个

constexpr int OBS_SIZE = SELF_OBS_END
                       + ZONE_VIEW_SLOTS * ZONE_OBS_DIM
                       + ALLY_OBS_DIM;                    // = 29

constexpr int ALLY_OBS_BEGIN = OBS_SIZE - ALLY_OBS_DIM;   // = 26

static_assert(ALLY_OBS_BEGIN == ZONE_BEGIN + ZONE_VIEW_SLOTS * ZONE_OBS_DIM,
              "队友块必须紧跟在食物圈块后面，中间不能有空洞");

// ★ 维度账的硬校验：任何一块改了而这里忘了改，编译期就会炸。
//   （这个项目因为"观测尾部只有 12 槽、循环却按 8 个圈写"越界过 96 字节，
//     所以每一段的算式都要能被 static_assert 钉住。）
static_assert(OBS_SIZE == OBSTACLE_DIM + FOOD_SECTOR_DIM + SELF_SCALAR_DIM
                        + ZONE_VIEW_SLOTS * ZONE_OBS_DIM + ALLY_OBS_DIM,
              "观测总维度与各分块之和对不上");
static_assert(OBS_SIZE == 29, "回滚后必须是 29 维 —— 改了这里就要同步所有文档");

// ---------------------------------------------------------------------------
//  ★★ 食物圈：给【当前圈 + 下一个圈】，不是"前 N 个圈"
//
//  ★ 为什么这样设计
//
//    旧做法是把"前 4 个圈"的方位写进观测 —— 那是一个**写死的数量假设**：
//    圈数不是 4 的时候（1 个 / 6 个 / 0 个）它就不对了。
//    而实际配置里 zone_count 可以是 0~8。
//
//    改成"当前圈 + 下一个圈"之后，**观测与圈数彻底解耦**：
//    不管世界里有几个圈，蛇看到的永远是
//        "现在该去哪"（当前圈的方位）+ "接下来去哪"（下一个圈的方位）。
//
//  ★ 为什么"下一个"也给
//    它让蛇能学【提前出发】：本轮快结束（甚至空窗期）时先往下一个圈挪，
//    而不是等当前圈关门才开始跑。这是轮换玩法想要的预判行为。
//
//  ★ 每圈 3 维的含义（2026-09-26 修正）
//        sin(相对方位) / cos(相对方位) / min(1, 距离像素 / 2000)
//
//    ⚠️ 第 3 维**原来是常量**（当前圈恒 1、下一个圈恒 0），
//       只表示"在哪个槽位"，白白占着一维而没有信息。
//       现在换成了真正的距离 —— 这是本次重设计的两件事之一。
//
//    ★ 距离用 min(1, 像素/2000) 归一化：
//      2000px 覆盖 2000x2000 地图的对角量级；地图更大时会饱和到 1，
//      但那时"反正很远"这个信息已经够用了。
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
//  ★★★ 动作空间：3 → 6（2026-09-26 负责人决定：把"加速"加进动作空间）
//
//  ★ 为什么必须加 —— 加速是一个**真实且重要**的决策
//
//    在此之前 `NeuralBrain` 硬编码 `action.boost = false`，
//    也就是说**神经网络永远不加速**，而加速是 250 → 450 px/s（1.8 倍）。
//    它白扔了 1.8 倍速度。
//
//  ★ 加速不是纯赚，是一个权衡（⚠️ 这一点必须记住）
//
//    `Snake::update()`：`m_timeSinceEat += dt * (isBoosting ? 1.5f : 1.0f)`
//
//        速度        250 → 450 px/s（1.8 倍）
//        饿死窗口      20 → 13.3 秒（×2/3）
//        **每像素饥饿成本 ×1.5**
//
//    净收益：加速让你在饿死前多走 20% 的路（6000px vs 5000px），
//    但每像素贵 50%。所以"该不该加速"是一个**真正的策略问题** ——
//    正是应该交给网络去学的。
//
//    ⚠️ 而且它还有第二层影响：`hungerLeft` 同时决定 ⑤「吃饱余韵」的窗口，
//       所以加速也会**缩短那笔占回报约 60% 的正收益的持续时间**。
//
//  ★ 编码方式：action = steerIndex + 3 * boostFlag
//
//        0 1 2  = 左转 / 直行 / 右转      （普通速度）
//        3 4 5  = 左转 / 直行 / 右转      （加速）
//
//  ★⚠️ 已知风险（必须记住）
//    项目实测过"**观测**维度从 35 加到 39 会让策略坍缩"。
//    动作维度翻倍的风险**未知** —— 输出层从 3 变 6，
//    每个动作的采样概率被摊薄，探索需要更多样本。
//    如果训练明显变差，**第一个该怀疑的就是这里**，
//    回退方法：把 ACT_SIZE 改回 3 并让 steerOfAction 忽略 boost 位。
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  ★★★ 动作空间：6 →【3】回滚（2026-09-28，与观测回滚同一次改动）
//
//  ★ 为什么一起回滚
//    29 维那份代码的 `ACT_SIZE = 3`（左 / 直 / 右，**没有加速**）。
//    要让"29 维的观测 + 现在的代码"自洽，动作空间必须跟着回到 3 ——
//    否则网络要同时学"8 方向的旧观测"和"6 个带加速的动作"，
//    那就是一次动两个变量，出了结果无法归因（本项目反复踩过）。
//
//  ★ 保留 boost 接口，但恒为 false（重要）
//    `boostOfAction()` 仍然存在、仍然被 `NeuralBrain` / `EpisodeRunner` 调用，
//    只是**永远返回 false**。这样做的好处：
//      · 调用的地方一行都不用改（减少改动面）
//      · 如果以后想再试加速，只要把这个常量改回 6、恢复 boostOfAction 的判别式
//    代价：神经网络蛇**永远不加速**（这正是 29 维时代的行为）。
//
//  ⚠️ 加速进动作空间的收益（实测"加速占比 43%→67%"）会一并失去 ——
//    这是回滚的已知代价，如实登记。
// ---------------------------------------------------------------------------
constexpr int ACT_STEER_COUNT = 3;   // 转向档位数（左 / 直 / 右）
constexpr int ACT_SIZE        = 3;   // ★ 回滚到 3 —— 没有加速档

// 动作下标 → 转向档位（0=左 1=直 2=右）
//   ★ ACT_SIZE == ACT_STEER_COUNT 时 action 本身就是档位下标，
//     `% ACT_STEER_COUNT` 是恒等变换，保留它是为了"改回 6 时不用动这里"。
inline int steerIndexOfAction(int action) noexcept {
    return (action >= 0 && action < ACT_SIZE) ? (action % ACT_STEER_COUNT) : 1;
}

// 动作下标 → 是否加速
//   ★ 3 动作空间下恒为 false（没有加速档）。写成"下标落在转向档之外"这个
//     判别式本身仍然正确，所以改回 6 时**不需要动这一行**。
inline bool boostOfAction(int action) noexcept {
    return (action >= 0 && action < ACT_SIZE) && (action >= ACT_STEER_COUNT);
}

// 转向档位 + 加速 → 动作下标（唯一的方向）
//   ⚠️ 3 动作空间下 boost 参数被忽略（没有对应的档位）。
//      调用方传 true 也不会越界，只是行为等于不加速。
inline int actionOf(int steerIndex, bool /*boost*/) noexcept {
    const int s = (steerIndex < 0) ? 0 : ((steerIndex >= ACT_STEER_COUNT)
                                          ? ACT_STEER_COUNT - 1 : steerIndex);
    return s;
}


// ---------------------------------------------------------------------------
//  ★★ 帧堆叠（frame stacking）—— 给纯 MLP 补上"记忆"
//
//  ★ 为什么需要它（这是当前最大的能力缺口）
//
//    策略网络是纯 MLP：**没有隐藏状态、没有历史、没有时钟**。
//    它每一步只看当前这一帧，看完就忘。于是它结构上【表示不了】任何
//    需要时序的行为 —— "记住刚才从哪来"、"预判对手下一步"、"先占位再收口"。
//
//    实测表现就是：能学会反应式的行为（朝食物走、绕圈找食），
//    但一上"合作 / 包围 / 用饥饿淘汰对手"就卡住 ——
//    给多少奖励项都没用，因为网络【表达不出来】。
//
//  ★ 做法
//    把最近 K 帧的观测首尾拼起来当输入。网络于是能看见"变化趋势"
//    （"距离在缩小"、"它正在朝我这边转"），这是时序行为的最小前提。
//
//  ★★ 安全开关：stackFrames = 1 时【逐位等价于旧行为】
//
//    K=1 时拼出来的向量就是当前帧本身，长度 = OBS_SIZE，
//    所以"退回旧版本" = 改这一个常量 + 重新编译，**不需要改任何逻辑**。
//    这是刻意设计的：帧堆叠有触发策略坍缩的风险（见下方教训），
//    所以必须能一键回到已验证好的状态。
//
//  ⚠️⚠️ 实测结论（2026-09-24）：**帧堆叠在这个项目上是负收益，K 越大越差**
//
//    任务：食物圈·位置随机（最难的、也是最诚实的能力判据）
//    同为 1500 局，取训练后半程 750 局的每局平均：
//
//        K=1（无记忆）  总回报 112.3   每局吃 5.87   存活 2248   死亡率 68%
//        K=2            总回报  70.2   每局吃 1.69   存活 1535   死亡率 92%
//        K=3            总回报  94.5   每局吃 3.14   存活 1983   死亡率 82%
//
//    也就是说：**加记忆反而全面变差**，而且 K=2 比 K=3 更差（不是单调的，
//    说明主因不是"信息不够"，而是【输入维度变大】本身在伤害学习）。
//
//    这和本项目另一条实测教训完全一致：观测从 35 加到 39 维就让策略锁死过。
//    Xavier 初始化按 1/fan_in 缩放，输入维度一变，初始 logits 的尺度就变，
//    而当前奖励下（没有存活奖励、每局必吃 −10）策略本来就脆弱。
//
//  ★ 所以这里保持 K=1，并且**框架保留**（FrameStack 是完整可用的）：
//    以后如果换了优化器/降低了梯度方差，值得重新试一次。
//    改这一个常量即可，不需要改任何逻辑。
//
//  ★ 那么"时序行为"这个能力缺口怎么办
//    本轮实验说明：**瓶颈不在观测，在优化过程**（单轨迹 REINFORCE 方差太大）。
//    下一步该优先做的不是结构改动，而是降方差 / 并行环境拿更多样本。
//    详见 docs/90 的相关章节。
// ---------------------------------------------------------------------------
constexpr int STACK_FRAMES   = 1;                        // 堆几帧（1 = 关闭）
constexpr int NN_INPUT_SIZE  = OBS_SIZE * STACK_FRAMES;  // 网络真正的输入维度

static_assert(STACK_FRAMES >= 1, "至少要堆 1 帧");

// 策略网络的隐藏层宽度。
// ★ 放在这里（而不是散在 Policy / Trainer 各处）是为了让
//   "输入维度 / 动作数 / 隐藏层宽度" 三件事只有一处定义 ——
//   否则改了维度而忘了改某一处，就会出现"编码 87 维喂给 29 维网络"。
constexpr int HIDDEN_WIDTH = 32;

// ---------------------------------------------------------------------------
//  ★ 队友信息的归一化常数
//
//  ALLY_POS_NORM：队友的"前向/右向偏移"除以多少变成 [-1,1]
//     取 1200px —— 沿用 docs/70 已定稿的"地图与导航类观测按 1200px 归一化"
//     的约定（食物扇区用的是 800px、圈距离用的是 2000px，各有各的依据）。
//     这里取 1200 的理由：2000² 地图的对角长约 2800px，而队友是"全局可见"的，
//     1200px 覆盖了"同队两条蛇正常可能拉开的距离"，再远应该给同一个饱和值。
//
//  ALLY_COUNT_NORM：队友数量除以多少变成 [0,1]
//     取 3.0 —— 满编 3 条队友时正好是 1.0。本项目的队伍规模上限就是 3。
// ---------------------------------------------------------------------------
constexpr double ALLY_POS_NORM   = 1200.0;
constexpr double ALLY_COUNT_NORM = 3.0;

// 把一帧观测打成一行（排查"编码到底给网络喂了什么"）
inline void dumpObservation(const double* obs, char* out, int outSize) {
    int n = 0;
    for (int i = 0; i < OBS_SIZE && n < outSize - 8; ++i) {
        n += std::snprintf(out + n, static_cast<std::size_t>(outSize - n),
                           "%.2f ", obs[i]);
    }
    if (n < outSize) out[n] = '\0';
}

// 食物圈最多支持几个（对应 Perception::zones）。
// 布局在 arena::ZoneLayout 里，这里只负责编码，不关心几个圈。
constexpr int MAX_ZONES = 8;

// ★ `ZONE_OBS_SLOTS` 与 `MAX_ZONES` **不是一个概念**，别再合并：
//       MAX_ZONES      = 世界最多能建几个圈（arena::ZoneLayout / GameConfig 的上限）
//       ZONE_OBS_SLOTS = 观测向量里放得下几个圈
//   两个数字曾经共用一个常量，于是"世界建了 4 个圈、观测按 8 个写"，
//   循环写到 obs[32..43] —— 越界 12 个 double（96 字节）。
//   见下面第 4) 段的注释，以及 devlogs 里那次 0xC0000005。
//   （两个常量的定义都在文件顶部，这里只留说明。）

// 策略网络：输入 NN_INPUT_SIZE 维（= 29 x STACK_FRAMES）→ [32] → 3
//
//   STACK_FRAMES = 1 时输入 29 维（旧行为）；= 3 时 87 维（带记忆）。
using Policy = nn::MLP<double, NN_INPUT_SIZE, ACT_SIZE, 1, HIDDEN_WIDTH,
                       nn::tanhAct, nn::dtanhAct>;

// ---------------------------------------------------------------------------
//  ★★ 价值网络 V(s)：Actor-Critic 的 critic（2026-09-26 新增）
//
//  ★ 为什么需要它（这是「越训越差」的正解）
//
//  纯 REINFORCE 的 advantage 是 `G − baseline`，而 baseline 是一个【跨局标量】。
//  它只能减掉"平均回报"，减不掉"**同一个状态在不同局里回报不同**"这件事 ——
//  于是同一个状态在好局里被强化、在坏局里被削弱，**梯度互相抵消**，
//  净信号接近零，剩下的是噪声。实测单局回报从 −167 到 +499（差 30 倍），
//  这个抵消足以让策略被噪声推进"永远同一个动作"的吸引域并且自锁。
//
//  价值网络给的是【按状态】的基线：
//      adv = GAE(s_t) = "这一步比【这个状态下】的平均水平好多少"
//  相同状态不会再互相抵消，方差从"整局运气"降到"TD 误差"。
//
//  ★ 为什么隐藏层宽度复用 HIDDEN_WIDTH（而不是另起一个常数）
//    两个网络吃**同一个观测**、做**同一件事的两种估计**（一个估动作、一个估价值）。
//    宽度写第二份就等于多一个"改了这边忘了那边"的机会 —— 本项目在这类
//    "同一个东西有两份"上栽过两次（观测编码分叉、世界推进两份）。
//    真要分开调时再显式加常量，而不是现在就先埋一个。
//
//  ★ 输出 1 维、线性激活：V(s) 是回归量，可以取任意实数，不加激活。
//    （和策略网络的 logits 一样，激活是"外面"的事。）
// ---------------------------------------------------------------------------
using ValueNet = nn::MLP<double, NN_INPUT_SIZE, 1, 1, HIDDEN_WIDTH,
                         nn::tanhAct, nn::dtanhAct>;

constexpr double PI_D = 3.14159265358979323846;

// ---------------------------------------------------------------------------
//  把网格偏移量换算成食物扇区（0..7）
//
//  扇区定义（由原来那句 sector = int(((atan2(dr,dc)+π)/(2π))*8) % 8 推导而来，
//  行是"前"、列是"右"）：
//
//      sector 0 : ang ∈ [-π, -7π/8)     正前偏左
//      sector 1 : ang ∈ [-7π/8, -3π/4)
//      sector 2 : ang ∈ [-3π/4, -π/2)   右上
//      sector 3 : ang ∈ [-π/2, -π/4)    正前偏右
//      sector 4 : ang ∈ [-π/4, 0)       右下
//      sector 5 : ang ∈ [0, π/4)        正右
//      sector 6 : ang ∈ [π/4, π/2)      左下
//      sector 7 : ang ∈ [π/2, π)        正后 / 左上
//
//  ★★ 为什么不用那句浮点式子？因为一个真实踩过的 bug：
//
//  角平分线（-135°、-45°、45°、135°）上的格子恰好落在扇区边界，
//  浮点式算出的 `t*8` 会是「差 1 个 ulp 的整数」——例如正上方
//  （dr=-16, dc=0）算出来是 1.999999999999998 而不是 2.0，
//  static_cast<int> 截断成 1，格子被判到隔壁扇区。
//
//  更糟的是结果**取决于表达式的书写顺序**：`(ang+P)/(2P)*8` 与 `8*(ang+P)/(2P)`
//  经编译器重结合后末位不同。实测两份"看起来一样"的编码实现，
//  在 240 万次比对中最大差异 0.96，正是这些边界格造成的。
//  见 devlogs/2026-09-22.md。
//
//  所以这里改成**纯整数比较**，边界一律归小的扇区号，
//  结果完全确定：与浮点无关、与编译器无关、与优化等级无关。
// ---------------------------------------------------------------------------
inline int foodSector(int dr, int dc) noexcept {
    if (dc > 0) {                                    // 右半平面
        if (dr < 0) return (-dr > dc) ? 3 : 2;       // 右上；正对角线归 2
        if (dr > 0) return ( dr > dc) ? 4 : 5;       // 右下；正对角线归 5
        return 5;                                    // 正右
    }
    if (dc < 0) {                                    // 左半平面
        if (dr < 0) return (-dr > -dc) ? 1 : 0;      // 左上；正对角线归 0
        if (dr > 0) return ( dr > -dc) ? 6 : 7;      // 左下；正对角线归 7
        return 7;                                    // 正左
    }
    return (dr < 0) ? 0 : 4;                         // dc == 0：正前 / 正后
}

// 角度归一化到 [-π, π]
inline double wrapAngle(double a) noexcept {
    while (a >  PI_D) a -= 2.0 * PI_D;
    while (a < -PI_D) a += 2.0 * PI_D;
    return a;
}

// ---------------------------------------------------------------------------
//  把感知编码成 29 维观测
//  返回：最近食物的世界距离（像素）；视野里没食物时返回一个大数。
//        训练时塑形奖励要用它；游戏里可以忽略。
// ---------------------------------------------------------------------------
inline double encodeObservation(const arena::Perception& p, double* obs) {
    const arena::VisionGrid& g = p.grid;

    // 1) ★ 八个方向的障碍距离 —— 【回滚到单通道】
    //
    //  ★ 为什么退回单通道（这是回滚 29 维的一部分）
    //    33 维版本把这里拆成了"墙 4 方向 + 敌身 8 方向"两个独立通道，
    //    理由是"网络分不清墙和敌人"。那件事本身是对的，但实测净效果为负
    //    （5 种子平均每局吃 3.68 → 2.33，锁死 1/5 → 2/5，见文件头的对照表），
    //    所以负责人决定回滚。
    //
    //  ★ 掩码用 `BLOCK_MASK_NO_ALLY`：**队友不算障碍**。
    //    规则里队友之间可以互相穿过（friendlyFire = false），
    //    把队友当墙等于给网络喂一个**错误的物理模型**。
    //    ⚠️ 这一条是 33 维时代修的**真 bug**，回滚时【特意保留】——
    //       它不是"33 维的取舍"，而是与真实规则一致性的修正。
    //
    //    方向编号 0..7 = 前, 右前, 右, 右后, 后, 左后, 左, 左前（见 VisionGrid 的 dr/dc）。
    for (int d = 0; d < OBSTACLE_DIM; ++d) {
        obs[d] = g.rayDistance(d, arena::VisionGrid::BLOCK_MASK_NO_ALLY);
    }

    // 2) 八个扇区里最近的食物距离
    for (int s = 0; s < FOOD_SECTOR_DIM; ++s) obs[FOOD_BEGIN + s] = 1.0;

    double nearest = 1.0e9;
    for (int r = 0; r < arena::VISION_ROWS; ++r) {
        for (int c = 0; c < arena::VISION_COLS; ++c) {
            const std::uint8_t v = g.cells[r][c];
            if (!(v & (arena::CELL_FOOD | arena::CELL_CORPSE))) continue;

            const double dr = static_cast<double>(r - arena::VISION_CENTER_ROW);
            const double dc = static_cast<double>(c - arena::VISION_CENTER_COL);
            const double distPx = std::sqrt(dr * dr + dc * dc) * p.cellSize;
            if (distPx < nearest) nearest = distPx;

            // 扇区 = 方位角分 8 份。注意 atan2(行, 列) —— 行是"前"，列是"右"
            const int sector = foodSector(r - arena::VISION_CENTER_ROW,
                                          c - arena::VISION_CENTER_COL);

            const double norm = std::min(1.0, distPx / 800.0);
            if (norm < obs[FOOD_BEGIN + sector]) obs[FOOD_BEGIN + sector] = norm;
        }
    }

    // 3) 自身状态
    obs[SELF_BEGIN + 0] = static_cast<double>(p.self.speed) / 450.0;
    obs[SELF_BEGIN + 1] = std::min(1.0, static_cast<double>(p.self.length) / 60.0);
    obs[SELF_BEGIN + 2] = std::sin(static_cast<double>(p.self.heading));
    obs[SELF_BEGIN + 3] = std::cos(static_cast<double>(p.self.heading));

    // 4) ★ 食物圈（当前圈 + 下一个圈，每圈 3 维）
    //
    //    每个圈给：相对自身朝向的 sin/cos(方位角) + 是否轮到它供食。
    //
    //    ★ 为什么方位角要减去自身 heading：
    //      "圈在左前方"永远对应同一组数值，与绝对方位无关 ——
    //      这样网络学到的方向特征可以复用到任何朝向。
    //
    //    ★ 第 3 维【回滚为"是否轮到它供食"】（当前圈恒 1、下一个圈恒 0）
    //      33 维版本把它换成了"到圈心的归一化距离"，理由是"知道往哪走但
    //      不知道多远是导航的硬伤"。那件事本身成立，但实测净效果为负，
    //      所以随观测一起回滚。
    //
    //      ⚠️ 保留这一维（而不是删掉）是刻意的：
    //        它是唯一的"槽位身份"显式信号，而且是**旧版语义** ——
    //        回滚必须逐维对齐旧版，否则 "29 维" 就只是维度数字相同、
    //        语义却不同，历史数据无法对照。
    {
        const int n = static_cast<int>(p.zones.size());

        // ★ 两个槽位：0 = 当前圈，1 = 下一个圈
        //   下标由 Arena/ZoneSchedule 给（−1 = 没有圈），**本函数不重推调度规则**。
        const int idx[2] = { p.activeZoneIndex, p.nextZoneIndex };

        for (int slot = 0; slot < ZONE_VIEW_SLOTS; ++slot) {
            const int base = ZONE_BEGIN + slot * ZONE_OBS_DIM;
            const int zi   = idx[slot];

            // ★★ 【特意保留】不可见的圈 = 当作"没有圈"
            //
            //  ⚠️ 这一条不是 33 维的取舍，而是**修掉"假圈"这个真 bug**：
            //     "下一个圈"如果本轮根本不供食，却在观测里给出方位，
            //     那就是**指向一块没有食物的空地**的假信号 —— 会教坏网络。
            //     所以回滚时【保留】它，见 FoodZoneView::visible。
            const bool zoneOk = (zi >= 0 && zi < n)
                             && p.zones[static_cast<std::size_t>(zi)].visible;

            if (zoneOk) {
                const arena::FoodZoneView& z = p.zones[static_cast<std::size_t>(zi)];

                // 方位角 = 圈心方向 − 自身朝向，再转成 sin/cos。
                const double rel = wrapAngle(
                    std::atan2(static_cast<double>(z.pos.y - p.self.pos.y),
                               static_cast<double>(z.pos.x - p.self.pos.x))
                    - static_cast<double>(p.self.heading));

                obs[base + 0] = std::sin(rel);
                obs[base + 1] = std::cos(rel);
                // 第 3 维（回滚后）：
                //   slot 0（当前圈）恒为 1 —— 它正在供食
                //   slot 1（下一个圈）恒为 0 —— 还没轮到它
                obs[base + 2] = (slot == 0) ? 1.0 : 0.0;
            } else {
                // 没有圈（或这个槽位没有对应的圈）→ 中性哨兵：
                //   方向 0/0 表示"没有方向"，第 3 维 1 表示"不适用"。
                //   ★ 用 (0, 0, 1) 这个组合唯一标识"没有圈" ——
                //     它和"圈在正前方且贴着"（0, 1, 0）不会混淆。
                obs[base + 0] = 0.0;
                obs[base + 1] = 0.0;
                obs[base + 2] = 1.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  5) ★ 队友（最后 3 维）
    //
    //  ★ 为什么必须加这一段
    //
    //    `Perception.allies` 一直都在算（源码注释写着"合作的前提是知道队友在哪"），
    //    但**编码从来没读过它** —— 也就是说网络收到的是一个
    //    "看不见队友"的世界。再加上旧版把队友身体算进障碍距离，
    //    它眼里的队友就是"一堵能撞死我的墙"。
    //
    //    在这种观测下，**"合作"不是难学，而是不可观测**。
    //    详见 docs/85-智能涌现课程设计.md 第 0 节。
    //
    //  ★ 为什么是这 3 维
    //
    //    前向偏移 / 右向偏移：用**自身朝向的旋转坐标**，和圈的 sin/cos(相对方位)
    //    表达的是同一件事 —— "队友在我左前方"永远对应同一组数值。
    //    好处是网络可以对"方向"复用同一组特征，而不必重新学一套绝对方位。
    //
    //    队友数量：让网络区分"孤身一人"和"身边有队友"，
    //    这是"要不要配合"的前提条件。
    //
    //  ★ 为什么【不】把"最近敌人"再单独编码一份
    //
    //    ⚠️ 这段注释 2026-09-28 更新（随 29 维回滚）：障碍射线**又合并回
    //       单通道**了（掩码 = 墙|敌身|敌头|自己），所以"最近的敌人在哪"
    //       确实能从障碍射线里**间接**读出来 —— 33 维时代那句
    //       "那个理由已经不成立"现在不再适用。
    //
    //    但仍然不额外编码"敌人距离/方位"，理由和旧版一致：
    //       8 条障碍射线已经是有方向、有距离的表达，
    //       再补一对 sin/cos 方位 + 距离只是把同一件事写第二遍。
    //       **重复编码不增加可观测性，只增加维度**（而维度越少越容易训）。
    //
    //    ★ 注意区分：`Perception::headToEnemyBodyDist`（用于 rSnakeDanger 扣分）
    //      仍然没有直接进观测 —— 但它现在**可以**从敌身射线近似读出来，
    //      因为危险区半径 100px 远小于视野半径 512px。
    //      这样"被扣分"和"能看见"第一次对上了。
    // -----------------------------------------------------------------------
    {
        // 默认：没有队友 → 中性值（偏移 0、数量 0）
        obs[ALLY_OBS_BEGIN + 0] = 0.0;
        obs[ALLY_OBS_BEGIN + 1] = 0.0;
        obs[ALLY_OBS_BEGIN + 2] = 0.0;

        if (!p.allies.empty()) {
            // allies 已由 PerceptionBuilder 按距离由近到远排好序，取最近的
            const arena::VisibleSnake& a = p.allies.front();

            // ★ 直接用感知里已经算好的 polar 量，【不自己再做旋转】
            //
            //   PerceptionBuilder 里 bearing 的定义是：
            //       bearing = atan2(dy, dx) − heading ∈ [−π, π]
            //               = 对方方位角相对【自己朝向】的偏移
            //   这就是我们要的东西 —— 0 = 正前方，>0 = 偏右手边。
            //
            //   ⚠️ 第一版我自己把 headRel 做了一次坐标旋转，结果**左右搞反了**：
            //       我写的是 (dx·sh − dy·ch)，而正确的是 (−dx·sh + dy·ch)。
            //       原因：世界坐标的 **y 轴向下**，heading=0 时蛇朝 +x，
            //       而"右手边"是 +y（屏幕上往下）。
            //       这种 bug 极其隐蔽 —— 网络照样能学会，只是左右是镜像的。
            //
            //   ★ 教训：**能用上游已经算好且语义明确的量，就不要自己重推一遍。**
            //     这个项目已经因为"同一个东西算两遍"栽过两次了。
            const double d     = static_cast<double>(a.distance);
            const double theta = static_cast<double>(a.bearing);

            const double fwd   = d * std::cos(theta);   // 前向分量（+ = 前方）
            const double right = d * std::sin(theta);   // 右向分量（+ = 右方）

            obs[ALLY_OBS_BEGIN + 0] = std::max(-1.0, std::min(1.0, fwd   / ALLY_POS_NORM));
            obs[ALLY_OBS_BEGIN + 1] = std::max(-1.0, std::min(1.0, right / ALLY_POS_NORM));
        }

        // 队友数量：用【全局可见的队友数】（p.allies），不是"视野内"的。
        //
        // ★ 为什么用全局而不是视野内：合作需要知道"我还有没有队友"，
        //   而这不该受 1024px 视野限制 —— 视野外队友还活着这件事是常识。
        obs[ALLY_OBS_BEGIN + 2] = std::min(
            1.0, static_cast<double>(p.allies.size()) / ALLY_COUNT_NORM);
    }

    return nearest;
}

// 数组 → 网络输入
//
// ⚠️ 现在网络输入的维度是 NN_INPUT_SIZE（= OBS_SIZE × STACK_FRAMES），
//    不是 OBS_SIZE。传进来的必须已经是【拼好的堆叠向量】。
inline nn::vec<double, NN_INPUT_SIZE> toInput(const double* stacked) {
    nn::vec<double, NN_INPUT_SIZE> x;
    for (int i = 0; i < NN_INPUT_SIZE; ++i) x.a[i] = stacked[i];
    return x;
}

// ---------------------------------------------------------------------------
//  ★ 帧堆叠缓冲
//
//  ★ 它属于【调用方】，不属于编码函数
//    编码函数是纯函数（Perception → 一帧观测），它没有状态。
//    而"最近 K 帧"是一段跨帧的状态，必须由调用方持有
//    （训练里是 EpisodeRunner / Trainer，游戏里是 NeuralBrain）。
//
//  ★ 用法
//      FrameStack stack;                 // 每局开始 / 每条蛇一份
//      ...
//      double frame[OBS_SIZE];
//      encodeObservation(p, frame);
//      stack.push(frame, stacked);       // stacked 是 NN_INPUT_SIZE 维
//      net.forward(toInput(stacked));
//
//  ★ 初始状态：前 K−1 帧填【第一帧的拷贝】，而不是填 0
//    填 0 会造出"凭空出现的一段静止史"，而且 0 在观测里往往是个有意义的值
//    （比如 sin(方位)=0 表示正前方）。用第一帧填充，语义是
//    "在此之前世界一直是这个样子"，最自然。
// ---------------------------------------------------------------------------
class FrameStack {
public:
    void reset() { m_filled = 0; }

    // 压入一帧，并写出 NN_INPUT_SIZE 维的堆叠向量
    void push(const double* frame, double* stacked) {
        // 第 0 帧特殊处理：把整个历史都填成它（见上面的说明）
        if (m_filled == 0) {
            for (int k = 0; k < STACK_FRAMES; ++k) {
                for (int j = 0; j < OBS_SIZE; ++j) {
                    m_hist[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] = frame[j];
                }
            }
            m_filled = STACK_FRAMES;
        } else {
            // 整体前移一格，最新帧放最后
            for (int k = 0; k + 1 < STACK_FRAMES; ++k) {
                m_hist[static_cast<std::size_t>(k)] = m_hist[static_cast<std::size_t>(k + 1)];
            }
            for (int j = 0; j < OBS_SIZE; ++j) {
                m_hist[static_cast<std::size_t>(STACK_FRAMES - 1)][static_cast<std::size_t>(j)] = frame[j];
            }
        }

        // 按【从旧到新】的顺序拼出来
        //   K=1 时这里就是"把 frame 拷过去"，与旧行为逐位一致 —— 这是安全开关的关键
        for (int k = 0; k < STACK_FRAMES; ++k) {
            for (int j = 0; j < OBS_SIZE; ++j) {
                stacked[k * OBS_SIZE + j] =
                    m_hist[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
            }
        }
    }

private:
    std::array<std::array<double, OBS_SIZE>, STACK_FRAMES> m_hist{};
    int m_filled = 0;
};

// 只拿堆叠向量、不需要历史时用（例如把一帧当成 K 帧输入做调试）
inline void fillStackedFromSingle(const double* frame, double* stacked) {
    for (int k = 0; k < STACK_FRAMES; ++k) {
        for (int j = 0; j < OBS_SIZE; ++j) {
            stacked[k * OBS_SIZE + j] = frame[j];
        }
    }
}

// 离散动作 → steer
//
// ★ 2026-09-28 回滚后动作空间是 3 个（左/直/右），动作下标 == 转向档位下标，
//   所以 `steerIndexOfAction` 是恒等映射。保留这一层是为了：
//     · 与历史口径一致（29 维时代就是这么写的）
//     · 以后把 ACT_SIZE 改回 6 时，这里**不需要动** ——
//       因为它已经正确地把"动作下标"先归约成"转向档位"再映射。
inline float steerOfAction(int action) {
    return arena::steerOfIndex(steerIndexOfAction(action),
                               arena::ActionSpace::Discrete3);
}

}  // namespace rl
