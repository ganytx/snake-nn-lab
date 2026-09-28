// ============================================================================
//  train_probe.cpp — REINFORCE 最小可行性验证
//
//  只回答一个问题：
//      「20 维扇形观测 + 这套奖励设计 + REINFORCE，到底能不能让蛇学会觅食？」
//
//  ★ 为什么先做这个，而不是先抽 Arena / 搭训练 UI
//
//    "在便宜的实验里失败，比在昂贵的架构里失败好。"
//    如果这里学不动，那多半是【观测编码或奖励设计】的问题，
//    跟用 C++ 还是 Python、有没有 UI 都无关。先把这个问题回答掉。
//
//  ★ 判据
//
//    必须和【随机策略基线】比 —— 同一套世界、同样的局数，
//    只把动作换成均匀随机。没有这个零点，"回报在涨"是没有意义的。
//
//  这个文件是【一次性验证工具】，不参与游戏本体，也不需要图形。
//  运行：build\train_probe.exe
// ============================================================================

#include "entities/Snake.h"
#include "entities/FoodManager.h"
#include "systems/CollisionSystem.h"
#include "agent/Agent.h"
#include "agent/PerceptionBuilder.h"
#include "brain/Action.h"
#include "brain/BrainRegistry.h"
#include "arena/Arena.h"         // ★★ 世界推进的唯一实现（本文件不再自带一份）
#include "rl/encoder.h"          // ★★ 观测编码的唯一实现（本文件不再自带拷贝）
#include "rl/Trainer.h"          // ★★ 梯度更新的唯一实现（游戏内训练也调它）
#include "rl/TrainTask.h"        // ★★ 训练任务的默认参数（游戏内训练用同一份）
#include "rl/EpisodeRunner.h"    // ★★ "一局怎么跑 + 奖励怎么算"的唯一实现
#include "rl/Checkpoint.h"       // ★ ckpt::ensureDir（敌方快照写在 saves/ 下）
#include "nn/network.h"
#include "nn/loss.h"
#include "utils/ConsoleUtf8.h"   // ★ 控制台 UTF-8 输出（否则中文乱码）

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>      // 命令行参数解析要用 std::string
#include <vector>

// ---------------------------------------------------------------------------
//  ★ 这里【故意没有】超参数常量块
//
//  2026-09-24 删除。原先这里有一整个 `namespace cfg`，里面写着
//  MAP_W / FOOD_MAX / MAX_TICKS / GAMMA / LR / R_ALIVE / R_EAT / R_DIE /
//  R_SHAPE / SEED —— 而它们【全部是零引用的僵尸常量】：
//  真正的参数一律来自 rl::makeTaskConfig()（见下面的 main）。
//
//  ⚠️ 它不是一个无害的残留，而是一个【会主动误导人的陷阱】：
//
//      · 常量里写着 R_ALIVE = 0.01 / R_DIE = -1.0 / MAX_TICKS = 1200
//      · 而 TaskConfig 里实际生效的是完全另一组数（2026-09-24 已整体重写）
//
//  任何人打开这个文件都会以为当前奖励是前者，据此去分析"为什么学不会"——
//  方向从一开始就是错的。这正是本项目反复吃亏的那类问题：
//  **现象伪装成逻辑错误时，先查配置**；而这里连配置本身都是假的。
//
//  ★ 纪律：**调参只改 src/rl/TrainTask.h 一处。**
//    本文件不许再定义任何超参数 —— 想临时做实验就用下面 main() 里的
//    命令行开关覆盖 TaskConfig（实验必须在命令行上看得见，不能藏在代码里）。
// ---------------------------------------------------------------------------

// 策略网络【直接用 rl 里那一个定义】，不再在这里另起一份。
// 这样"网络形状"和"编码维度"永远绑在一起 —— 改了 OBS_SIZE，
// 网络的输入层会自动跟着变，不会出现"编码 32 维喂给 20 维网络"这种事故。
using Policy = rl::Policy;
constexpr int OBS = rl::OBS_SIZE;
constexpr int ACT = rl::ACT_SIZE;

using nn::vec;   // 让下面的 vec<double, N> 直接可用

// ★★ 诊断开关：打印每局"开局的世界长什么样"（负责人查"每局都一样"）
//    ⚠️ 必须在 `runner.begin()` 【之后】立刻读，否则读到的是上一局的残留。
bool gDumpEpisodeWorld = false;

// ---------------------------------------------------------------------------
//  离散动作 → steer
//
//  ★ 直接用 rl::steerOfAction —— 动作换算也【只有一份】。
//    这个函数和 NeuralBrain 在游戏里用的是同一个，
//    所以"训练时按 0/1/2 出动作"和"游戏里按 0/1/2 出动作"必然一致。
// ---------------------------------------------------------------------------
double steerOf(int action) {
    return static_cast<double>(rl::steerOfAction(action));
}

// ---------------------------------------------------------------------------
//  训练数据记录
//
//  ★ 为什么要写文件而不是只打控制台
//
//  没有曲线的时候调参全靠猜：只看"第 1900 局回报 13.30"这一个点，
//  根本分不清是【震荡】【收敛】还是【发散】。而这三者的处理方式完全不同。
//
//  输出的 CSV 用【长格式】：每行一个"指标采样点"，列固定。
//
//      kind,episode,value,tag
//      train,1,3.47,reward          ← 每局训练回报
//      train,1,235,survival         ← 每局存活帧数
//      train,1,2,food               ← 每局吃到几个
//      train,1,0.17,baseline        ← 滑动基线
//      eval,1,0.12,reward           ← 训练后贪心评估（只有几行）
//
//  ★ 为什么用长格式而不是宽格式（每局一行的多列表格）：
//    指标会不断增加（以后要加 loss、梯度范数、熵…），宽格式每加一个指标
//    就要改读取方的列号；长格式加一个指标只是多一个 tag 值，
//    **读取方一行都不用改**。这对一个还在长的项目更重要。
//
//  文件位置：训练程序的工作目录（约定是 build\），文件名 train_metrics.csv
//  游戏里从主菜单进「训练数据」就能看到曲线。
// ---------------------------------------------------------------------------
struct MetricsWriter {
    std::FILE* fp = nullptr;

    bool open(const char* path) {
        fp = std::fopen(path, "w");
        if (!fp) return false;
        std::fprintf(fp, "kind,episode,value,tag\n");
        return true;
    }
    void close() {
        if (fp) { std::fclose(fp); fp = nullptr; }
    }
    void row(const char* kind, long episode, double value, const char* tag) {
        if (!fp) return;
        std::fprintf(fp, "%s,%ld,%.6g,%s\n", kind, episode, value, tag);
    }
};

// ---------------------------------------------------------------------------
//  跑一整局
//
//  ★ 本文件【不再自己写"一局怎么跑"】——
//    那是 rl::EpisodeRunner 的事，游戏内的「训练模式」用的是同一份。
//    这里只负责：把 Runner 接到 Arena 的 controller 上、跑到结束、必要时学一次。
// ---------------------------------------------------------------------------
rl::EpisodeStats runEpisode(arena::Arena& arena, rl::EpisodeRunner& runner,
                            rl::Trainer& trainer, std::mt19937& rng,
                            rl::RunMode mode)
{
    const float dt = 1.0f / 60.0f;

    arena.setController([&](std::size_t i, const arena::Perception& p) {
        return runner.decide(i, p, rng);
    });

    runner.begin(mode);

    // ★★ 诊断：世界"开局长什么样"—— 必须在 begin() 【之后】立刻读
    //
    //  ⚠️ 踩过的坑：我一开始在【训练循环里、runEpisode 返回之后】打印，
    //     而 runEpisode 内部会 `runner.begin()` → `m_arena.reset()`，
    //     所以那个位置读到的是**上一局的残留**，看起来像"圈位从不改变"。
    //     **读"开局状态"必须紧跟 begin()。**
    if (gDumpEpisodeWorld) {
        std::printf("  [开局] 种子=%u", arena.worldSeed());
        for (std::size_t k = 0; k < arena.food().zones().size(); ++k) {
            const auto& z = arena.food().zones()[k];
            std::printf("  圈%zu=(%.0f,%.0f)", k, z.x, z.y);
        }
        if (!arena.agents().empty()) {
            const Snake& s0 = arena.agents()[0].body;
            std::printf("  蛇0=(%.0f,%.0f)朝向=%.2f", s0.getX(), s0.getY(), s0.getAngle());
        }
        std::printf("\n");
    }

    while (!runner.done()) runner.tick(dt);

    arena.setController(nullptr);

    // ★ 局末更新一次网络 —— 这一步就是"学习"发生的地方。
    //   梯度累加的四步顺序在 rl::Trainer::finishEpisode 里，只有那一份。
    if (mode == rl::RunMode::Train) {
        trainer.finishEpisode();
        trainer.countEpisode();
    }
    return runner.stats();
}

// ---------------------------------------------------------------------------
//  ★ 命令行开关：让【实验条件】写在命令行上，而不是藏在代码里
//
//  ★ 为什么要这样做
//
//  调参必须有判据（项目铁律）。而"判据"的前提是【知道这一次跑的是什么条件】。
//  如果为了做一次实验就去改源码里的常数，那么：
//      · 跑完以后没人知道当时跑的是哪一版
//      · 两次实验之间可能顺手改了别的东西，无法归因
//      · 忘了改回来，默认值被污染
//
//  所以：**实验条件走命令行，跑完在开头把生效的配置【原样回显】**。
//  这样一份日志自己就说清了它是怎么跑出来的。
//
//  用法：
//      train_probe.exe                       跑默认任务（食物圈）
//      train_probe.exe --task zone-mini      食物圈·单圈（最小任务，调试用）
//      train_probe.exe --episodes 300        只跑 300 局（快速看趋势）
// ---------------------------------------------------------------------------
struct CliArgs {
    rl::TaskKind kind     = rl::TaskKind::FoodZone;
    int          episodes = 2000;
    int          seconds  = 0;    // 0 = 用任务预设的单局时长
    bool         dumpObs  = false; // 只打印观测向量然后退出（排查编码用）

    // -----------------------------------------------------------------------
    //  ★★ 奖励消融（ablation）开关 —— 因果对照实验用
    //
    //  ★ 它解决什么问题
    //
    //  实测现象：FoodZone 训练到几百局之后，策略会【锁死】成几乎只做一个动作
    //  （动作熵掉到接近 0），蛇绕圈直到饿死。而"总回报里某一项占比高"
    //  只能说明【相关】，不能说明【因果】。
    //
    //  要分清因果，唯一的办法是**把某一项单独关掉再训一遍**，同 seed 对比。
    //  这几个开关就是为这件事存在的。
    //
    //  ★ 为什么放在命令行而不是改源码里的常数
    //
    //  项目铁律："调参必须有判据"。如果为了做一次实验就去改 TrainTask.h，
    //  那么跑完之后没人知道当时跑的是哪一版，两组实验之间也可能顺手改了
    //  别的东西 —— 无法归因。走命令行则日志开头会原样回显条件。
    //
    //  ★ 默认全关 = 现状【逐位不变】。这是刻意的回滚保证。
    // -----------------------------------------------------------------------
    bool ablNoFed      = false;   // rFedMax = 0       （关掉"吃饱余韵"）
    // ⚠️ no-approach 于 2026-09-26 失效：④ 已整项删除，没有可关的东西了。
    //    保留这个开关的**解析**（不报错），这样老脚本不会因为未知参数而中断；
    //    生效时会在日志里显示"（已失效）"。
    bool ablNoApproach = false;   // 【已失效】④ 整项已删除
    bool ablNoHunger   = false;   // rHungerMax = 0    （关掉"快饿死"扣分）
    bool ablNoEat      = false;   // rEat = 0          （反向对照：没有正收益）

    // ★ 关掉"贴近敌方身体"的每帧惩罚（2026-09-26 补充）
    //
    //  ★ 为什么单独需要这一个
    //
    //  它是唯一一个"**持续扣分，但观测里看不见其来源**"的项：
    //      · 敌人身体被混进 8 条障碍射线，与墙、与自己不可区分
    //      · 而 Perception 里算好的 headToEnemyBodyDist **没有进观测**
    //  于是蛇会因为"头靠近敌人"持续被扣分，却无法针对性地躲。
    //
    //  这个消融就是要问一句：**这笔持续的钱，是不是在推着蛇不敢靠食物圈**
    //  （食物圈在 0.31 半径处，所有人都在那儿，而危险区半径 100px）。
    bool ablNoSnake    = false;   // rSnakeDanger = 0

    // -----------------------------------------------------------------------
    //  ★ 随机种子（2026-09-25 新增）—— 用来判断"结论是不是这一个种子的运气"
    //
    //  ★ 为什么必须有它
    //
    //  单种子的一次运行**无法区分"这个现象是普遍的"和"这一次碰巧如此"**。
    //  本轮实测就出现过：同一个任务、同一套参数，训练曲线在
    //  "一路变好"和"锁死不动"之间摇摆 —— 而这两种结论对应的修法完全不同。
    //  所以任何结论都要在多个种子下重复，单种子只能当线索，不能当证据。
    //
    //  ★ 默认 20260921u = 原来的固定值，所以【不带 --seed 时行为逐位不变】。
    // -----------------------------------------------------------------------
    unsigned seed = 20260921u;

    // -----------------------------------------------------------------------
    //  ★ 学习侧开关（2026-09-25 新增）—— 用来验证"问题出在学习算法"这个假设
    //
    //  ★ 为什么加这几个而不是直接改 TrainTask.h 的常数
    //
    //  负责人怀疑「越训越差」的根因在优化过程而不是奖励设计。
    //  要验证这个假设，就必须能把**学习侧**单独动、而奖励一分不动 ——
    //  否则改完分不清是哪一边起的作用。
    //
    //  ★ 全部默认 0 = 用任务预设值 = 不带参数时行为逐位不变。
    // -----------------------------------------------------------------------
    double lrOverride      = 0.0;   // >0 时覆盖 TaskConfig::lr
    double gradClip        = 0.0;   // >0 时启用梯度裁剪（全局 L2 范数上限）
    double entropyOverride = -1.0;  // >=0 时覆盖 entropyCoef

    // ★★ 用哪套 advantage（2026-09-26）
    //     baseline = 纯 REINFORCE（原行为，逐位不变）
    //     critic   = Actor-Critic + GAE（价值网络当 baseline）
    std::string learner = "baseline";   // 或 "critic"
    double valueLr      = 0.0;          // >0 时覆盖价值网络学习率
    double gaeLambda    = -1.0;         // >=0 时覆盖 GAE λ
    bool   advNormalize = true;         // advantage 是否 z-score 标准化

    // -----------------------------------------------------------------------
    //  ★★ 任务结构拆解开关（2026-09-28 新增）
    //
    //  ★ 为什么需要它们 —— 实测驱动的
    //
    //  同一套学习链路、同一套超参、同样 1500 局 × 30 秒：
    //      zone-mini（1200²、1 蛇、1 个居中圈 r=216） → 每局吃 9.32、回报一路上涨 ✅
    //      zone-random（2000²、6 蛇、2 个圈相距 800~1200）→ 每局吃 0.66、见顶回落 ❌
    //
    //  **学习链路是好的，它在简单任务上完全 work。**
    //  但 zone-random 与 zone-mini 之间差了【三个变量】：
    //      ① 队伍/蛇数（6 条 vs 1 条）      → 抢食、被撞死、观测里有一堆对手
    //      ② 圈是否乱跑（800~1200px 外 vs 始终在视野内）
    //      ③ 地图大小（2000² vs 1200²）
    //  不拆开就永远无法归因 —— 这正是本项目反复踩过的"一次动多个变量"。
    //
    //  ★ 这些开关【只改任务参数】，默认全部为"不覆盖"，
    //    所以不带参数时行为逐位不变。
    // -----------------------------------------------------------------------
    int    snakesPerTeam  = 0;    // >0 时覆盖（0 = 不覆盖，用任务预设）
    int    teamCount      = 0;    // >0 时覆盖
    double gammaOverride  = -1.0; // >0 时覆盖 TaskConfig::gamma 与 Trainer::gamma
    double foodRate       = -1.0; // >=0 时覆盖 foodPerSecondOverride
    double mapSize        = 0.0;  // >0 时覆盖地图边长（正方形）
};

// 任务名的字符串 → 枚举。故意用【和 taskKindName 一致】的中文别名 + 英文别名，
// 因为命令行里打中文在 Windows 控制台很麻烦。
bool parseTaskKind(const std::string& s, rl::TaskKind& out) {
    if (s == "foodzone" || s == "food-zone" || s == "zone") {
        out = rl::TaskKind::FoodZone;  return true;
    }
    if (s == "zone-mini" || s == "mini" || s == "zonemini") {
        out = rl::TaskKind::ZoneMini;  return true;
    }
    if (s == "ahead" || s == "ahead-food" || s == "aheadfood") {
        out = rl::TaskKind::AheadFood; return true;
    }
    if (s == "scatter" || s == "random" || s == "nozone") {
        out = rl::TaskKind::Scatter;   return true;
    }
    if (s == "zone-random" || s == "zonerandom" || s == "zr") {
        out = rl::TaskKind::ZoneRandom; return true;
    }
    return false;
}

// 把任务配置按消融开关改掉。
//
// ★ 单独抽成一个函数，是为了让"实验条件怎么落到配置上"只有一处 ——
//   散在 parseCli 里的话，加一个开关就要再改一次解析逻辑。
// ★ 返回值 = 实际生效的开关列表（给日志回显用，空串表示什么都没关）。
std::string applyAblation(rl::TaskConfig& t, const CliArgs& a) {
    std::string on;
    auto note = [&](const char* n) {
        if (!on.empty()) on += ",";
        on += n;
    };
    if (a.ablNoFed)      { t.rFedMax      = 0.0; note("no-fed"); }
    // ⚠️ ④ 已整项删除（2026-09-26），这里没有可置零的目标了。
    //    仍显式报告一次，避免"以为关掉了其实什么都没关"的误解。
    if (a.ablNoApproach) { note("no-approach(已失效:④已删除)"); }
    if (a.ablNoHunger)   { t.rHungerMax   = 0.0; note("no-hunger"); }
    if (a.ablNoEat)      { t.rEat         = 0.0; note("no-eat"); }
    return on;
}

CliArgs parseCli(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--task" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (!parseTaskKind(v, a.kind)) {
                std::printf("【警告】未知任务名 \"%s\"，沿用默认（食物圈）\n", v.c_str());
            }
        } else if (k == "--episodes" && i + 1 < argc) {
            const int n = std::atoi(argv[++i]);
            if (n > 0) a.episodes = n;
        } else if (k == "--seconds" && i + 1 < argc) {
            // ★ 覆盖单局时长（秒）—— 用来做"时长是否影响学习"的对照实验
            const int n = std::atoi(argv[++i]);
            if (n > 0) a.seconds = n;
        } else if (k == "--ablate" && i + 1 < argc) {
            // ★ 可以一次关多项：--ablate no-fed,no-approach
            std::string v = argv[++i];
            std::string cur;
            v += ',';                                  // 末尾补一个分隔符，简化循环
            for (char c : v) {
                if (c == ',') {
                    if      (cur == "no-fed")      a.ablNoFed      = true;
                    else if (cur == "no-approach") a.ablNoApproach = true;
                    else if (cur == "no-hunger")   a.ablNoHunger   = true;
                    else if (cur == "no-eat")      a.ablNoEat      = true;
                    else if (!cur.empty()) {
                        std::printf("【警告】未知消融项 \"%s\"（可选：no-fed / "
                                    "no-approach / no-hunger / no-eat）\n", cur.c_str());
                    }
                    cur.clear();
                } else {
                    cur += c;
                }
            }
        } else if (k == "--seed" && i + 1 < argc) {
            // ★ 多种子复现用。默认值不变 = 不带这个参数时行为逐位不变。
            const unsigned s = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
            if (s != 0u) a.seed = s;
        } else if (k == "--lr" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v > 0.0) a.lrOverride = v;
        } else if (k == "--gradclip" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v > 0.0) a.gradClip = v;
        } else if (k == "--entropy" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v >= 0.0) a.entropyOverride = v;
        } else if (k == "--learner" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "baseline" || v == "critic") {
                a.learner = v;
            } else {
                std::printf("【警告】未知学习器 \"%s\"（可选：baseline / critic），"
                            "沿用 baseline\n", v.c_str());
            }
        } else if (k == "--valuelr" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v > 0.0) a.valueLr = v;
        } else if (k == "--gaelambda" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v >= 0.0) a.gaeLambda = v;
        } else if (k == "--no-advnorm") {
            a.advNormalize = false;
        } else if (k == "--snakes" && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            if (v > 0) a.snakesPerTeam = v;
        } else if (k == "--teams" && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            if (v > 0) a.teamCount = v;
        } else if (k == "--gamma" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v > 0.0 && v < 1.0) a.gammaOverride = v;
        } else if (k == "--foodrate" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v >= 0.0) a.foodRate = v;
        } else if (k == "--mapsize" && i + 1 < argc) {
            const double v = std::atof(argv[++i]);
            if (v > 0.0) a.mapSize = v;
        } else if (k == "--dump-obs") {
            a.dumpObs = true;
        } else if (k == "--help" || k == "-h") {
            std::printf("用法: train_probe [--task foodzone|zone-mini|ahead|scatter|zone-random]"
                        " [--episodes N] [--seconds S] [--seed N] [--dump-obs]\n"
                        "      [--lr X] [--gradclip X] [--entropy X]\n"
                        "      [--learner baseline|critic] [--valuelr X] [--gaelambda X]"
                        " [--no-advnorm]\n"
                        "      [--ablate no-fed,no-approach,no-hunger,no-eat]\n");
            std::exit(0);
        } else {
            std::printf("【警告】忽略未知参数 \"%s\"\n", k.c_str());
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
//  ★ 奖励构成累加器
//
//  ★ 为什么必须有它（这是本次排查的关键工具）
//
//  实测遇到过：ZoneMini 任务里**连随机策略都能拿到 613 的平均回报**，
//  而它平均只吃 0.01 个食物。也就是说 613 分几乎全部来自【别的某一项】。
//  只看"总回报"是永远查不出这件事的 —— 总回报是 8 个分量的和，
//  涨了也不知道是谁的功劳，更不知道是谁在【主导】。
//
//  这正是 docs/30 第 6 节记的教训："不要只看总回报"。
//  游戏内训练面板早就有「奖励构成」这一栏，但**命令行工具一直没有**，
//  于是命令行跑出来的数字无法归因。这里补上同一件事。
// ---------------------------------------------------------------------------
struct RewardBreakdown {
    // ★ 2026-09-24 奖励重写后只剩 5 项，所以这里也只有 5 个累加器。
    double total    = 0.0;
    double die      = 0.0;   // ① 记为正数 = 被扣了多少
    double eat      = 0.0;   // ②
    double hunger   = 0.0;   // ③ 负数
    // ④ approach 于 2026-09-26 整项删除（差分项会惩罚"吃掉食物"）
    double fed      = 0.0;   // ⑤
    // ★ ⑤ 诊断：被跳过的"开局自带余韵"（2026-09-26 深夜）
    //   ★ 判据：**它应当恒为 0** —— 非 0 说明"开局不加 fed"的门控坏了。
    double fedSkipped = 0.0;
    double escape   = 0.0;   // ⑥′ 逃生奖励（正数）★ 2026-09-26 取代 snake
    double hunt     = 0.0;   // ⑦ 敌方头贴到我身体的奖励
    double kill     = 0.0;   // ⑧ 击杀奖励
    double disp     = 0.0;   // ⑩′ 饥饿时位移不足扣分（负数）★ 2026-09-26 取代 turn
    double wallD    = 0.0;   // ⑪ 墙危险距离扣分（负数）★ 2026-09-26 深夜
    double enemyD   = 0.0;   // ⑪ 敌身危险距离扣分（负数）★ 2026-09-26 深夜
    double ticks    = 0.0;
    double food     = 0.0;
    int    episodes = 0;
    int    deaths   = 0;
    // ★ 诊断：两个事件式项的触发频率
    //   恒为 0 = "从不触发" → 该去查【触发条件】，而不是调系数
    double escapeC  = 0.0;
    double dispFr   = 0.0;
    // ★ ⑪ 诊断：危险区内的帧数（"常触发"还是"触发得很深"）
    double wallFr   = 0.0;
    double enemyFr  = 0.0;
    // ★★ 死因分布 —— ⑪ 危险距离扣分【是否有效】的唯一判据
    //
    //  ★ 为什么必须分开记死因
    //    "撞墙死"是 ⑪ 要治的目标。只看 ⑪ 的扣分总额没有意义 ——
    //    那只说明"它在扣分"，不说明"蛇学会了躲"。
    //    **奖励项"在生效"和"教会了行为"是两回事**，
    //    这个项目已经在这上面栽过（见 docs/90 第 6 节）。
    int deathWall   = 0;
    int deathSnake  = 0;
    int deathStarve = 0;

    // ★ ④ 的逐帧探针已随该奖励项一起删除（2026-09-26）
    //
    //  保留这条注记是因为它的历史价值：
    //  正是靠"首帧距离"打印出 31622.8（= sqrt(1e9)，一个哨兵值）
    //  才发现"有食物分支的门限写错"那个把 99.66% 的回报变成 bug 常数的
    //  严重问题。**教训：看到一个"物理上不可能"的数，立刻停下来查。**

    void add(const rl::EpisodeStats& s) {
        total  += s.totalReward;
        die    += s.diePenalty;
        eat    += s.eatGain;
        hunger += s.hungerGain;
        fed    += s.fedGain;
        fedSkipped += s.fedSkippedBirth;
        escape += s.escapeGain;
        hunt   += s.huntGain;
        kill   += s.killGain;
        disp   += s.dispGain;
        wallD  += s.wallDangerGain;
        enemyD += s.enemyDangerGain;
        escapeC += s.escapeCount;
        dispFr  += s.dispPenaltyFrames;
        wallFr  += s.wallDangerFrames;
        enemyFr += s.enemyDangerFrames;
        ticks  += s.ticks;
        food   += s.eaten;
        ++episodes;
        if (s.died) ++deaths;
        // ★ 死因分类（⑪ 的判据）
        switch (s.cause) {
            case arena::DeathCause::Wall:       ++deathWall;   break;
            case arena::DeathCause::Snake:      ++deathSnake;  break;
            case arena::DeathCause::Starvation: ++deathStarve; break;
            default: break;
        }
    }

    // 打印【每局平均】的构成，并算出每一项占总回报的百分比
    void print(const char* title) const {
        if (episodes <= 0) return;
        const double n = static_cast<double>(episodes);
        std::printf("\n【奖励构成 · %s】按【每局平均】列（共 %d 局）\n", title, episodes);
        std::printf("   总回报     %10.3f\n", total / n);
        std::printf("   ------------------------------------------------\n");

        // ★ 用【绝对值】算占比：否则正负相抵时百分比会失去意义
        double absSum = std::abs(die) + std::abs(eat) + std::abs(hunger)
                      + std::abs(fed)
                      + std::abs(escape) + std::abs(hunt) + std::abs(kill)
                      + std::abs(disp)
                      + std::abs(wallD) + std::abs(enemyD);
        if (absSum <= 0.0) absSum = 1.0;

        const double per = 1.0 / n;
        auto row = [&](const char* name, double v) {
            std::printf("   %-10s %10.4f   %6.2f%%\n",
                        name, v * per, 100.0 * std::abs(v) / absSum);
        };
        row("die",      -die);
        row("eat",       eat);
        row("hunger",    hunger);
        row("fed",       fed);
        row("escape(6)", escape);
        row("hunt(7)",   hunt);
        row("kill(8)",   kill);
        row("disp(10)",  disp);
        row("wall(11)",  wallD);
        row("enemy(11)", enemyD);
        std::printf("   ------------------------------------------------\n");
        std::printf("   存活帧数   %10.1f   每局吃 %.2f 个   死亡率 %.0f%%\n",
                    ticks / n, food / n, 100.0 * deaths / n);

        // ★★ 死因分布 —— ⑪ 危险距离扣分【是否有效】的唯一判据
        //
        //  ★ 为什么这是唯一判据
        //    "扣了多少分"只说明项在生效；**"撞墙死的比例有没有下降"
        //    才说明蛇学会了躲。** 两者必须分开看。
        //  ⚠️ 前提：改动前后要用同一套任务参数与种子，否则不可比。
        std::printf("   死因分布   撞墙 %d (%.0f%%) | 撞蛇 %d (%.0f%%) | 饿死 %d (%.0f%%) | 未死 %d\n",
                    deathWall,   100.0 * deathWall   / n,
                    deathSnake,  100.0 * deathSnake  / n,
                    deathStarve, 100.0 * deathStarve / n,
                    episodes - deaths);
        std::printf("   ★ 判据: ⑪ 是否有效的唯一证据是【撞墙死的比例下降】，"
                    "而不是 wall(11) 变大\n");

        // ★★ 开工 fed 门控的看门狗（2026-09-26 深夜）
        //
        //  ★ 为什么必须显式报出来
        //    负责人要求"开局那 10 秒不加 fed"。而出生时
        //    `resetStarvation()` 把计时归零，与"刚吃到"无法区分 ——
        //    这个门控很容易在以后被人无意改坏（改回按 hungerLeft 判断就复活）。
        //    `fedSkippedBirth` 是**被跳过的金额**：
        //        = 0  → 门控正常工作（开局没加分）
        //        > 0  → **门控坏了**，正在白送
        std::printf("   开局 fed 门控: 被跳过的白送金额 %+.3f/局  (应为 0；>0 = 门控坏了)\n",
                    fedSkipped * per);
        // ★★ 2026-09-26 深夜：⑪ 危险距离扣分的"实际强度"
        //
        //  ★ 为什么必须报"每秒扣多少"而不只是总额
        //    ⑪ 是**每帧**项，它的强度只有换算成秒才有可比性。
        //    负责人问"reward 没有因为撞死而显著改变" ——
        //    下面这两行就是回答那个问题的数：
        //       次数多、金额小  → 项在生效，但强度不足以改变行为
        //       次数=0          → 判据根本没接上（那才是 bug）
        if (ticks > 0.0) {
            const double secs = ticks / n / 60.0;   // 每局平均秒数
            std::printf("   ---- ⑪ 危险距离（每局 %.1f 秒）----\n", secs);
            // ⚠️ 显示时取负号：内部存的是负数（扣分），但"扣 3.2 分/局"比
            //    "扣 −3.2 分/局"好读。判据不变（绝对值越大 = 扣得越多）。
            std::printf("      贴墙   扣 %8.3f 分/局   生效 %6.1f 帧 | 即 %.2f 分/秒（占该局 %.1f%% 时间）\n",
                        -wallD * per, wallFr / n,
                        secs > 0.0 ? -wallD / per / secs : 0.0,
                        100.0 * wallFr / ticks);
            std::printf("      贴敌身 扣 %8.3f 分/局   生效 %6.1f 帧 | 即 %.2f 分/秒（占该局 %.1f%% 时间）\n",
                        -enemyD * per, enemyFr / n,
                        secs > 0.0 ? -enemyD / per / secs : 0.0,
                        100.0 * enemyFr / ticks);
            std::printf("      ⚠️ 判据: 金额小 = 强度不够改变行为；帧数=0 = 判据没接上（那才是 bug）\n");
        }
        // ★ 事件式项的触发频率 —— 恒为 0 说明"从不触发"，要查条件而不是调系数
        //
        //  ⚠️ 2026-09-26：`escape` 曾经恒为 0，原因**不是**"蛇不靠近敌人"，
        //     而是 `Perception` 的 `headToEnemyBodyDist` **每帧没重置** ——
        //     敌人一出视野，读数就冻结在最后一次看到的值，于是
        //     "进入 <250px"会触发、"退出 >375px"永远不触发。
        //     修法见 `Perception::resetFrameState()`。
        //     所以这一行现在是判断该 bug 是否复发的**哨兵**：
        //     `逃生奖励触发` 长期为 0 就该去查它。
        std::printf("   逃生奖励触发 %.2f 次/局 | 连续转弯扣分 %.1f 帧/局\n",
                    escapeC / n, dispFr / n);
        // ★ ⑪ 危险距离扣分（2026-09-26 深夜）
        //   帧数 = "常触发"还是"触发得很深"；配合上面的金额一起看
        std::printf("   危险距离: 贴墙 %.1f 帧/局 (%.1f%%) | 贴敌身 %.1f 帧/局 (%.1f%%)\n",
                    wallFr / n, 100.0 * wallFr / (ticks > 0.0 ? ticks : 1.0),
                    enemyFr / n, 100.0 * enemyFr / (ticks > 0.0 ? ticks : 1.0));
        std::printf("   * ④「靠近食物」已于 2026-09-26 整项删除"
                    "（差分项会惩罚「吃掉食物」）\n");
    }
};

// ===========================================================================
//  ★★ 滑动窗口指标 —— 「策略有没有在学 / 有没有锁死」的客观判据
//
//  ★ 为什么不能只看单局抽样
//
//  实测（FoodZone 2500 局）单局抽样的样子：
//        第 200 局   吃 13 个、存活 3600 帧   ← 看起来学会了
//        第 1000 局  吃  0 个、存活  274 帧   ← 看起来什么都没学会
//        第 1700 局  吃  5 个、存活 3600 帧   ← 又好了
//  同一份权重在不同局之间能差 30 倍以上，**单局抽样完全分不出
//  "在进步"和"在两种模式之间乱跳"** —— 而这两种情况的修法完全不同。
//
//  所以要的是【最近 N 局的滑动平均】，而不是"第 N 局那一局"。
//
//  ★ 动作熵为什么是"锁死"的判据
//
//      H = −Σ p·log p        （p 由【实际执行】的动作次数归一化）
//      H = log(3) = 1.0986   → 三个动作等概率，完全没有偏好
//      H = 0                 → 永远只做一个动作，完全刻板
//  它比"看一眼 policy 那三个数"可靠，因为它是跨 100 局统计出来的，
//  不会被某一局的偶然采样骗到。
//
//  ★ 为什么用【实际执行】的动作而不是网络输出的概率
//
//  概率可能看起来还在动，但只要 argmax 固定，行为就是刻板的。
//  而且要一起算进 ε-greedy 那 5% 的随机步 —— 那才是作用在世界上的动作。
// ===========================================================================
struct RollingWindow {
    static constexpr int N = 100;

    // ★ 是否显示 critic 诊断（由 main 按 learner 模式设置）
    bool showCritic = false;

    int    n        = 0;
    double food     = 0.0;   // 累计吃食数
    double ticks    = 0.0;   // 累计存活帧
    double fedRate  = 0.0;   // 累计 fed 每帧速率
    int    framesFed      = 0;
    int    framesStarving = 0;
    int    framesTotal    = 0;
    long   act[rl::ACT_SIZE] = {};   // ★ 按 ACT_SIZE 归零（回滚后 = 3 个动作）

    // ★ 死因分布（2026-09-25 新增）
    //
    //  ★ 为什么需要它：现象"蛇一直转圈"到底是被什么结束的，
    //    三种死因的处理方式完全不同：
    //        撞墙 → 转向/边界行为问题
    //        撞蛇 → 躲避/竞争问题
    //        饿死 → 觅食问题（这才是"转圈到饿死"的字面含义）
    //    没有这个分布，"转圈"只是肉眼观感，无法归因。
    int deathWall = 0, deathSnake = 0, deathStarve = 0;

    // ★ 每项"实际生效帧数"（用来分清"常触发"还是"触发得很深"）
    //  ★ 为什么必须有它 —— 本轮真实踩到的推理错误
    //    实测 snake 项 −10.66/局，我据此想反推"它有多少帧在危险区内"，
    //    算出来 71 帧 —— 但这个反推【依赖"每帧都满额"这个假设】，
    //    而梯度是线性的：贴着边界待 1000 帧也能凑出同样的总和。
    //    两种情况的成因和修法完全不同，只看总和分不出来。
    long   snakeFrames  = 0;
    long   hungerFrames = 0;

    // ★ 步长诊断（「越训越差」的判据）
    //   均值看典型步长，**最大值**看有没有"一脚踹飞"的异常局 ——
    //   后者才是把学好的东西踩坏的元凶，只看均值会把它平均掉。
    double gradNormSum = 0.0, gradNormMax = 0.0;
    double updNormSum  = 0.0, updNormMax  = 0.0;
    double paramNormLast = 0.0;

    // ★ 熵项单独的梯度范数（2026-09-26 深夜新增）
    //   ★ 为什么：熵项若与策略梯度同量级，说明策略其实是被熵项推着走的 ——
    //     而熵项只喊"别太确定"，不含任何任务信息。
    double entGradSum = 0.0;

    // ★★ 响应幅度（2026-09-27 新增）—— 见 rl::EpisodeStats 里那一整段说明
    //
    //  这三个数回答的是一个比"回报/熵"更基础的问题：
    //      **网络的输出到底有没有随输入变化？**
    //  一个"永远输出同一个动作"的退化网络，和"健康但偏好某动作"的网络，
    //  在动作概率面板上可以长得一模一样（都是 0.07 0.07 0.60）。
    //
    //  判据：
    //      greedySwitchRate ≈ 0 且 topGreedyFrac ≈ 1  → 网络无视输入（退化）
    //      logitSpreadAvg 小到 2 以下                  → 对输入的响应极弱
    long   greedySwitch = 0;      // 贪心动作变化次数
    long   greedyFrames = 0;      // 参与统计的帧数
    long   greedyAct[rl::ACT_SIZE] = {};
    double spreadSum = 0.0;
    double spreadMin = 1.0e30, spreadMax = -1.0e30;

    // ★★ 有多少【局】是"锁死"的
    //
    //  ★ 为什么必须按【局】统计，而不是只看全局平均
    //    "整局死咬一个动作"是一个**每局都会发生或不发生的事件**，
    //    而全局平均值会把它摊平：一半的局锁死、一半的局正常，
    //    平均切换率看起来"还行"，但锁死的那一半局已经废了。
    //    实测就是这样：seed22222 在 @500/@600 两个窗口里
    //      切换率 0.0158 / 0.0017（几乎不切换）
    //      → 说明**大多数局整局只做一个动作**，而不是"偶尔发呆"。
    //
    //  ★ 判据（与 print() 里的告警用同一对阈值，避免两处判据打架）
    //      某局的 topFrac > 0.90 且 帧数 >= 100  → 记为锁死局
    //    "帧数 >= 100" 是为了排除"开局就撞死"的短局 ——
    //    那种局本来就没几帧，占比高是必然的，不代表策略刻板。
    int    lockedEpisodes = 0;     // topFrac > 0.90 的局数
    int    countedEpisodes = 0;    // 参与判定的局数（帧数 >= 100 的局）

    // ★ critic 诊断（只有 actor-critic 模式有值）
    //   ★ 这是"要不要继续往下做"的唯一判据：
    //     valueCorr 持续接近 0 = 价值网络没学到 = GAE 是空的，
    //     那就不该继续往上堆机制，而应该停下来查原因。
    double valueCorrSum = 0.0, valueCorrMin = 1.0e18;
    double valueLossSum = 0.0;
    double advStdSum    = 0.0;

    void clear() { *this = RollingWindow{}; }

    // ★ 每局累加响应幅度
    void addGreedy(const rl::EpisodeStats& s) {
        greedySwitch += s.greedySwitchCount;
        greedyFrames += s.greedyFrames;
        for (int i = 0; i < rl::ACT_SIZE; ++i) greedyAct[i] += s.greedyActionCount[i];
        spreadSum += s.logitSpreadSum;
        if (s.greedyFrames > 0) {
            if (s.logitSpreadMin < spreadMin) spreadMin = s.logitSpreadMin;
            if (s.logitSpreadMax > spreadMax) spreadMax = s.logitSpreadMax;
        }
        // ★ 按【局】判定"锁死"—— 见 lockedEpisodes 的说明
        if (s.greedyFrames >= 100) {
            ++countedEpisodes;
            if (s.topGreedyFraction() > 0.90) ++lockedEpisodes;
        }
    }

    // 锁死局占比（0..1）。判据：这个数大 = 多数局整局只做一个动作
    double lockedFrac() const {
        return (countedEpisodes > 0)
             ? static_cast<double>(lockedEpisodes) / static_cast<double>(countedEpisodes) : 0.0;
    }

    // 贪心动作切换率：健康应明显 > 0；≈ 0 = 网络无视输入
    double greedySwitchRate() const {
        return (greedyFrames > 0)
             ? static_cast<double>(greedySwitch) / static_cast<double>(greedyFrames) : 0.0;
    }

    // 出现最多的那个贪心动作占多少帧：≈ 1 = 整局死咬一个动作
    double topGreedyFrac() const {
        long tot = 0, best = 0;
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            tot += greedyAct[i];
            if (greedyAct[i] > best) best = greedyAct[i];
        }
        return (tot > 0) ? static_cast<double>(best) / static_cast<double>(tot) : 0.0;
    }

    // 出现最多的那个贪心动作的【下标】（0..5），-1 = 没有样本
    int topGreedyAction() const {
        long best = 0; int bi = -1;
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            if (greedyAct[i] > best) { best = greedyAct[i]; bi = i; }
        }
        return bi;
    }

    double spreadAvg() const {
        return (greedyFrames > 0) ? spreadSum / static_cast<double>(greedyFrames) : 0.0;
    }

    void addStep(double gNorm, double uNorm, double pNorm) {
        gradNormSum += gNorm;  if (gNorm > gradNormMax) gradNormMax = gNorm;
        updNormSum  += uNorm;  if (uNorm > updNormMax)  updNormMax  = uNorm;
        paramNormLast = pNorm;
    }

    void addEntGrad(double eNorm) { entGradSum += eNorm; }

    void addCritic(double corr, double vloss, double advStd) {
        valueCorrSum += corr;
        if (corr < valueCorrMin) valueCorrMin = corr;
        valueLossSum += vloss;
        advStdSum    += advStd;
    }

    void add(const rl::EpisodeStats& s) {
        ++n;
        food    += s.eaten;
        ticks   += s.ticks;
        fedRate += s.fedRateSum;
        framesFed      += s.framesFed;
        framesStarving += s.framesStarving;
        framesTotal    += s.ticks;
        snakeFrames    += s.dangerFrames;
        hungerFrames   += s.hungerFrames;
        for (int i = 0; i < rl::ACT_SIZE; ++i) act[i] += s.actCount[i];
        addGreedy(s);                       // ★ 响应幅度探针
        switch (s.cause) {
            case arena::DeathCause::Wall:       ++deathWall;   break;
            case arena::DeathCause::Snake:      ++deathSnake;  break;
            case arena::DeathCause::Starvation: ++deathStarve; break;
            default: break;   // 没死（到局末）或未知
        }
    }

    bool full() const { return n >= N; }

    // 实际动作占比 —— ★ 6 个动作【合并成 3 个转向档位】
    //
    //  ★ 为什么合并
    //    动作空间是 3 转向 × 2 速度（ACT_SIZE = 6），但**"锁死"是转向层面的现象**：
    //    "永远左转"无论加速与否都是刻板行为。
    //    合并之后这三项仍然是"转向偏好"，与改动作空间之前的语义完全一致，
    //    历史数据也还能对照。
    //  ⚠️ 想看速度档偏好，见 print() 里额外输出的 boost 占比。
    double pOf(int steerIdx) const {
        double tot = 0.0, hit = 0.0;
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            tot += static_cast<double>(act[i]);
            if (rl::steerIndexOfAction(i) == steerIdx) hit += static_cast<double>(act[i]);
        }
        return (tot > 0.0) ? hit / tot : 1.0 / rl::ACT_STEER_COUNT;
    }

    // 加速帧占比（2026-09-26 新增：加速进了动作空间，值得单独看）
    double boostFrac() const {
        double tot = 0.0, hit = 0.0;
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            tot += static_cast<double>(act[i]);
            if (rl::boostOfAction(i)) hit += static_cast<double>(act[i]);
        }
        return (tot > 0.0) ? hit / tot : 0.0;
    }

    // 动作熵（0 = 完全刻板，log3 ≈ 1.0986 = 完全均匀）
    //   ★ 在【3 个转向档位】上算 —— 理由同上（锁死是转向层面的现象）
    double entropy() const {
        double h = 0.0;
        for (int i = 0; i < rl::ACT_STEER_COUNT; ++i) {
            const double p = pOf(i);
            if (p > 1e-12) h -= p * std::log(p);
        }
        return h;
    }

    double meanFood()      const { return n ? food / n : 0.0; }
    double meanTicks()     const { return n ? ticks / n : 0.0; }
    double fedRatePerTick()const { return ticks > 0 ? fedRate / ticks : 0.0; }
    double fedFrac()       const { return framesTotal > 0
                                        ? static_cast<double>(framesFed) / framesTotal : 0.0; }
    double starveFrac()    const { return framesTotal > 0
                                        ? static_cast<double>(framesStarving) / framesTotal : 0.0; }

    // 一行摘要（写进 CSV，供游戏里的曲线界面以后使用）
    void writeRows(MetricsWriter& w, long ep) const {
        if (n <= 0) return;
        w.row("roll", ep, meanFood(),            "food");
        w.row("roll", ep, meanTicks(),           "survival");
        w.row("roll", ep, entropy(),             "actH");
        w.row("roll", ep, pOf(0),                "pL");
        w.row("roll", ep, pOf(1),                "pF");
        w.row("roll", ep, pOf(2),                "pR");
        w.row("roll", ep, boostFrac(),           "boost");
        w.row("roll", ep, fedRatePerTick(),      "fedRate");
        w.row("roll", ep, fedFrac(),             "fedFrac");
        w.row("roll", ep, starveFrac(),          "starveFrac");
        // ★★ 响应幅度 —— "网络有没有在看输入"的直接判据
        w.row("roll", ep, greedySwitchRate(),    "greedySwitchRate");
        w.row("roll", ep, topGreedyFrac(),       "topGreedyFrac");
        w.row("roll", ep, spreadAvg(),           "logitSpread");
        w.row("roll", ep, spreadMin,             "logitSpreadMin");
        w.row("roll", ep, spreadMax,             "logitSpreadMax");
        w.row("roll", ep, lockedFrac(),          "lockedFrac");
    }

    void print(long ep, double secs) const {
        if (n <= 0) return;
        std::printf("  [滑窗%3d局 @%ld] 吃 %5.2f | 存活 %6.1f | 动作熵 %.3f "
                    "(左%.2f 直%.2f 右%.2f) | 加速%.0f%%\n",
                    N, ep, meanFood(), meanTicks(), entropy(),
                    pOf(0), pOf(1), pOf(2), 100.0 * boostFrac());
        std::printf("                    每帧速率: fed %+.3f   "
                    "状态: 余韵 %4.1f%% | 饥饿 %4.1f%%   (%.0fs)\n",
                    fedRatePerTick(),
                    100.0 * fedFrac(), 100.0 * starveFrac(), secs);
        std::printf("                    死因: 撞墙 %3d | 撞蛇 %3d | 饿死 %3d | 未死 %3d\n",
                    deathWall, deathSnake, deathStarve,
                    n - deathWall - deathSnake - deathStarve);
        // ★ 生效帧数占比 —— 分清"常触发"和"触发得深"
        if (framesTotal > 0) {
            std::printf("                    生效帧占比: 蛇危险区 %5.2f%% | 饥饿窗口 %5.2f%%"
                        "   (即每局 %5.1f 帧 / %5.1f 帧)\n",                        100.0 * static_cast<double>(snakeFrames) / framesTotal,
                        100.0 * static_cast<double>(hungerFrames) / framesTotal,
                        static_cast<double>(snakeFrames) / n,
                        static_cast<double>(hungerFrames) / n);
        }
        // ★ 步长诊断 —— 「越训越差」最可能的直接成因
        //
        //   看两件事：
        //     ① updateNorm 均值相对 paramNorm 的比例（步子是不是太大）
        //     ② gradNormMax 相对均值（有没有异常大的局在踹参数）
        if (n > 0 && paramNormLast > 0.0) {
            const double avgG = gradNormSum / n;
            const double avgU = updNormSum / n;
            std::printf("                    步长: |grad| 均 %.3f 峰 %.3f | |Δw| 均 %.4f 峰 %.4f"
                        " | |w| %.2f  →  单步相对 %.3f%%\n",
                        avgG, gradNormMax, avgU, updNormMax, paramNormLast,
                        100.0 * avgU / paramNormLast);
            std::printf("                          (峰值/均值: |grad| %.1fx  |Δw| %.1fx)\n",
                        avgG > 0 ? gradNormMax / avgG : 0.0,
                        avgU > 0 ? updNormMax / avgU : 0.0);

            // ★ 熵项 vs 策略梯度（2026-09-26 深夜）
            //
            //  ★ 判据：熵项应当【远小于】总梯度。
            //    若 熵项/总梯度 是个不小的比例，说明策略很大程度上是被熵项
            //    推着走的 —— 而熵项只喊"别太确定"，不含任何任务信息，
            //    那"越训越差"就是它造成的。
            const double avgE = entGradSum / n;
            std::printf("                    熵项: |grad_ent| 均 %.5f  =  总梯度的 %.2f%%\n",
                        avgE, avgG > 1.0e-12 ? 100.0 * avgE / avgG : 0.0);
            if (avgG > 1.0e-12 && avgE / avgG > 0.30) {
                std::printf("                    【警告】熵项占比 >30%% —— "
                            "策略正在被熵项主导，检查 entropyCoef 的符号与大小\n");
            }
        }

        // ===============================================================
        //  ★★★ 响应幅度 —— "网络到底有没有在看输入"的唯一直接判据
        //
        //  ★ 为什么它比回报/熵更基础
        //    回报和熵都是【输出的统计】。一个"永远输出同一个动作"的
        //    退化网络，和一个"健康但偏好某动作"的网络，这两项可以
        //    长得一模一样 —— 都是 `policy 0.07 0.07 0.60`。
        //    区别只能是：前者对任何观测都这样，后者随局面变化。
        //    这一行量测的正是那个区别。
        //
        //  ★ 读法
        //      switch率 ≈ 0 且 最大动作占比 ≈ 1.0  → **网络无视输入**
        //      logit 幅度小（个位数以下）           → 对输入的响应极弱
        //    ⚠️ 幅度大不等于策略对（也可能自信地做错事），
        //       但幅度**小**一定是"网络没在区分局面"，那是最根本的问题。
        // ===============================================================
        if (greedyFrames > 0) {
            const int ta = topGreedyAction();
            static const char* kActName[6] = {
                "左转", "直行", "右转", "左转+加速", "直行+加速", "右转+加速"
            };
            std::printf("                    响应幅度: 贪心动作切换率 %.4f/帧 | "
                        "最大动作占比 %.1f%% (%s) | logit 幅度 均 %.2f 最小 %.2f 最大 %.2f\n",
                        greedySwitchRate(), 100.0 * topGreedyFrac(),
                        (ta >= 0 && ta < rl::ACT_SIZE) ? kActName[ta] : "?",
                        spreadAvg(), spreadMin, spreadMax);
            // ★★ 锁死局占比 —— 比平均值更该看的那一个数
            //
            //  ★ 为什么它比上面的"平均切换率"更重要
            //    "整局只做一个动作"是每局发生/不发生的事件，平均值会摊平它。
            //    实测 seed22222 @600：平均切换率 0.0017（已经很低），
            //    但真正该看的是"有多少局是这样"—— 那才决定训练数据里
            //    有多少比例是废的。
            std::printf("                    锁死局占比: %5.1f%%  (%d / %d 局，判据: 单局最大动作占比>90%%)\n",
                        100.0 * lockedFrac(), lockedEpisodes, countedEpisodes);
            if (greedySwitchRate() < 0.02 && topGreedyFrac() > 0.80) {
                std::printf("                    【警告】贪心动作几乎从不变化 —— "
                            "网络对输入没有反应，argmax 恒定。\n"
                            "                            这会让【敌方/队友】(argmax) "
                            "整局只做一个动作、撞死，\n"
                            "                            而学习中的蛇因为按概率采样看起来还有行为。\n");
            }
        }
        // ★ critic 诊断 —— 价值网络到底有没有学起来
        if (showCritic && n > 0) {
            std::printf("                    ★ critic: V↔目标 相关 均 %+.3f (最差 %+.3f) | "
                        "MSE %.2f | adv 标准差 %.3f\n",
                        valueCorrSum / n, valueCorrMin,
                        valueLossSum / n, advStdSum / n);
            std::printf("                      判据: 相关持续 <=0 = V 没学到 → GAE 是空的，"
                        "该停下来查\n");
        }
    }
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ★ 先把控制台输出代码页切成 UTF-8，否则下面的中文全是乱码。
    //   见 src/utils/ConsoleUtf8.h 的说明（默认代码页 936 会花屏）。
    textutil::enableUtf8Console();
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const CliArgs cli = parseCli(argc, argv);

    // ⚠️ 必须放在 parseCli 【之后】—— 否则 --seed 改不动这一份全局随机序列。
    //    `Arena` 的出生几何等地方还在用全局 rand()（见 docs/30 缺陷 #25），
    //    所以这里也要跟着种子走，多种子实验才是真的"不同世界"。
    std::srand(cli.seed);

    rl::TaskConfig task = rl::makeTaskConfig(cli.kind, cli.seed);
    // ★ 命令行覆盖单局时长（对照实验用；不改就是任务预设值）
    if (cli.seconds > 0) task.maxTicks = cli.seconds * 60;
    // ★★ 应用奖励消融开关 —— 必须在这里（拿到 task 之后、建世界之前）。
    //    返回值是"实际生效的开关"，下面要回显到日志里。
    const std::string ablated = applyAblation(task, cli);
    // ★ 学习侧覆盖（0 / 负数 = 用任务预设，不改）
    if (cli.lrOverride > 0.0)      task.lr          = cli.lrOverride;
    if (cli.entropyOverride >= 0.0) task.entropyCoef = cli.entropyOverride;

    // -----------------------------------------------------------------------
    //  ★★ 任务结构覆盖（2026-09-28）—— 用来拆开 zone-random 的三个变量
    //
    //  ★ 顺序讲究：必须在 `makeArenaConfig` 之前改完，否则改的是"已经拷进
    //    ArenaConfig 的副本"。这也是本项目踩过的坑（改了 TaskConfig 却
    //    发现世界没变 —— 因为 Arena 早就建好了）。
    // -----------------------------------------------------------------------
    if (cli.teamCount > 0 || cli.snakesPerTeam > 0) {
        if (cli.teamCount > 0)      task.teamCount     = cli.teamCount;
        if (cli.snakesPerTeam > 0)  task.snakesPerTeam = cli.snakesPerTeam;
        // ★ 队表必须跟着重建 —— 它是 teamCount/snakesPerTeam 的派生量，
        //   不重建就会出现"蛇数变了但角色表还是旧的"这种静默不一致。
        task.rebuildTeamTables();
    }
    if (cli.gammaOverride > 0.0)    task.gamma = cli.gammaOverride;
    if (cli.foodRate >= 0.0)        task.foodPerSecondOverride = cli.foodRate;
    if (cli.mapSize > 0.0) { task.mapWidth = cli.mapSize; task.mapHeight = cli.mapSize; }
    const int EPISODES = cli.episodes;

    std::printf("======================================================================\n");
    std::printf(" REINFORCE 训练（命令行）\n");
    std::printf(" 问题：%d 维观测 + 这套奖励 + REINFORCE，能不能学会觅食？\n", rl::OBS_SIZE);
    std::printf(" 判据：和【随机策略基线】比。没有这个零点，回报在涨也没意义。\n");
    std::printf("======================================================================\n");
    std::printf(" 观测 %d 维 | 动作 %d 个 | 隐藏层 %d | 局数 %d | 单局上限 %d 帧\n",
                rl::OBS_SIZE, rl::ACT_SIZE, rl::HIDDEN_WIDTH, EPISODES, task.maxTicks);
    // ⚠️ 饥饿时间和单局时长的关系是"会不会饿死"的关键 —— 必须回显出来
    std::printf(" 单局 %.0fs | 饥饿 %.0fs | 局内最多饿 %d 次\n",
                task.maxTicks / 60.0, task.starveTime,
                static_cast<int>(task.maxTicks / 60.0 / task.starveTime));
    std::printf(" 地图 %.0fx%.0f | 初始长度 %d 节\n",
                task.mapWidth, task.mapHeight, task.initSegs);

    // ★ 回显【本次实验的生效配置】—— 让日志自己说清它是怎么跑出来的
    //
    //  ⚠️⚠️ 这里曾经有一个中文被拆坏的 bug（2026-09-25 修）
    //
    //    旧写法是逐字符 `static_cast<char>(wn[k] & 0xFF)` —— 它假设
    //    "宽字符的低 8 位就是 ASCII"。这对 ASCII 成立，对中文**完全不成立**：
    //    一个汉字在 UTF-16 里的码点 > 0xFF，取低 8 位得到的是半个字节，
    //    于是任务名显示成 `�i4 ��Mnnb + 4 �`，排查时极易被误当成
    //    "任务选错了"或"配置读坏了"。
    //
    //    ★ 正确做法：宽字符 → **码点** → 按 UTF-8 重新编码。
    //      （不用 std::u8string / char8_t —— 那是 C++20，本项目是 C++17。）
    //  ⚠️ 另外 printf 的 `%ls` 在 MinGW 下对宽字符串不可靠，所以不指望它，
    //     只依赖上面这个显式转换。
    std::printf(" ---- 本次实验条件 ----\n");
    {
        // 宽字符串 → UTF-8（逐码点编码，1~4 字节）
        const wchar_t* wn = rl::taskKindName(task.kind);
        char nb[256] = {0};
        std::size_t o = 0;
        for (std::size_t k = 0; wn[k] != L'\0' && o + 4 < sizeof(nb); ++k) {
            const unsigned long cp = static_cast<unsigned long>(wn[k]);
            if (cp < 0x80UL) {
                nb[o++] = static_cast<char>(cp);
            } else if (cp < 0x800UL) {
                nb[o++] = static_cast<char>(0xC0UL | (cp >> 6));
                nb[o++] = static_cast<char>(0x80UL | (cp & 0x3FUL));
            } else if (cp < 0x10000UL) {
                nb[o++] = static_cast<char>(0xE0UL | (cp >> 12));
                nb[o++] = static_cast<char>(0x80UL | ((cp >> 6) & 0x3FUL));
                nb[o++] = static_cast<char>(0x80UL | (cp & 0x3FUL));
            } else {
                nb[o++] = static_cast<char>(0xF0UL | (cp >> 18));
                nb[o++] = static_cast<char>(0x80UL | ((cp >> 12) & 0x3FUL));
                nb[o++] = static_cast<char>(0x80UL | ((cp >> 6) & 0x3FUL));
                nb[o++] = static_cast<char>(0x80UL | (cp & 0x3FUL));
            }
        }
        nb[o] = '\0';
        std::printf(" 任务      : %s\n", nb);
    }
    std::printf(" 蛇        : %d 条（%d 个学习）\n",
                task.snakeCount(), task.learnerCount());
    std::printf(" 圈        : %d 个 | 半径比 %.3f | 供食 %.0fs + 空窗 %.0fs\n",
                task.zoneCount, task.zoneRadiusRatio,
                task.zoneSpawnSeconds, task.zoneGapSeconds);
    std::printf(" 食物      : %.2f 个/秒（总）\n", task.foodPerSecondOverride);
    std::printf(" 奖励      : die %.1f | eat %.1f | hunger %.3f(封顶) | fed %.1f/秒\n",
                task.rDie, task.rEat, task.rHungerMax, task.rFedMax);
    std::printf("            逃生 +%.2f(进%.0fpx/出%.0fpx) | 猎杀优势 +%.2f | 击杀 +%.1f\n",
                task.rEscape, task.escapeEnterDist,
                task.escapeEnterDist * task.escapeExitRatio,
                task.rHuntAdvantage, task.rKill);
    std::printf("            位移不足 %.1f/帧（饿时<%.0fs 且最近%.0fs 位移<%.0fpx 才扣）\n",
                task.dispPenalty, task.hungerWarnSec,
                task.dispWindowSec, task.dispMinPx);
    // ★ ⑪ 危险距离扣分（2026-09-26 深夜新增，负责人要求）
    //   必须回显"每帧 → 每秒"的换算 + 曲线指数，否则很容易误读强度
    std::printf("            危险距离 墙 %.2f/帧(贴脸) <%.0fpx | 敌身 %.2f/帧(贴脸) <%.0fpx | 曲线指数 %.1f\n",
                task.rWallDanger, task.wallDangerDist,
                task.rEnemyDanger, task.enemyDangerDist,
                task.dangerCurvePower);
    std::printf("                       (幂曲线 peak×(1−d/%.0f)^%.0f，外面几乎无感、贴脸剧痛)\n",
                task.wallDangerDist, task.dangerCurvePower);
    // ★ 标定自检：把"撞死一次大约扣多少"直接算出来给负责人看
    //
    //  ★ 为什么要在日志里算这个
    //    负责人给的目标是"撞死能到 −5 左右"。这个数**不是常数本身**，
    //    而是 常数 × 距离 ÷ 速度 的积分 —— 光看常数看不出对不对。
    //    这个标定已经错过两次（第一次只报峰值、第二次系数少写 10 倍），
    //    所以现在把它算出来回显，改完立刻能核对目标有没有达成。
    //
    //  ★ 积分下限是【碰撞距离】不是 0
    //    碰撞判定是"头【中心】到边界 < headRadius"，所以蛇只走完
    //    150px → headRadius 这一段。
    //  ★ (a) 连续积分（我用它做标定基准）
    //  ★ (b) 逐帧求和（更接近实际实现 —— 它每帧采一次样）
    //    两者都报出来，差额就是"离散化误差"，避免再把模型当精确值。
    {
        const double lim  = static_cast<double>(task.wallDangerDist);
        const double v    = 250.0;         // SNAKE_SPEED，基础速度
        const double rHit = 18.0;          // SNAKE_HEAD_RADIUS，碰撞时的头心距离
        const double pw   = task.dangerCurvePower;

        // (a) 连续：∫(1−d/lim)^pw dd 在 [rHit, lim] 上 = lim/(pw+1) · (1 − rHit/lim)^(pw+1)
        const double uHit   = 1.0 - rHit / lim;
        const double areaC  = lim / (pw + 1.0) * std::pow(uHit, pw + 1.0);
        const double costC  = task.rWallDanger * areaC / v * 60.0;

        // (b) 逐帧：每帧前进 v/60 px，从 rHit 走到 lim
        //
        //  ⚠️ 每帧的扣分已经【含 dt】（它是"每帧"的金额），所以累加之后
        //     **必须再乘帧率** 才是"每次撞死"的总分 —— 不乘就少了 60 倍。
        //     2026-09-26 我的第一版就漏了这个 60，日志打出 −0.40 而
        //     连续积分是 −22.67，两者本该接近。**两个数不一致就是 bug 的信号。**
        const double step   = v / 60.0;
        double costD = 0.0;
        for (double d = rHit; d < lim; d += step) {
            costD += task.rWallDanger * std::pow(1.0 - d / lim, pw) / 60.0;
        }
        costD *= 60.0;   // ← 换算成"每次撞死"的总分（与连续积分同口径）

        std::printf("                       ★ 标定「直冲撞墙一次」: 连续积分 %.1f 分 | 逐帧求和 %.1f 分\n",
                    costC, costD);
        std::printf("                         （积分下限 = 碰撞距离 %.0fpx，不是 0）\n", rHit);
        std::printf("                       ⚠️ 加速时穿越更快 → 该值约减半；斜擦过去也更小\n");
    }
    std::printf("            ⚠️ ④「靠近食物」已于 2026-09-26 整项删除\n");
    std::printf("            饥饿窗口 %.0fs | 吃饱余韵 %.0fs\n",
                task.hungerWarnSec, task.fedDuration);
    std::printf(" 熵系数    : %.4f | ε-greedy %.3f | lr %.3f | γ %.3f\n",
                task.entropyCoef, task.explorationEps, task.lr, task.gamma);
    // ★ 消融条件必须原样回显 —— 否则两组对照实验的日志长得一模一样，
    //   事后无法归因（这正是"调参必须有判据"要求的那件事）。
    std::printf(" 消融      : %s\n",
                ablated.empty() ? "无（所有奖励项按原值生效）"
                                : (ablated + "  ← 这些项已被置 0").c_str());
    // ★ 种子也要回显 —— 多种子对照实验里，日志必须能自证是哪一次
    std::printf(" 种子      : %u\n", task.seed);
    // ★ 学习侧条件（「越训越差」的排查必须能看见这一步走了多大）
    std::printf(" 学习      : lr %.4f | 熵系数 %.4f | 梯度裁剪 %s\n",
                task.lr, task.entropyCoef,
                (cli.gradClip > 0.0) ? "开" : "关(默认)");
    if (cli.gradClip > 0.0) std::printf("            裁剪上限 %.4f\n", cli.gradClip);
    // ⚠️ 这一段必须用 cli / task 里**已经确定的值**来打印，
    //    不能引用 trainer —— trainer 是在这之后才构造的。
    const bool wantCritic = (cli.learner == "critic");
    const double effValueLr = (cli.valueLr > 0.0) ? cli.valueLr : 0.02;
    const double effGae     = (cli.gaeLambda >= 0.0) ? cli.gaeLambda : 0.95;
    if (wantCritic) {
        std::printf(" 学习器    : ★ actor-critic + GAE"
                    " | 价值网络 lr %.4f | λ %.3f | advantage 标准化 %s\n",
                    effValueLr, effGae, cli.advNormalize ? "开" : "关");
        std::printf("             ⚠️ γ=%.3f、60Hz → 有效视野约 %.1f 帧 ≈ %.1f 秒\n",
                    task.gamma, 1.0 / (1.0 - task.gamma),
                    1.0 / (1.0 - task.gamma) / 60.0);
    } else {
        std::printf(" 学习器    : baseline（纯 REINFORCE，与原行为逐位一致）\n");
    }
    std::printf(" ----------------------\n\n");

    std::mt19937 rng(task.seed);

    // ★ 世界在 arena::Arena 里、网络更新在 rl::Trainer 里、
    //   "一局怎么跑 + 奖励怎么算"在 rl::EpisodeRunner 里 —— 三者都只有一份，
    //   和游戏内的训练模式共用。见 docs/45、docs/70。
    arena::registerBuiltinBrains();

    // -----------------------------------------------------------------------
    //  ★ 顺序要求：Trainer 必须在 Arena 【之前】建
    //
    //  ★ 为什么
    //    自对弈的敌方脑（"nn-enemy"）在 `Arena::reset()` 里创建，
    //    那时它会去读训练器写的【冻结快照】。如果 Trainer 还没建好、
    //    快照还没写，敌方就会加载失败 → 变成随机蛇，
    //    而且会一直随机到第 100 局第一次同步为止（那段数据白跑）。
    //
    //    所以先建 Trainer → 写一次初始快照 → 再建 Arena。
    //    （游戏内的训练模式也是同样的顺序，见 LiveTrainingScene::restartTraining）
    // -----------------------------------------------------------------------
    rl::Trainer trainer(task.seed);
    trainer.gamma            = task.gamma;
    trainer.lr               = task.lr;
    trainer.baselineMomentum = task.baselineMomentum;
    trainer.entropyCoef      = task.entropyCoef;
    trainer.gradClipNorm     = cli.gradClip;      // 0 = 不裁剪（默认）
    // ★★ 学习器：baseline（原行为）/ critic（价值网络 + GAE）
    trainer.mode = (cli.learner == "critic") ? rl::LearningMode::Critic
                                             : rl::LearningMode::Baseline;
    if (cli.valueLr > 0.0)   trainer.valueLr    = cli.valueLr;
    if (cli.gaeLambda >= 0.0) trainer.gaeLambda = cli.gaeLambda;
    trainer.normalizeAdv     = cli.advNormalize;
    // ★★★ 快照每局同步（2026-09-26 深夜）
    //
    //  ★ 为什么改：默认是 100，即队友/敌方读的快照**最多落后 100 局**，
    //    而学习者每局都在更新 —— 负责人报过"队 0 的其余两条蛇
    //    与第一条存在明显行为差异"，根因就是这个。
    //  ⚠️ 游戏内训练模式（LiveTrainingScene）也必须设同一个值，
    //     否则两个入口的行为不一致。**两边都是 1。**
    trainer.mirrorInterval   = 1;

    ckpt::ensureDir();                  // 快照写在 saves/ 下
    if (!trainer.writeMirrorSnapshot()) {
        std::printf("【警告】敌方快照写入失败 —— 敌方会是随机网络\n\n");
    }

    arena::Arena arena(rl::makeArenaConfig(task));

    rl::EpisodeRunner runner(arena, &trainer, task);
    // ★ 学习对象固定是第 0 条（Arena 里第一条出生的蛇）。
    //   默认值就是 0，这里显式写出来是为了让"学哪条"这件事在代码里看得见。
    runner.setLearnerIndex(0);

    // 奖励构成累加器：随机基线 / 全程 / 后半程 / 贪心评估
    RewardBreakdown accRandom, accAll, accTail, accEval;

    // ★ 滑动窗口指标（每 100 局滚动一次）—— "有没有在学 / 有没有锁死"的判据
    RollingWindow roll;
    roll.showCritic = (trainer.mode == rl::LearningMode::Critic);

    const bool dumpObs = cli.dumpObs;

    // -----------------------------------------------------------------------
    //  基线：随机策略
    // -----------------------------------------------------------------------
    if (dumpObs) {
        // ★ 打断点式的诊断：跑一局，把第 1/2/3 帧喂给网络的观测打出来
        std::printf("【观测转储】前 3 帧（每帧 %d 维）\n", rl::OBS_SIZE);
        arena.setController([&](std::size_t i, const arena::Perception& p) {
            return runner.decide(i, p, rng);
        });
        runner.begin(rl::RunMode::Train);

        // -------------------------------------------------------------------
        //  ★★ 圈的时间线追踪（2026-09-26 新增）
        //
        //  ★ 为什么必须按【时间】采样，而不是只看 t=0
        //    负责人报："一个圈一直不产食物，而且一直存在"。
        //    只看开局那一帧**分不出**"这一轮没轮到它"和"它永远不会轮到" ——
        //    必须跨过至少两次轮换（一轮 = 供食 9s + 空窗 5s = 14s）才能判定。
        //
        //  ★ 判据
        //    每一轮结束时打一行：活跃下标 + 每个圈的位置/食物数。
        //    如果**活跃下标一直不变**，或**某个圈的食物数恒为 0**，那就是 bug。
        // -------------------------------------------------------------------
        {
            const float cycleSec = arena.zoneSchedule().cycleDuration();
            std::printf("【圈时间线】每 %.0f 秒（= 一轮）打一次，共 4 轮\n", cycleSec);
            std::printf("  %-8s %-6s %-6s  %s\n", "时刻", "活跃", "下一个", "各圈: 位置 / 食物数");
            int lastCycle = -1;
            for (int f = 0; f < 4 * 60 * 20 && !runner.done(); ++f) {
                runner.tick(1.0f / 60.0f);
                const float zt = arena.food().zoneTimer();
                const int cyc = static_cast<int>(zt / cycleSec);
                if (cyc != lastCycle) {
                    lastCycle = cyc;
                    std::printf("  %6.1fs  %-6d %-6d ",
                                zt, arena.activeZoneIndex(), arena.nextZoneIndex());
                    for (std::size_t k = 0; k < arena.food().zones().size(); ++k) {
                        const auto& z = arena.food().zones()[k];
                        std::printf("| [%zu](%.0f,%.0f)食%d ", k, z.x, z.y, z.foodCount);
                    }
                    std::printf("\n");
                }
            }
            std::printf("\n");
        }

        // -------------------------------------------------------------------
        //  ★★ 观测层验证：「下一个圈」在【什么时候】才对 AI 可见（2026-09-27 定稿）
        //
        //  ★ 负责人定义的规格
        //    "不要一开始就有预告圈，预告圈是在活跃圈产食物**还剩 3s** 后出现的，
        //      然后中间有一定时间 gap，然后再被激活。"
        //
        //  ★ 为什么必须单独验这一条
        //    改动的作用点在**观测编码**上，而上面那段"圈时间线"打的是
        //    **世界状态** —— 世界对，不代表喂给网络的向量里有那 3 维。
        //
        //  ★ 判据（看 [27..29] 这 3 个位置）
        //      预告期之前 → (0.00, 0.00, 1.00) 哨兵 = "没有圈"，**AI 看不到**
        //      预告期之内 → 非零方位 + 距离，**AI 看得到**，才可能"提前出发"
        //
        //    距离那一维用 `min(1, 距离/2000)` 归一化，所以正常值不会是 1.00。
        // -------------------------------------------------------------------
        {
            const float lead = arena.zoneSchedule().previewLeadSeconds;
            const float spaw = arena.zoneSchedule().spawnDuration;
            const int nextBase = rl::ZONE_BEGIN + rl::ZONE_OBS_DIM;
            std::printf("【观测验证】「下一个圈」何时对 AI 可见（obs[%d..%d]）\n",
                        nextBase, nextBase + rl::ZONE_OBS_DIM - 1);
            std::printf("  规格: 供食 %.0fs + 空窗 %.0fs，预告在供食【还剩 %.0fs】时出现\n",
                        static_cast<double>(spaw),
                        static_cast<double>(arena.zoneSchedule().gapDuration),
                        static_cast<double>(lead));
            std::printf("  %-8s %-6s %-8s %-6s   %-22s %s\n",
                        "时刻", "阶段", "预告期", "活跃", "下一个圈 obs[sin,cos,距]", "判定");

            int earlySeen = 0, earlyTotal = 0;   // 预告期【之前】误可见的次数
            int preSeen   = 0, preTotal   = 0;   // 预告期之内可见的次数

            // ★ 自己造一份感知（不复用 runner 的，见下面注释里的坑）
            agent::PerceptionBuilder pb;
            arena::Perception percep;
            // ⚠️ 必须选一条【活着的】蛇：`PerceptionBuilder::build` 在
            //    `!self.isAlive()` 时直接 return，**连圈都不填**（圈是全局信息，
            //    那是个已知的小瑕疵）。上面那版固定用 0 号，而 0 号常常几秒就死了，
            //    于是读到的永远是空圈 —— 又浪费了一轮。
            //    这里每次采样都现场挑第一条活蛇。
            std::size_t selfIdx = 0;

            for (int s = 1; s <= 30; ++s) {
                for (int f = 0; f < 60; ++f) arena.step(1.0f / 60.0f);

                const float zt  = arena.food().zoneTimer();
                const bool  sp  = arena.zoneSchedule().spawning(zt);
                const bool  pre = arena.zoneSchedule().previewing(zt);

                // ★★ 必须【自己现编码一帧】，不能用 runner.lastObservation()
                //
                //  ⚠️ 这里踩过一个坑：`lastObservation()` 是【学习那条蛇最后一次决策】
                //     的观测。而那条蛇常常几秒就撞死了 —— 它一死，后续
                //     `runner.decide()` 不再被调用，那份观测就**冻结在死前那一帧**。
                //     于是整个采样窗口读到的都是同一个哨兵值，
                //     看起来像"可见性改动没生效"，其实是**读数来源错了**。
                //
                //  ★ 正确做法：直接对当前世界跑一次 `build` + `encodeObservation`，
                //    拿到的才是"此时此刻这个世界会喂给网络什么"。
                selfIdx = arena.agents().size();          // 非法值 = "没找到活蛇"
                for (std::size_t k = 0; k < arena.agents().size(); ++k) {
                    if (arena.agents()[k].isAlive()) { selfIdx = k; break; }
                }
                // ⚠️ 找不到活蛇时【必须跳过这次采样】，不能沿用上一局的 percep ——
                //    否则会把"上一局残留的观测"和"这一局的预告期标志"配对，
                //    统计出假的"预告期之前误可见"（踩过一次）。
                const bool haveSample = (selfIdx < arena.agents().size());
                if (!haveSample) continue;

                pb.build(arena.view(), selfIdx, percep);
                double obsNow[rl::OBS_SIZE] = {0};
                rl::encodeObservation(percep, obsNow);

                const double a0 = obsNow[static_cast<std::size_t>(nextBase + 0)];
                const double a1 = obsNow[static_cast<std::size_t>(nextBase + 1)];
                const double a2 = obsNow[static_cast<std::size_t>(nextBase + 2)];
                const bool seen = !(a0 == 0.0 && a1 == 0.0 && a2 == 1.0);

                if (pre) { ++preTotal;   if (seen) ++preSeen; }
                else     { ++earlyTotal; if (seen) ++earlySeen; }

                if (s <= 16) {   // 跨过一次换轮就够判读，免得刷屏
                    std::printf("  %6.1fs  %-6s %-8s %-6d   (%+.2f, %+.2f, %.2f)   %s\n",
                                static_cast<double>(zt), sp ? "供食" : "空窗",
                                pre ? "是" : "否",
                                arena.activeZoneIndex(), a0, a1, a2,
                                seen ? "可见" : "哨兵(不可见)");
                    if (s <= 9) {
                        std::printf("          [内部] view.pre=%d 圈数=%zu",
                                    arena.view().zonePreviewing ? 1 : 0,
                                    percep.zones.size());
                        for (std::size_t k = 0; k < percep.zones.size(); ++k) {
                            std::printf(" z%zu{a=%d v=%d}", k,
                                        percep.zones[k].active ? 1 : 0,
                                        percep.zones[k].visible ? 1 : 0);
                        }
                        std::printf("  活跃下标=%d 下一个下标=%d\n",
                                    percep.activeZoneIndex, percep.nextZoneIndex);
                    }
                }
            }
            std::printf("  --------------------------------------------------\n");
            std::printf("  预告期之前: %d/%d 次采样【误】可见  ← 应当为 0\n",
                        earlySeen, earlyTotal);
            std::printf("  预告期之内: %d/%d 次采样可见\n", preSeen, preTotal);
            std::printf("  判据: 预告期之前必须【全是哨兵】(否则又变成开局就预告)，"
                        "预告期之内必须【全部可见】。\n\n");
        }

        // -------------------------------------------------------------------
        //  ★ 用【随机策略】再跑一次同一段时间线
        //
        //  ★ 为什么必须用随机策略做这件事的对照
        //    学习中的那条蛇常常**几秒就撞死了**，一局根本活不过一轮（14 秒），
        //    于是"圈会不会轮换"这件事在它身上**永远观测不到**。
        //    随机策略同样会死，但下面这个循环不看 runner.done() ——
        //    它只用来看【世界里的圈】随时间怎么变，与蛇的生死无关。
        //    （Arena 在蛇全死之后仍然会被 step 推进，圈调度照常运行。）
        // -------------------------------------------------------------------
        {
            const float cycleSec = arena.zoneSchedule().cycleDuration();
            std::printf("【圈时间线 · 强制推进 60 秒，不看蛇的死活】\n");
            std::printf("  %-8s %-6s %-6s  %s\n", "时刻", "活跃", "下一个", "各圈: 位置 / 食物数");
            int lastCycle = -1;
            for (int f = 0; f < 60 * 60; ++f) {
                arena.step(1.0f / 60.0f);
                const float zt = arena.food().zoneTimer();
                const int cyc = static_cast<int>(zt / cycleSec);
                if (cyc != lastCycle) {
                    lastCycle = cyc;
                    std::printf("  %6.1fs  %-6d %-6d ",
                                zt, arena.activeZoneIndex(), arena.nextZoneIndex());
                    for (std::size_t k = 0; k < arena.food().zones().size(); ++k) {
                        const auto& z = arena.food().zones()[k];
                        std::printf("| [%zu](%.0f,%.0f)食%d ", k, z.x, z.y, z.foodCount);
                    }
                    std::printf("\n");
                }
            }
            std::printf("\n");
        }
        runner.begin(rl::RunMode::Train);
        std::printf("  【逐帧：输入观测 + 网络输出】—— 用来判定「网络有没有在看输入」\n");
        for (int f = 0; f < 3; ++f) {
            runner.tick(1.0f / 60.0f);
            char buf[1024];
            rl::dumpObservation(runner.lastObservation().data(), buf, sizeof(buf));
            std::printf("  t=%d  %s\n", f, buf);

            // ★ 同一帧里把网络的 logits 和 argmax 也打出来
            //    ★ 为什么必须【同一帧】：观测和输出分开看，就分不清
            //      "网络看到了 X 却选了 Y" 还是 "网络根本没看到 X"。
            const auto& lg = trainer.net().forward(
                rl::toInput(runner.lastStackedInput().data()));
            const int am = nn::argmax(lg);
            static const char* kAct[6] = {
                "左转", "直行", "右转", "左转+加速", "直行+加速", "右转+加速"
            };
            double zmax = lg.a[0], zmin = lg.a[0];
            for (int i = 1; i < rl::ACT_SIZE; ++i) {
                if (lg.a[i] > zmax) zmax = lg.a[i];
                if (lg.a[i] < zmin) zmin = lg.a[i];
            }
            std::printf("       logits:");            for (int i = 0; i < rl::ACT_SIZE; ++i) std::printf(" %+7.3f", lg.a[i]);
            std::printf("   argmax=%d(%s)  幅度=%.3f\n", am, kAct[am], zmax - zmin);
            if (runner.done()) { std::printf("       （本局已结束）\n"); break; }
        }
        std::printf("  说明（29 维布局）：前 %d 维=障碍射线（墙|敌身|敌头|自己） | "
                    "接着 %d 维=食物扇区 | 4 维=速度/长度/朝向 | "
                    "每 3 维一个圈(方位sin/cos, 是否轮到它供食) | 最后 3 维=队友\n\n",
                    rl::OBSTACLE_DIM, rl::FOOD_SECTOR_DIM);

        // -------------------------------------------------------------------
        //  ★ 圈诊断（2026-09-26 新增）
        //
        //  ★ 为什么要打这个
        //    负责人报："一局开始有两个食物圈，其中一个【一直不产食物】且一直存在"。
        //
        //    按设计，"下一个圈"确实提前出现、且本轮不供食 —— 但**它应该轮换**。
        //    所以必须能在【世界那个时刻】直接看到：
        //        · 世界里到底有几个圈
        //        · 哪个下标是"当前活跃"、哪个是"下一个"
        //        · 每个圈里实际有多少食物
        //        · 圈的半径、归一化布局是否是随机的
        //    只看观测向量是分不出"没轮到"和"永远不会轮到"的。
        //
        //  ⚠️ 这一段只在 --dump-obs 分支里跑，**不参与任何世界推进** ——
        //     所以不会影响 replay_check 的指纹。
        // -------------------------------------------------------------------
        {
            const auto& zs = arena.food().zones();
            const int n  = static_cast<int>(zs.size());
            const int ac = arena.activeZoneIndex();
            const int nx = arena.nextZoneIndex();
            const float zt = arena.food().zoneTimer();
            const auto& sched = arena.zoneSchedule();

            std::printf("【圈诊断】世界里 %d 个圈 | 当前活跃下标 %d | 下一个下标 %d\n", n, ac, nx);
            std::printf("  调度: randomOrder=%s  randomPosition=%s | 供食 %.0fs + 空窗 %.0fs = 一轮 %.0fs\n",
                        sched.randomOrder ? "是" : "否",
                        arena.config().zoneLayout.randomPosition ? "是" : "否",
                        sched.spawnDuration, sched.gapDuration, sched.cycleDuration());
            std::printf("  圈计时器 t=%.2fs（本轮已过 %.2fs）| 累计轮次 = %d\n",
                        zt, sched.elapsedInCycle(zt),
                        static_cast<int>(zt / sched.cycleDuration()));
            for (int i = 0; i < n; ++i) {
                const auto& z = zs[static_cast<std::size_t>(i)];
                std::printf("   圈[%d] pos(%.0f, %.0f) r=%.0f 食物数=%d %s%s\n",
                            i, z.x, z.y, z.radius, z.foodCount,
                            (i == ac) ? "[当前活跃]" : "",
                            (i == nx) ? "[下一个]" : "");
            }
            // ★ 关键：把"接下来几轮会轮到谁"算出来（用唯一的调度实现）
            std::printf("  接下来 6 轮的活跃圈下标: ");
            for (int k = 0; k < 6; ++k) {
                const float tk = zt + static_cast<float>(k) * sched.cycleDuration();
                std::printf("%d ", sched.activeZone(tk, n));
            }
            std::printf("\n");
            if (n > 0) {
                int distinct = 0;
                bool seen[16] = { false };
                for (int k = 0; k < 16 && k < 16; ++k) {
                    const float tk = zt + static_cast<float>(k) * sched.cycleDuration();
                    const int zi = sched.activeZone(tk, n);
                    if (zi >= 0 && zi < 16 && !seen[zi]) { seen[zi] = true; ++distinct; }
                }
                std::printf("  ★ 未来 16 轮里出现过的不同圈数 = %d", distinct);
                if (distinct <= 1 && n > 1) {
                    std::printf("   <<< ⚠️ 只用到 1 个圈！另一个圈【永远不会供食】\n");
                } else {
                    std::printf("   （正常：轮换到了多个圈）\n");
                }
            }
            std::printf("\n");
        }
        arena.setController(nullptr);
        return 0;
    }

    {
        double sumR = 0, sumT = 0, sumE = 0; int deaths = 0;
        const int N = 200;
        for (int i = 0; i < N; ++i) {
            const auto s = runEpisode(arena, runner, trainer, rng, rl::RunMode::Random);
            sumR += s.totalReward; sumT += s.ticks; sumE += s.eaten;
            if (s.died) ++deaths;
            accRandom.add(s);          // ★ 随机基线也要能归因
        }
        std::printf("【随机策略基线】%d 局\n", N);
        std::printf("  平均回报 %8.2f   平均存活 %6.1f 帧   平均吃 %4.2f 个   死亡率 %.0f%%\n\n",
                    sumR / N, sumT / N, sumE / N, 100.0 * deaths / N);
    }
    accRandom.print("随机策略基线");

    std::printf("网络参数 %d 个\n\n", trainer.paramCount());

    // -----------------------------------------------------------------------
    //  训练数据记录（给游戏里的「训练数据」界面用）
    // -----------------------------------------------------------------------
    MetricsWriter metrics;
    if (!metrics.open("train_metrics.csv")) {
        std::printf("【警告】无法写入 train_metrics.csv —— 训练照跑，但【没有曲线可看】\n");
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::printf("  局数     平均回报    均存活    均吃   基线     用时\n");
    std::printf("  ---------------------------------------------------------\n");

    // ★ 2026-09-26：统计"轨迹被缓冲上限截断"的局数（缺陷 #46 的看门狗）
    //
    //   ★ 为什么要有它：截断是【静默】的 —— 不崩不报错，现象只是"学不会"，
    //     历史上因此一直没被发现（90 秒的局只学了前 4096 帧）。
    //     上限已提到 8192，这个计数器保证"再截断一定看得见"。
    int truncatedEpisodes = 0;

    // =======================================================================
    //  ★★★ 全程累计的【实际执行动作】直方图（2026-09-28 新增）
    //
    //  ★ 为什么必须有它 —— 滑窗是每 100 局【清零重算】的
    //    于是"某一项长期偏低"这种【系统性】偏差会被窗口掩盖：
    //    如果每个窗口里"直行"都只有 19%，那说明**从头到尾就没学会直行**，
    //    而不是"最近 100 局偏低"。这两种情况的含义完全不同。
    //
    //  ★ 判据（用 `ahead` 任务时尤其清楚）
    //    那个任务把食物放在蛇的【正前方】，最优策略就是【一直直行】。
    //    如果全程直行占比远低于随机水平的 1/3，
    //    说明策略结构性地做不到"该直行时直行"。
    // =======================================================================
    long globalAct[rl::ACT_SIZE]       = {};
    long globalGreedyAct[rl::ACT_SIZE] = {};

    for (int ep = 1; ep <= EPISODES; ++ep) {
        // ★ 只在【前 5 局】打印"开局世界长什么样"（见 runEpisode 里那段）
        //
        //  ★ 为什么需要它（负责人要求"每局种子不同"）
        //    "每局换种子"这件事没有别的外部痕迹：种子换了、世界长得不一样，
        //    但日志里的回报/存活帧数看不出**原因**。
        //    把它打出来，"是不是真的每局都换了"一眼可验。
        //  ★ 种子公式：`spawnSeed + reset 次数`（见 Arena::beginRoundSetup 的说明），
        //    所以同一个任务种子跑出来的"第 k 局"永远相同 —— 对照实验仍可比。
        //  ⚠️ 开关【只在这里设】。以前在循环末尾又设了一次 true，
        //     结果它一直是 true，把每一局都打了出来（踩过一次）。
        gDumpEpisodeWorld = (ep <= 5);

        const auto s = runEpisode(arena, runner, trainer, rng, rl::RunMode::Train);
        if (trainer.lastEpisodeTruncated) ++truncatedEpisodes;

        // ★ 全程动作直方图（不清零）
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            globalAct[i]       += s.actCount[i];
            globalGreedyAct[i] += s.greedyActionCount[i];
        }

        metrics.row("train", ep, s.totalReward,      "reward");
        metrics.row("train", ep, s.ticks,            "survival");
        metrics.row("train", ep, s.eaten,            "food");
        metrics.row("train", ep, s.died ? 1 : 0,     "died");
        metrics.row("train", ep, trainer.baseline(), "baseline");
        // ★ 步长诊断：梯度范数 / 参数被移动的距离 / 参数自身范数
        metrics.row("train", ep, trainer.lastGradNorm,   "gradNorm");
        metrics.row("train", ep, trainer.lastUpdateNorm, "updateNorm");
        metrics.row("train", ep, trainer.lastParamNorm,  "paramNorm");
        // ★ 熵项单独的梯度范数 —— 用来判断"策略是被任务推着走，还是被熵项推着走"
        metrics.row("train", ep, trainer.lastEntropyGradNorm, "entGradNorm");
        // ★★★ 2026-09-27：ε-greedy 随机步的排除是否真的生效
        //
        //  ★ 为什么必须记这两个数（不是"顺便多打两个"）
        //    这次修的是"随机抽出来的动作被当成策略自己的选择去更新"。
        //    如果标志位没传对，**现象和修之前一模一样（学不会）**，
        //    无法从回报曲线上区分"修好了"和"根本没修上"。
        //    所以这个修复必须自带一个可读的判据：
        //
        //        policySteps + exploreSteps == 本局步数(T)
        //        exploreSteps / (两者之和)  ≈ explorationEps (0.05)
        //
        //      比例 ≈ 5%   → 排除逻辑正确
        //      比例 == 0%  → 标志没传进来（修复没生效）
        //      比例 == 100%→ 条件写反了（把策略步全排除了，学不到东西）
        metrics.row("train", ep, trainer.lastPolicySteps,      "policySteps");
        metrics.row("train", ep, trainer.lastExplorationSteps, "exploreSteps");

        // ★ 累加【奖励构成】—— 用来回答"总回报到底是哪一项贡献的"
        accAll.add(s);
        if (ep > EPISODES / 2) accTail.add(s);   // 后半程，看"学到了什么"

        // ★ 累加滑动窗口 —— 用来回答"最近 100 局到底在变好还是锁死"
        roll.add(s);
        roll.addStep(trainer.lastGradNorm, trainer.lastUpdateNorm, trainer.lastParamNorm);
        roll.addEntGrad(trainer.lastEntropyGradNorm);
        if (trainer.mode == rl::LearningMode::Critic) {
            roll.addCritic(trainer.lastValueCorr, trainer.lastValueLoss, trainer.lastAdvStd);
        }

        // ★ 守卫：非有限的回报会顺着累加梯度污染权重，最后变成 nan 网络。
        if (!std::isfinite(trainer.lastReturn()) || !std::isfinite(trainer.baseline())) {
            std::printf("【警告】第 %d 局出现非有限值：return=%.17g baseline=%.17g\n",
                        ep, trainer.lastReturn(), trainer.baseline());
            trainer.reset(task.seed);
        }

        if (ep % 100 == 0 || ep == 1) {
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            // ★ 带上 policy 概率 —— 判断"有没有在学"的最直接窗口。
            //   三项都 ≈ 0.333 说明梯度没起作用；某项占优说明学到了。
            std::printf("  %5d   %9.2f   %7.1f   %5.2f  %8.2f   %5.0fs   "
                        "policy %.2f %.2f %.2f\n",
                        ep, s.totalReward, static_cast<double>(s.ticks),
                        static_cast<double>(s.eaten), trainer.baseline(), secs,
                        runner.probL(), runner.probF(), runner.probR());
        }

        // ★★ 每满 100 局：输出一次滑动窗口指标，并写进 CSV
        //
        //  ★ 为什么是"满 100 就清空重来"而不是"每局都算最近 100 局"
        //    后者要在每一局都输出一行，CSV 会膨胀 100 倍，而曲线看得见的
        //    信息量完全一样。分段窗口是这里最省的等价做法。
        if (roll.full()) {
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            roll.print(ep, secs);
            roll.writeRows(metrics, ep);
            roll.clear();
        }
    }

    // -----------------------------------------------------------------------
    //  ★★★ 全程动作直方图（不清零）—— 看某个动作是否被【系统性】回避
    //
    //  ★ 为什么滑窗不够
    //    滑动窗口每 100 局就【清零重算】。如果每个窗口里"直行"都只有 19%，
    //    那不可能是偶然 —— 说明**从头到尾就没有直行过**。
    //    而窗口读数只会显示"这一窗偏低"，看不出"系统性"。
    // -----------------------------------------------------------------------
    {
        long totA = 0, totG = 0;
        for (int i = 0; i < rl::ACT_SIZE; ++i) { totA += globalAct[i]; totG += globalGreedyAct[i]; }
        // ★ 表按【6 个动作】定义（固定长度），只用到前 ACT_SIZE 个。
        //   回滚后 ACT_SIZE = 3，所以"加速"那三项目前不会被打印。
        //   保留它们是为了以后把 ACT_SIZE 改回 6 时这里不用动。
        static const char* kName[6] = {
            "左转", "直行", "右转", "左转+加速", "直行+加速", "右转+加速"
        };
        std::printf("\n【全程动作直方图】（%ld 帧，跨全部 %d 局，【不清零】）\n", totA, EPISODES);
        std::printf("   %-12s %11s %8s | %11s %8s\n", "动作", "实际执行", "占比", "贪心选择", "占比");
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            std::printf("   %-12s %11ld %7.2f%% | %11ld %7.2f%%\n",
                        kName[i], globalAct[i],
                        totA > 0 ? 100.0 * static_cast<double>(globalAct[i]) / static_cast<double>(totA) : 0.0,
                        globalGreedyAct[i],
                        totG > 0 ? 100.0 * static_cast<double>(globalGreedyAct[i]) / static_cast<double>(totG) : 0.0);
        }
        long st[3] = { 0, 0, 0 }, gs[3] = { 0, 0, 0 };
        for (int i = 0; i < rl::ACT_SIZE; ++i) {
            st[rl::steerIndexOfAction(i)] += globalAct[i];
            gs[rl::steerIndexOfAction(i)] += globalGreedyAct[i];
        }
        std::printf("   ---- 转向档位（合并两个速度档，与面板口径一致）----\n");
        const char* sn[3] = { "左转", "直行", "右转" };
        for (int i = 0; i < 3; ++i) {
            std::printf("   %-12s %11ld %7.2f%% | %11ld %7.2f%%\n", sn[i],
                        st[i], totA > 0 ? 100.0 * static_cast<double>(st[i]) / static_cast<double>(totA) : 0.0,
                        gs[i], totG > 0 ? 100.0 * static_cast<double>(gs[i]) / static_cast<double>(totG) : 0.0);
        }
        std::printf("   ⚠️ 随机水平 = 33.33%%。某一档【长期】远低于它 = 策略结构性回避该动作。\n");
    }

    // -----------------------------------------------------------------------
    //  ★ 看门狗：轨迹有没有被缓冲上限截断（缺陷 #46）
    //
    //  ★ 为什么必须显式打印而不是"没消息就是好消息"
    //    截断是静默的，历史上正因如此一直没被发现 ——
    //    90 秒的局只学了前 4096 帧，尾部 24% 的奖励连同死亡罚全丢。
    //    这里明确报"一局都没截断"，才算真的验证过。
    // -----------------------------------------------------------------------
    if (truncatedEpisodes > 0) {
        std::printf("\n【严重警告】%d / %d 局的轨迹被上限截断（maxStepsPerEpisode = %d）。\n"
                    "          这些局的尾部奖励【没有参与学习】—— 请把 maxStepsPerEpisode 调大。\n",
                    truncatedEpisodes, EPISODES, trainer.maxStepsPerEpisode);
    } else {
        std::printf("\n[看门狗] 轨迹截断：0 / %d 局（maxStepsPerEpisode = %d，上限充足）\n",
                    EPISODES, trainer.maxStepsPerEpisode);
    }

    // -----------------------------------------------------------------------
    //  训练后评估（贪心，不再采样）
    // -----------------------------------------------------------------------
    {
        std::printf("\n训练结束，用【贪心策略】评估 100 局：\n");
        double sumR = 0, sumT = 0, sumE = 0; int deaths = 0;
        const int N = 100;
        for (int i = 0; i < N; ++i) {
            const auto s = runEpisode(arena, runner, trainer, rng, rl::RunMode::Eval);
            sumR += s.totalReward; sumT += s.ticks; sumE += s.eaten;
            if (s.died) ++deaths;

            metrics.row("eval", i + 1, s.totalReward, "reward");
            metrics.row("eval", i + 1, s.ticks,       "survival");
            metrics.row("eval", i + 1, s.eaten,       "food");
            accEval.add(s);
        }
        std::printf("  平均回报 %8.2f   平均存活 %6.1f 帧   平均吃 %4.2f 个   死亡率 %.0f%%\n",
                    sumR / N, sumT / N, sumE / N, 100.0 * deaths / N);
    }
    accEval.print("训练后 · 贪心评估");

    // -----------------------------------------------------------------------
    //  ★ 奖励构成汇总 —— "总回报到底是哪一项贡献的"
    //
    //  ★ 为什么必须打这个
    //
    //  实测：ZoneMini 任务里**连随机策略都能拿到 613 的平均回报**，
    //  而它平均只吃 0.01 个食物。只看总回报会得出"训练有效"的错误结论
    //  （631 > 613），实际上两者的区别只是"存活帧数"，
    //  而且有某一项在【主导】整个回报 —— 必须点名它是谁。
    // -----------------------------------------------------------------------
    accAll.print("训练全程");
    accTail.print("训练后半程");

    metrics.close();

    trainer.net().save("train_probe_weights.txt");
    std::printf("\n权重已存到 train_probe_weights.txt\n");
    std::printf("训练数据已存到 train_metrics.csv（游戏里从主菜单进「训练数据」可以看曲线）\n");
    std::printf("======================================================================\n");
    return 0;
}
