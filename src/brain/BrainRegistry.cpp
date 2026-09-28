#include "BrainRegistry.h"
#include "RuleBrain.h"
#include "HumanBrain.h"
#include "NeuralBrain.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

namespace arena {

namespace {

// ---------------------------------------------------------------------------
//  ★ 造一个"加载冻结快照"的神经网络（2026-09-26 深夜抽出，消除重复）
//
//  ★ 为什么抽成【文件作用域的普通函数】而不是函数内的 lambda
//    "nn-enemy"（对手）和 "nn-mate"（队友）必须读同一个文件、走同一条
//    加载路径，只有语义不同。写成两份的话，以后改快照路径或改消费逻辑
//    时必然只改一处 —— 这个项目已经因为"同一个规则有两份实现"栽过三次
//    （观测编码分叉、世界推进两份、画圈三份）。
//
//    ⚠️ 但**不能写成 `registerBuiltinBrains()` 里的局部 lambda**：
//       `reg.reg()` 的工厂参数是 `std::function<...>`，而传进去的
//       `[] { return makeSnapshotBrain(...); }` 是个**没有捕获**的 lambda，
//       它引用不到外层那个局部变量 —— 编译报
//       "`makeSnapshotBrain` is not captured"。
//       放到文件作用域就没有这个问题：它不依赖任何局部状态。
//       （这也是它比"给 lambda 加 [&] 捕获"更好的地方：不引入生命周期疑问。）
//
//  ★ 那个"设完必须清空"是干嘛的（别删）
//    `g_pendingEnemySnapshot` 是【全局】的，而 NeuralBrain 构造函数
//    一见它非空就加载快照并直接 return。
//    生产完立刻清空，否则**之后创建的每一个 NeuralBrain 都会去读快照** ——
//    包括学习中的那条蛇（缺陷 #43，见 devlogs/2026-09-26.md 5.7 节）。
//
//  ★ 为什么每次创建前都要重设
//    因为上一次调用刚把它清空了。而 `Arena::reset()` 每局都会重新创建全部
//    大脑，所以**每局都会重新读一次文件** —— 这正是"每局同步最新权重"的
//    实现方式：同步点就是这里，不需要额外的同步代码。
// ---------------------------------------------------------------------------
std::unique_ptr<Brain> makeSnapshotBrain(const char* snapshotPath) {
    NeuralBrain::setPendingEnemySnapshot(snapshotPath);
    std::unique_ptr<Brain> b(new NeuralBrain());
    NeuralBrain::setPendingEnemySnapshot(std::string());   // ★ 消费完必须清
    return b;
}

}  // namespace

BrainRegistry& BrainRegistry::instance() {
    // 函数内静态变量：C++11 起保证线程安全且只构造一次，
    // 天然避免了"静态初始化顺序"问题
    static BrainRegistry s_instance;
    return s_instance;
}

void BrainRegistry::reg(const BrainInfo& info,
                        std::function<std::unique_ptr<Brain>()> factory) {
    // id 已存在 → 覆盖旧的（改算法时不必先删注册，也方便反复实验）
    auto it = std::find_if(m_infos.begin(), m_infos.end(),
                           [&](const BrainInfo& i) { return i.id == info.id; });
    if (it != m_infos.end()) {
        const std::size_t idx = static_cast<std::size_t>(it - m_infos.begin());
        m_infos[idx]     = info;
        m_factories[idx] = std::move(factory);
        return;
    }
    m_infos.push_back(info);
    m_factories.push_back(std::move(factory));
}

bool BrainRegistry::has(const std::string& id) const {
    return std::any_of(m_infos.begin(), m_infos.end(),
                       [&](const BrainInfo& i) { return i.id == id; });
}

std::unique_ptr<Brain> BrainRegistry::create(const std::string& id) const {
    for (std::size_t i = 0; i < m_infos.size(); ++i) {
        if (m_infos[i].id == id) {
            return m_factories[i] ? m_factories[i]() : nullptr;
        }
    }
    return nullptr;
}

std::unique_ptr<Brain> BrainRegistry::createWithWeights(
        const std::string& id, const std::string& weightsPath) const {
    // ⚠️⚠️ 这个函数以前是个【空壳】：它 `(void)weightsPath;` 之后直接调 create(id)，
    //      也就是说 weightsPath 参数**被丢掉了**，调用方以为指定了权重、其实没有。
    //
    //      更糟的是它是一个"看起来能用"的接口 —— 我照着它去实现权重选择，
    //      绕了很久才发现真正生效的路径是 NeuralBrain 内部的静态待定值。
    //
    //  ★ 现在如实转发到那条真正的路径：
    //      NeuralBrain 的工厂会读 `NeuralBrain::pendingWeights()`。
    //      所以这里把参数写进待定值，再 create —— 语义与实现终于一致。
    //
    //  ★ 注意它的副作用：pendingWeights 是【全局】的，所以这会改掉
    //      "所有用 nn 的队伍"的权重。当前架构下这没问题（一张网络所有队共用），
    //      但不要用它去做"每队不同权重"——那需要每队一份网络，是另一件事。
    if (id == "nn") {
        arena::NeuralBrain::setPendingWeights(weightsPath);
    }
    return create(id);
}

void BrainRegistry::clear() {
    m_infos.clear();
    m_factories.clear();
}

// ===========================================================================
//  内置算法注册
//
//  新增算法的唯一改动点就是这里加一段 reg(...) —— UI 代码一行都不用改。
// ===========================================================================
void registerBuiltinBrains() {
    auto& reg = BrainRegistry::instance();

    // ---- 规则 AI（基线）----
    // 已经改写成"只吃 Perception"，和神经网络看到的世界完全一样
    //
    // ⚠️ displayName 会被画到界面上。这里【只用 ASCII 括号】，不用全角（）
    //    —— SimHei 是 GB2312 时代的字体，全角括号和箭头这类字符可能没有字形，
    //    画出来就是一串方块。
    reg.reg(RuleBrain{}.info(),
            [] { return std::unique_ptr<Brain>(new RuleBrain()); });

    // ---- 玩家（鼠标控制）----
    // 玩家也是一个算法。可视化对局里把它挂给某一队的某个 agent，
    // 训练模式下完全不创建它 —— 这样训练和观战共用同一套世界代码。
    reg.reg(HumanBrain{}.info(),
            [] { return std::unique_ptr<Brain>(new HumanBrain()); });

    // ---- 神经网络（REINFORCE 训练出来的策略）----
    // 构造一次只为取 info()：displayName 会如实反映"有没有找到权重文件"，
    // 所以对局设置界面上一眼就能看出是不是真加载上了
    reg.reg(NeuralBrain{}.info(),
            [] { return std::unique_ptr<Brain>(new NeuralBrain()); });

    // -----------------------------------------------------------------------
    //  ---- ★ 神经网络 · 敌方快照（自对弈用）----
    //
    //  ★ 负责人要求："我想把敌人也改成神经网络"
    //
    //  ★ 它和 "nn" 的区别
    //      "nn"        —— 加载【用户选的/自动找到的】权重（我方用）
    //      "nn-enemy"  —— 加载【训练器定期写的冻结快照】（敌方用）
    //
    //  ★ 为什么必须分成两个 id，而不是共用 "nn"
    //    两者要读的文件不同，而且【语义完全不同】：
    //      我方网络是学习中的那条（在训练模式下）；
    //      敌方网络是它的一份【过去某时刻的拷贝】，只决策不学习。
    //    共用一个 id 就没法在界面上区分，也没法给它们不同的加载路径。
    //
    //  ★ 为什么快照路径是写死的字符串
    //    Trainer（rl/ 层）和 NeuralBrain（brain/ 层）都不能 include 对方 ——
    //    分层铁律。所以约定一个文件名，两边各自写一遍字面量，
    //    并在两处都注明"要和对方保持一致"。这比加一条跨层依赖好。
    // -----------------------------------------------------------------------
    reg.reg(BrainInfo{"nn-enemy", "神经网络 · 敌方快照",
                      "加载训练器定期同步的冻结权重（自对弈的对手）",
                      "REINFORCE", true, true},
            [] {
                // ⚠️ 这个路径必须和 Trainer::mirrorPath 一致（两边不能互相 include）
                return makeSnapshotBrain("saves/enemy_snapshot.txt");
            });

    // -----------------------------------------------------------------------
    //  ---- ★ 神经网络 · 队友快照（2026-09-26 深夜，负责人要求）----
    //
    //  ★ 负责人原话
    //    "让队友每一局都同步最新的权重，和另一名队友一样。"
    //
    //  ★ 它解决的是什么问题
    //    队 0 有两条蛇，但 `teamAlgorithms` 是【按队】配的，所以两条都挂 "nn"。
    //    而 "nn" 走的是 `weights/policy.txt` / `train_probe_weights.txt`
    //    —— 这两个文件训练时**根本不写**。于是队友要么加载失败变成随机蛇，
    //    要么加载到一份与训练无关的旧权重，而且**整场训练都不再变**。
    //    它却是个真实实体：能抢食物、能挡路、能撞死学习中的那条。
    //    **一个"不受奖励引导、却持续影响学习者"的实体**是纯噪声来源。
    //
    //  ★ 改成读快照之后的行为
    //    `Arena::reset()` 每局重建大脑 → 每次都重读快照文件 →
    //    队友每局都拿到【上一局结束时】的最新权重，和敌方完全同步。
    //    ⚠️ 仍然是"每局一次"，不是"本局内实时" —— 因为 Trainer 更新的是
    //      它自己那份 `m_net`，局内不会回写文件。这与负责人"每一局都同步"
    //      的要求一致。
    //
    //  ★ 为什么用独立的 id 而不是让队 0 直接挂 "nn-enemy"
    //    界面上要能区分"队友"和"对手"；而且两者的语义不同
    //    （一个是我方、一个是敌方），共用一个 id 以后没法分别调整。
    //    ⚠️ 代价：两条蛇读的是【同一个文件】，所以局内它们权重完全相同 ——
    //      这既是"同步"的字面效果，也意味着它们会高度相关地行动。
    // -----------------------------------------------------------------------
    reg.reg(BrainInfo{"nn-mate", "神经网络 · 队友快照",
                      "加载训练器同步的最新权重（队友，每局同步一次）",
                      "REINFORCE", true, true},
            [] {
                return makeSnapshotBrain("saves/enemy_snapshot.txt");
            });

    // 调试用：把当前注册表打印出来，方便确认链路通了
    if (!reg.list().empty()) {
        std::cout << "[arena] 已注册的算法：" << std::endl;
        for (const auto& b : reg.list()) {
            std::cout << "  - " << b.id << " : " << b.displayName << std::endl;
        }
    }
}
}  // namespace arena
