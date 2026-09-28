#pragma once

// ============================================================================
//  Brain.h — AI 算法必须实现的接口
//
//  这是"游戏与算法分离"的接缝。任何算法（规则、神经网络、你自己的）都只需要：
//    · 读一个 const Perception&
//    · 返回一个 Action
//  它完全不需要知道 Snake、GameScene、SFML 的存在。
//
//  ★ 分离的验收判据：把整个 src/brain/ 删掉，src/sim/ 必须照样能编译。
// ============================================================================

#include "Perception.h"
#include "Action.h"

#include <cstdint>
#include <memory>
#include <string>

namespace arena {

// ---------------------------------------------------------------------------
//  算法的自描述信息
//
//  UI、存档、跑分记录都只认 id，所以【新增算法不需要改任何 UI 代码】。
// ---------------------------------------------------------------------------
struct BrainInfo {
    std::string id;             // 唯一标识，如 "rule"、"nn-survivor"
    std::string displayName;    // 界面上显示的名字，如 "规则 AI（基线）"
    std::string description;    // 一句话说明
    std::string author;         // 可选
    bool needsWeights = false;  // 是否需要加载权重文件（神经网络 true）
    bool trainable    = false;  // 是否参与学习
};

// ---------------------------------------------------------------------------
//  算法基类
// ---------------------------------------------------------------------------
class Brain {
public:
    virtual ~Brain() = default;

    // ---- 必须实现 ----

    virtual const BrainInfo& info() const = 0;

    // 看感知，出动作。每 tick 调用一次（智能体层可能按决策频率节流）
    virtual Action think(const Perception& p) = 0;

    // ---- 生命周期钩子（按需重写）----

    // 一局开始时调用。神经网络用它来清理回合状态、重置随机数
    virtual void onEpisodeStart(std::uint32_t /*seed*/) {}

    // 自己死亡时调用
    virtual void onDeath() {}

    // 一局结束时调用。teamReward 是本队这一局的综合得分
    virtual void onEpisodeEnd(double /*teamReward*/) {}

    // ---- 训练钩子（只有可学习算法需要重写）----

    // 智能体层把这一步的经验交回算法；学不学、什么时候学，由算法自己决定。
    // 训练模式的每 tick 调用一次。
    virtual void observeStep(const Perception& /*p*/,
                             const Action& /*a*/,
                             double /*reward*/) {}

    // 整场对抗结束（跑分模式下可能跑了 N 局）
    virtual void onMatchEnd() {}

    // ---- 工具 ----

    virtual bool isTrainable() const { return info().trainable; }
};

// ---------------------------------------------------------------------------
//  Brain 实例的生命周期约定
//
//  采用【per-snake 实例 + 共享模型】：
//    · 同队的每条蛇各自持有一个 Brain 对象 —— 这样每条蛇可以有独立的记忆状态，
//      将来换成 RNN 不需要改架构
//    · 但内部网络权重通过 std::shared_ptr<Model> 共享 —— 同队学的是同一个策略
//
//  所以 Brain 的构造函数通常长这样：
//      NeuralBrain(std::shared_ptr<const Policy> sharedPolicy, uint32_t seed);
// ============================================================================

}  // namespace arena
