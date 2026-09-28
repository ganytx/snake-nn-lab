#pragma once

// ============================================================================
//  RuleBrain.h — 规则 AI（优先级行为树）
//
//  ★ 输入是【只吃 arena::Perception】——
//    它拿到的信息和神经网络一模一样，没有上帝视角。
//    否则"规则 AI vs 神经网络"这个比较毫无意义。
//
//  ★★ 决策优先级（2026-09-26 按负责人要求重排）
//
//      0. 保命      躲障碍（墙 / 蛇身）—— 最高优先级，命中即返回
//      1. 找食物    视野里的食物 / 尸体（近了才看得见，视野 ≈512px）
//      2. 找圈      ★ 新增：视野里没食物时，朝【当前活跃圈】走
//                   当前圈在空窗期就朝【下一个圈】走（提前出发）
//      3. 攻击敌人  ★ 新增：只有在"我占优势"时才动手（见 canHunt）
//      4. 漫游      以上都不成立时随机走
//
//  ★ 为什么必须加"找圈"（这是本次重写的核心）
//
//    旧版只有【躲障碍 > 找食物 > 漫游】。而食物只在圈里生成，
//    视野半径 ≈512px，食物圈半径 150px —— 蛇只要不在圈附近，
//    视野里就【什么都没有】，于是它只能"漫游"。
//    结果就是**靠运气撞到圈**，表现成"傻"。
//
//    ★ 关键：`Perception.zones` 一直提供**全局**的圈方位与距离
//      （`FoodZoneView::bearing` / `distance`，不受视野限制）——
//      也就是说"该往哪走"这个信息**一直都在**，只是旧版从来没用过它。
//      这不是加信息，是把已有信息接上。
//
//  ★ 为什么"攻击敌人"排在最后，而且要加优势判定
//
//    规则是「**我的头**碰到**对方身体**才死」。所以"靠近敌人"这件事
//    对主动方是**危险**的。无条件冲上去等于自杀。
//    `canHunt()` 要求同时满足两条才动手：
//        · 我的头离对方身体足够远（我还安全）
//        · 对方的头离我的身体足够近（它更危险）
//    这正是"围猎"的几何条件 —— 用身体别住它，同时把头让开。
//
//  ★ 本文件禁止 include 任何 SFML 头文件（算法层铁律）
// ============================================================================

#include "Brain.h"

#include <cstdint>

namespace arena {

class RuleBrain : public Brain {
public:
    RuleBrain();

    const BrainInfo& info() const override;
    Action think(const Perception& p) override;
    void   onEpisodeStart(std::uint32_t seed) override;

    // 重新随机反应时间与漫游方向（换一局/重生时用）
    void reset();

    // 走一遍完整的行为树，返回这一轮决策的动作。
    //
    // ⚠️ 注意：think() 会按 m_reactionTime 节流，未到反应时间就直接沿用上一次的动作
    //    （模拟生物的反应延迟）。想【无条件】重算一次，调这个函数。
    //    bench_sim 用它量"决策成本的上界"。
    Action decide(const Perception& p);

private:
    // 优先级 0：用网格射线判断前方是否危险，需要转向就填进 action
    // 返回 true 表示"已经决定了，不用再考虑别的"
    bool avoidObstacles(const Perception& p, Action& action);

    // 优先级 1：在视野网格里找最近的食物 / 尸体，算出转向量
    // 返回 false 表示视野里没有食物
    bool seekFood(const Perception& p, Action& action);

    // ★★ 优先级 2：圈内搜索（2026-09-26 晚新增）
    //
    //  ★ 为什么单独需要它
    //    `seekFood` 只在视野网格（≈512px）里找。而食物在【圈里随机位置】生成、
    //    圈半径只有 150px —— 蛇进了圈之后很可能"附近没食物，但圈的另一侧有"，
    //    于是 seekFood 返回 false。旧逻辑这时会掉到"找圈"，
    //    而找圈看的是【圈心方位】→ 走到圈心 → 没食物 → 继续抖 → 在圈里打转。
    //
    //  ★ 做法：朝圈的【切线】方向走（而不是朝圈心），沿圈扫一圈。
    //    返回 false 表示"我不在任何圈里"或"这个圈已经空了"。
    bool scrubZone(const Perception& p, Action& action);

    // 优先级 3：朝食物圈走（★ 2026-09-26 新增，含【提前出发】）
    //
    //   ★ 2026-09-26 晚改进：加了【提前出发】
    //     旧逻辑只在"当前圈空窗了"才切下一个圈 —— **开圈那一刻它才出发，永远迟到**。
    //     现在按"供食期还剩多少 + 走过去要多久"提前决策。
    //
    //   返回 false 表示世界里没有圈（比如 Scatter 任务）
    bool seekZone(const Perception& p, Action& action);

    // 优先级 4：攻击敌人（★ 2026-09-26 新增）
    //
    //   只在【我占优势】时动手，条件见 canHunt()。返回 true 表示已决定动作。
    bool attackEnemy(const Perception& p, Action& action);

    // "我是否有围猎优势"的几何判定（见 .cpp 里的详细说明）
    bool canHunt(const Perception& p) const;

    // ★★ 是否该加速（2026-09-26 晚新增）
    //
    //  ★ 为什么规则 AI 以前从不加速，而这是个明显的浪费
    //      加速把速度从 250 提到 450 px/s（1.8 倍），而且**当前实现里没有任何代价**
    //      （`Snake.h` 自己的注释写着"体节消耗待后续实现" —— 还没实现）。
    //      附带好处：体节沿轨迹按固定间距采样，加速时轨迹点变稀疏 →
    //      **身体被拉长、更松散 → 更容易别住敌人**。
    //
    //  ★ 为什么不能无脑一直加速
    //      速度越快，反应时间内的位移越大 —— 同一个反应延迟下更容易撞上去。
    //      所以只在"没有近处威胁"时加速。
    bool wantBoost(const Perception& p) const;

    // 可复现的伪随机（不用全局 rand()，那会破坏跑分的可复现性）
    float randUnit();
    float randRange(float lo, float hi);

    float  m_timer = 0.0f;          // 距上次决策过了多久
    float  m_reactionTime = 0.1f;   // 反应时间（秒），模拟生物延迟
    Action m_last;                  // 上一次的动作（未到反应时间时沿用它）
    float  m_wanderAngle = 0.0f;    // 漫游方向
    std::uint32_t m_rng = 1u;       // xorshift32 状态
};

}  // namespace arena
