#pragma once

// ============================================================================
//  Action.h — 算法输出的动作
//
//  设计取向：统一成「相对转向 + 加速开关」，动作空间由【对局配置】指定。
//
//  为什么用相对转向，而不是绝对目标朝向？
//    · 绝对角要求网络自己学"我朝哪 / 该朝哪"的关系，难学
//    · 相对角直接就是"往左打多少舵"，对神经网络友好，也是标准做法
//
//  为什么动作空间是对局级配置？
//    为了【公平】。现在的规则 AI 输出连续角度；如果神经网络只能出 3 个离散动作，
//    规则 AI 就白占了便宜，"哪个算法强"的结论不成立。
//    同一场对局里所有算法必须使用同一个 ActionSpace。
// ============================================================================

#include "Types.h"
#include <cmath>
#include <algorithm>

namespace arena {

// ---------------------------------------------------------------------------
//  一条蛇这一 tick 的动作
// ---------------------------------------------------------------------------
struct Action {
    // 转向量，范围 [-1, +1]：
    //   -1 = 满舵左转     0 = 直行     +1 = 满舵右转
    // 接口层会把它换算成蛇的目标朝向（见 docs/40 第 3 节）
    float steer = 0.0f;

    // 是否加速（代价机制待定，见缺陷 #21）
    bool boost = false;
};

// ---------------------------------------------------------------------------
//  动作空间
// ---------------------------------------------------------------------------
enum class ActionSpace {
    Continuous,   // steer 取任意 [-1, 1]
    Discrete3,    // {-1, 0, +1}
    Discrete5,    // {-1, -0.5, 0, +0.5, +1}   ← 对局默认
    Discrete7     // {-1, -2/3, -1/3, 0, +1/3, +2/3, +1}
};

// 该动作空间有多少个可选动作；Continuous 返回 0（表示"不是离散的"）
inline int actionCount(ActionSpace space) {
    switch (space) {
        case ActionSpace::Discrete3: return 3;
        case ActionSpace::Discrete5: return 5;
        case ActionSpace::Discrete7: return 7;
        case ActionSpace::Continuous: break;
    }
    return 0;
}

// 第 index 个档位对应的 steer 值。
// 把 [0, n-1] 线性映射到 [-1, +1]：n=3 时得到 -1, 0, +1；n=5 时得到 -1, -0.5, 0, 0.5, 1
inline float steerOfIndex(int index, ActionSpace space) {
    const int n = actionCount(space);
    if (n <= 1) return 0.0f;
    index = std::max(0, std::min(n - 1, index));
    return -1.0f + 2.0f * static_cast<float>(index) / static_cast<float>(n - 1);
}

// 找出最接近 steer 的档位下标（离散动作空间专用）
inline int nearestIndex(float steer, ActionSpace space) {
    const int n = actionCount(space);
    if (n <= 0) return -1;                       // Continuous 没有"档位"概念
    steer = std::max(-1.0f, std::min(1.0f, steer));
    int   best  = 0;
    float bestD = 1e30f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(steer - steerOfIndex(i, space));
        if (d < bestD) { bestD = d; best = i; }
    }
    return best;
}

// 把任意 steer 规范化到指定动作空间：
//   Continuous → 只做夹取
//   离散空间   → 量化到最近的档位
// 由智能体层在施加动作前调用，保证算法无法通过输出越界值"作弊"
inline float quantizeSteer(float steer, ActionSpace space) {
    steer = std::max(-1.0f, std::min(1.0f, steer));
    const int idx = nearestIndex(steer, space);
    if (idx < 0) return steer;                   // Continuous
    return steerOfIndex(idx, space);
}

}  // namespace arena
