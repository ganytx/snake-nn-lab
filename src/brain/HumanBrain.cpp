#include "HumanBrain.h"

#include <cmath>

// 算法层铁律：不 include 任何 SFML 头文件（utils/Math.h 也不行，它拖 SFML）
namespace {

constexpr float PI = 3.14159265358979323846f;

inline float normalizeAngle(float a) {
    while (a >  PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

namespace arena {

const BrainInfo& HumanBrain::info() const {
    static const BrainInfo s_info{
        "human",                       // id
        "玩家（鼠标控制）",              // displayName
        "人类操控：鼠标瞄准 + 空格加速",  // description
        "",                            // author
        /*needsWeights=*/false,
        /*trainable=*/false,
    };
    return s_info;
}

// ---------------------------------------------------------------------------
//  把鼠标位置 / 键盘操作翻译成统一的 Action。
//
//  转向语义和别的算法完全一致：steer ∈ [-1,1]，由智能体层换算成
//  「目标朝向 = 当前朝向 + steer × turnRate × dt」。
//  所以玩家的手感（缓动转向、满舵约 1/6 秒到最大角速度）没有任何变化。
// ---------------------------------------------------------------------------
Action HumanBrain::think(const Perception& p) {
    Action action;
    action.boost = m_input.boost;

    if (!p.self.alive) return action;

    // ---- 键盘备选：左/右硬转 ----
    if (m_input.turnLeft)  { action.steer = -1.0f; return action; }
    if (m_input.turnRight) { action.steer =  1.0f; return action; }

    // ---- 鼠标瞄准 ----
    if (!m_input.hasAim) return action;      // 还没动过鼠标 → 直行

    const float dx = m_input.aimX - p.self.pos.x;
    const float dy = m_input.aimY - p.self.pos.y;

    // 鼠标几乎贴在蛇头上时方向会剧烈抖动（除以接近 0 的向量）→ 忽略
    if (dx * dx + dy * dy < 25.0f) return action;   // 5px 以内

    // bearing：目标方向相对自己朝向的夹角
    const float bearing = normalizeAngle(std::atan2(dy, dx) - p.self.heading);

    // 差 90° 以上就满舵；快对准时按比例收力，形成"先快后慢"的手感
    action.steer = clampf(bearing / (PI * 0.5f), -1.0f, 1.0f);
    return action;
}

}  // namespace arena
