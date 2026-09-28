#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>

// ==========================================
// 数学工具函数 — 全部 inline，放在命名空间中避免污染全局
//
// 包含：角度处理、向量运算、插值、距离计算、随机数
// 全部是无状态的纯函数，不依赖任何外部状态。
// ==========================================
namespace Math {

    constexpr float PI = 3.14159265358979323846f;

    // ---- 角度 ----

    // 将角度规范化到 [-PI, PI] 范围，消除多圈旋转的累积
    inline float normalizeAngle(float angle) {
        while (angle > PI)  angle -= 2.0f * PI;
        while (angle < -PI) angle += 2.0f * PI;
        return angle;
    }

    // 计算从角度 from 到角度 to 的最短转向差（带符号，正=顺时针）
    inline float angleDifference(float from, float to) {
        return normalizeAngle(to - from);
    }

    // 将角度（弧度）转为单位方向向量
    inline sf::Vector2f angleToVector(float angle) {
        return sf::Vector2f(std::cos(angle), std::sin(angle));
    }

    // 将向量转为角度（弧度），atan2 自动处理象限
    inline float vectorToAngle(const sf::Vector2f& v) {
        return std::atan2(v.y, v.x);
    }

    // ---- 约束 ----

    // 将 value 限制在 [minVal, maxVal] 范围内
    inline float clamp(float value, float minVal, float maxVal) {
        return (value < minVal) ? minVal : ((value > maxVal) ? maxVal : value);
    }

    // ---- 插值 ----

    // 标量线性插值：a + (b-a)*t
    inline float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    // 二维向量线性插值
    inline sf::Vector2f lerp(const sf::Vector2f& a, const sf::Vector2f& b, float t) {
        return sf::Vector2f(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    }

    // ---- 距离 ----

    // 两点之间的欧几里得距离
    inline float distance(const sf::Vector2f& a, const sf::Vector2f& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // 两点之间距离的平方（避免 sqrt 开销，仅用于大小比较）
    inline float distanceSq(const sf::Vector2f& a, const sf::Vector2f& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

    // ---- 随机 ----

    // 生成 [min, max] 范围内的随机浮点数
    inline float randomFloat(float min, float max) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return min + r * (max - min);
    }

    // 生成 [min, max] 范围内的随机整数
    inline int randomInt(int min, int max) {
        return min + rand() % (max - min + 1);
    }
}
