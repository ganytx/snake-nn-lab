#pragma once

#include <SFML/Graphics.hpp>

// ==========================================
// 摄像机 — 延迟跟随玩家蛇，营造速度感
//
// 不使用硬绑定（摄像机中心 = 蛇头），而是用指数衰减 lerp。
// m_followSpeed 控制追赶速度：
//   - 值大 → 追得快 → 延迟小 → 画面紧贴蛇头
//   - 值小 → 追得慢 → 延迟大 → 蛇头"冲"在前面 → 速度感强
//
// 加速时调低 m_followSpeed，让摄像机落后更多，
// 给玩家一种"冲出去"的感觉。
// ==========================================
class Camera {
public:
    // 构造函数：指定地图大小和可视区域大小
    Camera(float mapWidth, float mapHeight, float viewWidth, float viewHeight);

    // 每帧调用：用指数衰减 lerp 让摄像机追向 m_target
    void update(float dt);

    // 设置追踪目标（通常是蛇头位置）
    void setTarget(const sf::Vector2f& target) { m_target = target; }

    // 设置跟随速度系数（值越小 = 延迟越大 = 速度感越强）
    void setFollowSpeed(float speed) { m_followSpeed = speed; }

    // 获取摄像机当前跟随速度（调试用）
    float getFollowSpeed() const { return m_followSpeed; }

    // 获取 SFML View 对象（用于 window.setView()）
    sf::View getView() const { return m_view; }

    // 获取摄像机当前中心位置（调试用）
    sf::Vector2f getCenter() const { return m_center; }

    // 获取摄像机目标位置（调试用）
    sf::Vector2f getTarget() const { return m_target; }

private:
    sf::View m_view;              // SFML 视图，控制渲染时的可见区域
    sf::Vector2f m_center;        // 摄像机当前位置（平滑插值后的值）
    sf::Vector2f m_target;        // 目标位置（蛇头实时位置）
    float m_followSpeed;          // 跟随速度系数
    float m_mapWidth, m_mapHeight;
    float m_viewWidth, m_viewHeight;
};
