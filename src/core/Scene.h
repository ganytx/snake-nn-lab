#pragma once

#include <SFML/Graphics.hpp>

// 前向声明，避免循环引用
class Game;

// ==========================================
// 场景基类 — 所有场景的抽象接口
//
// 持有 Game 指针用于场景切换（push/pop/switch）。
// 构造函数接收 Game*，Game 在 pushScene 时自动传入 this。
// ==========================================
class Scene {
public:
    explicit Scene(Game* game) : m_game(game) {}
    virtual ~Scene() = default;

    // 处理 SFML 事件
    virtual void handleEvent(const sf::Event& event) = 0;

    // 固定时间步更新（物理、移动、碰撞）
    virtual void fixedUpdate(float dt) = 0;

    // 可变时间步更新（输入、AI、动画）
    virtual void update(float dt) = 0;

    // 渲染
    virtual void render(sf::RenderWindow& window) = 0;

protected:
    Game* m_game;  // 供子类调用 pushScene/popScene/switchScene
};
