#pragma once

#include "../core/Scene.h"
#include "../systems/MenuBackdrop.h"

// ==========================================
// 主菜单 — 开始 / 设置 / 退出
// ==========================================
class MenuScene : public Scene {
public:
    explicit MenuScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // 菜单项数量：开始游戏 / 训练模式 / 训练数据 / 设置 / 退出
    // ★ 加一项只要改这里 + handleEvent 的分支 + render 的 labels 数组，
    //   三个地方的顺序必须一致（上/下键的取模靠它）。
    static constexpr int ITEM_COUNT = 5;

    sf::Font m_font;
    int m_selected = 0;  // 0=开始游戏 1=训练模式 2=训练数据 3=设置 4=退出

    // ★★ 背景动效（2026-09-27 负责人要求，**只有主菜单有**）
    //
    //  ★ 为什么放在这里而不是做一个"所有场景共用的背景"
    //    负责人明确说了"只有这个界面！"—— 游戏里加背景会干扰读画面，
    //    设置页加背景会让文字更难读。所以它是主菜单专属的装饰。
    systems::MenuBackdrop m_backdrop;
};
