#pragma once

#include "../core/Scene.h"

// ==========================================
// 暂停场景 — 半透明覆盖层，暂停游戏逻辑
//
// 按 ESC 或点"继续" → popScene 回到 GameScene
// 点"退出" → switchScene 回到 MenuScene
// ==========================================
class PauseScene : public Scene {
public:
    explicit PauseScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    sf::Font m_font;
    int m_selected = 0;  // 0=继续, 1=退出
};
