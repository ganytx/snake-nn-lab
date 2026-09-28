#include "PauseScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "MenuScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include <memory>

PauseScene::PauseScene(Game* game)
    : Scene(game), m_selected(0)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "PauseScene");
    }
}

void PauseScene::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::KeyPressed) {
        switch (event.key.code) {
            case sf::Keyboard::Escape:
                m_game->popScene();
                break;
            case sf::Keyboard::Up:
            case sf::Keyboard::Down:
                m_selected = 1 - m_selected;
                break;
            case sf::Keyboard::Enter:
            case sf::Keyboard::Space:
                if (m_selected == 0)
                    m_game->popScene();
                else
                    m_game->switchScene(std::make_unique<MenuScene>(m_game));
                break;
            default: break;
        }
    }
}

void PauseScene::fixedUpdate(float) {}
void PauseScene::update(float) {}

void PauseScene::render(sf::RenderWindow& window) {
    float w = Constants::WINDOW_WIDTH;
    float h = Constants::WINDOW_HEIGHT;

    // 半透明遮罩
    sf::RectangleShape overlay;
    overlay.setSize(sf::Vector2f(w, h));
    overlay.setFillColor(sf::Color(0, 0, 0, 150));
    window.draw(overlay);

    // 标题
    sf::Text title(L"暂停", m_font, 48);
    title.setFillColor(sf::Color(255, 255, 255));
    sf::FloatRect tb = title.getLocalBounds();
    title.setOrigin(tb.width / 2.0f, tb.height / 2.0f);
    title.setPosition(w / 2.0f, h / 2.0f - 80.0f);
    window.draw(title);

    // 按钮
    const wchar_t* labels[] = { L"继续游戏", L"退出到菜单" };
    for (int i = 0; i < 2; ++i) {
        sf::Text btn(labels[i], m_font, 28);
        btn.setFillColor(i == m_selected
            ? sf::Color(78, 205, 196) : sf::Color(180, 180, 180));
        sf::FloatRect bb = btn.getLocalBounds();
        btn.setOrigin(bb.width / 2.0f, bb.height / 2.0f);
        btn.setPosition(w / 2.0f, h / 2.0f + i * 50.0f);
        window.draw(btn);

        // 选中白框
        if (i == m_selected) {
            sf::RectangleShape box;
            box.setSize(sf::Vector2f(bb.width + 20.0f, bb.height + 10.0f));
            box.setOrigin(box.getSize().x / 2.0f, box.getSize().y / 2.0f);
            box.setPosition(w / 2.0f, h / 2.0f + i * 50.0f);
            box.setFillColor(sf::Color::Transparent);
            box.setOutlineColor(sf::Color(255, 255, 255, 180));
            box.setOutlineThickness(2.0f);
            window.draw(box);
        }
    }

    // 提示
    sf::Text hint(L"ESC 返回游戏", m_font, 16);
    hint.setFillColor(sf::Color(120, 120, 120));
    sf::FloatRect hb = hint.getLocalBounds();
    hint.setOrigin(hb.width / 2.0f, hb.height / 2.0f);
    hint.setPosition(w / 2.0f, h / 2.0f + 120.0f);
    window.draw(hint);
}
