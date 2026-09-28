#include "MenuScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "GameScene.h"
#include "ModeSelectScene.h"      // ★ 「开始游戏」现在先进模式选择页
#include "SettingsScene.h"
#include "TrainingScene.h"
#include "TrainingSetupScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include <memory>

MenuScene::MenuScene(Game* game)
    : Scene(game), m_selected(0)
    // ★ 背景动效的视口 = 窗口尺寸（见 MenuBackdrop 的说明）
    , m_backdrop(static_cast<float>(Constants::WINDOW_WIDTH),
                 static_cast<float>(Constants::WINDOW_HEIGHT))
{
    // 依次尝试中文字体（.ttf 优先，.ttc SFML可能不支持）
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "MenuScene");
    }
}

void MenuScene::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::KeyPressed) {
        switch (event.key.code) {
            case sf::Keyboard::Up:   m_selected = (m_selected - 1 + ITEM_COUNT) % ITEM_COUNT; break;
            case sf::Keyboard::Down: m_selected = (m_selected + 1) % ITEM_COUNT; break;
            case sf::Keyboard::Enter:
            case sf::Keyboard::Space:
                if (m_selected == 0)
                    // ★ 2026-09-27：先选【模式】，再在该模式的设置里定这一局
                    //   （原来是直接进单一玩法的"对局设置"）
                    m_game->switchScene(std::make_unique<ModeSelectScene>(m_game));
                else if (m_selected == 1)
                    // 训练：先进【训练设置】（和环境设置一样，先定好再开始）
                    m_game->switchScene(std::make_unique<TrainingSetupScene>(m_game));
                else if (m_selected == 2)
                    // 读 train_metrics.csv 画曲线（train_probe 跑完的结果）
                    m_game->switchScene(std::make_unique<TrainingScene>(m_game));
                else if (m_selected == 3)
                    m_game->pushScene(std::make_unique<SettingsScene>(m_game));
                else
                    m_game->closeWindow();
                break;
            default: break;
        }
    }
}

void MenuScene::fixedUpdate(float) {}

// ★★ 背景动效在 update() 里推进，不在 fixedUpdate()
//
//  ★ 为什么用 update 而不是 fixedUpdate
//    fixedUpdate 是固定步长（60Hz），世界推进必须用它以保证确定性。
//    而这个背景是**纯表现层**：掉帧时慢一点无所谓，也不需要可复现。
//    用 update 还能拿到真实的 dt（fixedUpdate 的 dt 恒为 1/60）。
void MenuScene::update(float dt) {
    m_backdrop.update(dt);
}

void MenuScene::render(sf::RenderWindow& window) {
    float cx = Constants::WINDOW_WIDTH / 2.0f;
    float cy = Constants::WINDOW_HEIGHT / 2.0f;

    // ---- ① 先画背景动效（它自己带底色，所以会盖满整个窗口）----
    //
    // ★ 顺序就是"半透明"的全部实现：背景先画、菜单后画，
    //   于是文字压在动效之上。反过来做的话菜单会被动效盖住。
    m_backdrop.render(window);

    // ---- ② 菜单文字后面垫一层暗色，保证任何背景都能读清 ----
    //
    // ★ 为什么需要它：背景里的蛇是亮青色的，正好游到标题后面时，
    //   标题（同样是青色系）会糊在一起。垫一层半透明黑就彻底解决，
    //   而且它本身也是"半透明画面"的一部分。
    {
        sf::RectangleShape panel(sf::Vector2f(520.0f, 400.0f));
        panel.setOrigin(panel.getSize().x / 2.0f, panel.getSize().y / 2.0f);
        panel.setPosition(cx, cy + 10.0f);
        panel.setFillColor(sf::Color(8, 12, 18, 170));
        window.draw(panel);
    }

    // 标题
    //
    // ★ 字号从 56 降到 44：新名字「神经网络蛇对抗训练」是 9 个字，
    //   56 号在 1280 宽的窗口里会顶到边框。
    sf::Text title(L"神经网络蛇对抗训练", m_font, 44);
    title.setFillColor(sf::Color(78, 205, 196));
    title.setStyle(sf::Text::Bold);
    sf::FloatRect tb = title.getLocalBounds();
    title.setOrigin(tb.width / 2.0f, tb.height / 2.0f);
    title.setPosition(cx, cy - 150.0f);
    window.draw(title);

    // 按钮列表
    const wchar_t* labels[] = { L"开始游戏", L"训练模式", L"训练数据", L"设置", L"退出" };
    for (int i = 0; i < ITEM_COUNT; ++i) {
        sf::Text btn(labels[i], m_font, 30);
        btn.setFillColor(i == m_selected
            ? sf::Color(78, 205, 196) : sf::Color(180, 180, 180));
        sf::FloatRect bb = btn.getLocalBounds();
        btn.setOrigin(bb.width / 2.0f, bb.height / 2.0f);
        btn.setPosition(cx, cy - 90.0f + i * 54.0f);
        window.draw(btn);

        // 选中项加白色边框
        if (i == m_selected) {
            sf::RectangleShape box;
            box.setSize(sf::Vector2f(bb.width + 20.0f, bb.height + 10.0f));
            box.setOrigin(box.getSize().x / 2.0f, box.getSize().y / 2.0f);
            box.setPosition(cx, cy - 90.0f + i * 54.0f);
            box.setFillColor(sf::Color::Transparent);
            box.setOutlineColor(sf::Color(255, 255, 255, 180));
            box.setOutlineThickness(2.0f);
            window.draw(box);
        }
    }

    // 提示
    sf::Text hint(L"上下键 选择    Enter 确认", m_font, 18);
    hint.setFillColor(sf::Color(120, 120, 120));
    sf::FloatRect hb = hint.getLocalBounds();
    hint.setOrigin(hb.width / 2.0f, hb.height / 2.0f);
    hint.setPosition(cx, cy + 200.0f);
    window.draw(hint);
}
