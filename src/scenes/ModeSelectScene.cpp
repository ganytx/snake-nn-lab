#include "ModeSelectScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "ModeSetupScene.h"
#include "GameScene.h"
#include "MenuScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include "../utils/Text.h"

#include <cmath>
#include <memory>
#include <string>

namespace {

const sf::Color kSelect = sf::Color(78, 205, 196);
const sf::Color kNormal = sf::Color(150, 150, 160);
const sf::Color kTitle  = sf::Color(255, 210, 90);

// 示意图区域的边距（相对窗口）
constexpr float DIAG_TOP    = 150.0f;   // 示意图上边
constexpr float DIAG_HEIGHT = 300.0f;   // 示意图高度

}  // namespace

ModeSelectScene::ModeSelectScene(Game* game)
    : Scene(game)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "ModeSelectScene");
    }

    // ★ 打开本页时，光标停在【上次玩过的模式】上
    //
    //  ★ 为什么不用"总是从模式 0 开始"
    //    玩家通常连续玩同一个模式，每次都回到第一个会很烦。
    //    而 `Game::matchMode()` 记着上一次开局用的模式，正好是"上次玩的那个"。
    m_index = static_cast<int>(m_game->matchMode());
    if (m_index < 0 || m_index >= game::MODE_COUNT) m_index = 0;

    // -----------------------------------------------------------------------
    //  ★★ 进入本页时就初始化【第三层配置】（这一局的配置）
    //
    //  ★ 为什么必须在这里做，而不是等按"开始"
    //    「设置」（小设置页）改的就是第三层配置。如果它还没被初始化，
    //    小设置页读到的就是【上一次残留】或者【默认构造的空配置】——
    //    而它不会报错，只会表现为"设置里显示的队数和实际开局的不一样"。
    //
    //  ★ 为什么用当前选中的模式初始化
    //    玩家一进这一页，看到的就是这个模式；他接着按「设置」或「开始」，
    //    期望的都是"这个模式"。所以进页面 = 开始配置这一局。
    //
    //  ⚠️ 副作用（已知、可接受）：在模式间左右切换时，第三层配置会被
    //     重新初始化成该模式的默认值 —— 也就是说"在模式 1 里改的设置，
    //     切到模式 2 再切回来就没了"。
    //     这是刻意的：第三层配置按定义【只属于一局】，而"切换模式"
    //     本身就是放弃上一局。要保留改动请直接用「开始」。
    // -----------------------------------------------------------------------
    m_game->beginMatch(mode());
}

// ===========================================================================
//  事件
// ===========================================================================
void ModeSelectScene::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::KeyPressed) return;

    switch (event.key.code) {
        case sf::Keyboard::Left:
            m_index = (m_index - 1 + game::MODE_COUNT) % game::MODE_COUNT;
            m_game->beginMatch(mode());   // 换模式 = 重新初始化这一局的配置
            break;
        case sf::Keyboard::Right:
            m_index = (m_index + 1) % game::MODE_COUNT;
            m_game->beginMatch(mode());
            break;

        case sf::Keyboard::Enter:
        case sf::Keyboard::Space:
            // ★ 开始：这一局的配置在进本页（或切换模式）时已经就绪，
            //   这里只需要把它敲定并进 GameScene。
            //
            //  ⚠️ GameScene 的构造函数里就会读这一局的配置，所以顺序上
            //     "先 beginMatch 再 new GameScene" 是硬要求 ——
            //     反过来的话它读到的是上一次的残留。
            //     本页的 beginMatch 都在切换模式时调过了，这里再调一次是
            //     幂等的（同一个模式、同一份默认值），但为了不产生
            //     "按 Enter 会把小设置里的改动冲掉"这种意外，这里【不调】。
            m_game->switchScene(std::make_unique<GameScene>(m_game, mode()));
            break;

        case sf::Keyboard::S:
        case sf::Keyboard::Up:
            // 进"这一局的设置"（小设置）—— 它只改 ③
            m_game->pushScene(std::make_unique<ModeSetupScene>(m_game, mode()));
            break;

        case sf::Keyboard::Escape:
            m_game->switchScene(std::make_unique<MenuScene>(m_game));
            break;

        default: break;
    }
}

void ModeSelectScene::fixedUpdate(float) {}
void ModeSelectScene::update(float) {}

// ===========================================================================
//  ★ 示意图：三个模式各画一张（**只画规则，不画蛇**）
//
//  ★ 为什么把蛇删掉了（负责人 2026-09-27："把蛇删了吧，太诡异了"）
//    第一版画的是"绕自己的头原地旋转 + 身体排成一条直线"，一眼就不对；
//    第二版改成沿蜿蜒路径游动、加了头大尾小和眼睛 —— 更像蛇了，
//    但**用几个圆去模拟生物本身就容易出戏**。
//
//  ★ 这三张图真正要传达的是【规则】：
//      模式 1：两个圈、一亮一暗、亮的那轮换
//      模式 2：食物撒满全图、蛇会不断加入
//      模式 3：几乎没有食物、尸体是唯一的粮
//    圈、食物点、尸体都是【几何图形】，本来就不需要"像什么"。
//    去掉蛇之后图更干净、规则更醒目。
//
//  ★ 队色仍用【色带】保留（"场上有几队"是玩法的一部分）：
//    底部一排小色块，每块代表一队 —— 一眼能数出队伍数，且不像生物。
//
//  ★ 共同约定：矩形 r 是"画布"，所有图形都画在 r 内部，不越界。
//    这样三种示意图能共用同一个布局，切换模式时位置不跳。
// ===========================================================================

// ===========================================================================
//  示意图：三个模式各画一张
//
//  ★ 共同约定：矩形 r 是"画布"，所有图形都画在 r 内部，不越界。
//    这样三种示意图能共用同一个布局，切换模式时位置不跳。
// ===========================================================================
// ===========================================================================
//  ★ 队色带（见头文件里的说明）

// ===========================================================================
void ModeSelectScene::drawZoneDiagram(sf::RenderWindow& w, const sf::FloatRect& r) {
    // 随机食物圈：地图上两个圈，一个亮（正在供食）一个暗（下一个），
    // 位置每轮互换 —— 所以让它缓慢地"呼吸"并在两个位置间轮换。
    const float t = m_anim.getElapsedTime().asSeconds();

    // 地图边框
    sf::RectangleShape border(sf::Vector2f(r.width, r.height));
    border.setPosition(r.left, r.top);
    border.setFillColor(sf::Color(18, 18, 32));
    border.setOutlineColor(sf::Color(70, 70, 90));
    border.setOutlineThickness(2.0f);
    w.draw(border);

    // 两个圈的位置（相对画布的比例）
    const float zoneR = 46.0f;
    const sf::Vector2f p0(r.left + r.width * 0.28f, r.top + r.height * 0.42f);
    const sf::Vector2f p1(r.left + r.width * 0.70f, r.top + r.height * 0.64f);

    // 每 2 秒换一次"谁是活跃圈" —— 这就是模式的核心规则
    const bool firstActive = (static_cast<int>(t / 2.0f) % 2) == 0;

    auto drawZone = [&](const sf::Vector2f& c, bool active) {
        sf::CircleShape z(zoneR);
        z.setOrigin(zoneR, zoneR);
        z.setPosition(c);
        if (active) {
            // 活跃圈：亮绿边 + 半透明填充 + 呼吸
            const float pulse = 0.5f + 0.5f * std::sin(t * 3.0f);
            z.setFillColor(sf::Color(78, 205, 196,
                                     static_cast<sf::Uint8>(26 + 26 * pulse)));
            z.setOutlineColor(sf::Color(78, 205, 196));
            z.setOutlineThickness(3.0f);
        } else {
            // 下一个圈：暗灰，只表示"位置已经定好了"
            z.setFillColor(sf::Color(40, 40, 52));
            z.setOutlineColor(sf::Color(100, 100, 115));
            z.setOutlineThickness(2.0f);
        }
        w.draw(z);

        // 活跃圈里撒几个食物点
        if (active) {
            for (int i = 0; i < 7; ++i) {
                const float a = static_cast<float>(i) * 0.9f + t * 0.4f;
                const float rr = zoneR * 0.68f * (0.35f + 0.6f * static_cast<float>(i % 3) / 2.0f);
                sf::CircleShape f(4.0f);
                f.setOrigin(4.0f, 4.0f);
                f.setPosition(c.x + std::cos(a) * rr, c.y + std::sin(a) * rr);
                f.setFillColor(sf::Color(120, 230, 140));
                w.draw(f);
            }
        }
    };

    drawZone(p0, firstActive);
    drawZone(p1, !firstActive);

    // ⚠️ 2026-09-27：这里原本还有两样东西，负责人都让删了 ——
    //    · 两个黄点（在两圈之间来回移动，表示"位置在轮换"）
    //    · 队色带（一排色块表示"几队"）
    //    理由：它们让图变花，却没让规则更清楚。
    //    "位置每轮随机"这件事由【圈的明暗交替】本身表达就够了；
    //    "几队"写在下方文字介绍里，那才是它该在的地方。
    //    **判据：示意图只留"不看文字也能看懂的规则"，其余一律去掉。**

    // 标注"当前供食"
    sf::Text lbl(L"亮圈=正在供食   暗圈=下一轮（位置已定）",
                 m_font, 15);
    lbl.setFillColor(sf::Color(150, 200, 150));
    lbl.setPosition(r.left + 12.0f, r.top + r.height - 26.0f);
    w.draw(lbl);
}

void ModeSelectScene::drawInfiniteDiagram(sf::RenderWindow& w, const sf::FloatRect& r) {
    // 无限模式：全图随机撒食物、没有圈、蛇会不断加入
    const float t = m_anim.getElapsedTime().asSeconds();

    sf::RectangleShape border(sf::Vector2f(r.width, r.height));
    border.setPosition(r.left, r.top);
    border.setFillColor(sf::Color(18, 18, 32));
    border.setOutlineColor(sf::Color(70, 70, 90));
    border.setOutlineThickness(2.0f);
    w.draw(border);

    // 满地图随机的食物点（位置由下标决定，稳定不闪）
    for (int i = 0; i < 46; ++i) {
        // 用整数哈希做"伪随机但确定"的分布，避免每帧乱跳
        const unsigned h = static_cast<unsigned>(i) * 2654435761u;
        const float fx = static_cast<float>(h % 1000u) / 1000.0f;
        const float fy = static_cast<float>((h / 1000u) % 1000u) / 1000.0f;
        sf::CircleShape f(3.5f);
        f.setOrigin(3.5f, 3.5f);
        f.setPosition(r.left + 10.0f + fx * (r.width - 20.0f),
                      r.top  + 10.0f + fy * (r.height - 20.0f));
        f.setFillColor(sf::Color(120, 230, 140, 170));
        w.draw(f);
    }

    // ⚠️ 2026-09-27：这里原本有一排 **20 个圆点**表示"蛇数从 5 涨到 20"。
    //    负责人："无限模式的人数点你仍然没有删掉" → **整段删除**。
    //
    //    ★ 我上次删的是【队色带】（一排横条）和【模式 3 的圆点阵】，
    //      漏了这里的 — 说明我核对"哪些是点"时只按名字找，没按形状找。
    //    ★ 判据（三次删减都是同一条）：
    //      **示意图只留"不看文字也能看懂的规则"**。
    //      "蛇数会从 5 涨到 20"是文字介绍该说的话 ——
    //      在图上排 20 个点既不是一条规则，又让画面变花。
    //
    //    删掉之后这张图只剩满图的食物点 —— 正好是这个模式的核心规则：
    //    **食物全图随机、管够**（这也是它和"大混战"最大的区别）。

    sf::Text lbl(L"食物全图随机生成；每 20 秒加入一条蛇（上限 20）",
                 m_font, 15);
    lbl.setFillColor(sf::Color(150, 200, 150));
    lbl.setPosition(r.left + 12.0f, r.top + r.height - 26.0f);
    w.draw(lbl);
}

void ModeSelectScene::drawBattleDiagram(sf::RenderWindow& w, const sf::FloatRect& r) {
    // 大混战：没有食物（只有极少保底），到处都是蛇，靠撞死别人吃尸体
    const float t = m_anim.getElapsedTime().asSeconds();

    sf::RectangleShape border(sf::Vector2f(r.width, r.height));
    border.setPosition(r.left, r.top);
    border.setFillColor(sf::Color(24, 16, 20));
    border.setOutlineColor(sf::Color(90, 60, 70));
    border.setOutlineThickness(2.0f);
    w.draw(border);

    // 极少的保底食物（只有 2 个，而且要能看出"很少"）
    for (int i = 0; i < 2; ++i) {
        const float fx = (i == 0) ? 0.22f : 0.78f;
        const float fy = (i == 0) ? 0.72f : 0.26f;
        sf::CircleShape f(4.0f);
        f.setOrigin(4.0f, 4.0f);
        f.setPosition(r.left + fx * r.width, r.top + fy * r.height);
        f.setFillColor(sf::Color(120, 230, 140));
        w.draw(f);
    }

    // ⚠️ 2026-09-27：这里原本画了"15 条蛇"的编号圆点阵（5 色 × 15 个），
    //    负责人要求"显示队伍人数的点没有删掉" → **整段删除**。
    //
    //    ★ 判据（和删队色带、删黄点同一条）：
    //      **示意图只留"不看文字也能看懂的规则"**。
    //      "这个模式有几条蛇、几条队" 是文字介绍该说的话 ——
    //      在图上排 15 个点既不构成一条规则，又让画面变花。
    //
    //    删掉之后这张图只剩两样东西：
    //      · 两个孤零零的绿食物点  → "食物极少"
    //      · 几个灰白尸体点        → "尸体是主要食物来源"
    //    这两样正好是这个模式的核心规则。

    // 尸体（灰白色点），表示"唯一的粮"
    for (int i = 0; i < 5; ++i) {
        const unsigned h = static_cast<unsigned>(i) * 668265263u + 11u;
        sf::CircleShape c(5.0f);
        c.setOrigin(5.0f, 5.0f);
        c.setPosition(r.left + 40.0f + static_cast<float>(h % 1000u) / 1000.0f * (r.width - 80.0f),
                      r.top  + 40.0f + static_cast<float>((h / 1000u) % 1000u) / 1000.0f * (r.height - 80.0f));
        c.setFillColor(sf::Color(210, 200, 190, 200));
        w.draw(c);
    }

    sf::Text lbl(L"食物极少 —— 撞死别人，吃掉尸体（灰点）才能活",
                 m_font, 15);
    lbl.setFillColor(sf::Color(255, 190, 140));
    lbl.setPosition(r.left + 12.0f, r.top + r.height - 26.0f);
    w.draw(lbl);
}

// ===========================================================================
//  渲染
// ===========================================================================
void ModeSelectScene::render(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);

    // ---- 左上角：返回提示 ----
    {
        sf::Text back(L"ESC 返回", m_font, 16);
        back.setFillColor(sf::Color(110, 110, 120));
        back.setPosition(24.0f, 20.0f);
        window.draw(back);
    }

    // ---- 标题：模式名 + 标语 ----
    {
        sf::Text name(game::modeName(mode()), m_font, 44);
        name.setFillColor(kSelect);
        name.setStyle(sf::Text::Bold);
        sf::FloatRect nb = name.getLocalBounds();
        name.setOrigin(nb.width / 2.0f, 0.0f);
        name.setPosition(W / 2.0f, 52.0f);
        window.draw(name);

        sf::Text tag(game::modeTagline(mode()), m_font, 20);
        tag.setFillColor(kTitle);
        sf::FloatRect tb = tag.getLocalBounds();
        tag.setOrigin(tb.width / 2.0f, 0.0f);
        tag.setPosition(W / 2.0f, 108.0f);
        window.draw(tag);
    }

    // ---- 左右箭头（表示可以切换）----
    {
        const wchar_t* arrows[2] = { L"<", L">" };
        const float xs[2] = { 44.0f, W - 44.0f };
        for (int i = 0; i < 2; ++i) {
            sf::Text a(arrows[i], m_font, 52);
            a.setFillColor(sf::Color(120, 200, 190));
            sf::FloatRect ab = a.getLocalBounds();
            a.setOrigin(ab.width / 2.0f, 0.0f);
            a.setPosition(xs[i], DIAG_TOP + 110.0f);
            window.draw(a);
        }
    }

    // ---- 示意图 ----
    {
        const float margin = 110.0f;
        sf::FloatRect r(margin, DIAG_TOP, W - margin * 2.0f, DIAG_HEIGHT);
        switch (mode()) {
            case game::GameMode::ZoneRandom: drawZoneDiagram(window, r);     break;
            case game::GameMode::Infinite:   drawInfiniteDiagram(window, r); break;
            case game::GameMode::Battle:     drawBattleDiagram(window, r);   break;
        }
    }

    // ---- 小字介绍（多行）----
    //
    //  ★ 介绍文本里带 '\n'，所以按行拆开分别画 —— sf::Text 不会自动换行。
    {
        const std::wstring desc = game::modeDescription(mode());
        float y = DIAG_TOP + DIAG_HEIGHT + 18.0f;
        std::size_t start = 0;
        while (start <= desc.size()) {
            const std::size_t nl = desc.find(L'\n', start);
            const std::wstring line = desc.substr(
                start, (nl == std::wstring::npos) ? std::wstring::npos : nl - start);
            if (!line.empty()) {
                sf::Text t(line, m_font, 17);
                t.setFillColor(sf::Color(175, 175, 190));
                t.setPosition(110.0f, y);
                window.draw(t);
            }
            y += 24.0f;
            if (nl == std::wstring::npos) break;
            start = nl + 1;
        }
    }

    // ---- 底部按钮：设置 / 开始 ----
    {
        const float by = static_cast<float>(Constants::WINDOW_HEIGHT) - 74.0f;

        auto button = [&](float cx, const wchar_t* label, bool primary) {
            sf::Text t(label, m_font, 26);
            t.setFillColor(primary ? sf::Color(24, 32, 34) : sf::Color(200, 200, 210));
            sf::FloatRect b = t.getLocalBounds();

            const float pw = b.width + 56.0f;
            const float ph = b.height + 24.0f;
            sf::RectangleShape box(sf::Vector2f(pw, ph));
            box.setOrigin(pw / 2.0f, ph / 2.0f);
            box.setPosition(cx, by);
            box.setFillColor(primary ? kSelect : sf::Color(44, 44, 56));
            box.setOutlineColor(primary ? kSelect : sf::Color(110, 110, 125));
            box.setOutlineThickness(2.0f);
            window.draw(box);

            t.setOrigin(b.width / 2.0f, b.height / 2.0f + b.top * 0.0f);
            t.setPosition(cx, by - b.height * 0.5f - b.top * 0.5f);
            window.draw(t);
        };

        button(W / 2.0f - 120.0f, L"设置  (S)", false);
        button(W / 2.0f + 120.0f, L"开始  (Enter)", true);

        sf::Text hint(L"左右键 换模式", m_font, 15);
        hint.setFillColor(sf::Color(110, 110, 120));
        sf::FloatRect hb = hint.getLocalBounds();
        hint.setOrigin(hb.width / 2.0f, 0.0f);
        hint.setPosition(W / 2.0f, static_cast<float>(Constants::WINDOW_HEIGHT) - 26.0f);
        window.draw(hint);
    }
}
