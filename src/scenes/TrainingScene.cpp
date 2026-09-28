#include "TrainingScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "MenuScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include "../utils/Text.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace {

// 曲线颜色（最多同时画几条；超出的循环使用）
const sf::Color CURVE_COLORS[] = {
    sf::Color(78, 205, 196),    // 青
    sf::Color(255, 200, 60),    // 黄
    sf::Color(255, 107, 107),   // 红
    sf::Color(140, 110, 255),   // 紫
};

// 一张图的作图区（屏幕像素）
//
// ⚠️ 作图区左边距和"指标列表"的位置必须错开：
//    第一版把指标列表竖着排在右侧、从 PLOT_TOP 开始，
//    结果它直接压在图上、和曲线糊在一起。
//    现在指标列表改成【横排在最上面一行】，图在它下面，互不遮挡。
constexpr float PLOT_LEFT   = 90.0f;
constexpr float PLOT_TOP    = 210.0f;
constexpr float PLOT_W      = 1080.0f;
constexpr float PLOT_H      = 370.0f;

// 指标列表（横排）的基线
constexpr float LIST_Y      = 128.0f;

}  // namespace

// ===========================================================================
TrainingScene::TrainingScene(Game* game)
    : Scene(game)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "TrainingScene");
    }

    reload();
}

// ===========================================================================
//  读数据 + 组装"视图"
// ===========================================================================
void TrainingScene::reload() {
    m_reason.clear();
    m_data = metrics::load(m_reason);

    m_views.clear();
    m_selected = 0;
    if (!m_data.ok) return;

    // 视图 1：训练回报 + 滑动基线（放一起才能看出"回报是不是真的在涨"）
    {
        View v;
        v.title = "训练回报（每局）与滑动基线";
        const metrics::Series* r = m_data.find("train", "reward");
        const metrics::Series* b = m_data.find("train", "baseline");
        if (r) v.series.push_back(r);
        if (b) v.series.push_back(b);
        if (!v.series.empty()) m_views.push_back(std::move(v));
    }
    // 视图 2：存活帧数
    {
        View v;
        v.title = "存活帧数（上限 1200）";
        if (const metrics::Series* s = m_data.find("train", "survival"))
            v.series.push_back(s);
        if (!v.series.empty()) m_views.push_back(std::move(v));
    }
    // 视图 3：每局吃到几个
    {
        View v;
        v.title = "每局吃到的食物数";
        if (const metrics::Series* s = m_data.find("train", "food"))
            v.series.push_back(s);
        if (!v.series.empty()) m_views.push_back(std::move(v));
    }
    // 视图 4：训练后的贪心评估（只有 100 局，横轴是评估局号）
    {
        View v;
        v.title = "训练后贪心评估（100 局）";
        if (const metrics::Series* s = m_data.find("eval", "reward"))
            v.series.push_back(s);
        if (!v.series.empty()) m_views.push_back(std::move(v));
    }
    {
        View v;
        v.title = "贪心评估：存活帧数";
        if (const metrics::Series* s = m_data.find("eval", "survival"))
            v.series.push_back(s);
        if (!v.series.empty()) m_views.push_back(std::move(v));
    }
}

// ===========================================================================
void TrainingScene::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::KeyPressed) return;

    switch (event.key.code) {
        case sf::Keyboard::Up:
            if (!m_views.empty())
                m_selected = (m_selected - 1 + static_cast<int>(m_views.size()))
                           % static_cast<int>(m_views.size());
            break;
        case sf::Keyboard::Down:
            if (!m_views.empty())
                m_selected = (m_selected + 1) % static_cast<int>(m_views.size());
            break;

        case sf::Keyboard::R:
            reload();
            m_notice = m_data.ok ? "已重新读取" : "读取失败";
            m_noticeTimer = 3.0f;
            break;

        case sf::Keyboard::Escape:
            m_game->switchScene(std::make_unique<MenuScene>(m_game));
            break;

        default: break;
    }
}

void TrainingScene::fixedUpdate(float) {}

void TrainingScene::update(float dt) {
    if (m_noticeTimer > 0.0f) m_noticeTimer -= dt;
}

// ===========================================================================
//  画一条曲线
//
//  ★ 横轴用【局号】而不是数组下标：
//    训练的 2000 局是连续的，但 eval 只有 100 局 ——
//    用下标会把它们画成同样宽，看起来一样长，容易误解。
// ===========================================================================
void TrainingScene::drawCurve(sf::RenderWindow& window,
                               const metrics::Series& s,
                               float left, float top, float w, float h,
                               const sf::Color& color,
                               float yMin, float yMax, int xMax) const
{
    if (s.x.size() < 2 || xMax <= 0) return;

    const float yRange = (yMax - yMin > 1.0e-6f) ? (yMax - yMin) : 1.0f;

    // 用顶点数组画折线：比逐段 draw 快得多，而且线是连续的
    sf::VertexArray line(sf::LineStrip, s.x.size());
    for (std::size_t i = 0; i < s.x.size(); ++i) {
        const float px = left + w * (s.x[i] / static_cast<float>(xMax));
        const float py = top + h * (1.0f - (s.y[i] - yMin) / yRange);
        line[i].position = sf::Vector2f(px, py);
        line[i].color    = color;
    }
    window.draw(line);
}

// ===========================================================================
void TrainingScene::render(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);

    // ---- 标题 ----
    sf::Text title(L"训练数据", m_font, 34);
    title.setFillColor(sf::Color(78, 205, 196));
    title.setStyle(sf::Text::Bold);
    title.setPosition(60.0f, 26.0f);
    window.draw(title);

    // ---- 数据来源（避免"我看的是旧数据"这种误会）----
    if (m_data.ok) {
        const std::string info = "数据源: " + m_data.sourcePath
                               + "    共 " + std::to_string(m_data.maxEpisode()) + " 局";
        sf::Text src(textutil::utf8(info), m_font, 16);
        src.setFillColor(sf::Color(150, 150, 150));
        src.setPosition(60.0f, 70.0f);
        window.draw(src);
    }

    // ---- 读失败：直接说清原因，不要显示一张空图 ----
    if (!m_data.ok) {
        sf::Text err(textutil::utf8("读不到训练数据：" + m_reason), m_font, 20);
        err.setFillColor(sf::Color(255, 140, 90));
        err.setPosition(60.0f, 170.0f);
        window.draw(err);

        sf::Text hint(L"先在 build\\ 目录下跑一次 train_probe.exe，再按 R 重新读取",
                      m_font, 18);
        hint.setFillColor(sf::Color(160, 160, 160));
        hint.setPosition(60.0f, 215.0f);
        window.draw(hint);
    } else if (m_views.empty()) {
        sf::Text none(L"文件读到了，但里面没有可画的曲线", m_font, 20);
        none.setFillColor(sf::Color(255, 200, 90));
        none.setPosition(60.0f, 170.0f);
        window.draw(none);
    } else {
        // ---- 指标列表（横排在最上面，选中项高亮 + 下划线）----
        float lx = PLOT_LEFT;
        for (std::size_t i = 0; i < m_views.size(); ++i) {
            const bool sel = (static_cast<int>(i) == m_selected);
            sf::Text item(textutil::utf8(m_views[i].title), m_font, 16);
            item.setFillColor(sel ? sf::Color(255, 255, 255)
                                  : sf::Color(120, 120, 130));
            item.setPosition(lx, LIST_Y);
            window.draw(item);

            const float iw = item.getLocalBounds().width;
            if (sel) {
                sf::RectangleShape under(sf::Vector2f(iw, 2.0f));
                under.setPosition(lx, LIST_Y + 24.0f);
                under.setFillColor(sf::Color(78, 205, 196));
                window.draw(under);
            }
            lx += iw + 34.0f;
        }

        // ---- 当前视图 ----
        const View& v = m_views[static_cast<std::size_t>(m_selected)];

        // 纵轴范围取所有曲线的并集，这样叠加的两条线才可比
        float yMin = 1.0e30f, yMax = -1.0e30f;
        for (const auto* s : v.series) {
            if (s->empty()) continue;
            yMin = std::min(yMin, s->minY());
            yMax = std::max(yMax, s->maxY());
        }
        if (yMin > yMax) { yMin = 0.0f; yMax = 1.0f; }
        // 上下留 8% 余量，免得曲线贴着边框
        const float pad = (yMax - yMin > 1.0e-6f) ? (yMax - yMin) * 0.08f : 1.0f;
        yMin -= pad;
        yMax += pad;

        // 横轴范围：所有曲线里最大的局号
        int xMax = 0;
        for (const auto* s : v.series) {
            if (!s->x.empty()) xMax = std::max(xMax, static_cast<int>(s->x.back()));
        }

        // ---- 作图区边框 ----
        sf::RectangleShape frame(sf::Vector2f(PLOT_W, PLOT_H));
        frame.setPosition(PLOT_LEFT, PLOT_TOP);
        frame.setFillColor(sf::Color(18, 18, 32, 220));
        frame.setOutlineColor(sf::Color(70, 70, 90));
        frame.setOutlineThickness(1.0f);
        window.draw(frame);

        // ---- 网格 + 纵轴刻度（4 条）----
        for (int i = 0; i <= 4; ++i) {
            const float t  = static_cast<float>(i) / 4.0f;
            const float gy = PLOT_TOP + PLOT_H * (1.0f - t);

            sf::RectangleShape grid(sf::Vector2f(PLOT_W, 1.0f));
            grid.setPosition(PLOT_LEFT, gy);
            grid.setFillColor(sf::Color(50, 50, 70, 180));
            window.draw(grid);

            const float val = yMin + (yMax - yMin) * t;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4g", static_cast<double>(val));
            sf::Text tick(buf, m_font, 14);
            tick.setFillColor(sf::Color(130, 130, 150));
            tick.setPosition(PLOT_LEFT - 70.0f, gy - 9.0f);
            window.draw(tick);
        }

        // ---- 曲线 ----
        for (std::size_t i = 0; i < v.series.size(); ++i) {
            drawCurve(window, *v.series[i],
                      PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H,
                      CURVE_COLORS[i % 4], yMin, yMax, xMax);
        }

        // ---- 图例 ----
        //
        // ⚠️ 这里原来用 snprintf 拼了一句带中文的 "平均/最高"，
        //    而 snprintf 产出的是【窄字符 std::string】——
        //    sf::Text 从 std::string 构造时按【本地编码】解释，不是 UTF-8，
        //    于是中文全变成方块 □□□。
        //    教训（项目铁律里就写着这条）：含中文的字符串要么用宽字符串 L"..."，
        //    要么过 textutil::utf8()。数值部分用 ASCII 就没这个问题。
        float lgx = PLOT_LEFT;
        const float ly = PLOT_TOP + PLOT_H + 16.0f;
        for (std::size_t i = 0; i < v.series.size(); ++i) {
            const sf::Color c = CURVE_COLORS[i % 4];

            sf::RectangleShape swatch(sf::Vector2f(22.0f, 4.0f));
            swatch.setPosition(lgx, ly + 9.0f);
            swatch.setFillColor(c);
            window.draw(swatch);

            const metrics::Series& s = *v.series[i];
            char buf[192];
            std::snprintf(buf, sizeof(buf), "%s.%s   avg %.4g   max %.4g",
                          s.kind.c_str(), s.tag.c_str(),
                          static_cast<double>(s.avgY()),
                          static_cast<double>(s.maxY()));
            sf::Text lab(buf, m_font, 16);
            lab.setFillColor(sf::Color(190, 190, 190));
            lab.setPosition(lgx + 30.0f, ly);
            window.draw(lab);

            lgx += 30.0f + lab.getLocalBounds().width + 44.0f;
        }
    }

    // ---- 底部操作提示 ----
    sf::Text hint(L"上下键 切换指标    R 重新读取    ESC 返回", m_font, 18);
    hint.setFillColor(sf::Color(120, 120, 120));
    hint.setPosition(60.0f, static_cast<float>(Constants::WINDOW_HEIGHT) - 46.0f);
    window.draw(hint);

    if (m_noticeTimer > 0.0f) {
        sf::Text notice(textutil::utf8(m_notice), m_font, 20);
        notice.setFillColor(sf::Color(120, 255, 120));
        notice.setPosition(W - 220.0f, static_cast<float>(Constants::WINDOW_HEIGHT) - 46.0f);
        window.draw(notice);
    }
}
