#include "Minimap.h"
#include "../utils/Math.h"
#include <sstream>
#include <iomanip>

// ==========================================
// 构造函数 — 计算小地图位置和缩放
//
// ★ 缩放系数只有一个，取"能装进控件"的那个比例（min），
//   然后把地图居中放进控件里。
//
//   地图 2000x2000 + 控件 150x150 时：scale = 0.075，offset 全为 0，
//   地图正好铺满控件。地图和控件比例不一致时才会出现留白，
//   但那也比"把地图拉变形"好 —— 变形会让所有位置读数失去意义。
// ==========================================
Minimap::Minimap(float mapW, float mapH, float displayW, float displayH)
    : m_mapW(mapW)
    , m_mapH(mapH)
    , m_displayW(displayW)
    , m_displayH(displayH)
{
    recomputeScale();

    // 屏幕右上角（默认1920屏幕，我们取窗口宽度减margin）
    // 位置会在render前根据窗口大小动态计算
    m_x = 1280.0f - displayW - 10.0f;
    m_y = 10.0f;
}

// ==========================================
// 重算缩放与留白
// ==========================================
void Minimap::recomputeScale() {
    // 防 0：地图尺寸理论上不会为 0，但配置里写个 0 进来就是除零崩溃；
    //       负值同样会让缩放系数变成负数，地图会整个画反。
    const float safeW = (m_mapW > 0.0f) ? m_mapW : 1.0f;
    const float safeH = (m_mapH > 0.0f) ? m_mapH : 1.0f;

    const float sx = m_displayW / safeW;
    const float sy = m_displayH / safeH;
    m_scale = (sx < sy) ? sx : sy;          // 取小的那个 = 等比装得下

    // 居中留白
    m_offsetX = (m_displayW - safeW * m_scale) * 0.5f;
    m_offsetY = (m_displayH - safeH * m_scale) * 0.5f;
}

// ==========================================
// 把世界坐标换算成"控件内"的像素坐标
//
// 注意返回值是【相对控件左上角】的，render 时还要再加控件在屏幕上的位置。
// ==========================================
sf::Vector2f Minimap::worldToWidget(float wx, float wy) const {
    return sf::Vector2f(m_offsetX + wx * m_scale,
                        m_offsetY + wy * m_scale);
}

// ==========================================
// 每帧更新 — 缓存所有蛇的位置和视野矩形
// ==========================================
void Minimap::update(const Snake& player,
                      const std::vector<Snake*>& allSnakes,
                      const sf::View& cameraView,
                      const std::vector<arena::FoodZone>* zones,
                      const arena::FoodZone* activeZone,
                      bool zoneSpawning)
{
    // 地图尺寸可能在运行中被改（对局重开），比例变了就得重算一次
    {
        const float safeW = (m_mapW > 0.0f) ? m_mapW : 1.0f;
        const float safeH = (m_mapH > 0.0f) ? m_mapH : 1.0f;
        const float sx = m_displayW / safeW;
        const float sy = m_displayH / safeH;
        const float s  = (sx < sy) ? sx : sy;
        if (s != m_scale) recomputeScale();
    }

    // 玩家位置
    m_playerDot = worldToWidget(player.getX(), player.getY());

    // AI蛇位置
    m_aiDots.clear();
    for (const auto* s : allSnakes) {
        if (!s->isAlive() || s->isPlayer()) continue;
        m_aiDots.push_back(worldToWidget(s->getX(), s->getY()));
    }

    // 视野矩形：左上角换算 + 尺寸按同一个 scale 缩放
    // ★ 宽高必须用【同一个】m_scale，否则视野框会被拉成和地图不一样的变形
    sf::Vector2f viewCenter = cameraView.getCenter();
    sf::Vector2f viewSize = cameraView.getSize();
    const sf::Vector2f topLeft =
        worldToWidget(viewCenter.x - viewSize.x / 2.0f,
                      viewCenter.y - viewSize.y / 2.0f);
    m_viewRect.left   = topLeft.x;
    m_viewRect.top    = topLeft.y;
    m_viewRect.width  = viewSize.x * m_scale;
    m_viewRect.height = viewSize.y * m_scale;

    // 坐标文字
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0);
    oss << "X:" << player.getX() << " Y:" << player.getY();
    m_coordText = oss.str();

    // ---- 食物圈标记 ----
    //
    // ★★ 2026-09-27 定稿（负责人定义）：**「下一个圈」只在预告期标出来**
    //
    //    "不要一开始就有预告圈，预告圈是在活跃圈产食物还剩 3s 后出现的。"
    //    ⚠️ 传入的这个 bool 现在语义是 **previewing（预告期）**，不再是 spawning。
    //       调用方（GameScene）已经改成传 `zoneSchedule().previewing(...)`。
    //       判据的唯一实现在 `ZoneSchedule::previewing()`，这里不重推算式。
    //
    //  ⚠️ 规则的另外两份实现在 `GameScene` 与 `LiveTrainingScene` 的渲染里；
    //     改这里/那里时必须三处一起改 —— 否则玩家看到的两个画面会互相矛盾。
    m_zoneMarks.clear();
    if (zones != nullptr) {
        m_zoneMarks.reserve(zones->size());
        for (const arena::FoodZone& z : *zones) {
            const bool isActive = (&z == activeZone);

            // ★ 非活跃圈只在预告期标；活跃圈永远标
            if (!isActive && !zoneSpawning) continue;

            ZoneMark m;
            m.pos      = worldToWidget(z.x, z.y);
            m.active   = isActive;
            m.spawning = isActive && zoneSpawning;   // 保留字段名，语义见调用方
            m_zoneMarks.push_back(m);
        }
    }
}

// ==========================================
// 渲染 — 屏幕右上角
// ==========================================
void Minimap::render(sf::RenderWindow& window) const {
    float screenW = static_cast<float>(window.getSize().x);

    // 动态计算位置（右上角，距边缘10px）
    float px = screenW - m_displayW - 10.0f;
    float py = 10.0f;

    // ★ 地图区的尺寸由【单一缩放系数】决定，不再等于控件尺寸。
    //   控件与地图比例一致时两者相等；不一致时地图居中出现留白。
    const float mapPixelW = m_mapW * m_scale;
    const float mapPixelH = m_mapH * m_scale;

    // ---- 半透明黑色背景（整个控件外框） ----
    sf::RectangleShape bg;
    bg.setSize(sf::Vector2f(m_displayW + 4.0f, m_displayH + 4.0f));
    bg.setPosition(px - 2.0f, py - 2.0f);
    bg.setFillColor(sf::Color(0, 0, 0, 140));
    bg.setOutlineColor(sf::Color(80, 80, 80, 100));
    bg.setOutlineThickness(1.0f);
    window.draw(bg);

    // ---- 地图边界矩形（等比，居中） ----
    sf::RectangleShape mapRect;
    mapRect.setSize(sf::Vector2f(mapPixelW, mapPixelH));
    mapRect.setPosition(px + m_offsetX, py + m_offsetY);
    mapRect.setFillColor(sf::Color(20, 20, 40, 200));
    mapRect.setOutlineColor(sf::Color(100, 100, 120));
    mapRect.setOutlineThickness(1.0f);
    window.draw(mapRect);

    // ---- ★ 食物圈标记 ----
    //
    // ★ 用【符号 + 颜色】而不是照抄大地图上的淡色填充圆：
    //   圈半径 150px 在 2000 地图上只占 7.5%，到了 150px 的小地图里
    //   只有 11 像素 —— 画个淡淡的圆根本看不出哪个是活跃圈。
    //   符号能区分"在供食 / 空窗 / 没轮到"，颜色强化这个区分。
    //
    // ★ 画在蛇【之前】：标记是背景信息，不该盖住蛇和玩家。
    for (const auto& zm : m_zoneMarks) {
        const float px2 = px + zm.pos.x;
        const float py2 = py + zm.pos.y;

        // 颜色：暖黄 = 正在供食；冷蓝 = 轮到它但空窗；暗灰 = 没轮到
        sf::Color col;
        if (zm.spawning)      col = sf::Color(255, 214, 90, 255);
        else if (zm.active)   col = sf::Color(120, 190, 255, 220);
        else                  col = sf::Color(140, 140, 150, 150);

        if (zm.active) {
            // 活跃圈：实心圆点（大一点，带白色细边，在深色底上更醒目）
            sf::CircleShape dot(4.0f);
            dot.setOrigin(4.0f, 4.0f);
            dot.setPosition(px2, py2);
            dot.setFillColor(col);
            dot.setOutlineThickness(1.0f);
            dot.setOutlineColor(sf::Color(255, 255, 255, 160));
            window.draw(dot);
        } else {
            // 未激活：交叉（✕），两条短粗线
            const float L = 4.0f;
            sf::RectangleShape bar(sf::Vector2f(L * 2.0f, 1.5f));
            bar.setOrigin(L, 0.75f);
            bar.setPosition(px2, py2);
            bar.setFillColor(col);

            bar.setRotation(45.0f);
            window.draw(bar);
            bar.setRotation(-45.0f);
            window.draw(bar);
        }
    }

    // ---- AI蛇（红色圆点，2px） ----
    sf::CircleShape dot(2.0f);
    dot.setOrigin(2.0f, 2.0f);
    dot.setFillColor(sf::Color(255, 80, 80));
    for (const auto& pos : m_aiDots) {
        dot.setPosition(px + pos.x, py + pos.y);
        window.draw(dot);
    }

    // ---- 玩家（绿色圆点，3px，稍大） ----
    sf::CircleShape playerDot(3.0f);
    playerDot.setOrigin(3.0f, 3.0f);
    playerDot.setFillColor(sf::Color(100, 255, 100));
    playerDot.setPosition(px + m_playerDot.x, py + m_playerDot.y);
    window.draw(playerDot);

    // ---- 视野矩形（白色细框） ----
    sf::RectangleShape viewRect;
    viewRect.setSize(sf::Vector2f(m_viewRect.width, m_viewRect.height));
    viewRect.setPosition(px + m_viewRect.left, py + m_viewRect.top);
    viewRect.setFillColor(sf::Color::Transparent);
    viewRect.setOutlineColor(sf::Color(255, 255, 255, 80));
    viewRect.setOutlineThickness(1.0f);
    window.draw(viewRect);

    // ---- 坐标文字（小地图下方） ----
    if (!m_coordText.empty()) {
        // 这里需要字体，但 Minimap 不管理字体
        // 坐标文字在 GameScene 中用 HUD 绘制
    }
}
