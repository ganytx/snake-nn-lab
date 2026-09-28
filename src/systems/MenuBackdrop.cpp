#include "MenuBackdrop.h"
#include "../utils/Math.h"

#include <algorithm>
#include <cmath>

namespace systems {

namespace {
constexpr float TWO_PI = 6.283185307179586f;
constexpr float kPi    = 3.141592653589793f;

// ===========================================================================
//  ★★ 蛇的尺寸：**直接照抄游戏里的比例**（2026-09-27 负责人要求）
//
//  ★ 负责人原话
//      "蛇的样子有点奇怪，你和游戏内的蛇样子统一一下吧。"
//
//  ★ 第一版为什么"奇怪"
//    我自己编了一套尺寸（体节半径 9.5→6.5 递减、alpha 150→105 渐隐、
//    间距 13、22 节）。而游戏里的蛇是：
//        · 所有体节**同一个半径、同一个颜色、不透明**（没有渐隐、没有变细）
//        · 头比体节**大一点**，颜色是 body +40（更亮）
//        · 两只黑眼睛，在头前方 8px、左右 6px 处
//    两者放在一起看就是"两种生物"。
//
//  ★ 现在怎么统一：用【同一套公式 × 一个缩放系数】
//    游戏的常量（`utils/Constants.h`，3000² 地图 · 1280 宽的窗口）：
//        SNAKE_HEAD_RADIUS     18.0
//        SNAKE_SEGMENT_RADIUS  16.0
//        SNAKE_SEGMENT_SPACING 20.0
//        SNAKE_INITIAL_SEGMENTS  12   （配置里的 snake_init_segs）
//        眼睛：半径 4.0，前方 8.0，左右 6.0
//        FOOD_RADIUS            6.0
//    地图短边 ↔ 窗口宽度的比例是 1280 / 3000 ≈ 0.427，但背景动效不需要
//    1:1 复刻地图（那样蛇只有 102px 长，太小看不清），所以取一个
//    **看得清又保持比例**的缩放：kScale = 0.70（蛇全长 ≈ 14 × 14 = 196px）。
//      ⚠️ 关键是【所有尺寸同乘一个系数】—— 这样头/身/间距/眼睛的相对
//         关系与游戏完全一致，只是整体大小不同。看起来就是同一种蛇。
// ===========================================================================
constexpr float kScale = 0.70f;

constexpr float kSegRadius  = 16.0f * kScale;   // ≈ 11.2
constexpr float kHeadRadius = 18.0f * kScale;   // ≈ 12.6
constexpr float kSegSpacing = 20.0f * kScale;   // ≈ 14.0
constexpr float kEyeRadius  = 4.0f  * kScale;   // ≈ 2.8
constexpr float kEyeForward = 8.0f  * kScale;   // ≈ 5.6
constexpr float kEyeSide    = 6.0f  * kScale;   // ≈ 4.2
constexpr float kFoodRadius = 6.0f  * kScale;   // ≈ 4.2（同 FOOD_RADIUS）
constexpr int   kSegCount   = 14;               // 比游戏的 12 略长，画面更舒展

// 蛇的颜色：队 4 的绿（见 Arena.cpp 的 TEAM_COLORS）。
//  ★ 用游戏里的调色板值，不自己配色 —— 这也是"统一"的一部分。
const sf::Color kSnakeColor(90, 220, 120);

// 把角度折到 (-π, π]，用于算"该往哪边转最近"
float wrapPi(float a) {
    while (a >  kPi) a -= TWO_PI;
    while (a <= -kPi) a += TWO_PI;
    return a;
}

// ★ 蛇的半径/间距都用 k* 常量（见上）。第一版是写在类成员里的一串
//   手调数值，那正是"和游戏里的蛇不一样"的来源 —— 现在只有一处定义。

}  // namespace

// ===========================================================================
MenuBackdrop::MenuBackdrop(float width, float height)
    : m_w(width), m_h(height)
{
    // 起点放画面中间偏左，朝向随意 —— 反正第一帧就会开始追食物
    m_headX = width * 0.30f;
    m_headY = height * 0.62f;
    m_heading = 0.3f;

    // 初始先铺 8 个食物（负责人要的"逐渐生成"，所以不是一上来就铺满）
    for (int i = 0; i < 8; ++i) spawnFood();

    // 轨迹先填满，否则第一帧蛇身是空的（会"从无到有"闪一下）
    const float total = static_cast<float>(kSegCount) * kSegSpacing;
    for (float d = 0.0f; d <= total; d += kSegSpacing * 0.5f) {
        m_trail.push_back(sf::Vector2f(m_headX - std::cos(m_heading) * d,
                                       m_headY - std::sin(m_heading) * d));
    }

    pickNewTarget();
}

// ---------------------------------------------------------------------------
//  专用随机（xorshift32）—— 见头文件里"为什么不用 rand()"的说明
// ---------------------------------------------------------------------------
float MenuBackdrop::rnd01() {
    m_rngState ^= m_rngState << 13;
    m_rngState ^= m_rngState >> 17;
    m_rngState ^= m_rngState << 5;
    return static_cast<float>(m_rngState & 0xFFFFFFu) / 16777216.0f;
}

float MenuBackdrop::rndRange(float lo, float hi) {
    return lo + rnd01() * (hi - lo);
}

// ---------------------------------------------------------------------------
void MenuBackdrop::spawnFood() {
    // 留出边距，别贴着窗口边缘（看着像被裁掉一半）
    const float m = 40.0f;
    Food f;
    f.x = rndRange(m, m_w - m);
    f.y = rndRange(m, m_h - m);
    // ★ 固定半径 = 游戏的 FOOD_RADIUS × kScale
    //   （第一版是 4~6.5 随机，而游戏里所有食物一样大 —— 又一处不一致）
    f.r = kFoodRadius;
    m_foods.push_back(f);
}

// ---------------------------------------------------------------------------
//  选下一个目标：取【离蛇最近】的那个食物
//
//  ★ 为什么是"最近"而不是"随机挑一个"
//    随机挑会让蛇满地图乱窜，看起来像无头苍蝇。
//    就近吃 → 走出来是一条连续的觅食路线，像真的在找吃的。
// ===========================================================================
void MenuBackdrop::pickNewTarget() {
    m_hasTarget = false;
    float best = 1.0e18f;
    for (std::size_t i = 0; i < m_foods.size(); ++i) {
        const float dx = m_foods[i].x - m_headX;
        const float dy = m_foods[i].y - m_headY;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; m_targetIdx = i; m_hasTarget = true; }
    }
}

// ---------------------------------------------------------------------------
void MenuBackdrop::update(float dt) {
    // 防止切窗口回来时 dt 巨大，一帧跳过去（表现层的常见坑）
    if (dt > 0.1f) dt = 0.1f;
    if (dt <= 0.0f) return;

    // ---- 1. 逐渐生成食物 ----
    m_spawnTimer += dt;
    if (m_spawnTimer >= m_spawnInterval) {
        m_spawnTimer = 0.0f;
        // 上限 26 个：够密就行，太多会糊成一片红点
        if (m_foods.size() < 26) spawnFood();
    }

    // ---- 2. 追目标 ----
    if (!m_hasTarget && !m_foods.empty()) pickNewTarget();

    float want = m_heading;
    if (m_hasTarget && m_targetIdx < m_foods.size()) {
        const Food& t = m_foods[m_targetIdx];
        const float dx = t.x - m_headX;
        const float dy = t.y - m_headY;
        const float dist = std::sqrt(dx * dx + dy * dy);

        // 到了 → 吃掉它
        //
        // ★ 判定半径比食物本身大一点（食物半径 + 14px）：
        //   太严的话蛇会绕着食物打转（有限转向速率下很难精确命中中心），
        //   看起来像"想吃又吃不到"。这是表现层，宽一点更好看。
        if (dist < t.r + 14.0f) {
            m_ripples.push_back({ t.x, t.y, 1.0f });
            m_foods.erase(m_foods.begin() + static_cast<std::ptrdiff_t>(m_targetIdx));
            pickNewTarget();
        } else {
            want = std::atan2(dy, dx);
        }
    }

    // ---- 3. 转向：每帧最多转 turnRate * dt ----
    //
    //  ★ 这就是"扭动"的一半：有限转向速率让蛇走弧线而不是折线。
    const float diff = wrapPi(want - m_heading);
    const float maxTurn = m_turnRate * dt;
    m_heading += Math::clamp(diff, -maxTurn, maxTurn);

    // ---- 4. 摆动：在朝向上叠一个正弦 ----
    //
    //  ★ 这是"扭动"的另一半。只有有限转向的话，直道上是笔直的；
    //    叠上正弦之后**直道也在左右摆**，才像蛇在游。
    m_wigglePhase += TWO_PI * m_wiggleHz * dt;
    if (m_wigglePhase > TWO_PI) m_wigglePhase -= TWO_PI;

    const float wiggled = m_heading + std::sin(m_wigglePhase) * m_wiggleAmp;
    m_drawDir = wiggled;      // ★ 渲染时画眼睛要用它（不是 m_heading，见 render）

    // ---- 5. 前进 ----
    m_headX += std::cos(wiggled) * m_speed * dt;
    m_headY += std::sin(wiggled) * m_speed * dt;

    // 撞到边界 → 掉头（不让它跑出去，也不用做真正的碰撞）
    const float pad = 26.0f;
    if (m_headX < pad || m_headX > m_w - pad ||
        m_headY < pad || m_headY > m_h - pad) {
        m_headX = Math::clamp(m_headX, pad, m_w - pad);
        m_headY = Math::clamp(m_headY, pad, m_h - pad);
        m_heading = wrapPi(m_heading + kPi);
        m_wigglePhase = 0.0f;
    }

    // ---- 6. 轨迹：只有走出半个体节间距才记一个点 ----
    //
    //  ★ 为什么要这个阈值：每帧都记点的话，轨迹点会密集到
    //    "按弧长插值"退化成"直接取点"，蛇身会一节一节地跳。
    if (m_trail.empty()) {
        m_trail.push_back(sf::Vector2f(m_headX, m_headY));
    } else {
        const sf::Vector2f& last = m_trail.back();
        const float dx = m_headX - last.x, dy = m_headY - last.y;
        if (dx * dx + dy * dy >= (kSegSpacing * 0.5f) * (kSegSpacing * 0.5f)) {
            m_trail.push_back(sf::Vector2f(m_headX, m_headY));
        }
    }

    // 轨迹长度封顶：够铺满身体再多留一段就行
    const std::size_t maxPts = static_cast<std::size_t>(kSegCount) * 4 + 8;
    if (m_trail.size() > maxPts) {
        m_trail.erase(m_trail.begin(),
                      m_trail.begin() + static_cast<std::ptrdiff_t>(m_trail.size() - maxPts));
    }

    // ---- 7. 涟漪衰减 ----
    for (auto& r : m_ripples) r.life -= dt * 1.8f;
    m_ripples.erase(
        std::remove_if(m_ripples.begin(), m_ripples.end(),
                       [](const Ripple& r) { return r.life <= 0.0f; }),
        m_ripples.end());
}

// ---------------------------------------------------------------------------
//  渲染
//
//  ★ 半透明的做法（两层）
//    ① 先铺一层不透明的深色底 —— 这样菜单文字永远看得清，
//       不会因为背后正好有条蛇而读不出来。
//    ② 动效元素本身用中等 alpha（食物 150~190、蛇 130~170）——
//       于是它"透"得像水印，又不会抢文字。
// ===========================================================================
void MenuBackdrop::render(sf::RenderWindow& window) const {
    // ---- 底色：竖直渐变的感觉用一个矩形 + 两个叠加矩形近似 ----
    {
        sf::RectangleShape bg(sf::Vector2f(m_w, m_h));
        bg.setFillColor(sf::Color(14, 18, 26));
        window.draw(bg);

        // 中间亮一点，四周暗（近似"暗角"）
        sf::RectangleShape glow(sf::Vector2f(m_w * 0.7f, m_h * 0.7f));
        glow.setOrigin(glow.getSize().x / 2.0f, glow.getSize().y / 2.0f);
        glow.setPosition(m_w / 2.0f, m_h / 2.0f);
        glow.setFillColor(sf::Color(30, 58, 68, 90));
        window.draw(glow);
    }

    // ---- 网格（"地图"的暗示；很淡，只是纹理）----
    {
        const float step = 60.0f;
        sf::RectangleShape v(sf::Vector2f(1.0f, m_h));
        v.setFillColor(sf::Color(90, 130, 150, 20));
        for (float x = 0.0f; x < m_w; x += step) {
            v.setPosition(x, 0.0f);
            window.draw(v);
        }
        sf::RectangleShape h(sf::Vector2f(m_w, 1.0f));
        h.setFillColor(sf::Color(90, 130, 150, 20));
        for (float y = 0.0f; y < m_h; y += step) {
            h.setPosition(0.0f, y);
            window.draw(h);
        }
    }

    // ---- 食物 ----
    //
    // ★ 尺寸走 kFoodRadius（= 游戏的 FOOD_RADIUS × kScale）——
    //   第一版用的 4~6.5 随机半径，和游戏的固定 6.0 不是一回事，
    //   看起来"食物比蛇还大"。
    for (const auto& f : m_foods) {
        // 外圈光晕（游戏里没有，但背景是暗色的，加一点光晕更好看，
        //  而且它很淡，不会改变"食物是一个小红点"的观感）
        sf::CircleShape halo(f.r * 2.4f);
        halo.setOrigin(halo.getRadius(), halo.getRadius());
        halo.setPosition(f.x, f.y);
        halo.setFillColor(sf::Color(255, 107, 107, 30));
        window.draw(halo);

        sf::CircleShape dot(f.r);
        dot.setOrigin(f.r, f.r);
        dot.setPosition(f.x, f.y);
        // ★ 颜色 = 游戏里普通食物的颜色（GameScene/FoodManager 用的 255,107,107）
        //   alpha 不同只是因为这里是半透明背景层
        dot.setFillColor(sf::Color(255, 107, 107, 200));
        window.draw(dot);
    }

    // ---- 吃掉的涟漪 ----
    for (const auto& r : m_ripples) {
        const float a = Math::clamp(r.life, 0.0f, 1.0f);
        const float rad = 8.0f + (1.0f - a) * 26.0f;
        sf::CircleShape ring(rad);
        ring.setOrigin(rad, rad);
        ring.setPosition(r.x, r.y);
        ring.setFillColor(sf::Color::Transparent);
        ring.setOutlineThickness(2.0f);
        ring.setOutlineColor(sf::Color(255, 210, 120,
                                       static_cast<sf::Uint8>(200.0f * a)));
        window.draw(ring);
    }

    // ---- 蛇 ----
    //
    // ★★ 画法与 `Snake::render()` 【逐条对应】（2026-09-27 统一外形）
    //
    //    游戏里的画法是：
    //      ① 先画**所有体节**：同一个半径 `m_segmentRadius`、
    //         同一个颜色 `m_color`、**完全不透明**、不渐隐不变细
    //      ② 再画头：半径 `m_headRadius`（比体节大 2px），
    //         颜色 = body 的每个通道 **+40**（更亮）
    //      ③ 最后画两只黑眼睛（只有 AI 画）：半径 4、
    //         位置在头前方 8px、左右各偏 6px
    //
    //    ⚠️ 第一版我在体节上加了**渐隐 + 变细**（alpha 150→105、半径 9.5→6.5），
    //       还时不时叠个高光 —— 那是我自己编的"好看"，不是这个游戏的蛇。
    //       负责人一眼就看出来了："蛇的样子有点奇怪。"
    //       **统一外形的正确做法是复刻画法，不是复刻感觉。**
    if (!m_trail.empty()) {
        // 按弧长在轨迹上取等距的 kSegCount 个点
        std::vector<sf::Vector2f> pts;
        pts.reserve(static_cast<std::size_t>(kSegCount));

        // 从头部开始（轨迹最后一个点最新）
        float acc = 0.0f;
        std::size_t idx = m_trail.size() - 1;
        pts.push_back(m_trail[idx]);
        float need = kSegSpacing;
        while (pts.size() < static_cast<std::size_t>(kSegCount) && idx > 0) {
            const sf::Vector2f& a = m_trail[idx];
            const sf::Vector2f& b = m_trail[idx - 1];
            const float dx = a.x - b.x, dy = a.y - b.y;
            const float segLen = std::sqrt(dx * dx + dy * dy);
            if (segLen <= 1.0e-4f) { --idx; continue; }

            if (acc + segLen >= need) {
                const float t = (need - acc) / segLen;
                pts.push_back(sf::Vector2f(b.x + dx * t, b.y + dy * t));
                need += kSegSpacing;
            } else {
                acc += segLen;
                --idx;
            }
        }

        // ---- ① 体节：全部同半径同色（和游戏一样，不渐隐、不变细）----
        //
        //  ★ 画序也必须和游戏一致：`Snake::render()` 是**先画头、再画体节**，
        //    于是体节会盖住头的一小部分。看着像头略小一点，
        //    但那正是游戏里的样子 —— 统一外形要连画序一起照抄。
        //    （`pts[0]` 是头，因为轨迹最后一个点就是蛇头。）
        sf::CircleShape segShape(kSegRadius);
        segShape.setOrigin(kSegRadius, kSegRadius);
        segShape.setFillColor(sf::Color(kSnakeColor.r, kSnakeColor.g,
                                        kSnakeColor.b, 200));

        // ---- ② 头部：比体节大，颜色每个通道 +40（和游戏一致）----
        if (!pts.empty()) {
            const sf::Vector2f hp = pts[0];
            sf::CircleShape headShape(kHeadRadius);
            headShape.setOrigin(kHeadRadius, kHeadRadius);
            headShape.setPosition(hp);
            headShape.setFillColor(sf::Color(
                static_cast<sf::Uint8>(std::min(255, kSnakeColor.r + 40)),
                static_cast<sf::Uint8>(std::min(255, kSnakeColor.g + 40)),
                static_cast<sf::Uint8>(std::min(255, kSnakeColor.b + 40)),
                210));
            window.draw(headShape);
        }

        for (const auto& p : pts) {
            segShape.setPosition(p);
            window.draw(segShape);
        }

        // ---- ③ 眼睛：黑、前方 8×kScale、左右 6×kScale（最后画，在最上层）----
        if (!pts.empty()) {
            const sf::Vector2f hp = pts[0];
            //  ★ 用**实际朝向**（含摆动）算前向量，不是用 m_heading ——
            //    否则蛇在扭动时眼睛会"看着直走的方向"，很怪。
            const sf::Vector2f fwd(std::cos(m_drawDir), std::sin(m_drawDir));
            const sf::Vector2f left(-fwd.y, fwd.x);
            sf::CircleShape eye(kEyeRadius);
            eye.setOrigin(kEyeRadius, kEyeRadius);
            eye.setFillColor(sf::Color::Black);
            eye.setPosition(hp.x + fwd.x * kEyeForward + left.x * kEyeSide,
                            hp.y + fwd.y * kEyeForward + left.y * kEyeSide);
            window.draw(eye);
            eye.setPosition(hp.x + fwd.x * kEyeForward - left.x * kEyeSide,
                            hp.y + fwd.y * kEyeForward - left.y * kEyeSide);
            window.draw(eye);
        }
    }
}

}  // namespace systems
