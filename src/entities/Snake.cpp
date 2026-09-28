#include "Snake.h"
#include "../utils/Math.h"
#include "../utils/Constants.h"
#include <algorithm>
#include <cmath>

// ==========================================
// 构造函数 — 初始化蛇的位置、方向、身体
//
// 在蛇出生时预先填充一条"直线轨迹"，
// 这样蛇一出现就有完整的身体，而不是只有头。
// ==========================================
Snake::Snake(float startX, float startY, float startAngle,
             const sf::Color& bodyColor, bool isPlayer)
    : m_x(startX)
    , m_y(startY)
    , m_angle(startAngle)
    , m_targetAngle(startAngle)
    , m_currentSpeed(Constants::SNAKE_SPEED)
    , m_headRadius(Constants::SNAKE_HEAD_RADIUS)
    , m_segmentRadius(Constants::SNAKE_SEGMENT_RADIUS)
    , m_segmentSpacing(Constants::SNAKE_SEGMENT_SPACING)
    , m_normalSpeed(Constants::SNAKE_SPEED)
    , m_boostSpeed(Constants::SNAKE_BOOST_SPEED)
    , m_turnRate(Constants::SNAKE_TURN_RATE)
    , m_turnEase(Constants::SNAKE_TURN_EASE)
    , m_segmentCount(Constants::SNAKE_INITIAL_SEGMENTS)
    , m_color(bodyColor)
    , m_isPlayer(isPlayer)
{
    // 每条蛇的转向起伏初相位都随机，否则一群蛇会整齐划一地一起快一起慢
    m_turnWobblePhase = Math::randomFloat(0.0f, Math::PI * 2.0f);

    // 在出生点铺一条直线轨迹，让蛇一出现就有完整的身体（而不是只有头）
    prefillTrail();
}

// ==========================================
// 预填充轨迹 — 在出生点沿"朝向的反方向"铺一条直线
//
// 目的：蛇一出现就有完整的身体。
// 长度按"当前体节数需要的总长"再多留 20% 余量，这样即使随后
// setSegmentCount() 把蛇调长，体节也不会全部挤在尾巴上。
// ==========================================
void Snake::prefillTrail() {
    m_trail.clear();

    // ★ 紧凑出生：整条身体堆在一个点上（见 setCompactSpawn 的说明）
    //
    //   注意 still 要放【足够的点数】—— updateSegments() 是按轨迹点的
    //   累计弧长插值的，点太少会插不出 12 个体节。
    //   这里按原逻辑的密度铺点，只是把位置全部设成同一个点。
    if (m_compactSpawn) {
        const float needed = m_segmentCount * m_segmentSpacing * 1.2f;
        const float step   = std::max(1.0f, m_segmentSpacing * 0.5f);
        for (float d = 0.0f; d < needed; d += step) {
            m_trail.push_back(sf::Vector2f(m_x, m_y));
        }
        m_trail.push_back(sf::Vector2f(m_x, m_y));   // 补终点，保证点数足够
        updateSegments();
        return;
    }

    sf::Vector2f forward = Math::angleToVector(m_angle);
    sf::Vector2f backward(-forward.x, -forward.y);

    // 需要覆盖的总长度（体节间距 × 体节数，留 20% 余量）
    float needed = m_segmentCount * m_segmentSpacing * 1.2f;
    // 采样步长取间距的一半：点足够密，updateSegments() 插值才不会出现折角
    float step = std::max(1.0f, m_segmentSpacing * 0.5f);

    for (float d = 0.0f; d < needed; d += step) {
        m_trail.push_back(sf::Vector2f(m_x + backward.x * d,
                                       m_y + backward.y * d));
    }
    // 补上终点，保证轨迹长度一定 >= needed
    m_trail.push_back(sf::Vector2f(m_x + backward.x * needed,
                                   m_y + backward.y * needed));

    updateSegments();
}

// ==========================================
// 每帧更新 — 蛇的核心逻辑
//
// 6个步骤按顺序执行：
//   1. 平滑转向（限制 turnRate）
//   2. 速度平滑过渡（lerp）
//   3. 移动头部
//   4. 记录新头部到 trail
//   5. 沿 trail 计算体节位置
//   6. 清理 trail 尾部
// ==========================================
void Snake::update(float dt) {
    if (!m_isAlive) return;

    // 存活时间（用于转向起伏的相位）
    // 注意不能复用 m_timeSinceEat —— 它每吃一口东西就会被清零
    m_age += dt;

    // ---- 饥饿计时（加速时消耗×1.5） ----
    m_timeSinceEat += dt * (m_isBoosting ? 1.5f : 1.0f);

    // ---- 第1步：平滑转向（两层限制，不是匀速） ----
    //
    // 第1层 · 机动性上限：不管怎么转，一帧转过的角度不超过 turnRate*dt。
    //
    // 第2层 · 缓动：本帧想转的角度 = 角度差 × m_turnEase × dt。
    //   角度差大（离目标远）→ 想要的角度超过上限 → 按上限全速转；
    //   角度差小（快对准了）→ 想要的角度低于上限 → 自动减速，平滑贴合目标。
    //   所以整个转头过程是"先快后慢"的曲线，而不是匀速转到位。
    //
    // 另外 AI 蛇的"上限"本身还会随时间缓慢起伏（wobble），
    // 于是同一条蛇在不同时刻的转速也不同，进一步去掉机械感。
    float rateScale = 1.0f;
    if (m_turnWobbleAmp > 0.0f) {
        rateScale = 1.0f + m_turnWobbleAmp *
                    std::sin(m_age * m_turnWobbleSpeed + m_turnWobblePhase);
    }
    float maxTurn = m_turnRate * rateScale * dt;

    float diff = Math::angleDifference(m_angle, m_targetAngle);
    // 缓动系数乘 dt 后夹到 1 以内：否则一帧可能转过头，出现来回抖动
    float ease = std::min(1.0f, m_turnEase * dt);
    float turn = Math::clamp(diff * ease, -maxTurn, maxTurn);
    m_angle += turn;
    m_angle = Math::normalizeAngle(m_angle);

    // ---- 第2步：速度平滑过渡 ----
    // 加速/减速不是瞬间切换，而是用 lerp 平滑过渡
    float targetSpeed = m_isBoosting ? m_boostSpeed : m_normalSpeed;
    m_currentSpeed = Math::lerp(m_currentSpeed, targetSpeed, dt * 5.0f);

    // ---- 第3步：移动头部 ----
    // 注意：不再限制在地图内，碰撞系统会检测撞墙并触发死亡
    sf::Vector2f forward = Math::angleToVector(m_angle);
    m_x += forward.x * m_currentSpeed * dt;
    m_y += forward.y * m_currentSpeed * dt;

    // ---- 第4步：记录头部轨迹 ----
    m_trail.push_front(sf::Vector2f(m_x, m_y));

    // ---- 第5步：沿轨迹计算体节位置 ----
    updateSegments();

    // ---- 第6步：清理尾部旧轨迹 ----
    trimTrail();
}

// ==========================================
// 体节位置计算 — Trail Buffer 算法的核心
//
// 从头部出发，沿 trail 向后走，
// 每隔 segmentSpacing 距离放置一个体节。
// 当目标距离落在两个轨迹点之间时，线性插值得到精确位置。
//
// 复杂度 O(segmentCount + trailWalked)，每帧只走一遍。
// ==========================================
void Snake::updateSegments() {
    m_segmentPositions.clear();
    if (m_trail.empty()) return;

    float accumulatedDist = 0.0f;
    auto it = m_trail.begin();
    auto prev = m_trail.begin();

    for (int i = 0; i < m_segmentCount; ++i) {
        // 第 i 个体节应距离头部 (i+1) * spacing
        float targetDist = (i + 1) * m_segmentSpacing;

        // 沿轨迹向前走，累计距离直到达到目标
        while (it != m_trail.end() && accumulatedDist < targetDist) {
            prev = it;
            ++it;
            if (it != m_trail.end()) {
                accumulatedDist += Math::distance(*prev, *it);
            }
        }

        if (accumulatedDist >= targetDist && it != m_trail.end()) {
            // 目标在两个轨迹点之间 → 线性插值
            float overshoot = accumulatedDist - targetDist;
            float segLen = Math::distance(*prev, *it);
            if (segLen > 0.001f) {
                float t = overshoot / segLen;
                m_segmentPositions.push_back(sf::Vector2f(
                    it->x + (prev->x - it->x) * t,
                    it->y + (prev->y - it->y) * t
                ));
            } else {
                m_segmentPositions.push_back(*it);
            }
        } else {
            // 轨迹不够长，用当前最后一个有效点
            m_segmentPositions.push_back(*prev);
        }
    }
}

// ==========================================
// 清理过旧的轨迹点
//
// 轨迹只需保留"所有体节都需要"的长度。
// 从头部开始测量，超过所需长度的尾部全部删除。
// 这防止 trail 无限增长导致内存和性能问题。
// ==========================================
void Snake::trimTrail() {
    float maxTrailLength = m_segmentCount * m_segmentSpacing + 100.0f;

    float totalLen = 0.0f;
    auto it = m_trail.begin();
    auto prev = m_trail.begin();

    while (it != m_trail.end() && totalLen < maxTrailLength) {
        prev = it;
        ++it;
        if (it != m_trail.end()) {
            totalLen += Math::distance(*prev, *it);
        }
    }

    if (it != m_trail.end()) {
        m_trail.erase(it, m_trail.end());
    }
}

// ==========================================
// 食物计数 — 每吃满 FOOD_SEGMENTS (5) 个 +1 体节
// ==========================================
void Snake::addFoodEaten() {
    m_foodCounter++;
    if (m_foodCounter >= m_foodNeeded) {
        m_foodCounter -= m_foodNeeded;
        m_segmentCount += 1;
    }
    resetStarvation();  // 吃东西重置饥饿计时
}

// ==========================================
// 尸体食物计数 — 每吃满 CORPSE_FOOD_SEGMENTS (2) 个 +1 体节
// ==========================================
void Snake::addCorpseEaten() {
    m_corpseCounter++;
    if (m_corpseCounter >= m_corpseNeeded) {
        m_corpseCounter -= m_corpseNeeded;
        m_segmentCount += 1;
    }
    resetStarvation();  // 吃尸体也重置饥饿计时
}

// ==========================================
// 直接设置体节数 — 出生时设定初始长度（玩家按配置，AI随机）
//
// ⚠ 只应该在"出生那一刻"调用：本函数会按新的体节数重建一条
//   直线轨迹，游戏中途调用会让身体瞬间拉成一条直线。
//   游戏内自然生长走的是 addFoodEaten()/addCorpseEaten() 的
//   m_segmentCount += 1，那条路径不动轨迹，所以不会有这个问题。
// ==========================================
void Snake::setSegmentCount(int count) {
    m_segmentCount = count;
    prefillTrail();
}

// ==========================================
// 设置转向个性 — 主要给AI用
//
// 缓动系数决定"多急"，起伏幅度/快慢决定"转速随时间怎么晃"。
// 参数都做了下限保护，避免出现转不动（卡死）或抖动的极端情况。
// ==========================================
void Snake::setTurnPersonality(float turnEase, float wobbleAmp, float wobbleSpeed) {
    m_turnEase        = std::max(0.5f, turnEase);
    m_turnWobbleAmp   = Math::clamp(wobbleAmp, 0.0f, 0.9f);  // 上限0.9：保证转速恒为正
    m_turnWobbleSpeed = std::max(0.0f, wobbleSpeed);
    m_turnWobblePhase = Math::randomFloat(0.0f, Math::PI * 2.0f);
}

// ==========================================
// 标记死亡 — 碰撞系统检测到致命碰撞时调用
// ==========================================
void Snake::die() {
    m_isAlive = false;
}

// ==========================================
// 击杀+1 — 你的身体被别人撞到导致对方死亡时调用
// ==========================================
void Snake::addKill() {
    m_kills++;
}

// ==========================================
// 重置饥饿计时 — 吃东西或刚出生时调用
// ==========================================
void Snake::resetStarvation() {
    m_timeSinceEat = 0.0f;
}

// ==========================================
// 应用配置 — 覆盖构造时的默认参数
// ==========================================
void Snake::applyConfig(float speed, float boost, float turn, float spacing,
                         int foodNeeded, int corpseNeeded) {
    m_normalSpeed = std::max(10.0f, speed);
    m_boostSpeed = std::max(20.0f, boost);
    m_turnRate = std::max(0.5f, turn);
    m_segmentSpacing = std::max(2.0f, spacing);
    m_currentSpeed = m_normalSpeed;
    m_foodNeeded = std::max(1, foodNeeded);
    m_corpseNeeded = std::max(1, corpseNeeded);
}

// ==========================================
// 增加分数
// ==========================================
void Snake::addScore(int points) {
    m_score += points;
}

// ==========================================
// 设置头部贴图 — 玩家蛇用图片替代圆形头部
//
// 调用方（GameScene）负责加载并持有 sf::Texture，
// 这里只存指针引用，不管理纹理生命周期。
// 如果不设置贴图（AI蛇），头部渲染回退为圆形。
// ==========================================
void Snake::setHeadTexture(const sf::Texture& texture) {
    m_headTexture = &texture;
    m_headSprite.setTexture(*m_headTexture);
    // 设置旋转中心为图片中心，这样旋转时绕着图片中心转
    sf::FloatRect bounds = m_headSprite.getLocalBounds();
    m_headSprite.setOrigin(bounds.width / 2.0f, bounds.height / 2.0f);
}

// ==========================================
// 渲染蛇的完整外观
//
// 绘制顺序（从底到顶）：
//   1. 所有体节 — 圆形，蛇身颜色
//   2. 头部 — 比体节稍大、颜色稍亮的圆形
//   3. 眼睛 — 两个小白点，指示朝向
// ==========================================
void Snake::render(sf::RenderWindow& window) const {
    if (!m_isAlive) return;

    // ---- 加速时的亮度脉冲系数 ----
    float bright = 1.0f;
    if (m_isBoosting) {
        // 正弦波产生"一阵一阵"的亮度脉动
        float pulse = (std::sin(m_timeSinceEat * 15.0f) + 1.0f) * 0.5f;  // 0~1
        bright = 1.0f + pulse * 0.35f;  // 亮度在 100%~135% 之间波动
    }

    // ---- 体节（加速时整体变亮） ----
    sf::CircleShape segmentShape(m_segmentRadius);
    segmentShape.setOrigin(m_segmentRadius, m_segmentRadius);
    sf::Color segColor = m_color;
    if (m_isBoosting) {
        segColor.r = std::min(255, static_cast<int>(segColor.r * bright));
        segColor.g = std::min(255, static_cast<int>(segColor.g * bright));
        segColor.b = std::min(255, static_cast<int>(segColor.b * bright));
    }
    segmentShape.setFillColor(segColor);

    for (const auto& pos : m_segmentPositions) {
        segmentShape.setPosition(pos);
        window.draw(segmentShape);
    }

    // ---- 头部 ----
    if (m_headTexture) {
        // 玩家：图片头部（不调亮度，因为贴图无法简单改色）
        m_headSprite.setPosition(m_x, m_y);
        m_headSprite.setRotation(m_angle * 180.0f / Math::PI);
        float texW = m_headSprite.getLocalBounds().width;
        float texH = m_headSprite.getLocalBounds().height;
        m_headSprite.setScale((m_headRadius * 2.0f) / texW,
                              (m_headRadius * 2.0f) / texH);
        window.draw(m_headSprite);
    } else {
        // AI：圆形头部（加速时也变亮）
        sf::CircleShape headShape(m_headRadius);
        headShape.setOrigin(m_headRadius, m_headRadius);
        headShape.setPosition(m_x, m_y);
        sf::Color headColor = m_color;
        headColor.r = std::min(255, static_cast<int>((headColor.r + 40) * bright));
        headColor.g = std::min(255, static_cast<int>((headColor.g + 40) * bright));
        headColor.b = std::min(255, static_cast<int>((headColor.b + 40) * bright));
        headShape.setFillColor(headColor);
        window.draw(headShape);
        // ---- 眼睛（只有AI画） ----
        sf::Vector2f forward = Math::angleToVector(m_angle);
        sf::Vector2f left(-forward.y, forward.x);
        sf::CircleShape eye(4.0f);
        eye.setOrigin(4.0f, 4.0f);
        eye.setFillColor(sf::Color::Black);
        eye.setPosition(m_x + forward.x * 8.0f + left.x * 6.0f,
                        m_y + forward.y * 8.0f + left.y * 6.0f);
        window.draw(eye);
        eye.setPosition(m_x + forward.x * 8.0f - left.x * 6.0f,
                        m_y + forward.y * 8.0f - left.y * 6.0f);
        window.draw(eye);
    }

    
}
