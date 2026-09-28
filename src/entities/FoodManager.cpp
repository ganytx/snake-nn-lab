#include "FoodManager.h"
#include "../utils/Constants.h"
#include "../utils/Math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ==========================================
// 构造函数
// ==========================================
FoodManager::FoodManager(float mapWidth, float mapHeight)
    : m_mapWidth(mapWidth)
    , m_mapHeight(mapHeight)
{
}

// ==========================================
//  应用配置
// ==========================================
void FoodManager::applyConfig(float foodPerSecond, int foodSegs, int corpseRatio,
                               int corpseScore, int corpseSegs,
                               float foodLifetime, float corpseLifetime) {
    // 防止非法值导致异常
    m_foodPerSecond = (foodPerSecond > 0.0f) ? foodPerSecond : 0.0f;
    m_foodSegs      = std::max(1, foodSegs);
    m_corpseRatio   = std::max(1, corpseRatio);
    m_corpseScore   = std::max(1, corpseScore);
    m_corpseSegs    = std::max(1, corpseSegs);
    m_foodLifetime   = (foodLifetime   > 0.0f) ? foodLifetime   : 1.0e9f;
    m_corpseLifetime = (corpseLifetime > 0.0f) ? corpseLifetime : 1.0e9f;
}

// ==========================================
//  每帧更新 — 食物生成 + 超时清理 + 圈内统计
//
//  ★ 为什么用【累加器】而不是"计时器到点生成"：
//    生成速率是「蛇数 / 9」个每秒，会出现 0.22、1.33 这种小数。
//    累加器能正确处理小数 —— 攒够 1 个才真的生成一个，
//    所以 2 条蛇时大约每 4.5 秒出一个，18 条蛇时每秒出两个。
//    用整数计时器就只能四舍五入，蛇数一变配比就歪了。
// ==========================================
void FoodManager::update(float dt) {
    m_zoneTimer += dt;

    // ---- 1. 分发食物生成 ----
    if (m_foodPerSecond > 0.0f && m_sched.spawning(m_zoneTimer)) {
        m_spawnAccum += m_foodPerSecond * dt;

        const int active = m_sched.activeZone(m_zoneTimer,
                                              static_cast<int>(m_zones.size()));

        while (m_spawnAccum >= 1.0f) {
            m_spawnAccum -= 1.0f;

            // ★★ 存量上限（负责人要求"食物存在上限"，2026-09-27）
            //
            //  ★ 为什么"到了上限也必须把累加器减掉"
            //    上面那行 `m_spawnAccum -= 1.0f` 是在本判断【之前】做的，
            //    所以到了上限只是"这一个不生成"，累加器不会越攒越多。
            //    如果反过来写成"满了就整个跳过 while"，累加器会在满的时候
            //    一路涨到几百 —— 之后只要被吃掉一个，就会瞬间补上一大片。
            //
            //  ★ 为什么用 `continue` 而不是 `break`
            //    两种写法在"这一帧全都满了"时等价（下面的分支一个也不会执行），
            //    但 `continue` 的语义更直白：**这一轮不生成，接着算下一轮**。
            if (m_foodCap > 0 && m_foodCount >= m_foodCap) continue;

            if (active >= 0 && active < static_cast<int>(m_zones.size())) {
                spawnFoodInZone(m_zones[static_cast<std::size_t>(active)]);
            } else {
                // 没有配置圈 → 退化成全图随机撒（老行为）
                spawnRandomFood();
            }
            ++m_foodCount;
        }
    }

    // ---- 2. 累加寿命 + 移除超时食物 ----
    //
    // ★ 以前只有尸体食物会过期，普通食物永久留在地图上。
    //   食物圈玩法要求普通食物也有寿命（docs/70 第 4 节），
    //   否则一轮没吃完的食物会无限堆积，玩法退化。
    for (auto& f : m_foods) {
        f.lifetime += dt;
    }
    m_foods.erase(
        std::remove_if(m_foods.begin(), m_foods.end(),
            [this](const Food& f) {
                return f.isCorpse ? (f.lifetime >= m_corpseLifetime)
                                  : (f.lifetime >= m_foodLifetime);
            }),
        m_foods.end()
    );

    // ---- 3. 重新统计（感知要用的"每圈几个" + 上限要用的"普通食物总数"）----
    //
    // ★ 为什么统计放在最后：上面刚删过食物，这时候统计才是当前真值。
    //   而且感知是在 Arena::step() 里、食物更新【之前】读的 ——
    //   也就是说感知读到的是上一帧末的统计。差一帧，无所谓，
    //   但必须知道这一点，否则会误以为"统计错了"。
    //
    // ★ 两者一起算：都是"遍历一遍 m_foods"，没必要走两趟。
    recountZoneFood();
}

// ==========================================
//  重新统计各圈食物数 + 普通食物总数
//
//  ★ 为什么"普通食物总数"是【重算】而不是"生成 +1、被吃 -1"
//    增量维护要在【每一个】改到食物的路径上都写对：圈内生成、全图生成、
//    指定位置生成、种子撒、被吃掉（removeFood）、寿命到期（上面的擦除）。
//    漏一处就是一个"上限永远不生效"或"上限提前生效"的静默 bug。
//    反正每帧本来就要遍历一遍统计圈内数量，顺手多数一个计数器不会更慢。
//
//  ⚠️ 尸体【不计入】：它是别的蛇死了留下的，不该被自然生成的上限卡住
//     （大混战的核心食物来源就是尸体）。
// ==========================================
void FoodManager::recountZoneFood() {
    for (auto& z : m_zones) z.foodCount = 0;

    m_foodCount = 0;
    for (const auto& f : m_foods) {
        if (f.isCorpse) continue;
        ++m_foodCount;

        for (auto& z : m_zones) {
            if (z.contains(f.position.x, f.position.y)) {
                ++z.foodCount;
                break;      // 圈之间不重叠，算一次就够
            }
        }
    }
}

// ==========================================
//  全图随机撒 n 个食物
// ==========================================
void FoodManager::seedUniform(int n) {
    // ★ 守上限：要到上限了就少撒（负责人要求"初始时有大量食物"，但别冲破上限）
    for (int i = 0; i < n; ++i) {
        if (m_foodCap > 0 && m_foodCount >= m_foodCap) break;
        spawnRandomFood();
        ++m_foodCount;
    }
    recountZoneFood();
}

// ==========================================
//  在指定位置放一个普通食物
//
//  ★ 用途：最小学习验证（"朝前觅食"任务）。
//    食物位置【完全确定】，所以最优策略是显而易见的 ——
//    如果连这个都学不会，问题就在学习链路，不在奖励设计。
// ==========================================
void FoodManager::spawnAt(float x, float y) {
    // 夹在地图内，别放到外面去（调用方算出来的位置可能贴边）
    const float margin = Constants::FOOD_RADIUS + 2.0f;
    const float cx = Math::clamp(x, margin, m_mapWidth  - margin);
    const float cy = Math::clamp(y, margin, m_mapHeight - margin);

    m_foods.emplace_back(cx, cy,
        Constants::FOOD_RADIUS,
        Constants::FOOD_SCORE,
        m_foodSegs,
        false  // isCorpse = false
    );
    // ★ 用重算而不是 ++m_foodCount ——
    //   计数器只能有一个来源（recountZoneFood），否则就会出现
    //   "这条路径 +1 了、那条路径忘了 +1"的静默偏差。
    //   本函数不在每帧热路径上（只在开局定点食物时调一次），重算的开销无所谓。
    recountZoneFood();
}

// ==========================================
//  蛇死亡时生成尸体食物
//
//  每隔 corpseRatio 个体节生成一个食物，
//  避免一条长蛇死后铺满整个地图。
// ==========================================
void FoodManager::spawnCorpseFood(const Snake& snake) {
    const auto& segments = snake.getSegments();
    for (size_t i = 0; i < segments.size(); i += static_cast<size_t>(m_corpseRatio)) {
        Food food(
            segments[i].x, segments[i].y,
            Constants::CORPSE_FOOD_RADIUS,
            m_corpseScore,
            m_corpseSegs,
            true
        );
        m_foods.push_back(food);
    }
}

// ==========================================
//  移除食物 — 被蛇吃掉时调用
// ==========================================
void FoodManager::removeFood(size_t index) {
    if (index < m_foods.size()) {
        // 使用 swap-and-pop 技巧：O(1) 删除，不保持顺序
        m_foods[index] = std::move(m_foods.back());
        m_foods.pop_back();
    }
}

// ==========================================
//  在圈内生成一个食物
//
//  ★ 用【圆内均匀采样】而不是极坐标里对 r 均匀采样。
//    对 r 均匀采样会让食物挤在圆心附近（面积 ∝ r²，半径小的环面积小）。
//    做法：先在 [0,R] 均匀取 r，再取 r' = R*sqrt(u) 做修正；
//    这里直接用的是标准写法：r = R * sqrt(u)。
// ==========================================
void FoodManager::spawnFoodInZone(const arena::FoodZone& z) {
    auto u = []() {
        return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    };

    const float ang = u() * 6.283185307179586f;
    const float r   = z.radius * std::sqrt(u());

    const float px = z.x + std::cos(ang) * r;
    const float py = z.y + std::sin(ang) * r;

    // 别生成到地图外面去（圈本来就该在地图内，这是兜底）
    const float margin = Constants::FOOD_RADIUS + 2.0f;
    const float cx = Math::clamp(px, margin, m_mapWidth  - margin);
    const float cy = Math::clamp(py, margin, m_mapHeight - margin);

    m_foods.emplace_back(cx, cy,
        Constants::FOOD_RADIUS,
        Constants::FOOD_SCORE,
        m_foodSegs,
        false);
}

// ==========================================
//  全图随机位置生成一个食物（只在没有配置圈时用）
// ==========================================
void FoodManager::spawnRandomFood() {
    float x = Math::randomFloat(20.0f, m_mapWidth - 20.0f);
    float y = Math::randomFloat(20.0f, m_mapHeight - 20.0f);
    m_foods.emplace_back(x, y,
        Constants::FOOD_RADIUS,
        Constants::FOOD_SCORE,
        m_foodSegs,
        false  // isCorpse = false
    );
}

// ==========================================
//  渲染所有食物
//
//  普通食物：红色圆点
//  尸体食物：稍大，橙红色，半透明（随时间逐渐更透明）
// ==========================================
void FoodManager::render(sf::RenderWindow& window) const {
    sf::CircleShape shape;
    shape.setOrigin(shape.getRadius(), shape.getRadius());

    for (const auto& food : m_foods) {
        shape.setRadius(food.radius);
        shape.setOrigin(food.radius, food.radius);
        shape.setPosition(food.position);

        if (food.isCorpse) {
            // 尸体食物：橙红色，随时间变淡
            float alpha = 1.0f - (food.lifetime / m_corpseLifetime);
            alpha = Math::clamp(alpha, 0.2f, 1.0f);
            shape.setFillColor(sf::Color(255, 140, 60, static_cast<sf::Uint8>(255 * alpha)));
        } else {
            // 普通食物：亮红色。也随时间轻微变淡，让"快过期了"看得出来
            float alpha = 1.0f - (food.lifetime / m_foodLifetime) * 0.6f;
            alpha = Math::clamp(alpha, 0.4f, 1.0f);
            shape.setFillColor(sf::Color(255, 107, 107, static_cast<sf::Uint8>(255 * alpha)));
        }

        window.draw(shape);
    }
}
