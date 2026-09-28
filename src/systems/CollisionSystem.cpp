#include "CollisionSystem.h"
#include <cstdio>
#include "../utils/Math.h"
#include "../utils/Constants.h"

#include <algorithm>
#include <functional>

namespace {

// 空间网格的格子边长。
//   太小 → 覆盖同样范围要遍历很多格，查询变慢
//   太大 → 每格里塞的实体太多，退化成线性扫描
// 128px ≈ 4 个蛇身直径；8000x8000 的地图切成 63x63 ≈ 4000 格，清空成本可忽略。
constexpr float GRID_CELL_SIZE = 128.0f;

}  // namespace

CollisionSystem::CollisionSystem(float mapWidth, float mapHeight)
    : m_mapWidth(mapWidth)
    , m_mapHeight(mapHeight)
{
    m_grid.init(m_mapWidth, m_mapHeight, GRID_CELL_SIZE);
}

void CollisionSystem::applyConfig(float forgiveness, int killScore) {
    m_forgiveness = forgiveness;
    m_killScore = killScore;
}

// ==========================================
// 检查所有碰撞 — 每帧对每条活蛇执行
// ==========================================
void CollisionSystem::checkAll(std::vector<Snake*>& snakes, FoodManager& foodManager) {
    rebuildIndex(snakes, foodManager);

    const std::size_t n = snakes.size();
    for (std::size_t i = 0; i < n; ++i) {
        Snake* snake = snakes[i];
        if (!snake->isAlive()) continue;

        // 保持与优化前完全相同的执行顺序（撞墙 → 撞蛇 → 吃食物），
        // 这样对局结果可比，优化只改变复杂度、不改变行为。
        checkWallCollision(*snake);
        checkSnakeCollision(*snake, snakes, i);
        checkFoodCollision(*snake, foodManager);
    }

    // 延迟移除被吃掉的食物。
    // removeFood() 用的是 swap-and-pop，必须【按下标从大到小】删，
    // 否则删掉小下标时会把后面还没处理的下标搬走，导致删错东西。
    if (!m_eatenList.empty()) {
        std::sort(m_eatenList.begin(), m_eatenList.end(), std::greater<std::size_t>());
        for (std::size_t idx : m_eatenList) {
            foodManager.removeFood(idx);
        }
        m_eatenList.clear();
    }
}

// ==========================================
// 重建空间网格
// ==========================================
void CollisionSystem::rebuildIndex(std::vector<Snake*>& snakes, FoodManager& foodManager) {
    m_grid.init(m_mapWidth, m_mapHeight, GRID_CELL_SIZE);   // 尺寸不变时不会重新分配

    for (std::size_t i = 0; i < snakes.size(); ++i) {
        Snake* snake = snakes[i];
        if (!snake->isAlive()) continue;      // 死蛇不参与碰撞，也不该挡住别人

        const auto& segs = snake->getSegments();
        for (const auto& seg : segs) {
            m_grid.insert(seg.x, seg.y, static_cast<int>(i), sim::GridKind::Body);
        }

        // 蛇头单独一类：对头相撞的规则和撞身体不同（双方都死，且不给击杀分）
        const sf::Vector2f head = snake->getHeadPosition();
        m_grid.insert(head.x, head.y, static_cast<int>(i), sim::GridKind::Head);
    }

    auto& foods = foodManager.getFoods();
    for (std::size_t i = 0; i < foods.size(); ++i) {
        m_grid.insert(foods[i].position.x, foods[i].position.y,
                      static_cast<int>(i), sim::GridKind::Food);
    }

    // 重置本帧的"已吃掉"标记（容量不变，不会重新分配）
    m_foodEaten.assign(foods.size(), 0);
    m_eatenList.clear();
}

// ==========================================
// 墙壁碰撞 — 蛇头超出地图边界则死亡
// ==========================================
void CollisionSystem::checkWallCollision(Snake& snake) {
    float r = snake.getHeadRadius() - m_forgiveness;
    float x = snake.getX();
    float y = snake.getY();

    if (x - r < 0.0f || x + r > m_mapWidth ||
        y - r < 0.0f || y + r > m_mapHeight) {
        snake.die();
    }
}

// ==========================================
// 蛇-蛇碰撞 — 蛇头碰到其他蛇的身体 / 蛇头
//
// 规则：
//   - 不检测自己撞自己（用户需求）
//   - 头部碰到其他蛇的身体 → 头部所属蛇死亡，对方得击杀分
//   - 对头相撞 → 双方都死，双方都不得分
//   - 忽略已死亡的蛇
// ==========================================
void CollisionSystem::checkSnakeCollision(Snake& snake,
                                          const std::vector<Snake*>& allSnakes,
                                          std::size_t snakeIndex) {
    const sf::Vector2f headPos = snake.getHeadPosition();
    const float headR = snake.getHeadRadius();

    // 查询半径：要覆盖"头半径 + 对方最大半径"的最大可能命中距离。
    // 蛇头半径 18 > 体节半径 16，所以用 18+18 再加一点余量，
    // 避免实体刚好落在格子边界时被漏掉。
    const float queryR = headR * 2.0f + 4.0f;

    m_grid.forEachNearby(headPos.x, headPos.y, queryR,
        [&](const sim::GridItem& item) -> bool {
            // ★ 必须先验类型标签，再解释 payload！
            //   蛇和食物共用同一个 payload 字段（各自存自己数组的下标），
            //   如果拿食物下标去 allSnakes[] 取，就是越界访问（会直接崩溃）。
            if (item.kind == sim::GridKind::Food) return true;   // 食物不参与蛇-蛇碰撞
            if (item.payload == static_cast<int>(snakeIndex)) return true;  // 自己

            Snake* other = allSnakes[static_cast<std::size_t>(item.payload)];
            if (!other->isAlive()) return true;

            // 队友：默认不碰撞、可穿过（需求方选定的规则）。
            // 只在"双方都有队伍且同队"时才跳过 —— teamId = -1 表示无队伍，
            // 那种情况下仍然按普通蛇处理。
            if (!m_friendlyFire) {
                const int ta = snake.getTeamId();
                const int tb = other->getTeamId();
                if (ta >= 0 && ta == tb) return true;
            }

            const sf::Vector2f p(item.x, item.y);
            const float dist = Math::distance(headPos, p);

            if (item.kind == sim::GridKind::Body) {
                if (dist < headR + other->getSegmentRadius() - m_forgiveness) {
                    snake.die();
                    other->addKill();
                    other->addScore(m_killScore);
                    return false;      // 已经死了，不用再找
                }
            } else if (item.kind == sim::GridKind::Head) {
                if (dist < headR + other->getHeadRadius() - m_forgiveness) {
                    snake.die();
                    other->die();
                    return false;
                }
            }
            return true;   // 继续找
        });
}

// ==========================================
// 食物碰撞 — 蛇头碰到食物则吃掉
// ==========================================
void CollisionSystem::checkFoodCollision(Snake& snake, FoodManager& foodManager) {
    const sf::Vector2f headPos = snake.getHeadPosition();
    const float headR = snake.getHeadRadius();
    auto& foods = foodManager.getFoods();

    // 最远的可吃距离 = 头半径 + 最大食物半径（尸体食物 8px），再加 1px 余量
    const float queryR = headR + 8.0f + 1.0f;

    m_grid.forEachNearby(headPos.x, headPos.y, queryR,
        [&](const sim::GridItem& item) -> bool {
            if (item.kind != sim::GridKind::Food) return true;

            const std::size_t fi = static_cast<std::size_t>(item.payload);
            if (fi >= foods.size()) return true;
            if (m_foodEaten[fi]) return true;      // 本帧已经被别的蛇吃掉了

            const Food& food = foods[fi];
            if (Math::distance(headPos, food.position) < headR + food.radius) {
                if (food.isCorpse) snake.addCorpseEaten();
                else               snake.addFoodEaten();
                snake.addScore(food.scoreValue);

                m_foodEaten[fi] = 1;
                m_eatenList.push_back(fi);
            }
            return true;
        });
}
