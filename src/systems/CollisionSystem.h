#pragma once

#include "../entities/Snake.h"
#include "../entities/FoodManager.h"
#include "../sim/SpatialGrid.h"

#include <cstdint>
#include <vector>

// ==========================================
// 碰撞检测系统 — 处理蛇与世界的所有交互
//
// 检测项：
//   1. 蛇头 vs 墙壁 → 死亡
//   2. 蛇头 vs 食物 → 吃掉
//   3. 蛇头 vs 其他蛇身体 / 蛇头 → 死亡
//
// 不检测项：
//   - 蛇头 vs 自己身体 → 无伤害（用户需求）
//
// ★ 空间网格加速
//   原来每条蛇都要遍历"其他所有蛇的所有体节"（O(蛇²×节)）以及"所有食物"
//   （O(蛇×食物)）。实测这两项占了每帧耗时的 70% 以上。
//   现在每帧把所有体节 / 蛇头 / 食物插进一张均匀网格，
//   每条蛇只查询自己头周围一小圈，复杂度降到近似 O(实体总数)。
// ==========================================
class CollisionSystem {
public:
    CollisionSystem(float mapWidth, float mapHeight);

    void applyConfig(float forgiveness, int killScore);  // 应用配置

    // 队友之间能否互相撞死。默认 false = 队友可穿过（需求方选定的规则）
    void setFriendlyFire(bool on) { m_friendlyFire = on; }

    void checkAll(std::vector<Snake*>& snakes, FoodManager& foodManager);

private:
    // 每帧重建空间网格（把体节 / 蛇头 / 食物全部插进去）
    void rebuildIndex(std::vector<Snake*>& snakes, FoodManager& foodManager);

    // 蛇头 vs 墙壁
    void checkWallCollision(Snake& snake);

    // 蛇头 vs 食物。snakeIndex 是本条蛇在 snakes 数组里的下标
    void checkFoodCollision(Snake& snake, FoodManager& foodManager);

    // 蛇头 vs 其他蛇的身体 / 蛇头
    void checkSnakeCollision(Snake& snake, const std::vector<Snake*>& allSnakes,
                             std::size_t snakeIndex);

    float m_mapWidth, m_mapHeight;
    float m_forgiveness = 5.0f;
    int m_killScore = 10;
    bool m_friendlyFire = false;

    sim::SpatialGrid m_grid;

    // 食物是"延迟移除"的：网格里存的是食物的下标，
    // 如果边遍历边删（removeFood 用的是 swap-and-pop），后面的下标全会错位。
    // 所以先只做标记，等所有蛇都处理完再一次性删除。
    std::vector<std::uint8_t> m_foodEaten;   // 本帧已被吃掉的食物标记
    std::vector<std::size_t>  m_eatenList;   // 本帧被吃掉的食物的下标
};
