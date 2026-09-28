#pragma once

// ============================================================================
//  SpatialGrid.h — 均匀空间网格
//
//  解决的问题：碰撞检测和邻近查询现在是 O(n²) 的暴力双重循环。
//  实测（devlogs/2026-09-21.md 第五批）：30 条蛇时碰撞检测占每帧耗时的 76%，
//  其中食物检测又占碰撞段的 58%~95%。这是仿真里唯一的大热点。
//
//  做法：把地图切成固定边长的格子，每个实体按坐标落进一格。
//  查询"我附近有什么"时只遍历目标圆覆盖到的那几格，
//  复杂度从 O(实体总数) 降到 O(覆盖格数 + 格内实体数)。
//
//  ★ 实现要点：用「侵入式单向链表」的桶布局
//      m_head[cell] → 该格第一个实体在 m_items 里的下标（-1 表示空格）
//      m_next[i]    → 同格下一个实体的下标
//    好处是【一趟就能建表】，不需要先数一遍每格有几个再填。
//    而且 m_items / m_next 是连续数组，缓存友好。
//
//    ⚠️ 常见的性能陷阱：用 vector<vector<int>> 存桶。那样每次插入都可能
//       触发一次小分配，3900 个内层 vector 的分配开销会把网格的收益吃光。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sim {

// 网格里存的三类东西，查询时靠它区分
enum class GridKind : std::uint8_t {
    Body,   // 蛇身体节
    Head,   // 蛇头（单独一类：对头相撞双方都死，规则和撞身体不同）
    Food,   // 食物
};

struct GridItem {
    float    x = 0.0f;
    float    y = 0.0f;
    int      payload = 0;                  // Body/Head → 蛇的下标；Food → 食物下标
    GridKind kind = GridKind::Body;
};

class SpatialGrid {
public:
    // 每帧重建时调用。尺寸不变时不会重新分配内存。
    void init(float worldWidth, float worldHeight, float cellSize) {
        m_cellSize = std::max(1.0f, cellSize);
        m_invCell  = 1.0f / m_cellSize;
        m_cols = std::max(1, static_cast<int>(std::ceil(worldWidth  * m_invCell)));
        m_rows = std::max(1, static_cast<int>(std::ceil(worldHeight * m_invCell)));
        m_head.assign(static_cast<std::size_t>(m_cols) * m_rows, -1);
        m_items.clear();
        m_next.clear();
    }

    // 清空重建。不做增量更新 —— 蛇是连续移动的，
    // 维护增量的成本比重建还高，而且极易出错。
    void clear() {
        std::fill(m_head.begin(), m_head.end(), -1);
        m_items.clear();
        m_next.clear();
    }

    void insert(float x, float y, int payload, GridKind kind) {
        const int idx = static_cast<int>(m_items.size());
        m_items.push_back(GridItem{x, y, payload, kind});

        const int c = cellIndexOf(x, y);
        m_next.push_back(m_head[static_cast<std::size_t>(c)]);
        m_head[static_cast<std::size_t>(c)] = idx;
    }

    // 遍历以 (x,y) 为圆心、radius 为半径的圆【内部】的所有实体。
    //   · 落在覆盖格里但出了圆的，会被距离判断剔除
    //   · 回调签名 bool f(const GridItem&)，返回 false 可提前终止整个遍历
    //     （碰撞检测找到第一个致命碰撞后就不再需要往下找了）
    template <typename F>
    void forEachNearby(float x, float y, float radius, F&& f) const {
        const float r2 = radius * radius;

        const int c0 = clampCol(static_cast<int>(std::floor((x - radius) * m_invCell)));
        const int c1 = clampCol(static_cast<int>(std::floor((x + radius) * m_invCell)));
        const int r0 = clampRow(static_cast<int>(std::floor((y - radius) * m_invCell)));
        const int r1 = clampRow(static_cast<int>(std::floor((y + radius) * m_invCell)));

        for (int r = r0; r <= r1; ++r) {
            const int rowBase = r * m_cols;
            for (int c = c0; c <= c1; ++c) {
                for (int i = m_head[static_cast<std::size_t>(rowBase + c)];
                     i != -1;
                     i = m_next[static_cast<std::size_t>(i)]) {
                    const GridItem& it = m_items[static_cast<std::size_t>(i)];
                    const float dx = it.x - x;
                    const float dy = it.y - y;
                    if (dx * dx + dy * dy > r2) continue;
                    if (!f(it)) return;
                }
            }
        }
    }

    int         cols()      const { return m_cols; }
    int         rows()      const { return m_rows; }
    float       cellSize()  const { return m_cellSize; }
    std::size_t itemCount() const { return m_items.size(); }
    std::size_t cellCount() const { return m_head.size(); }

private:
    // 实体可能跑到地图外（长蛇的身体会拖出边界），所以格子下标必须夹取。
    // 夹到边缘格不会算错 —— 距离判断会把不在圆内的剔除掉。
    int clampCol(int c) const { return c < 0 ? 0 : (c >= m_cols ? m_cols - 1 : c); }
    int clampRow(int r) const { return r < 0 ? 0 : (r >= m_rows ? m_rows - 1 : r); }

    int cellIndexOf(float x, float y) const {
        const int c = clampCol(static_cast<int>(std::floor(x * m_invCell)));
        const int r = clampRow(static_cast<int>(std::floor(y * m_invCell)));
        return r * m_cols + c;
    }

    float m_cellSize = 128.0f;
    float m_invCell  = 1.0f / 128.0f;
    int   m_cols = 1;
    int   m_rows = 1;

    std::vector<int>      m_head;    // 每格的首个实体下标，-1 = 空格
    std::vector<int>      m_next;    // 同格的下一个实体下标
    std::vector<GridItem> m_items;   // 所有实体，连续存放
};

}  // namespace sim
