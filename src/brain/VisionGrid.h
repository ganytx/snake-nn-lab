#pragma once

// ============================================================================
//  VisionGrid.h — 视野网格（算法接口的核心）
//
//  把蛇头周围一定范围光栅化成一张固定尺寸的格子图，每格用 1 字节位掩码
//  标记"这里有什么"。**算法自己决定怎么用这张图**：
//     · 可以直接拍平喂网络
//     · 也可以先提取出食物/敌人的位置，组成一个小向量再喂网络
//
//  ★ 网格随蛇头朝向旋转（"前方永远朝上"）
//
//      第 0 行 = 最前方          ← 前方
//      第 16 行 = 蛇头所在行（中心）
//      第 31 行 = 最后方
//      第 0 列 = 最左，第 16 列 = 正前/正后，第 31 列 = 最右
//
//  为什么必须旋转：网络要做的是"左转 / 直行 / 右转"这种【相对自身】的决策。
//  如果网格是世界轴对齐的，同样的场景出现在东边和北边会落在网格的不同位置，
//  网络必须额外学会旋转不变性——白白浪费容量，而且极难学。
//  旋转之后"左边有障碍"永远出现在网格的同一片区域，学习难度降一个数量级。
//
//  尺寸：32 x 32 格，格边长默认 32px → 视野 1024 x 1024px
//        精度 ±16px，而蛇的转向半径 v/ω = 250/8 = 31px，误差范围内仍来得及转向
//
//  ★ 为什么是 32x32 而不是 50x50？
//    光栅化的成本 = 格数 x 每格约 10 次运算，**和视野大小无关、只和格数有关**。
//    缩小格边长只能缩小视野，格数不变、成本一分不降。
//    所以真正有效的杠杆是【减少格数】：
//        50x50 = 2500 格  →  32x32 = 1024 格（成本降到 41%）
//    视野随之从 2400px 缩到 1024px —— 地图是 2000x2000，
//    相当于从"看得见整张图"变成"只看得见自己周围半个地图"，博弈也更真实。
// ============================================================================

#include "Types.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace arena {

// 每格 1 字节位掩码。一格里有多种东西也能同时表达，
// 不像"单值类别"那样必须给它们定优先级。
constexpr std::uint8_t CELL_WALL       = 0x01;  // 墙 / 地图边界
constexpr std::uint8_t CELL_FOOD       = 0x02;  // 普通食物
constexpr std::uint8_t CELL_CORPSE     = 0x04;  // 尸体食物（更值钱，也说明附近刚死过蛇）
constexpr std::uint8_t CELL_ENEMY_BODY = 0x08;  // 敌方蛇身
constexpr std::uint8_t CELL_ENEMY_HEAD = 0x10;  // 敌方蛇头
constexpr std::uint8_t CELL_ALLY_BODY  = 0x20;  // 队友蛇身
constexpr std::uint8_t CELL_ALLY_HEAD  = 0x40;  // 队友蛇头
constexpr std::uint8_t CELL_SELF_BODY  = 0x80;  // 自己的身体

constexpr int VISION_COLS = 32;
constexpr int VISION_ROWS = 32;
constexpr int VISION_CENTER_ROW = VISION_ROWS / 2;   // 16
constexpr int VISION_CENTER_COL = VISION_COLS / 2;   // 16

struct VisionGrid {
    std::uint8_t cells[VISION_ROWS][VISION_COLS] = {};

    void clear() { std::memset(cells, 0, sizeof(cells)); }

    // -----------------------------------------------------------------------
    //  把一个点写进网格
    //
    //  dx, dy : 目标相对蛇头的【世界坐标】偏移（target - self）
    //  heading: 蛇头朝向（弧度）
    //  cellSize: 格边长（像素）
    //  返回：是否落在网格范围内
    //
    //  内部先把世界偏移旋转到"以蛇头朝向为前"的局部坐标系：
    //      前向分量 f =  dx·cosθ + dy·sinθ
    //      右向分量 r = -dx·sinθ + dy·cosθ
    //  再映射到行列：向前 → 行号变小；向右 → 列号变大。
    // -----------------------------------------------------------------------
    bool mark(float dx, float dy, float heading, float cellSize, std::uint8_t bits) {
        const float c = std::cos(heading);
        const float s = std::sin(heading);
        const float f =  dx * c + dy * s;    // 前向
        const float r = -dx * s + dy * c;    // 右向

        const int row = VISION_CENTER_ROW - static_cast<int>(std::floor(f / cellSize));
        const int col = VISION_CENTER_COL + static_cast<int>(std::floor(r / cellSize));

        if (row < 0 || row >= VISION_ROWS || col < 0 || col >= VISION_COLS) return false;
        cells[row][col] |= bits;
        return true;
    }

    // 直接按行列写（光栅化墙的时候用，那时已经算好行列了）
    void markCell(int row, int col, std::uint8_t bits) {
        if (row < 0 || row >= VISION_ROWS || col < 0 || col >= VISION_COLS) return;
        cells[row][col] |= bits;
    }

    std::uint8_t at(int row, int col) const {
        if (row < 0 || row >= VISION_ROWS || col < 0 || col >= VISION_COLS) return 0;
        return cells[row][col];
    }

    // -----------------------------------------------------------------------
    //  把网格坐标反算回"相对蛇头"的世界坐标偏移。
    //  算法想从网格里取出"食物的世界位置"时会用到它。
    //  取的是格子中心。
    // -----------------------------------------------------------------------
    void cellToRelative(int row, int col, float heading, float cellSize,
                        float& outDx, float& outDy) const {
        const float f = static_cast<float>(VISION_CENTER_ROW - row) * cellSize;
        const float r = static_cast<float>(col - VISION_CENTER_COL) * cellSize;
        const float c = std::cos(heading);
        const float s = std::sin(heading);
        outDx = f * c - r * s;
        outDy = f * s + r * c;
    }

    // -----------------------------------------------------------------------
    //  ★ 射线要拦哪些格子 —— 用一个【位掩码】参数化
    //
    //  ★ 为什么必须参数化（而不是写死一套）
    //
    //    第一版把五种格子硬写成一个集合：
    //        CELL_WALL | CELL_ENEMY_BODY | CELL_ENEMY_HEAD |
    //        CELL_ALLY_BODY | CELL_SELF_BODY
    //    于是**队友身体和墙被当成了同一种东西**。
    //
    //    ⚠️ 而实际上【队友之间可以互相穿过】（friendlyFire = false）。
    //      也就是说：那份观测在【撒谎】—— 它告诉网络"前面那堵墙会撞死你"，
    //      而那条"墙"其实是它能直接穿过去的队友。
    //
    //    对规则 AI（RuleBrain）来说这只是"多绕一点"，无所谓；
    //    但对**要学合作的神经网络**来说，这是喂了一个错误的物理模型 ——
    //    它会学会绕开队友，而不是跟队友配合。详见 docs/85。
    //
    //  ★ 默认值保持了【原来的语义】，所以 RuleBrain 的行为一字不变。
    // -----------------------------------------------------------------------
    using RayMask = std::uint8_t;

    // 旧行为：墙 + 敌方 + 自己 + 队友，全都算障碍
    static constexpr RayMask BLOCK_MASK_ALL =
        CELL_WALL | CELL_ENEMY_BODY | CELL_ENEMY_HEAD |
        CELL_ALLY_BODY | CELL_SELF_BODY;

    // ★ 与真实物理模型一致：队友能穿过，所以**不算障碍**
    //   （自己的头也不该重复算 —— 蛇头自己的格子由中心格表示）
    static constexpr RayMask BLOCK_MASK_NO_ALLY =
        CELL_WALL | CELL_ENEMY_BODY | CELL_ENEMY_HEAD | CELL_SELF_BODY;

    // 每个方向上的"到最近障碍的距离"（供算法快速取用；也可以自己在网格上算）
    //   dir  取 0..7，0 = 正前方，顺时针增加，每 45°
    //   mask 哪些格子算障碍，见上面的 BLOCK_MASK_*
    //   返回归一化距离 [0,1]，1 表示该方向一路畅通
    //
    //   ⚠️ 这里原来有一个 `float cellSize` 参数，但它**完全没被用到**
    //      ——射线是逐格步进的，`maxSteps = VISION_COLS` 就是全部。
    //      留着一个不起作用的参数会让人以为"改它能改射线精度"，所以删掉。
    float rayDistance(int dir, RayMask mask = BLOCK_MASK_ALL) const {
        static const int dr[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };
        static const int dc[8] = {  0,  1,  1,  1,  0, -1, -1, -1 };
        const int maxSteps = VISION_COLS;   // 足够走穿整个网格

        for (int step = 1; step <= maxSteps; ++step) {
            const int row = VISION_CENTER_ROW + dr[dir] * step;
            const int col = VISION_CENTER_COL + dc[dir] * step;
            if (row < 0 || row >= VISION_ROWS || col < 0 || col >= VISION_COLS) break;
            if (cells[row][col] & mask) {
                return static_cast<float>(step) / static_cast<float>(maxSteps);
            }
        }
        return 1.0f;
    }
};

}  // namespace arena
