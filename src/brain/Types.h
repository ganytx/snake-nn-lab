#pragma once

// ============================================================================
//  Types.h — 算法层共用的基础类型与限额
//
//  ★ 算法层铁律：禁止 #include 任何 SFML 头文件。
//
//  原因有两个：
//    1. 无头训练模式完全不创建窗口、不初始化图形上下文，拖着 SFML 没意义
//    2. 防止有人用 sf::Vector2f 把"算法层"和"仿真层"的类型系统偷偷黏在一起，
//       那样分层就名存实亡了
//
//  所以这里自己定义 Vec2（就是两个 float）。智能体层负责与 sf::Vector2f 互转。
// ============================================================================

namespace arena {

// 二维向量（普通聚合体，没有运算符重载，避免和 sf::Vector2f 混淆）
struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

namespace limits {

// 感知列表的数量上限。
// 设上限的目的：① 让算法的输入维度有界、可预测；② 避免一帧分配大量内存。
// 实现上按"距离由近到远"排序后截断，所以丢掉的是最远的，不影响生存判断。
constexpr int MAX_VISIBLE_SNAKES = 12;   // 视野内最多报告多少条其他蛇
constexpr int MAX_THREATS        = 16;   // 最多报告多少个"危险点"
constexpr int MAX_VISIBLE_FOODS  = 24;   // 最多报告多少个食物
constexpr int MAX_TEAMMATES      = 16;   // 队友全局感知的上限（不含自己）

// 每条蛇的身体采样间隔：每隔多少个体节取一个"危险点"。
// 取 1 表示每一节都给（精确但量大），取 3 表示每 3 节给一个（省 2/3）。
constexpr int BODY_SAMPLE_STEP = 2;

constexpr float PI     = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;

}  // namespace limits
}  // namespace arena
