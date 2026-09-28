#pragma once

// ==========================================
// 游戏常量定义 — 所有可调参数集中管理
//
// 后续阶段6加入配置系统后，这些默认值可被
// config 文件覆盖，但常量仍然作为兜底默认值。
// ==========================================

namespace Constants {

    // ----- 窗口 -----
    constexpr int WINDOW_WIDTH  = 1280;
    constexpr int WINDOW_HEIGHT = 720;
    constexpr const wchar_t* WINDOW_TITLE = L"神经网络蛇对抗训练";

    // ----- 地图（默认边长是原来 8000 的 1/4）-----
    // 地图小了之后的连带影响：视野网格 1024px 约占地图一半，
    // 蛇与蛇的遭遇频率大幅上升，"对抗 / 合作"才有机会学出来。
    constexpr float MAP_WIDTH  = 2000.0f;
    constexpr float MAP_HEIGHT = 2000.0f;

    // ----- 时间步长 -----
    constexpr float FIXED_DT = 1.0f / 60.0f;  // 固定物理更新间隔（60Hz）

    // ----- 蛇 -----
    constexpr float SNAKE_SPEED        = 250.0f;  // 基础移动速度（像素/秒）
    constexpr float SNAKE_BOOST_SPEED  = 450.0f;  // 加速速度
    constexpr float SNAKE_TURN_RATE    = 8.0f;    // 最大转向速度（弧度/秒）= 机动性上限
    // 转向缓动系数：转向速度 = 角度差 × 该系数，再受上面 SNAKE_TURN_RATE 封顶。
    // 有了它，转头是"离目标远时全速转、快对准时自动减速"的曲线，而不是匀速转到底。
    // 值越大越"急"，越小越"懒"。
    constexpr float SNAKE_TURN_EASE    = 6.0f;
    constexpr float SNAKE_HEAD_RADIUS  = 18.0f;   // 头部半径（比体节略大）
    constexpr float SNAKE_SEGMENT_RADIUS = 16.0f;  // 体节半径
    constexpr float SNAKE_SEGMENT_SPACING = 20.0f; // 体节间距（圆心距离）
    constexpr int   SNAKE_INITIAL_SEGMENTS = 5;   // 初始体节数（默认值，会被配置覆盖）

    // ----- 食物 -----
    constexpr float FOOD_RADIUS        = 6.0f;   // 普通食物半径
    constexpr float FOOD_SPAWN_INTERVAL = 0.5f;   // 食物自然生成间隔（秒）
    constexpr int   FOOD_SPAWN_COUNT   = 10;      // 每次生成数量
    constexpr int   FOOD_MAX_COUNT     = 1000;    // 地图上最大食物数量
    constexpr int   FOOD_SCORE         = 1;      // 每个食物得分
    constexpr int   FOOD_SEGMENTS      = 5;      // 吃几个食物才+1体节

    // ----- 尸体食物 -----
    constexpr float CORPSE_FOOD_RADIUS      = 8.0f;   // 尸体食物半径（比普通食物大）
    constexpr int   CORPSE_FOOD_SCORE       = 3;      // 每个尸体食物得分
    constexpr int   CORPSE_FOOD_SEGMENTS    = 2;      // 吃几个尸体食物才+1体节
    constexpr int   CORPSE_SEGMENT_RATIO    = 1;      // 蛇的每N个体节生成1个尸体食物
    constexpr float CORPSE_FOOD_LIFETIME    = 30.0f;  // 尸体食物存在时间（秒）

    // ----- 碰撞 -----
    constexpr float COLLISION_FORGIVENESS = 5.0f;     // 碰撞容错边距
    constexpr float MIN_SPAWN_DIST       = 200.0f;    // 生成位置与其他蛇头的最小距离

    // ----- AI -----
    constexpr int   AI_INITIAL_COUNT = 10;      // 初始AI数量
    constexpr int   AI_MAX_COUNT     = 30;      // 最大AI数量
    constexpr float AI_SPAWN_INTERVAL = 3.0f;   // AI生成间隔（秒）
    constexpr float AI_VISION_RADIUS  = 500.0f; // AI视野半径

    // AI 的决策反应时间（秒）：每条蛇在 [MIN, MAX] 之间随机取一个值，
    // 决定它多久才重新决策一次（模拟生物的反应延迟）
    constexpr float AI_REACTION_MIN  = 0.05f;
    constexpr float AI_REACTION_MAX  = 0.30f;

    // AI 的转向"个性"：每条蛇出生时随机抽一组，让它们的转头动作互不相同
    //   缓动系数 —— 越小越迟钝（转起来慢悠悠），越大越敏捷
    constexpr float AI_TURN_EASE_MIN = 3.5f;
    constexpr float AI_TURN_EASE_MAX = 9.0f;
    //   转速起伏幅度 —— 最大转速在 100%±该值 之间按正弦摆动（0 = 匀速）
    constexpr float AI_TURN_WOBBLE_AMP_MIN = 0.20f;
    constexpr float AI_TURN_WOBBLE_AMP_MAX = 0.55f;
    //   起伏快慢（弧度/秒）—— 越大摆动越快，越小越像"慢性子"
    constexpr float AI_TURN_WOBBLE_SPEED_MIN = 0.5f;
    constexpr float AI_TURN_WOBBLE_SPEED_MAX = 2.0f;

    // ----- 击杀 & 饥饿 -----
    constexpr int   KILL_SCORE       = 10;      // 击杀奖励分数
    constexpr float STARVATION_TIME  = 20.0f;   // 饥饿死亡时间（秒）

    // ----- 摄像机 -----
    constexpr float CAMERA_NORMAL_FOLLOW = 3.0f;  // 正常跟随速度
    constexpr float CAMERA_BOOST_FOLLOW  = 1.5f;  // 加速时跟随速度（更低=更延迟）

    // ----- 小地图 -----
    // ★ 必须是正方形：地图本身是正方形（2000x2000），小地图控件如果宽高不等，
    //   又按两个轴各自缩放，就会把地图横向拉伸。
    //   原来是 180x120 —— 横向被拉伸了 1.5 倍，蛇的位置和视野框全是错的。
    constexpr float MINIMAP_WIDTH  = 150.0f;
    constexpr float MINIMAP_HEIGHT = 150.0f;
    constexpr float MINIMAP_MARGIN = 10.0f;
}
