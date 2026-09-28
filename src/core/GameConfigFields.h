// ============================================================================
//  GameConfigFields.h — GameConfig 的【唯一字段表】（X-macro 表）
//
//  ⚠️⚠️ 这个文件【故意不加 #pragma once / include guard】⚠️⚠️
//
//  它是一张 X-macro 表，靠"被重复展开"工作：
//
//      Game.h   里展开一次 → 变成 GameConfig 的成员声明
//      Game.cpp 里展开一次 → 变成 loadConfig 的解析分支
//      Game.cpp 里再展开   → 变成 saveConfig 的写出语句
//
//  如果加了 include guard，同一个 .cpp 里的第二次、第三次展开就会被整段跳过，
//  症状是 "GAME_CONFIG_FIELDS was not declared in this scope" 加一片
//  "xxx was not declared in this scope"。**这个坑真的踩过一次。**
//
//  ⚠️ 宏定义里的注释一律用 `//`，不要用 `/* */`。
//     反斜杠续行的宏在预处理器眼里是【一整行】，块注释会把后面的内容一起吃掉。
//
//  ---- 怎么加一个新的配置项 ----
//    1. 在下面加一行 X_FLOAT / X_INT / X_BOOL(字段名, 配置键名, 默认值)
//    2. 在 GameConfig::sanitize() 里补上它的有效范围（如果它需要兜底）
//    3. 如果想在设置界面里能调，再去 SettingsScene 加一行 addF / addI
//    —— 读配置、写配置、结构体成员都会自动跟上，不需要再改别的地方
// ============================================================================

// X_INT / X_FLOAT / X_BOOL 的参数： (C++ 字段名, 配置文件里的键名, 默认值)
#define GAME_CONFIG_FIELDS(X_FLOAT, X_INT, X_BOOL)                \
    /* 地图                                                          */ \
    /* ★★ 2026-09-27：地图尺寸不再是"可调项"，而是【每个模式固定 3000×3000】   */ \
    /*    （负责人指定三个模式统一）。它仍然作为字段存在，因为 GameScene 和相机   */ \
    /*    要读它，而且 GameModes.cpp 会按模式写进去。                            */ \
    /*    ⚠️ 从设置界面上撤掉了 —— 玩家改它会让三个模式的地图不一致。            */ \
    X_FLOAT(mapWidth,            map_width,           3000.0f)    \
    X_FLOAT(mapHeight,           map_height,          3000.0f)    \
    /* 蛇 */                                                       \
    X_FLOAT(snakeSpeed,          snake_speed,          250.0f)    \
    X_FLOAT(snakeBoostSpeed,     snake_boost_speed,    450.0f)    \
    X_FLOAT(snakeTurnRate,       snake_turn_rate,        8.0f)    \
    X_INT  (snakeInitSegs,       snake_init_segs,          12)    \
    X_FLOAT(segmentSpacing,      segment_spacing,       20.0f)    \
    /* 食物 */                                                     \
    /* ⚠️ foodMaxCount 不在【通用规则】这张表里。                            */ \
    /*    2026-09-27 白天它曾因"拷进 ArenaConfig 后没有任何读取点"被删掉；    */ \
    /*    同一天负责人要求"食物存在上限"，于是它以                              */ \
    /*    `GameConfig::ModeConfig::foodMaxCount`（**每模式一份**）的形式回来， */ \
    /*    因为"上限多少"正是模式之间的分界线（模式 2 高、模式 3 低）。         */ \
    /*    它由 applyModeParams() 写、GameScene 转交给 ArenaConfig，            */ \
    /*    最终在 FoodManager::update() 的生成分支里生效。                      */ \
    X_INT  (foodSpawnCount,      food_spawn_count,          2)    \
    X_FLOAT(foodSpawnInterval,   food_spawn_interval,     2.0f)   \
    /* ★ 吃几个普通食物长一节。食物圈玩法定为 1（吃一个长 20px） */ \
    X_INT  (foodSegments,        food_segments,             1)    \
    /* ★ 普通食物寿命（秒）。食物圈玩法要求普通食物也会过期， */   \
    /*   否则一轮没吃完的食物无限堆积，玩法退化 */                 \
    X_FLOAT(foodLifetime,        food_lifetime,          20.0f)   \
    X_FLOAT(corpseLifetime,      corpse_lifetime,        30.0f)   \
    /* ---- 食物圈（玩法核心，见 docs/70）---- */                  \
    /* 圈数。0 = 关闭圈，退化成全图随机撒食物                              */ \
    /* ★★ 2026-09-27：圈数现在是【每个模式不同】的 —— 模式 1 = 2，         */ \
    /*    模式 2/3 = 0（全图撒）。由 game::applyModeParams() 写，          */ \
    /*    从设置界面上撤掉（改它会让模式定义失效）。                        */ \
    X_INT  (zoneCount,           zone_count,                2)    \
    /* 圈心距地图中心的距离 / min(地图宽,高)                             */ \
    /* ⚠️ 随机位置模式下不使用它（圈心是随机挑的）—— 只在固定布局时有效 */ \
    X_FLOAT(zoneCenterRatio,     zone_center_ratio,      0.31f)   \
    /* 圈半径 / min(地图宽,高)。0.05 × 3000 = 150px（与训练侧同一像素值） */ \
    X_FLOAT(zoneRadiusRatio,     zone_radius_ratio,      0.05f)   \
    /* 每个圈持续生成食物多少秒 */                                 \
    X_FLOAT(zoneSpawnSeconds,    zone_spawn_seconds,      9.0f)   \
    /* 两轮之间的空窗多少秒（不让优势方一路跟到底） */             \
    X_FLOAT(zoneGapSeconds,      zone_gap_seconds,        5.0f)   \
    /* ★★ 2026-09-27：foodPerSnake 移进 GameConfig::ModeConfig ——        */ \
    /*    它必须【每个模式不同】（模式 1 要 2 个/秒、模式 3 只有 1 个/秒）。 */ \
    /*    留在全局表里就会出现"改了一个模式、另两个也变了"。                */ \
    /* 尸体 */                                                     \
    X_INT  (corpseSegments,      corpse_segments,           2)    \
    X_INT  (corpseScore,         corpse_score,              3)    \
    /* 饥饿 / 碰撞 */                                              \
    X_FLOAT(starvationTime,      starvation_time,        20.0f)   \
    X_FLOAT(collisionForgive,    collision_forgive,       5.0f)   \
    /* 对局 */                                                     \
    /* ★★ 2026-09-27：teamCount / snakesPerTeam 移进 GameConfig::ModeConfig */ \
    /*    （每个模式默认队伍规模不同）。playerTeam 留着，三个模式都是 0。   */ \
    X_INT  (playerTeam,          player_team,               0)    \
    X_FLOAT(matchTimeLimit,      match_time_limit,        0.0f)   \
    X_BOOL (friendlyFire,        friendly_fire,             0)    \
    X_INT  (allyVisibleCount,    ally_visible_count,        2)
