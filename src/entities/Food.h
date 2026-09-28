#pragma once

#include <SFML/Graphics.hpp>

// ==========================================
// 食物实体 — 地图上可被蛇吃掉的物件
//
// 两种类型：
//   普通食物 — 自然随机刷新，小圆点，每吃一个 +1 体节
//   尸体食物 — 蛇死后遗留，比普通食物略大，每吃2个 +1 体节
// ==========================================
struct Food {
    sf::Vector2f position;   // 世界坐标
    float radius;            // 圆形半径
    int scoreValue;          // 吃掉后加多少分
    int segmentValue;        // 吃掉后对体节增长的贡献（尸体食物需要累积2个）
    bool isCorpse;           // 是否是尸体食物
    float lifetime;          // 已存在时间（尸体食物超时后消失，普通食物不受此限制）

    Food(float x, float y, float r, int score, int segVal, bool corpse)
        : position(x, y)
        , radius(r)
        , scoreValue(score)
        , segmentValue(segVal)
        , isCorpse(corpse)
        , lifetime(0.0f)
    {}
};
