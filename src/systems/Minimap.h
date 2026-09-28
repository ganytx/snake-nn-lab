#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include "../entities/Snake.h"
#include "../entities/FoodZone.h"

// ==========================================
// 小地图 — 屏幕右上角显示整张地图的缩略图
//
// 绿色圆点 = 玩家自己
// 红色圆点 = 其他蛇（AI）
// 白色矩形框 = 当前视野范围
// 食物圈标记 = 圈的位置与活跃状态（见下）
//
// ★ 只用一个缩放系数（不用 scaleX / scaleY 各算各的）
//
//   地图是正方形，控件如果宽高不等又分轴独立缩放，
//   就会把地图拉成非正方形 —— 原来 180x120 的控件让横向拉伸了 1.5 倍，
//   蛇的位置和视野框全是歪的，而且歪得不明显、很难发现。
//
//   现在：控件尺寸由调用方给，但缩放系数取【自适应等比】的那个，
//   地图按比例居中放进控件里。
//
// ★ 食物圈在小地图上怎么画（负责人定的方案：符号 + 颜色）
//
//   大地图上的淡色填充圆在缩略图里只剩几像素，根本看不出区别，
//   所以小地图上换成【符号】：
//
//      活跃圈（正在供食）  ● 实心圆，暖黄  ← 一眼就能看出"食物在这儿"
//      活跃圈（空窗期）    ● 实心圆，冷蓝  ← 还轮到它，但暂时不出食物
//      未激活圈            ✕ 交叉，暗灰    ← 只是标记位置
//
//   ⚠️ 这纯粹是【给人看】的。AI 拿到的是 Perception::zones 里的
//      方位/距离/食物数，和这里画什么完全无关。
// ==========================================
class Minimap {
public:
    Minimap(float mapW, float mapH, float displayW, float displayH);

    // 每帧更新玩家的位置信息
    //
    // zones / activeZone / spawning 用来画食物圈标记，可以不传（nullptr = 没有圈）。
    void update(const Snake& player,
                const std::vector<Snake*>& allSnakes,
                const sf::View& cameraView,
                const std::vector<arena::FoodZone>* zones = nullptr,
                const arena::FoodZone* activeZone = nullptr,
                bool zoneSpawning = false);

    // 渲染小地图到屏幕
    void render(sf::RenderWindow& window) const;

private:
    // 世界坐标 → 控件内的像素坐标（不含控件在屏幕上的位置）
    sf::Vector2f worldToWidget(float wx, float wy) const;

    // 重算缩放系数与居中留白（构造时算一次，地图尺寸变了会在 update 里再算）
    void recomputeScale();

    float m_mapW, m_mapH;        // 世界地图尺寸
    float m_displayW, m_displayH;// 小地图显示尺寸（控件外框）
    float m_x, m_y;              // 小地图在屏幕上的位置（旧版遗留，render 里按窗口宽度重算）
    float m_scale = 1.0f;        // ★ 唯一的缩放比例（世界 → 小地图），等比
    float m_offsetX = 0.0f;      // 地图在控件内居中时的左留白（像素）
    float m_offsetY = 0.0f;      // 地图在控件内居中时的上留白（像素）

    // 缓存的渲染数据
    std::vector<sf::Vector2f> m_aiDots;    // AI蛇在小地图上的位置
    sf::Vector2f m_playerDot;              // 玩家在小地图上的位置
    sf::FloatRect m_viewRect;              // 当前视野在小地图上的矩形
    std::string m_coordText;               // 坐标文字

    // ---- 食物圈标记 ----
    struct ZoneMark {
        sf::Vector2f pos;        // 控件内坐标
        bool  active   = false;
        bool  spawning = false;
    };
    std::vector<ZoneMark> m_zoneMarks;
};
