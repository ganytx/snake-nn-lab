#pragma once

// ============================================================================
//  TrainingScene.h — 训练数据（曲线）
//
//  ★ 为什么要有这个界面
//
//  以前调参只能看控制台打的几个点（"第 1900 局回报 13.30"）。
//  一个点分不清是【震荡】【收敛】还是【发散】—— 而这三者的处理方式完全不同：
//      震荡 → 降学习率 / 加基线
//      收敛 → 该换更大的网络或更难的任务了
//      发散 → 立刻停下，学习率太大
//  所以先要把"看得见"做出来，否则调参就是盲猜。
//
//  ★ 数据来自哪
//
//  tools/train_probe.cpp 每局写一行到 train_metrics.csv（长格式）。
//  本界面只读，不产生数据 —— 训练还是在 build\train_probe.exe 里跑。
//
//  操作：
//      ↑ / ↓      切换指标（回报 / 存活 / 吃食 / 基线 / 评估）
//      R          重新读文件（训练跑完不用重启游戏）
//      ESC        返回主菜单
// ============================================================================

#include "../core/Scene.h"
#include "../ui/TrainingMetrics.h"

#include <string>
#include <vector>

class TrainingScene : public Scene {
public:
    explicit TrainingScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    void reload();

    // 画一条曲线（自动按数据范围缩放）
    void drawCurve(sf::RenderWindow& window,
                   const metrics::Series& s,
                   float left, float top, float w, float h,
                   const sf::Color& color,
                   float yMin, float yMax, int xMax) const;

    sf::Font m_font;

    metrics::Dataset m_data;
    std::string      m_reason;      // 读失败的原因（直接显示在界面上）
    std::string      m_notice;      // 临时提示（例如"已重新读取"）
    float            m_noticeTimer = 0.0f;

    int m_selected = 0;             // 当前选中的指标（索引进 m_views）

    // 可选的"视图"：每个视图可以叠若干条曲线（比如 训练回报 + 基线）
    struct View {
        std::string title;
        std::vector<const metrics::Series*> series;
    };
    std::vector<View> m_views;
};
