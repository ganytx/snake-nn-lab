#pragma once

// ============================================================================
//  TrainingMetrics.h — 读训练数据（train_metrics.csv）
//
//  ★ 数据从哪来
//
//  tools/train_probe.cpp 每训完一局就写几行到 train_metrics.csv（CSV 长格式）：
//
//      kind,episode,value,tag
//      train,1,12,reward
//      train,1,1200,survival
//      eval,1,0.12,reward
//
//  ★ 为什么是长格式
//
//  宽格式（每局一行、多列）每加一个指标就要改读取方的列号；
//  长格式加指标只是多一个 tag 值，**读取方一行都不用改**。
//  对一个还在长的项目，这个差别很大。
//
//  ★ 文件位置
//
//  训练程序写在它的工作目录（约定是 build\）。游戏也从 build\ 运行，
//  所以直接找同目录即可；找不到再退回项目根的 build\train_metrics.csv。
// ============================================================================

#include <string>
#include <vector>

namespace metrics {

// ---------------------------------------------------------------------------
//  一条曲线
// ---------------------------------------------------------------------------
struct Series {
    std::string kind;      // "train" / "eval" / "base"
    std::string tag;       // "reward" / "survival" / "food" / "baseline"
    std::vector<float> x;  // 横轴（局数）
    std::vector<float> y;  // 纵轴（值）

    // 人看的名字，例如 "train / reward"
    std::string label() const { return kind + " / " + tag; }

    bool empty() const { return x.empty(); }

    float minY() const;
    float maxY() const;
    float avgY() const;
};

// ---------------------------------------------------------------------------
//  一整套训练数据
// ---------------------------------------------------------------------------
struct Dataset {
    std::vector<Series> series;
    std::string sourcePath;   // 从哪个文件读的（界面上显示，避免"看的是旧数据"）
    bool ok = false;

    // 按 (kind, tag) 找一条曲线；找不到返回 nullptr
    const Series* find(const std::string& kind, const std::string& tag) const;

    int maxEpisode() const;
};

// 读文件。失败时 ok = false，reason 里说明原因（文件不存在 / 格式不对 / 空）。
Dataset load(std::string& reason);

// 试着找训练数据文件，找到第一个存在的返回路径；都没有则返回空串
std::string findMetricsFile();

}  // namespace metrics
