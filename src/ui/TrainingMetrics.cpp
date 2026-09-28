#include "TrainingMetrics.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <windows.h>

namespace metrics {

// ---------------------------------------------------------------------------
//  Series 的小统计
// ---------------------------------------------------------------------------
float Series::minY() const {
    if (y.empty()) return 0.0f;
    return *std::min_element(y.begin(), y.end());
}

float Series::maxY() const {
    if (y.empty()) return 0.0f;
    return *std::max_element(y.begin(), y.end());
}

float Series::avgY() const {
    if (y.empty()) return 0.0f;
    double s = 0.0;
    for (float v : y) s += static_cast<double>(v);
    return static_cast<float>(s / static_cast<double>(y.size()));
}

// ---------------------------------------------------------------------------
const Series* Dataset::find(const std::string& kind, const std::string& tag) const {
    for (const auto& s : series) {
        if (s.kind == kind && s.tag == tag) return &s;
    }
    return nullptr;
}

int Dataset::maxEpisode() const {
    int m = 0;
    for (const auto& s : series) {
        if (!s.x.empty()) {
            const int last = static_cast<int>(s.x.back());
            if (last > m) m = last;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
//  找文件
//
//  训练程序写在它的工作目录（build\）。游戏也从 build\ 跑，
//  所以先找当前目录，再找 exe 旁边，最后找上一级的 build\。
// ---------------------------------------------------------------------------
static bool fileExists(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::string findMetricsFile() {
    const char* candidates[] = {
        "train_metrics.csv",
        "..\\train_metrics.csv",
        "build\\train_metrics.csv",
    };
    for (const char* c : candidates) {
        if (fileExists(c)) return c;
    }
    return std::string();
}

// ---------------------------------------------------------------------------
//  读文件
// ---------------------------------------------------------------------------
Dataset load(std::string& reason) {
    Dataset ds;
    reason.clear();

    ds.sourcePath = findMetricsFile();
    if (ds.sourcePath.empty()) {
        reason = "找不到 train_metrics.csv —— 先跑一次 build\\train_probe.exe";
        return ds;
    }

    std::ifstream f(ds.sourcePath);
    if (!f.is_open()) {
        reason = "打不开 " + ds.sourcePath + "（可能正被训练程序占用？）";
        return ds;
    }

    std::string line;
    if (!std::getline(f, line)) {
        reason = ds.sourcePath + " 是空文件";
        return ds;
    }
    if (line.find("kind") == std::string::npos) {
        reason = ds.sourcePath + " 格式不对（第一行不是 kind,episode,value,tag）";
        return ds;
    }

    // ---- 第一遍：把所有 (kind, tag) 组合收集出来 ----
    //
    // ★ 为什么要先扫一遍、而不是边读边 push_back：
    //   `std::vector` 扩容会让之前拿到的元素【引用失效】。
    //   如果边读边返回 `Series&` 再往里塞数据，扩容一次就写到野内存上了。
    //   （这类 bug 的症状是随机崩溃或者数据错乱，极难查。）
    //   先建好全部 Series 再填值，就完全没有这个问题。
    std::vector<std::string> lineBuf;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        lineBuf.push_back(line);
    }

    // 把一行 CSV 拆成 (kind, tag)；列数不够就返回 false
    auto splitKey = [](const std::string& l, std::string& kind, std::string& tag) -> bool {
        const std::size_t c1 = l.find(',');
        if (c1 == std::string::npos) return false;
        const std::size_t c2 = l.find(',', c1 + 1);
        if (c2 == std::string::npos) return false;
        const std::size_t c3 = l.find(',', c2 + 1);
        if (c3 == std::string::npos) return false;
        kind = l.substr(0, c1);
        tag  = l.substr(c3 + 1);
        return !kind.empty() && !tag.empty();
    };

    for (const auto& l : lineBuf) {
        std::string k, t;
        if (!splitKey(l, k, t)) continue;
        bool found = false;
        for (const auto& s : ds.series) {
            if (s.kind == k && s.tag == t) { found = true; break; }
        }
        if (!found) {
            Series s;
            s.kind = k;
            s.tag  = t;
            ds.series.push_back(std::move(s));
        }
    }

    // ---- 第二遍：填值（此时 series 已经不会再增删，引用稳定） ----
    int rows = 0;
    for (const auto& l : lineBuf) {
        std::string kind, tag;
        if (!splitKey(l, kind, tag)) continue;

        const std::size_t c1 = l.find(',');
        const std::size_t c2 = l.find(',', c1 + 1);
        const std::size_t c3 = l.find(',', c2 + 1);
        const float ep = static_cast<float>(std::atof(l.substr(c1 + 1, c2 - c1 - 1).c_str()));
        const float v  = static_cast<float>(std::atof(l.substr(c2 + 1, c3 - c2 - 1).c_str()));

        for (auto& s : ds.series) {
            if (s.kind == kind && s.tag == tag) {
                s.x.push_back(ep);
                s.y.push_back(v);
                ++rows;
                break;
            }
        }
    }

    if (rows == 0) {
        reason = ds.sourcePath + " 里没有任何数据行";
        return ds;
    }

    ds.ok = true;
    return ds;
}

}  // namespace metrics
