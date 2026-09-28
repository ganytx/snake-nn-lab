#pragma once

// ============================================================================
//  GameFont.h — 字体加载的【唯一实现】
//
//  ★ 为什么需要它（原来每个场景各写一遍，共 11 份）
//
//    改可移植性之前，11 个场景各自写着同一段：
//
//        if (!m_font.loadFromFile("C:/Windows/Fonts/simhei.ttf"))
//            if (!m_font.loadFromFile("C:/Windows/Fonts/msyh.ttc"))
//                m_font.loadFromFile("C:/Windows/Fonts/arial.ttf");
//
//    三个问题：
//      ① **写死了 C:/Windows/Fonts** —— 字体不在系统目录的机器上直接没有中文
//      ② **11 份拷贝** —— 以后想加一个候选路径要改 11 处（这个项目已经因为
//         "同一件事有多份实现"栽过好几次：观测编码分叉 0.96、画圈三份只改一份）
//      ③ **没有项目自带的字体** —— `assets/fonts/` 一直是空目录，等于没有兜底
//
//  ★ 现在的策略：**优先用项目自带的字体，系统字体只作为兜底**
//
//    查找顺序（第一个能加载成功的就用它）：
//        ① assets/fonts/simhei.ttf        ← ★ 项目自带，拷在 exe 旁边
//        ② assets/fonts/*.ttf             ← 自带目录里的任意 ttf（可换字体）
//        ③ C:/Windows/Fonts/simhei.ttf    ← 系统黑体（中文）
//        ④ C:/Windows/Fonts/msyh.ttc      ← 系统雅黑
//        ⑤ C:/Windows/Fonts/msyh.ttf
//        ⑥ C:/Windows/Fonts/arial.ttf     ← 最后兜底（无中文，至少不崩）
//
//  ★ 关于字形（这个项目踩过）
//    SFML 的 `sf::String` 从 `std::string` 构造时按**本地编码**解释，不是 UTF-8；
//    含中文的字面量要么用 `L"..."`、要么过 `textutil::utf8()`。
//    另外 **SimHei 缺 `↑↓←→` 和全角括号 `（）` 的字形**，UI 上要避开。
//
//  ★ 用法
//      sf::Font& font = ...;               // 场景自己持有的那份
//      textutil::loadGameFont(font);       // 一行，不用管候选列表
//
//    ⚠️ `sf::Font` 不能拷贝，每个场景必须持有自己的那一份 ——
//       这个函数只负责"往你给的那份里加载"，不返回共享对象。
// ============================================================================

#include <SFML/Graphics.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace textutil {

// 候选路径（按优先级）。
//
// ★ 为什么是函数而不是全局数组：`std::vector<std::string>` 全局构造会在
//   `main()` 之前跑，静态初始化顺序不受控；这里每次调用现建一份，
//   而字体加载只发生在场景构造时（一局几十次都不到），开销无所谓。
inline std::vector<std::string> fontCandidates() {
    return {
        // ---- ① 项目自带（首选）----
        "assets/fonts/simhei.ttf",
        "assets/fonts/msyh.ttf",
        "assets/fonts/font.ttf",
        // ---- ② 系统字体（兜底）----
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    };
}

// 依次尝试候选路径，加载成功返回 true。
//
// ★ 返回值为什么重要
//   加载失败时 SFML **不报错**，只是后面画出来的字全是空白/方块 ——
//   这个项目已经因为"静默失败"栽过很多次（权重加载失败、配置写错副本……）。
//   所以把结果返回出去，调用方想知道就能知道。
inline bool loadGameFont(sf::Font& font, std::string* usedPath = nullptr) {
    for (const std::string& p : fontCandidates()) {
        if (font.loadFromFile(p)) {
            if (usedPath != nullptr) *usedPath = p;
            return true;
        }
    }
    // 一个都加载不到：把候选列表打出来，省得猜"为什么没字"
    //   （WIN32 子系统没有控制台，但这行在命令行启动时能看到）
    std::printf("[GameFont] 字体加载失败！尝试过：\n");
    for (const std::string& p : fontCandidates()) {
        std::printf("           %s\n", p.c_str());
    }
    std::printf("           中文会显示为方块。把 simhei.ttf 放到 build/assets/fonts/ 下即可。\n");
    return false;
}

}  // namespace textutil
