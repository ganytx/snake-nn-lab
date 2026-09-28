#pragma once

// ============================================================================
//  Text.h — SFML 文本编码转换
//
//  ★ 坑一：sf::String 从 std::string 【隐式构造】时按 **Latin-1** 解释字节，
//    不是 UTF-8。所以
//          sf::Text t("剩余 120 秒", font, 20);     // ← 乱码
//    中文字符在 UTF-8 里是 3 个字节，被当成 3 个 Latin-1 字符显示。
//
//  ★ 坑二：sf::String::fromUtf8() 的行为依赖 SFML 版本与模板实例化细节，
//    在这个项目里出现过"部分位置仍然乱码"的现象，难以排查。
//
//  所以这里【不依赖 SFML 的 UTF-8 解码】，自己把 UTF-8 解成 UTF-16，
//  再用 sf::String(const std::wstring&) 构造 —— 这条路径是确定可靠的
//  （Windows 上 wchar_t 是 16 位，正好就是 UTF-16）。
//
//  用法：
//      sf::Text t(textutil::utf8(someUtf8String), font, 20);
//  纯字面量也可以直接写宽字符串：sf::Text t(L"中文", font, 20);
// ============================================================================

#include <SFML/Graphics.hpp>
#include <cstddef>
#include <string>

namespace textutil {

// 把 UTF-8 字节串解成 UTF-16（Windows 的 std::wstring）
inline std::wstring utf8ToWide(const std::string& s) {
    std::wstring out;
    out.reserve(s.size());

    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i < n) {
        const unsigned char c0 = static_cast<unsigned char>(s[i]);

        unsigned int cp   = 0;
        int          extra = 0;
        if (c0 < 0x80)             { cp = c0;          extra = 0; }
        else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F; extra = 1; }
        else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F; extra = 2; }
        else if ((c0 & 0xF8) == 0xF0) { cp = c0 & 0x07; extra = 3; }
        else { ++i; continue; }                 // 非法起始字节 → 跳过

        if (i + static_cast<std::size_t>(extra) >= n) break;   // 截断的序列

        bool ok = true;
        for (int k = 1; k <= extra; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }     // 不是续接字节
            cp = (cp << 6) | (cc & 0x3F);
        }
        i += static_cast<std::size_t>(extra) + 1;

        if (!ok) { out.push_back(L'?'); continue; }

        if (cp <= 0xFFFF) {
            out.push_back(static_cast<wchar_t>(cp));
        } else {
            // 补充平面 → 代理对
            cp -= 0x10000u;
            out.push_back(static_cast<wchar_t>(0xD800u + (cp >> 10)));
            out.push_back(static_cast<wchar_t>(0xDC00u + (cp & 0x3FFu)));
        }
    }
    return out;
}

// UTF-8 的 std::string → sf::String
inline sf::String utf8(const std::string& s) {
    return sf::String(utf8ToWide(s));
}

}  // namespace textutil
