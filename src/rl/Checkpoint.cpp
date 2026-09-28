#include "Checkpoint.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

// ============================================================================
//  ★ 2026-09-24 改用 std::filesystem，不再用 Win32 的 A 版 API
//
//  ★ 为什么要改（负责人报的问题：中文存档名在文件管理器里是乱码）
//
//    原来用的是 `CreateDirectoryA` / `FindFirstFileA` / `WIN32_FIND_DATAA`，
//    加上 `std::ofstream(std::string)` —— 这三处都把字符串按
//    **ANSI 本地代码页（中文 Windows = GBK）** 解释，而我们给的是 **UTF-8**。
//    于是中文文件名被当成 GBK 解码 → 乱码。
//    （ASCII 名（如 `ep000123_20260923_091530.txt`）两种编码一致，所以看不出问题
//      —— 这也是它一直没被发现的原因。）
//
//  ★ 为什么 std::filesystem 能解决
//    Windows 上 `std::filesystem::path` 内部是**宽字符 UTF-16**：
//      · 从 UTF-8 构造时明确按 UTF-8 解释（见 Checkpoint::path 的说明）
//      · 交给系统时走宽字符 API
//    而 NTFS 的文件名本来就是 UTF-16 —— 全程不经过 GBK，就不会乱码。
//
//  ★ 关于链接：<filesystem> 在 MinGW-w64 + GCC 13 下已经是标准库的一部分，
//    不需要 -lstdc++fs（那是 GCC 8 之前的事）。
//    src/nn/network.h 已经用了它且能正常链接，所以这里同样安全。
//
//  ⚠️ windows.h 只允许出现在【这里】，绝不能进 src/nn/network.h
//     （它会定义 IN / OUT / min / max 宏，而 network.h 的模板参数就叫 IN/OUT）。
// ============================================================================

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>     // MultiByteToWideChar / WideCharToMultiByte（UTF-8 转换）
#endif

namespace ckpt {

// ---------------------------------------------------------------------------
//  UTF-8 字符串 → 文件系统路径
//
//  ★ 为什么必须显式转，不能直接 `std::filesystem::path(std::string)`
//    那个构造函数在 Windows 上按【本地代码页】解释入参，中文会再次乱码。
//
//  ⚠️ 也不能用 std::u8string / char8_t —— 那是 C++20，本项目是 C++17
//     （实测编译报 "'u8string' is not a member of 'std'"）。
//     所以直接调 Win32 的 MultiByteToWideChar(CP_UTF8)。
// ---------------------------------------------------------------------------
std::filesystem::path fsPath(const std::string& utf8) {
#ifdef _WIN32
    if (utf8.empty()) return std::filesystem::path();
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                        static_cast<int>(utf8.size()), nullptr, 0);
    if (n <= 0) return std::filesystem::path(utf8);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()),
                          w.data(), n);
    return std::filesystem::path(w);
#else
    return std::filesystem::path(utf8);
#endif
}

bool ensureDir() {
    std::error_code ec;
    std::filesystem::create_directories(fsPath(kDir), ec);
    // 已存在不算失败（create_directories 本身对已存在返回 false 且不报错）
    return !ec;
}

std::vector<std::string> list() {
    std::vector<std::string> out;

    std::error_code ec;
    const std::filesystem::path dir = fsPath(kDir);
    if (!std::filesystem::exists(dir, ec)) return out;   // 目录不存在 = 没有存档

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;

        const std::filesystem::path& p = entry.path();
        if (p.extension() != ".txt") continue;           // 只认 .txt 存档

        // ★ 文件名转回 UTF-8 字符串
        //
        //   这一步是"乱码"的另一半：宽字符必须【编成 UTF-8】再交给界面，
        //   否则 sf::String::fromUtf8 会把它当 UTF-8 解码 → 界面显示乱码。
        //   用 WideCharToMultiByte(CP_UTF8) 显式做（同 fsPath 的理由：C++17 没有 u8string）。
        const std::wstring wname = p.filename().wstring();
#ifdef _WIN32
        const int n = ::WideCharToMultiByte(CP_UTF8, 0, wname.c_str(),
                                            static_cast<int>(wname.size()),
                                            nullptr, 0, nullptr, nullptr);
        if (n > 0) {
            std::string s(static_cast<std::size_t>(n), '\0');
            ::WideCharToMultiByte(CP_UTF8, 0, wname.c_str(),
                                  static_cast<int>(wname.size()),
                                  s.data(), n, nullptr, nullptr);
            out.push_back(std::move(s));
        }
#else
        out.emplace_back(wname.begin(), wname.end());
#endif
    }

    // 文件名里带 yyyymmdd_hhmmss，所以字典序就是时间序 —— 老的在前、新的在后
    std::sort(out.begin(), out.end());
    return out;
}

std::string makeName(long episodes) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    tm = *std::localtime(&t);
#endif

    std::ostringstream o;
    o << "ep" << std::setw(6) << std::setfill('0') << episodes
      << '_'
      << std::put_time(&tm, "%Y%m%d_%H%M%S")
      << ".txt";
    return o.str();
}

}  // namespace ckpt
