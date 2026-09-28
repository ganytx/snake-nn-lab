#pragma once

// ============================================================================
//  ConsoleUtf8.h — 让控制台工具的中文输出【可读】
//
//  ★ 为什么需要它（这是一个真实踩过的坑）
//
//  本项目的控制台工具（nn_selftest / bench_sim / train_probe / replay_check）
//  用 `printf` 输出**中文**，而这些中文字符串在源码里是 **UTF-8** 编码。
//
//  但 Windows 控制台的默认代码页是 **936（GBK）**，不是 UTF-8。
//  于是控制台拿 GBK 去解释 UTF-8 字节 —— 用户看到的是**满屏乱码**。
//
//  ⚠️ 表现极具误导性：程序其实**完全正常**（退出码 0、数字都对），
//     只是人读不了。用户的第一反应会是"这些测试是不是全挂了"。
//     实测确认过：`chcp` 返回 936 时，工具输出全部花屏。
//
//  ★ 修法
//
//  在 main() 开头调用 `SetConsoleOutputCP(CP_UTF8)`，把控制台的输出代码页
//  切到 UTF-8（65001）。此调用**只影响当前控制台窗口**，
//  进程退出即恢复，不改系统设置、不改注册表。
//
//  ★ 为什么不用 `system("chcp 65001")`
//
//  · 会额外 fork 一个进程，且在某些环境下被禁用
//  · 输出会多一行 "Active code page: 65001"，污染日志
//  · 需要包含 <cstdlib>，而直接调 Win32 API 更准确
//
//  ★ 为什么用 `#ifdef _WIN32` 包起来
//
//  这是 Win32 API，其它平台没有。包起来之后这些工具仍然可以在
//  非 Windows 上编译（虽然本项目目前只针对 Windows）。
//
//  ★ 为什么放 `src/utils/` 而不是各个工具各写一份
//
//  这个项目的铁律是「同一个东西只有一份实现」——已经因为两份实现栽过两次。
//  四个工具都要这行代码，所以它必须只有一处。
// ============================================================================

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace textutil {

// 把【控制台输出】代码页切成 UTF-8，使 printf 出来的中文可读。
//
// 返回 true 表示切换成功；失败也不致命（只是可能显示乱码），所以调用方
// 可以忽略返回值 —— 但**不要**因为失败就中止程序。
inline bool enableUtf8Console() noexcept {
#ifdef _WIN32
    // CP_UTF8 = 65001
    return ::SetConsoleOutputCP(CP_UTF8) != 0;
#else
    return true;   // 非 Windows 平台默认就是 UTF-8，无需处理
#endif
}

}  // namespace textutil
