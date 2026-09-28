#pragma once

// ============================================================================
//  BrainRegistry.h — 算法注册表
//
//  解决的问题：让"用户可以在游戏内选择不同的算法"这件事，不需要 UI 认识任何
//  具体算法类型。UI 只做三件事：
//      1. list() 拿到所有算法的 BrainInfo
//      2. 用户选一个 id
//      3. create(id) 拿到实例
//
//  新增一个算法 = 写一个 Brain 子类 + 在 registerBuiltinBrains() 里加一行。
//  UI 代码一行都不用改。
// ============================================================================

#include "Brain.h"

#include <functional>
#include <string>
#include <vector>

namespace arena {

class BrainRegistry {
public:
    static BrainRegistry& instance();

    // 注册一个算法。id 重复时覆盖旧的（方便热重载实验）
    void reg(const BrainInfo& info,
             std::function<std::unique_ptr<Brain>()> factory);

    // 所有已注册算法的信息（按注册顺序）
    const std::vector<BrainInfo>& list() const { return m_infos; }

    bool has(const std::string& id) const;

    // 创建实例；id 不存在时返回 nullptr
    std::unique_ptr<Brain> create(const std::string& id) const;

    // 创建需要额外参数的实例（神经网络要加载权重文件）
    // 基类默认忽略 weightsPath；需要权重的算法自己重写
    std::unique_ptr<Brain> createWithWeights(const std::string& id,
                                             const std::string& weightsPath) const;

    // 清空（测试用）
    void clear();

private:
    BrainRegistry() = default;

    std::vector<BrainInfo> m_infos;
    std::vector<std::function<std::unique_ptr<Brain>()>> m_factories;
};

// ---------------------------------------------------------------------------
//  注册所有内置算法。在 main() 里显式调用一次。
//
//  为什么用显式函数而不是静态初始化宏？
//    · 初学者友好，看得见调用点，断点好下
//    · 注册顺序可控
//    · 不会踩 C++ 静态初始化顺序的坑
// ---------------------------------------------------------------------------
void registerBuiltinBrains();

}  // namespace arena
