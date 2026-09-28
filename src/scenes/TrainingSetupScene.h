#pragma once

// ============================================================================
//  TrainingSetupScene.h — 训练设置：开始训练之前先定环境
//
//  ★ 为什么要有这个界面（负责人建议）
//
//  第一版把这些参数绑在【训练模式里的按键】上（1/2/3 改蛇数…）。那样有两个问题：
//    1. 按键改参数本来就不好发现 —— 按了没反应，不知道是没编译还是没生效
//    2. 更糟：改蛇数会【重建世界、清空网络】——
//       训练到一半不小心按到，之前的进度全没了
//
//  所以改成和「开始游戏 → 对局设置」一样的做法：
//  **先在一个界面里把环境定好，再进训练**。这也是项目里已有的交互习惯。
//
//  ★ 这里改的是"任务"，不是"游戏平衡"
//
//  这些参数只影响训练模式，不写进 config/settings.cfg，
//  也不影响「开始游戏」里的对局。
//
//  操作（和对局设置一致）：
//      上下键 选择    左右键 调整    Enter 开始训练    ESC 返回
// ============================================================================

#include "../core/Scene.h"
#include "../rl/TrainTask.h"

#include <functional>
#include <string>
#include <vector>

class TrainingSetupScene : public Scene {
public:
    explicit TrainingSetupScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

    // 进训练模式时用它取配置。
    //
    // ★ 做成"静态的待定配置"是因为切场景后本场景就被销毁了，
    //   而训练模式需要拿到这里选好的参数。
    //   同时只会存在一个训练设置界面，所以一份就够。
    //
    //   给出两个版本：本场景要改它（非 const），训练模式只读它（const）。
    //   —— 比在调用处 const_cast 干净。
    static rl::TaskConfig&       pendingConfig();
    static const rl::TaskConfig& pendingConfigRead();

    // -----------------------------------------------------------------------
    //  ★ 进训练模式时要【预加载】的存档文件名（2026-09-24 负责人要求）
    //
    //  ★ 为什么不放进 TaskConfig
    //    TaskConfig 描述的是"训练任务"（地图/奖励/蛇数），
    //    而"加载哪份权重"是【会话选择】而不是任务定义 ——
    //    把它塞进 TaskConfig 会让 rl/ 那一层知道"文件里存了什么"，
    //    破坏"rl 只管算法、不管磁盘"的分层。
    //
    //  ★ 空字符串 = 不加载，从头练（默认）
    // -----------------------------------------------------------------------
    static std::string&       pendingPreload();
    static const std::string& pendingPreloadRead();

    // -----------------------------------------------------------------------
    //  ★ 待定的【队伍算法】选择（负责人要求：像主模式那样能选神经网络/规则）
    //
    //  ★ 为什么单独一份，不塞进 TaskConfig
    //    TaskConfig 描述"训练任务"（地图/奖励/蛇数），属于 rl/ 层；
    //    而"哪一队用神经网络"是【会话选择】，且它要影响
    //    EpisodeRunner 的 [0, learnerCount) 索引范围。
    //    混进 TaskConfig 会让 rl/ 层的语义变模糊（"任务"里为什么有 UI 选择）。
    //
    //  ★ 两值语义
    //      engine  = "nn"   → 学习中的网络所在的队（且必须只有一队是 nn）
    //      engine  = ""     → 没有队用神经网络（纯规则 AI 对局，不学习）
    //
    //  ★ 用【类静态】而不是函数内 static
    //    类静态数据成员在整个程序里只有一份（定义在 .cpp 里），
    //    两个设置界面共享它，不会因为编译单元不同而各拿到一份。
    // -----------------------------------------------------------------------
    static std::string&       pendingEngineTeam();
    static const std::string& pendingEngineTeamRead();

    // ★ 供「队伍设置」界面用 —— 它直接改 pendingConfig()，改完让本界面刷新行数。
    //   两个界面共享同一份配置，所以不需要额外的传参机制。
    void refreshRows() { buildRows(); }

private:
    // 一行设置：名字 + 取值（给界面显示）+ 调整（左右键）
    //
    // ★ 用 std::function 而不是裸函数指针 —— 这样 lambda 能就地写在构造函数里，
    //   不需要为了取函数指针而把每个 lambda 都写成无捕获的。
    struct Row {
        std::wstring label;
        std::function<std::wstring(const rl::TaskConfig&)> get;
        std::function<void(rl::TaskConfig&, int delta)>    adjust;
    };
    std::vector<Row> m_rows;

    // ★「队伍设置」永远是【最后一行】。回车时按这个判断是"跳转"还是"开训"，
    //   不用往 Row 里塞 std::function（那样构造函数里拿不到 this/game）。
    int  teamRowIndex() const { return static_cast<int>(m_rows.size()) - 1; }

    void buildRows();   // 构造/重建行表（队伍设置改完蛇数后要重算显示）

    sf::Font m_font;
    int      m_row = 0;
};
