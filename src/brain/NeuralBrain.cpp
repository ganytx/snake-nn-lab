#include "NeuralBrain.h"
#include "../nn/loss.h"      // nn::argmax 在这里
#include "../rl/Checkpoint.h"   // ★ 用户选中的存档在 saves/ 下

#include <cstdio>
#include <iostream>

namespace arena {

namespace {

// 依次尝试这些路径找权重文件。
// 训练程序把文件写在 build\ 下，而游戏是从项目根目录启动的，
// 所以两个位置都要找 —— 否则用户训练完了，游戏却"找不到权重"。
const char* kCandidates[] = {
    "weights/policy.txt",
    "build/train_probe_weights.txt",
    "train_probe_weights.txt",
};

// ★ 用户在界面上选中的那份存档（空 = 没选，走上面的自动查找）
std::string g_pendingWeights;

// ★ 敌方要用的冻结快照（空 = 不走这条路）
std::string g_pendingEnemySnapshot;

}  // namespace

void NeuralBrain::setPendingWeights(const std::string& name) {
    g_pendingWeights = name;
}

const std::string& NeuralBrain::pendingWeights() {
    return g_pendingWeights;
}

void NeuralBrain::setPendingEnemySnapshot(const std::string& pathOrName) {
    g_pendingEnemySnapshot = pathOrName;
}

const std::string& NeuralBrain::pendingEnemySnapshot() {
    return g_pendingEnemySnapshot;
}

// ---------------------------------------------------------------------------
NeuralBrain::NeuralBrain() {
    m_policy.init(nn::Init::Xavier, 20260921u);   // 先随机初始化（找不到权重的兜底）

    // ---- ① 敌方快照优先（自对弈：敌人 = 学习网络的一份冻结拷贝）----
    //
    //  ★ 为什么它排在用户选择【前面】
    //    它是"这一条蛇的身份"决定的（敌方蛇就该用快照），
    //    而 pendingWeights 是"我方要用哪份权重"——两者不该互相干扰。
    if (!g_pendingEnemySnapshot.empty()) {
        // ⚠️ 这里【不能】用 ckpt::path() —— 那个函数会拼上 saves/ 前缀，
        //    而快照路径本身就是完整的相对路径（"saves/enemy_snapshot.txt"）。
        //    实测踩过：拼成 "saves/saves/enemy_snapshot.txt" → 加载一直失败，
        //    而日志只说"加载失败"，看不出是路径拼错了。
        const std::filesystem::path full = ckpt::fsPath(g_pendingEnemySnapshot);
        if (m_policy.load(full)) {
            m_loaded = true;
            m_path   = full.string();
        }
        // ★ 失败时【不退回自动查找】—— 见头文件里的说明
        m_info.id          = "nn";
        m_info.displayName = m_loaded ? "神经网络·敌方快照 (已加载)"
                                      : "神经网络·敌方快照 (加载失败, 随机!)";
        m_info.description = m_loaded
            ? ("敌方 = 学习网络的冻结快照: " + m_path)
            : ("敌方快照加载失败, 这一局它只是随机蛇: " + g_pendingEnemySnapshot);
        m_info.author       = "REINFORCE";
        m_info.needsWeights = true;
        m_info.trainable    = true;
        std::cout << "[NeuralBrain] " << m_info.description << std::endl;
        return;
    }

    // ---- ② 用户在界面上选过 → 优先加载它 ----
    //
    // ★ 为什么"选了就只试它，不退回自动查找"
    //   如果选了 A 却加载失败、然后偷偷去加载 B，
    //   用户会看到"我选的那份好像生效了"，而实际跑的是另一份 ——
    //   这种"没报错但行为不是你要的"最难查。
    //   所以选了就只用它；失败就如实显示"加载失败"，不猜。
    if (!g_pendingWeights.empty()) {
        const std::filesystem::path full = ckpt::path(g_pendingWeights);
        if (m_policy.load(full)) {
            m_loaded = true;
            m_path   = full.string();
        }
    } else {
        // ---- ② 没选 → 走原来的自动查找 ----
        for (const char* path : kCandidates) {
            if (m_policy.load(path)) {
                m_loaded = true;
                m_path   = path;
                break;
            }
        }
    }

    m_info.id          = "nn";
    // ★ displayName 里【如实标注】三种状态，一眼能看出到底加载上没有
    if (m_loaded) {
        m_info.displayName = "神经网络 (已加载权重)";
    } else if (!g_pendingWeights.empty()) {
        m_info.displayName = "神经网络 (选的那份加载失败!)";
    } else {
        m_info.displayName = "神经网络 (未找到权重, 随机)";
    }
    m_info.description = m_loaded
        ? ("从 " + m_path + " 加载的 REINFORCE 策略")
        : (g_pendingWeights.empty()
              ? "没有找到权重文件, 现在是随机策略"
              : ("指定的权重加载失败: " + g_pendingWeights
                 + " (维度不匹配？改了 OBS_SIZE 的旧存档读不了)"));
    m_info.author      = "REINFORCE";
    m_info.needsWeights = true;
    m_info.trainable    = true;

    // WIN32 子系统没有控制台，这行输出一般看不到；
    // 但用命令行启动 build\snake_game.exe 时能看到，方便排查
    std::cout << "[NeuralBrain] " << m_info.description << std::endl;
}

const BrainInfo& NeuralBrain::info() const {
    return m_info;
}

// ---------------------------------------------------------------------------
Action NeuralBrain::think(const Perception& p) {
    Action action;
    if (!p.self.alive) return action;

    // ---- 感知 → 观测（和训练时走的是【同一个函数】）----
    double frame[rl::OBS_SIZE];
    rl::encodeObservation(p, frame);

    // ★ 帧堆叠：和训练时【同一套】做法
    //
    //  ⚠️ 这里的 m_frameStack 是【这条蛇私有】的成员 ——
    //     每条神经网络蛇都有自己的历史，不能共用（否则会互相污染记忆）。
    //     而"每局开始要清零"由 Brain::onEpisodeStart() 负责（见 onEpisodeStart）。
    double stacked[rl::NN_INPUT_SIZE];
    m_frameStack.push(frame, stacked);

    // ---- 前向 → 取概率最大的动作 ----
    const auto& logits = m_policy.forward(rl::toInput(stacked));
    const int best = nn::argmax(logits);

    // ★★★ 2026-09-28：观测/动作回滚到 29 维 + 3 动作
    //
    //  历史：2026-09-26 动作空间曾从 3 扩到 6（3 转向 × 2 速度），
    //        目的是让网络学会用加速（实测加速占比从 43% 升到 67%）。
    //        但同期的观测从 29 扩到 33 维，5 种子对照是**净负结果**
    //        （每局吃 3.68 → 2.33，锁死 1/5 → 2/5），
    //        所以 2026-09-28 观测与动作**一起回滚**。
    //
    //  现在：ACT_SIZE = 3，`boostOfAction()` 恒为 false ——
    //        **神经网络蛇不加速**（这正是 29 维时代的行为）。
    //
    //  两个辅助函数（steerIndexOfAction / boostOfAction）仍然是**唯一**的译码处，
    //  训练侧（EpisodeRunner）和这里用的是同一对函数 —— 不会分叉。
    //  ⚠️ 以后若把 ACT_SIZE 改回 6，这个函数**一行都不用改**。
    action.steer = rl::steerOfAction(best);
    action.boost = rl::boostOfAction(best);
    return action;
}

// ---------------------------------------------------------------------------
//  ★ 每局开始时清空帧堆叠历史
//
//  ★ 为什么必须清
//    不清的话，新一局的头 K 帧会拿【上一局最后几帧】当"记忆" ——
//    那些画面里的蛇位置、食物、圈位置全都和现在无关。
//    现象是"每局开头几十帧行为莫名其妙"，而且很难联想到是历史没清。
//
//  ★ 为什么放在 onEpisodeStart 而不是构造函数
//    构造函数只在算法被创建时跑一次，而"一局开始"要跑很多次。
//    Arena 建蛇时会调 onEpisodeStart（见 Arena::reset）。
// ---------------------------------------------------------------------------
void NeuralBrain::onEpisodeStart(std::uint32_t /*seed*/) {
    m_frameStack.reset();
}

}  // namespace arena
