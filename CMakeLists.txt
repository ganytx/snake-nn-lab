cmake_minimum_required(VERSION 3.16)
project(snake_nn_lab LANGUAGES CXX)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release 优化：神经网络的训练循环是纯计算，-O2 和 -O0 能差十倍以上
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ===========================================================================
#  ★★ 工具链路径：**可覆盖**，不再写死
#
#  ★ 为什么改（可移植性）
#    原来这里写死了 `C:/mingw64` 和 `C:/SFML-2.6.2`。
#    换一台机器、或者把项目换个目录，只要 MinGW / SFML 装在别处就直接配不过，
#    而报错信息是 find_package 找不到 SFML —— 看不出是"路径写死了"。
#
#  ★ 覆盖方式（优先级从高到低）
#      ① CMake 命令行：  -DMINGW_ROOT=D:/mingw64 -DSFML_ROOT=D:/SFML-2.6.2
#      ② 环境变量：      set MINGW_ROOT=D:\mingw64
#      ③ 都不给 → 用下面这两个默认值（开发本机就是这两个）
#
#  ⚠️ 编译器与 make 程序【不在这里设】—— 它们必须在 generator 校验之前就定好，
#     而 `-G "MinGW Makefiles"` 的校验发生在**读本文件之前**。
#     所以那部分在 `cmake/Toolchain-mingw.cmake` 里。
#     用 build.bat 会自动带上它；手动配 cmake 时见 README 的命令。
# ===========================================================================
if(NOT DEFINED MINGW_ROOT)
    if(DEFINED ENV{MINGW_ROOT})
        set(MINGW_ROOT "$ENV{MINGW_ROOT}")
    else()
        set(MINGW_ROOT "C:/mingw64")
    endif()
endif()
if(NOT DEFINED SFML_ROOT)
    if(DEFINED ENV{SFML_ROOT})
        set(SFML_ROOT "$ENV{SFML_ROOT}")
    else()
        set(SFML_ROOT "C:/SFML-2.6.2")
    endif()
endif()
message(STATUS "MinGW: ${MINGW_ROOT}")
message(STATUS "SFML : ${SFML_ROOT}")

# 设置 SFML 路径
set(SFML_DIR "${SFML_ROOT}/lib/cmake/SFML")

# 查找 SFML
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

# ---------------------------------------------------------------------------
#  两个"拷贝 DLL"用的目录常量
#
#  ⚠️ 必须在【所有 add_custom_command 之前】定义 ——
#     `add_custom_command(TARGET ...)` 里的 `${SFML_BIN}` 是**立即展开**的，
#     定义写在后面就等于展开成空字符串（那个坑很难看出来：
#     报错是"copy 失败"或干脆静默少拷一个 DLL）。
# ---------------------------------------------------------------------------
set(SFML_BIN  "${SFML_ROOT}/bin")
set(MINGW_BIN "${MINGW_ROOT}/bin")

# 收集所有源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# ---------------------------------------------------------------------------
#  目标 1：游戏本体（神经网络的"训练环境"）
# ---------------------------------------------------------------------------
add_executable(snake_game WIN32 ${SOURCES})

# 把 src/ 加入头文件搜索路径，这样游戏代码可以直接 #include "nn/network.h"
target_include_directories(snake_game PRIVATE "${CMAKE_SOURCE_DIR}/src")

# 链接 SFML 库
target_link_libraries(snake_game PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
)

# 添加 SFML DLL 搜索路径（运行时需要）
target_link_directories(snake_game PRIVATE "${SFML_ROOT}/bin")

# ---------------------------------------------------------------------------
#  目标 2：MLP 自检程序（纯控制台，不依赖 SFML，秒级跑完）
#
#  它只编译 tools/nn_selftest.cpp 这一个文件，
#  不参与 src/ 的 GLOB —— 否则会把整个游戏代码拉进来。
# ---------------------------------------------------------------------------
add_executable(nn_selftest tools/nn_selftest.cpp)
target_include_directories(nn_selftest PRIVATE "${CMAKE_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
#  目标 3：仿真性能计时探针
#
#  它需要真实的 Snake / FoodManager / CollisionSystem，所以链接 SFML；
#  但全程不创建窗口、不加载纹理，因此不需要图形上下文。
# ---------------------------------------------------------------------------
add_executable(bench_sim
    tools/bench_sim.cpp
    src/entities/Snake.cpp
    src/entities/FoodManager.cpp
    src/systems/CollisionSystem.cpp
    src/agent/PerceptionBuilder.cpp
    src/brain/RuleBrain.cpp
)
target_include_directories(bench_sim PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(bench_sim PRIVATE sfml-graphics sfml-window sfml-system)

# ---------------------------------------------------------------------------
#  ★★ 学习链路自证工具（2026-09-28 新增为正式目标）
#
#  ★ 它们是干什么的
#    排查"训练学不会"时，最大的困难是**没法判定"学习器本身对不对"** ——
#    贪吃蛇太复杂，学不会可能有一百个原因。
#
#    解法：造一个【答案已知的最小问题】，用**和 Trainer 完全相同的**
#    网络类型与梯度四步去解它。跑得对 ⇒ 学习器是对的，问题在别处。
#
#      bandit_check        单状态多臂老虎机（第 2 臂 μ 最优）
#                          → 证明梯度四步 / 采样 / 基线 / 更新顺序正确
#      statebandit_check   上下文老虎机（最优策略必须是"看输入决定动作"）
#                          → 证明网络【能】学会条件策略
#
#  ★ 为什么值得做成正式目标
#    这两条是**整个排查里最硬的证据**，而且不依赖 SFML、秒级跑完。
#    以前它们不在 CMake 目标里，只能靠翻日志；现在可以当场跑给人看。
#
#  ⚠️ 另外三个（`sens_check` / `perturb_check` / `win_scale_test`）需要读
#     `train_probe_weights.txt`，是"针对某次训练结果"的诊断，留在 tools/ 里
#     按需手动编译，不进默认目标（避免没有权重文件时白编译）。
# ---------------------------------------------------------------------------
foreach(_t bandit_check statebandit_check)
    add_executable(${_t} tools/${_t}.cpp)
    target_include_directories(${_t} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    # 用 MinGW 运行时 DLL 才能双击运行（与 nn_selftest 同理）
    add_custom_command(TARGET ${_t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MINGW_BIN}/libgcc_s_seh-1.dll"
        "${MINGW_BIN}/libstdc++-6.dll"
        "${MINGW_BIN}/libwinpthread-1.dll"
        $<TARGET_FILE_DIR:${_t}>
    )
endforeach()

# ---------------------------------------------------------------------------
#  目标 4：REINFORCE 最小可行性验证（一次性工具，纯控制台）
# ---------------------------------------------------------------------------
add_executable(train_probe
    tools/train_probe.cpp
    src/entities/Snake.cpp
    src/entities/FoodManager.cpp
    src/systems/CollisionSystem.cpp
    src/agent/PerceptionBuilder.cpp
    src/brain/RuleBrain.cpp
    src/brain/BrainRegistry.cpp
    src/brain/HumanBrain.cpp
    src/brain/NeuralBrain.cpp
    src/arena/Arena.cpp
    src/rl/Checkpoint.cpp
)
target_include_directories(train_probe PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(train_probe PRIVATE sfml-graphics sfml-window sfml-system)

add_custom_command(TARGET train_probe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SFML_BIN}/sfml-graphics-2.dll"
    "${SFML_BIN}/sfml-window-2.dll"
    "${SFML_BIN}/sfml-system-2.dll"
    "${MINGW_BIN}/libgcc_s_seh-1.dll"
    "${MINGW_BIN}/libstdc++-6.dll"
    "${MINGW_BIN}/libwinpthread-1.dll"
    $<TARGET_FILE_DIR:train_probe>
)


# Windows 特定：拷贝所有需要的 DLL 到可执行文件目录
#   ⚠️ SFML_BIN / MINGW_BIN 定义在文件上方（必须在使用之前，见那里的说明）

add_custom_command(TARGET snake_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SFML_BIN}/sfml-graphics-2.dll"
    "${SFML_BIN}/sfml-window-2.dll"
    "${SFML_BIN}/sfml-system-2.dll"
    "${SFML_BIN}/openal32.dll"
    "${MINGW_BIN}/libgcc_s_seh-1.dll"
    "${MINGW_BIN}/libstdc++-6.dll"
    "${MINGW_BIN}/libwinpthread-1.dll"
    $<TARGET_FILE_DIR:snake_game>
    # 拷贝 assets 目录到 build 目录（图片、字体等资源）
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/assets"
    "$<TARGET_FILE_DIR:snake_game>/assets"
)

# 自检程序需要 MinGW 运行时 DLL 才能双击运行
add_custom_command(TARGET nn_selftest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MINGW_BIN}/libgcc_s_seh-1.dll"
    "${MINGW_BIN}/libstdc++-6.dll"
    "${MINGW_BIN}/libwinpthread-1.dll"
    $<TARGET_FILE_DIR:nn_selftest>
)

# ---------------------------------------------------------------------------
#  把配置模板拷到 build\config\
#
#  ★ 为什么需要这一步
#
#    run.bat 用 `start "" "%BUILD_DIR%\snake_game.exe"` 启动，而 **`start` 会把
#    子进程的工作目录设成 exe 所在目录** —— 所以游戏实际是在 build\ 下运行的。
#    Game::loadConfig() 读相对路径 config/settings.cfg，读到的就是
#    build\config\settings.cfg。
#
#    而在这之前 CMake 【从来没有】把 config\ 拷进 build\，于是：
#      · 游戏读不到根目录那份配置，一直用内置默认值
#      · 设置界面保存时写出 build\config\settings.cfg，
#        而根目录那份永远不变 —— 两份副本各说各话
#
#  ★ 运行时的权威配置是 `build\config\settings.cfg`（游戏读写的就是它）。
#    根目录的 `config/settings.cfg` 是【模板】：
#    想改默认值 → 改根目录那份，然后重新 build.bat 让它覆盖过去。
#    已经跑过游戏、被游戏改过的那份不要指望它会被自动合并 —— 见 docs/00。
# ---------------------------------------------------------------------------
add_custom_command(TARGET snake_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "$<TARGET_FILE_DIR:snake_game>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/config/settings.cfg"
    "$<TARGET_FILE_DIR:snake_game>/config/settings.cfg"
)

# ---------------------------------------------------------------------------
#  ★★ 把 saves\ 里【你挑好的存档】同步到 build\saves\（2026-09-27 新增）
#
#  ★ 为什么需要这一步（这是"设了没用"的又一个实例）
#
#    `ckpt::kDir = "saves"` 是**相对工作目录**的路径，而 run.bat 的 `start`
#    把工作目录设成了 exe 所在目录 → 游戏读写的是 **build\saves\**。
#    这和配置文件的坑（见上面 config 那段）是同一个坑。
#
#    后果很隐蔽：负责人在项目根 `saves\` 里放了 3 份自己挑过的存档
#    （`另一种.txt` / `喜欢转.txt` / `略显智能.txt`），而游戏里"每条蛇的权重"
#    只能选到 `build\saves\` 里那一份自动产物 —— **3 份存档从来没被游戏看见过**。
#    无限模式"所有蛇随机搭载存档里的神经网络"因此退化成"全部都是同一份"。
#
#  ★ 为什么是【逐个文件拷贝】而不是 `copy_directory` 全量镜像
#
#    ⚠️ 负责人明确要求：**`auto_latest.txt`（自动最新）和 `enemy_snapshot.txt`
#       （敌人快照）不要同步过去。**
#
#    这两份是【训练器每局覆盖写的自动产物】：
#      · `auto_latest.txt`    —— 训练自动保存的"最新"
#      · `enemy_snapshot.txt` —— `nn-enemy` / `nn-mate` 两个算法 id 读的冻结快照
#    它们不是"你挑好的存档"，混进随机池只会让"随机搭载"看起来像没生效
#    （连着抽到同一份自动产物）。所以这里显式排除它们，
#    只同步你手动放进根 `saves\` 的那些。
#
#  ★ 为什么是【脚本枚举目录】而不是 CMake 自己逐个拷（2026-09-27 踩过一次）
#
#    ⚠️⚠️ 第一版写的是 `file(GLOB)` + `add_custom_command(copy_if_different ...)`，
#       结果**编译直接挂**：
#
#           Error copying file (if different) from
#           ".../saves/鍙︿竴绉?txt" to ".../build/saves/"
#
#       根因：本机的 cmake **跑在 GBK 代码页下**，中文文件名在生成
#       `build.make` 时就被转码成乱码了（`另一种.txt` → `鍙︿竴绉?txt`，
#       那个 `?` 就是"转换不回去"的证据）。
#
#    ★ 教训（比这次修复本身更重要）：
#      **绝不要让构建系统把非 ASCII 文件名放到命令行上。**
#      本项目的存档名是中文（`另一种.txt` 等），所以这条路走不通。
#
#    修法：把"枚举 + 拷贝"整个交给 `tools/sync_saves.ps1`。
#      PowerShell 直接从文件系统读目录项，文件名全程不经过任何代码页转换。
#      CMake 命令行里只剩 **纯 ASCII 的脚本路径**。
#
#  ★ 排除哪两份自动产物、为什么
#    见 `tools/sync_saves.ps1` 顶部（那份说明是权威的）。
#    运行时"随机池"侧的同一规则是 `ckpt::isAutoGenerated()` —— 两处要一致。
# ---------------------------------------------------------------------------
add_custom_command(TARGET snake_game POST_BUILD
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
            -File "${CMAKE_SOURCE_DIR}/tools/sync_saves.ps1"
            -SourceDir "${CMAKE_SOURCE_DIR}/saves"
            -DestDir "$<TARGET_FILE_DIR:snake_game>/saves"
    COMMENT "Syncing saves\\ -> build\\saves\\ (PowerShell, avoids GBK mojibake)"
)

# 计时探针链接了 SFML，所以还要拷贝 SFML 的 DLL
add_custom_command(TARGET bench_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SFML_BIN}/sfml-graphics-2.dll"
    "${SFML_BIN}/sfml-window-2.dll"
    "${SFML_BIN}/sfml-system-2.dll"
    "${MINGW_BIN}/libgcc_s_seh-1.dll"
    "${MINGW_BIN}/libstdc++-6.dll"
    "${MINGW_BIN}/libwinpthread-1.dll"
    $<TARGET_FILE_DIR:bench_sim>
)

# ---------------------------------------------------------------------------
#  目标 5：世界指纹回归工具（replay_check）
#
#  它是"重构世界推进代码"时的唯一硬证据：
#  固定配置 + 固定种子 + 固定动作序列跑一遍，输出确定性指纹。
#  重构前后两次输出【逐字节相同】，才能说"行为没变"。
#
#  为什么需要它：GameScene 是图形程序，没法在无头环境里自动比对画面。
#  这与 bench_sim 的「标墙对拍」是同一套思路。
# ---------------------------------------------------------------------------
add_executable(replay_check
    tools/replay_check.cpp
    src/entities/Snake.cpp
    src/entities/FoodManager.cpp
    src/systems/CollisionSystem.cpp
    src/agent/PerceptionBuilder.cpp
    src/brain/RuleBrain.cpp
    src/brain/BrainRegistry.cpp
    src/brain/HumanBrain.cpp
    src/brain/NeuralBrain.cpp
    src/arena/Arena.cpp
    src/rl/Checkpoint.cpp
)
target_include_directories(replay_check PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(replay_check PRIVATE sfml-graphics sfml-window sfml-system)

add_custom_command(TARGET replay_check POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SFML_BIN}/sfml-graphics-2.dll"
    "${SFML_BIN}/sfml-window-2.dll"
    "${SFML_BIN}/sfml-system-2.dll"
    "${MINGW_BIN}/libgcc_s_seh-1.dll"
    "${MINGW_BIN}/libstdc++-6.dll"
    "${MINGW_BIN}/libwinpthread-1.dll"
    $<TARGET_FILE_DIR:replay_check>
)

