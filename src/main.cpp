#include "core/Game.h"
#include "scenes/MenuScene.h"
#include "brain/BrainRegistry.h"

#include <memory>

// ==========================================
// 程序入口 — 从主菜单开始
//
// 启动时先注册所有 AI 算法（见 src/brain/BrainRegistry.h）。
// 这一步必须在创建任何场景之前完成，因为"实验场"场景要靠注册表
// 列出可选算法。
// ==========================================
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    arena::registerBuiltinBrains();

    Game game;
    game.pushScene(std::make_unique<MenuScene>(&game));
    game.run();
    return 0;
}
