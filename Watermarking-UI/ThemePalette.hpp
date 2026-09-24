#pragma once

#include <string>
#include <QColor>
#include <WatermarkCore.hpp>

/*!
 *  \brief  Theme color palettes for CUDA, Eigen, and OpenCL compute backends
 *  \author Dimitris Karatzas
 */

// UI color scheme definition
struct ThemePalette {
    QColor accent;
    QColor deep;
    QColor ink;
    QColor surface = QColor("#131313");
    QColor surfaceHover = QColor("#2a2a2a");
    QColor surfacePressed = QColor("#363636");
    QColor text = QColor("#e5e2e1");
    QColor muted = QColor("#8b90a0");
    QColor error = QColor("#ff7474");
    QColor warning = QColor("#f7c86f");
    QColor canvas = QColor("#0e0e0e");
    QColor divider = QColor("#f4f4f4");
    QColor dividerInk = QColor("#181818");
};

// Return the theme color palette tailored to the active compute backend
inline const ThemePalette& themePalette() {
    static const ThemePalette palette = [] {
        const std::string backend = WatermarkCore::getBackendName();
        if (backend == "cuda")
            return ThemePalette{QColor("#52eca8"), QColor("#20b26c"), QColor("#041d13")};
        if (backend == "eigen")
            return ThemePalette{QColor("#a855f7"), QColor("#7e22ce"), QColor("#140424")};
        return ThemePalette{QColor("#00f2ff"), QColor("#4a8eff"), QColor("#04181b")};
    }();
    return palette;
}
