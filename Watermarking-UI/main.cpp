#include "WatermarkingUI.h"
#include "ThemePalette.hpp"
#include <QFile>
#include <QFont>
#include <QFontDatabase>
#include <QCursor>
#include <QGuiApplication>
#include <QIcon>
#include <QRect>
#include <QScreen>
#include <QString>
#include <QStringList>
#include <QtWidgets/QApplication>

/*!
 *  \brief  Application entry point and GUI initialization
 *  \author Dimitris Karatzas
 */

// Application entry point initializing Qt and launching the main window
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/assets/watermark_icon.ico"));
    // Share painted widget colors with the stylesheet
    QFile styleFile(":/stylesheets/main.qss");
    styleFile.open(QFile::ReadOnly);
    QString stylesheet = QString::fromUtf8(styleFile.readAll());
    const ThemePalette& palette = themePalette();
    stylesheet.replace("@ACCENT_HOVER@", palette.accent.lighter(110).name());
    stylesheet.replace("@ACCENT_LIGHT@", palette.accent.name());
    stylesheet.replace("@ACCENT_BORDER@", palette.accent.name());
    stylesheet.replace("@ACCENT_DARK@", palette.deep.name());
    stylesheet.replace("@ACCENT_INK@", palette.ink.name());
    stylesheet.replace("@ACCENT@", palette.accent.name());
    stylesheet.replace("@SURFACE@", palette.surface.name());
    stylesheet.replace("@SURFACE_HOVER@", palette.surfaceHover.name());
    stylesheet.replace("@TEXT@", palette.text.name());
    stylesheet.replace("@MUTED@", palette.muted.name());
    stylesheet.replace("@ERROR@", palette.error.name());
    stylesheet.replace("@WARNING@", palette.warning.name());
    stylesheet.replace("@CANVAS@", palette.canvas.name());
    app.setStyleSheet(stylesheet);
    // apply the font
    const int fontId = QFontDatabase::addApplicationFont(":/assets/fonts/SofiaSansSemiCondensed.ttf");
    if (fontId != -1) {
        const QStringList families = QFontDatabase::applicationFontFamilies(fontId);
        if (!families.isEmpty())
            app.setFont(QFont(families.first(), 10));
    }
    // show the main window
    WatermarkingUI window;
    window.show();
    // Center the main window on the active display monitor
    QScreen* startupScreen = QGuiApplication::screenAt(QCursor::pos());
    if (startupScreen == nullptr)
        startupScreen = QGuiApplication::primaryScreen();
    QRect frame = window.frameGeometry();
    frame.moveCenter(startupScreen->availableGeometry().center());
    window.move(frame.topLeft());
    return app.exec();
}
