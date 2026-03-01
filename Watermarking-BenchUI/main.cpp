#include "WatermarkingBenchUI.h"
#include <QFile>
#include <QFont>
#include <QFontDatabase>
#include <QIcon>
#include <QString>
#include <QtWidgets/QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/assets/watermark_icon.ico"));
    // apply the stylesheet
    QFile styleFile(":/stylesheets/main.qss");
    styleFile.open(QFile::ReadOnly);
    app.setStyleSheet(QString(styleFile.readAll()));
    // apply the font
    const int fontId = QFontDatabase::addApplicationFont(":/assets/fonts/TitilliumWeb-Regular.ttf");
    if (fontId != -1)
        app.setFont(QFont(QFontDatabase::applicationFontFamilies(fontId).at(0), 10));
    // show the main window
    WatermarkingBenchUI window;
    window.show();
    return app.exec();
}
