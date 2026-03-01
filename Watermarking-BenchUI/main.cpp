#include "WatermarkingBenchUI.h"
#include <QtWidgets/QApplication>
#include <QFile>
#include <QFontDatabase>
#include <QIcon>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/assets/watermark_icon.ico"));

    QFile styleFile(":/stylesheets/main.qss");
    styleFile.open(QFile::ReadOnly);
    // apply the stylesheet
    app.setStyleSheet(QString(styleFile.readAll()));
    QFontDatabase::addApplicationFont(":/assets/fonts/TitilliumWeb-Regular.ttf");

    WatermarkingBenchUI window;
    window.show();
    return app.exec();
}
